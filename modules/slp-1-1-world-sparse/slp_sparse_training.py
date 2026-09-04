"""Deterministic bounded maximum-likelihood training for the sparse candidate.

This is a library boundary, not an OMF checkpoint boundary. It uses only the
pretraining corpus for optimizer updates, evaluates only the molecular-
validation corpus, and never performs validation-selected stopping.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import platform
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from slp_sparse_architecture import (
    SparseTypedWorldModel,
    WorldPrediction,
    negative_log_likelihood_terms,
)
from slp_sparse_corpus import (
    CorpusIndex,
    DeterministicHierarchicalSampler,
    MaterializedBatch,
    RecordLocation,
    SparseShard,
)


REPORT_SCHEMA = "slp.sparse-training-report/v1"


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 731
    epochs: int = 12
    draws_per_epoch: int = 32
    batch_size: int = 8
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    evaluation_batch_size: int = 64
    d_model: int = 16
    nhead: int = 4
    encoder_layers: int = 1
    decoder_layers: int = 1
    ffn_multiplier: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        positive_ints = (
            self.epochs,
            self.draws_per_epoch,
            self.batch_size,
            self.evaluation_batch_size,
            self.d_model,
            self.nhead,
            self.encoder_layers,
            self.decoder_layers,
            self.ffn_multiplier,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in positive_ints
        ):
            raise ValueError("training counts and model dimensions must be positive integers")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative and finite")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive and finite")
        if not 0 <= self.dropout <= 0.5:
            raise ValueError("dropout must be between zero and 0.5")


@dataclass(frozen=True)
class TrainingOutcome:
    model: SparseTypedWorldModel
    report: dict[str, Any]


@dataclass(frozen=True)
class PredictionBatch:
    record_id: tuple[str, ...]
    query_id: tuple[tuple[str, ...], ...]
    parameters: torch.Tensor
    likelihood_type: torch.Tensor
    query_mask: torch.Tensor


@dataclass
class _ShardAccessAudit:
    loads_by_phase: dict[str, int]
    max_records_loaded: int = 0
    max_target_values_loaded: int = 0

    @classmethod
    def create(cls) -> "_ShardAccessAudit":
        return cls(loads_by_phase=defaultdict(int))

    def load(self, corpus: CorpusIndex, shard_index: int, phase: str) -> SparseShard:
        reference = corpus.shards[shard_index]
        if reference.records > corpus.bounds["maxRecordsPerShard"]:
            raise ValueError("shard record bound drifted after corpus admission")
        if reference.target_values > (
            reference.records * corpus.bounds["maxTargetsPerRecord"]
        ):
            raise ValueError("shard target bound drifted after corpus admission")
        shard = corpus.load_shard(shard_index)
        self.loads_by_phase[phase] += 1
        self.max_records_loaded = max(self.max_records_loaded, shard.reference.records)
        self.max_target_values_loaded = max(
            self.max_target_values_loaded, shard.reference.target_values
        )
        return shard


@dataclass(frozen=True)
class _Evaluation:
    nll_sum: float
    observed_targets: int
    by_source: dict[str, tuple[float, int]]
    by_species: dict[str, tuple[float, int]]
    by_species_source: dict[str, tuple[float, int]]
    prediction_sha256: str

    @property
    def per_observed_target_nll(self) -> float:
        return self.nll_sum / self.observed_targets


def train_sparse_world(
    pretrain: CorpusIndex,
    molecular_validation: CorpusIndex,
    config: TrainingConfig,
) -> TrainingOutcome:
    """Fit a fixed-epoch candidate and return deterministic in-memory evidence."""

    _validate_corpus_boundary(pretrain, molecular_validation)
    model_config = pretrain.world_config(
        d_model=config.d_model,
        nhead=config.nhead,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        ffn_multiplier=config.ffn_multiplier,
        dropout=config.dropout,
    )
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_threads = torch.get_num_threads()
    previous_cpu_rng = torch.random.get_rng_state()
    try:
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        torch.default_generator.manual_seed(config.seed)
        model = SparseTypedWorldModel(model_config).cpu()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        access = _ShardAccessAudit.create()
        sampler, location_source = _build_pretrain_sampler(pretrain, config.seed, access)
        validation_before = _evaluate(
            model,
            molecular_validation,
            config.evaluation_batch_size,
            access,
            "validation-before",
        )
        epoch_mean_record_nll: list[float] = []
        epoch_schedule_sha256: list[str] = []
        source_draws = {source: 0 for source in pretrain.sources}
        source_draws_by_epoch: list[dict[str, int]] = []
        for epoch in range(config.epochs):
            schedule = _epoch_schedule(sampler, config.draws_per_epoch, epoch)
            epoch_schedule_sha256.append(_schedule_sha256(schedule))
            epoch_source_draws = _scheduled_source_counts(
                schedule, location_source, pretrain.sources
            )
            source_draws_by_epoch.append(epoch_source_draws)
            for source, count in epoch_source_draws.items():
                source_draws[source] += count
            model.train()
            total_record_nll = 0.0
            total_records = 0
            for start in range(0, len(schedule), config.batch_size):
                batch_locations = schedule[start : start + config.batch_size]
                record_nll_sum, records = _train_batch(
                    model,
                    optimizer,
                    pretrain,
                    batch_locations,
                    config.gradient_clip_norm,
                    access,
                    f"pretrain-epoch-{epoch}",
                )
                total_record_nll += record_nll_sum
                total_records += records
            if total_records != config.draws_per_epoch:
                raise ValueError("pretraining schedule record count drifted")
            epoch_mean_record_nll.append(total_record_nll / total_records)
        validation_after = _evaluate(
            model,
            molecular_validation,
            config.evaluation_batch_size,
            access,
            "validation-after",
        )
        parameter_digest = model_parameter_sha256(model)
        validation_report = _comparison(validation_before, validation_after)
        report: dict[str, Any] = {
            "schema": REPORT_SCHEMA,
            "config": asdict(config),
            "modelConfig": model_config.as_dict(),
            "parameterCount": model.count_parameters(),
            "modelParameterSha256": parameter_digest,
            "validationPredictionSha256": validation_after.prediction_sha256,
            "corpora": {
                "pretrain": _corpus_identity(pretrain),
                "molecularValidation": _corpus_identity(molecular_validation),
            },
            "isolation": {
                "pretrainRole": pretrain.role,
                "validationRole": molecular_validation.role,
                "trajectoryGeneOverlap": [],
                "benchmarkLabelsPresent": False,
                "validationUsedForOptimization": False,
                "selection": "fixed-final-epoch",
            },
            "training": {
                "objective": {
                    "name": "mean-per-record-observed-typed-nll",
                    "scheduledRecordWeighting": "equal",
                    "withinRecordTargetWeighting": "equal-observed-target",
                    "scheduleHierarchy": "source-perturbation-replicate-record",
                },
                "epochOptimizationMeanRecordNll": epoch_mean_record_nll,
                "epochScheduleSha256": epoch_schedule_sha256,
                "sourceDraws": source_draws,
                "sourceDrawsByEpoch": source_draws_by_epoch,
                "optimizer": "AdamW",
            },
            "validation": validation_report,
            "streaming": {
                "denseRecordByDictionaryTargetAllocated": False,
                "maxShardRecordsLoaded": access.max_records_loaded,
                "maxShardTargetValuesLoaded": access.max_target_values_loaded,
                "shardLoadsByPhase": dict(sorted(access.loads_by_phase.items())),
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "device": "cpu",
                "deterministicAlgorithms": True,
                "threads": 1,
            },
            "checkpointProduced": False,
        }
        _validate_finite_report(report)
        report["reportSha256"] = _canonical_sha256(report)
        return TrainingOutcome(model=model, report=report)
    finally:
        torch.random.set_rng_state(previous_cpu_rng)
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(
            previous_deterministic, warn_only=previous_warn_only
        )


def model_parameter_sha256(model: SparseTypedWorldModel) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0" + value.tobytes(order="C"))
    return digest.hexdigest()


def equal_record_negative_log_likelihood(
    prediction: WorldPrediction,
    target: torch.Tensor,
    target_observed: torch.Tensor,
) -> torch.Tensor:
    """Mean each record's observed typed NLL, then weight records equally."""

    terms = negative_log_likelihood_terms(prediction, target, target_observed)
    if terms.ndim != 2:
        raise ValueError("record-balanced likelihood requires [record, query] tensors")
    observed_per_record = target_observed.sum(dim=1)
    if (observed_per_record <= 0).any():
        raise ValueError("every optimized record requires an observed target")
    record_nll = terms.sum(dim=1) / observed_per_record.to(terms.dtype)
    return record_nll.mean()


def iter_sparse_predictions(
    model: SparseTypedWorldModel,
    molecular_validation: CorpusIndex,
    batch_size: int = 64,
) -> Iterable[PredictionBatch]:
    """Yield bounded deterministic validation predictions with provenance IDs."""

    if molecular_validation.role != "molecular-validation":
        raise ValueError("prediction may read only the molecular-validation role")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("prediction batch_size must be a positive integer")
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for shard_index in range(len(molecular_validation.shards)):
                shard = molecular_validation.load_shard(shard_index)
                for start in range(0, shard.records, batch_size):
                    rows = list(range(start, min(start + batch_size, shard.records)))
                    batch = molecular_validation.materialize_batch(shard, rows)
                    prediction = model(batch.world)
                    query_ids: list[tuple[str, ...]] = []
                    for row in rows:
                        panel = int(shard.arrays["query_panel_index"][row])
                        panel_start = int(molecular_validation.panel_indptr[panel])
                        panel_stop = int(molecular_validation.panel_indptr[panel + 1])
                        query_ids.append(
                            tuple(
                                str(molecular_validation.query_id[int(item)])
                                for item in molecular_validation.panel_query_index[
                                    panel_start:panel_stop
                                ]
                            )
                        )
                    yield PredictionBatch(
                        record_id=batch.provenance.record_id,
                        query_id=tuple(query_ids),
                        parameters=prediction.parameters.detach().clone(),
                        likelihood_type=prediction.likelihood_type.detach().clone(),
                        query_mask=prediction.query_mask.detach().clone(),
                    )
    finally:
        model.train(was_training)


def _validate_corpus_boundary(
    pretrain: CorpusIndex, molecular_validation: CorpusIndex
) -> None:
    if pretrain.role != "pretrain":
        raise ValueError("optimizer input must have the pretrain role")
    if molecular_validation.role != "molecular-validation":
        raise ValueError("evaluation input must have the molecular-validation role")
    overlap = sorted(pretrain.trajectory_genes & molecular_validation.trajectory_genes)
    if overlap:
        raise ValueError(
            "validation intervention genes occur in pretrain quantitative trajectories: "
            + ", ".join(overlap)
        )
    compatibility = (
        "entity_feature_dim",
        "species_feature_dim",
        "entity_types",
        "context_types",
        "action_types",
        "covariates",
        "readouts",
        "feature_pack_revision",
        "feature_pack_sha256",
        "normalization_id",
        "value_space",
    )
    mismatches = [
        name
        for name in compatibility
        if getattr(pretrain, name) != getattr(molecular_validation, name)
    ]
    if mismatches:
        raise ValueError(
            "pretrain and molecular-validation model contracts differ: "
            + ", ".join(mismatches)
        )
    for taxon in sorted(
        set(pretrain.species_taxa) & set(molecular_validation.species_taxa)
    ):
        if (
            pretrain.species_feature_value[taxon]
            != molecular_validation.species_feature_value[taxon]
            or pretrain.species_feature_present[taxon]
            != molecular_validation.species_feature_present[taxon]
        ):
            raise ValueError(
                f"species feature contract differs for shared taxon {taxon}"
            )


def _build_pretrain_sampler(
    corpus: CorpusIndex, seed: int, access: _ShardAccessAudit
) -> tuple[DeterministicHierarchicalSampler, dict[RecordLocation, int]]:
    locations: list[RecordLocation] = []
    sources: list[int] = []
    perturbations: list[str] = []
    replicates: list[str] = []
    location_source: dict[RecordLocation, int] = {}
    for shard_index in range(len(corpus.shards)):
        shard = access.load(corpus, shard_index, "pretrain-sampler")
        arrays = shard.arrays
        for row_index in range(shard.records):
            location = RecordLocation(shard_index, row_index)
            source = int(arrays["source_index"][row_index])
            locations.append(location)
            sources.append(source)
            perturbations.append(str(arrays["perturbation_id"][row_index]))
            replicates.append(str(arrays["replicate_id"][row_index]))
            location_source[location] = source
    sampler = DeterministicHierarchicalSampler(
        sources,
        perturbations,
        replicates,
        corpus.source_weights,
        seed,
        locations,
    )
    return sampler, location_source


def _scheduled_source_counts(
    schedule: Sequence[RecordLocation],
    location_source: dict[RecordLocation, int],
    source_names: Sequence[str],
) -> dict[str, int]:
    counts = {name: 0 for name in source_names}
    for location in schedule:
        counts[source_names[location_source[location]]] += 1
    return counts


def _epoch_schedule(
    sampler: DeterministicHierarchicalSampler, draws: int, epoch: int
) -> tuple[RecordLocation, ...]:
    seed_document = json.dumps(
        {
            "domain": "slp.sparse-training-epoch-schedule/v1",
            "baseSeed": sampler.seed,
            "epoch": epoch,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    epoch_seed = int.from_bytes(hashlib.sha256(seed_document).digest()[:8], "big")
    epoch_sampler = DeterministicHierarchicalSampler(
        sampler.source_index,
        sampler.perturbation_id,
        sampler.replicate_id,
        sampler.source_weights,
        epoch_seed,
        sampler.locations,
    )
    return epoch_sampler.schedule(draws)


def _schedule_sha256(schedule: Sequence[RecordLocation]) -> str:
    document = [
        [location.shard_index, location.row_index] for location in schedule
    ]
    return _canonical_sha256(document)


def _train_batch(
    model: SparseTypedWorldModel,
    optimizer: torch.optim.Optimizer,
    corpus: CorpusIndex,
    locations: Sequence[RecordLocation],
    gradient_clip_norm: float,
    access: _ShardAccessAudit,
    phase: str,
) -> tuple[float, int]:
    rows_by_shard: dict[int, list[int]] = defaultdict(list)
    for location in locations:
        rows_by_shard[location.shard_index].append(location.row_index)
    batches: list[MaterializedBatch] = []
    for shard_index in sorted(rows_by_shard):
        shard = access.load(corpus, shard_index, phase)
        batch = corpus.materialize_batch(shard, rows_by_shard[shard_index])
        batches.append(batch)
    total_records = len(locations)
    if total_records <= 0:
        raise ValueError("pretraining batch has no scheduled records")
    optimizer.zero_grad(set_to_none=True)
    record_nll_sum = 0.0
    for batch in batches:
        records = int(batch.target_observed.shape[0])
        loss = equal_record_negative_log_likelihood(
            model(batch.world), batch.target_value, batch.target_observed
        )
        (loss * (records / total_records)).backward()
        record_nll_sum += float(loss.detach().item()) * records
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    return record_nll_sum, total_records


def _evaluate(
    model: SparseTypedWorldModel,
    corpus: CorpusIndex,
    batch_size: int,
    access: _ShardAccessAudit,
    phase: str,
) -> _Evaluation:
    if corpus.role != "molecular-validation":
        raise ValueError("evaluation may read only the molecular-validation role")
    digest = hashlib.sha256()
    digest.update(b"slp.sparse-predictions/v1\0")
    total_sum = 0.0
    total_count = 0
    by_source: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0])
    by_species: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0])
    by_species_source: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0])
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for shard_index in range(len(corpus.shards)):
                shard = access.load(corpus, shard_index, phase)
                for start in range(0, shard.records, batch_size):
                    rows = list(range(start, min(start + batch_size, shard.records)))
                    batch = corpus.materialize_batch(shard, rows)
                    prediction = model(batch.world)
                    terms = negative_log_likelihood_terms(
                        prediction, batch.target_value, batch.target_observed
                    )
                    _update_prediction_digest(
                        digest, corpus, shard, rows, batch, prediction.parameters
                    )
                    for local_row in range(len(rows)):
                        observed = batch.target_observed[local_row]
                        count = int(observed.sum().item())
                        value = float(terms[local_row][observed].sum().item())
                        source_index = int(batch.provenance.source_index[local_row].item())
                        source = corpus.sources[source_index]
                        species = str(
                            int(batch.provenance.species_taxon[local_row].item())
                        )
                        joint = f"{species}|{source}"
                        total_sum += value
                        total_count += count
                        _accumulate(by_source[source], value, count)
                        _accumulate(by_species[species], value, count)
                        _accumulate(by_species_source[joint], value, count)
    finally:
        model.train(was_training)
    if total_count <= 0:
        raise ValueError("molecular validation has no observed targets")
    return _Evaluation(
        nll_sum=total_sum,
        observed_targets=total_count,
        by_source=_freeze_groups(by_source),
        by_species=_freeze_groups(by_species),
        by_species_source=_freeze_groups(by_species_source),
        prediction_sha256=digest.hexdigest(),
    )


def _update_prediction_digest(
    digest: Any,
    corpus: CorpusIndex,
    shard: SparseShard,
    rows: Sequence[int],
    batch: MaterializedBatch,
    parameters: torch.Tensor,
) -> None:
    identifiers: list[dict[str, Any]] = []
    for row in rows:
        panel = int(shard.arrays["query_panel_index"][row])
        start = int(corpus.panel_indptr[panel])
        stop = int(corpus.panel_indptr[panel + 1])
        query_indices = corpus.panel_query_index[start:stop]
        identifiers.append(
            {
                "recordId": str(shard.arrays["record_id"][row]),
                "queryIds": [str(corpus.query_id[int(item)]) for item in query_indices],
            }
        )
    digest.update(
        json.dumps(identifiers, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for tensor in (
        parameters,
        batch.world.likelihood_type,
        batch.world.query_mask,
    ):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0" + value.tobytes(order="C"))


def _comparison(before: _Evaluation, after: _Evaluation) -> dict[str, Any]:
    if before.observed_targets != after.observed_targets:
        raise ValueError("validation target population changed during training")

    def compare_groups(
        earlier: dict[str, tuple[float, int]], later: dict[str, tuple[float, int]]
    ) -> dict[str, dict[str, float | int]]:
        if set(earlier) != set(later):
            raise ValueError("validation strata changed during training")
        result: dict[str, dict[str, float | int]] = {}
        for key in sorted(earlier):
            before_sum, before_count = earlier[key]
            after_sum, after_count = later[key]
            if before_count != after_count or before_count <= 0:
                raise ValueError("validation stratum target count changed")
            initialization_nll = before_sum / before_count
            final_nll = after_sum / after_count
            result[key] = {
                "observedTargets": before_count,
                "initializationPerObservedTargetNll": initialization_nll,
                "finalPerObservedTargetNll": final_nll,
                "descriptiveImprovement": initialization_nll - final_nll,
            }
        return result

    by_source = compare_groups(before.by_source, after.by_source)
    by_species = compare_groups(before.by_species, after.by_species)
    by_species_source = compare_groups(
        before.by_species_source, after.by_species_source
    )
    return {
        "metric": "mean-nll-per-observed-molecular-target",
        "comparison": "descriptive-initialization-to-fixed-final-epoch",
        "decisionUse": "frozen-molecular-gate-only",
        "scientificBaselineComparison": False,
        "overall": {
            "observedTargets": before.observed_targets,
            "initializationPerObservedTargetNll": before.per_observed_target_nll,
            "finalPerObservedTargetNll": after.per_observed_target_nll,
            "descriptiveImprovement": (
                before.per_observed_target_nll - after.per_observed_target_nll
            ),
        },
        "bySource": by_source,
        "bySpecies": by_species,
        "bySpeciesSource": by_species_source,
        "minimumSourceDescriptiveImprovement": min(
            float(value["descriptiveImprovement"]) for value in by_source.values()
        ),
        "minimumSpeciesDescriptiveImprovement": min(
            float(value["descriptiveImprovement"]) for value in by_species.values()
        ),
    }


def _corpus_identity(corpus: CorpusIndex) -> dict[str, Any]:
    return {
        "datasetId": corpus.dataset_id,
        "version": corpus.version,
        "role": corpus.role,
        "contentDigest": corpus.content_digest,
        "trajectoryGenes": sorted(corpus.trajectory_genes),
    }


def _accumulate(accumulator: list[float | int], value: float, count: int) -> None:
    accumulator[0] = float(accumulator[0]) + value
    accumulator[1] = int(accumulator[1]) + count


def _freeze_groups(
    value: dict[str, list[float | int]]
) -> dict[str, tuple[float, int]]:
    return {
        key: (float(items[0]), int(items[1])) for key, items in sorted(value.items())
    }


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_finite_report(value: object) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _validate_finite_report(item)
    elif isinstance(value, list):
        for item in value:
            _validate_finite_report(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("training report contains a non-finite number")
