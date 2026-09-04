"""Deterministic bounded maximum-likelihood training for the sparse candidate.

This library reads molecular targets only from the pretraining corpus. Held
truth is structurally outside the production process; prediction consumes only
an admitted target-free query snapshot.
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
    PredictionQueryIndex,
    RecordLocation,
    SparseShard,
)


REPORT_SCHEMA = "slp.sparse-training-report/v2"


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 731
    epochs: int = 12
    draws_per_epoch: int = 32
    batch_size: int = 8
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    prediction_batch_size: int = 64
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
            self.prediction_batch_size,
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
        numeric = (
            self.learning_rate,
            self.weight_decay,
            self.gradient_clip_norm,
            self.dropout,
        )
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in numeric
        ):
            raise ValueError("training real-valued settings must be finite numbers")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative and finite")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive and finite")
        if not 0 <= self.dropout <= 0.5:
            raise ValueError("dropout must be between zero and 0.5")


@dataclass(frozen=True)
class TrainingOutcome:
    model: SparseTypedWorldModel
    report: dict[str, Any]


@dataclass(frozen=True)
class PredictionBatch:
    profile_id: tuple[str, ...]
    source_id: tuple[str, ...]
    species_taxon: tuple[int, ...]
    centering_group: tuple[str, ...]
    perturbation_id: tuple[str, ...]
    intervention_ids: tuple[tuple[str, ...], ...]
    readout_ids: tuple[tuple[str, ...], ...]
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


def train_sparse_world(
    pretrain: CorpusIndex,
    config: TrainingConfig,
) -> TrainingOutcome:
    """Fit a fixed-epoch model from pretraining targets only."""

    if pretrain.role != "pretrain":
        raise ValueError("optimizer input must have the pretrain role")
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
        parameter_digest = model_parameter_sha256(model)
        report: dict[str, Any] = {
            "schema": REPORT_SCHEMA,
            "config": asdict(config),
            "modelConfig": model_config.as_dict(),
            "parameterCount": model.count_parameters(),
            "modelParameterSha256": parameter_digest,
            "corpora": {
                "pretrain": _corpus_identity(pretrain),
            },
            "isolation": {
                "pretrainRole": pretrain.role,
                "benchmarkLabelsPresent": False,
                "heldTruthAccessible": False,
                "predictionQueryUsedForOptimization": False,
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
    molecular_query: PredictionQueryIndex,
    batch_size: int = 64,
) -> Iterable[PredictionBatch]:
    """Yield bounded predictions from a structurally target-free query snapshot."""

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("prediction batch_size must be a positive integer")
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for shard_index in range(len(molecular_query.shards)):
                pending: list[dict[str, object]] = []
                for record in molecular_query.iter_records(shard_index):
                    pending.append(record)
                    if len(pending) < batch_size:
                        continue
                    batch = molecular_query.materialize(pending)
                    prediction = model(batch.world)
                    yield PredictionBatch(
                        profile_id=tuple(str(item["profileId"]) for item in pending),
                        source_id=tuple(str(item["sourceId"]) for item in pending),
                        species_taxon=tuple(int(item["speciesTaxon"]) for item in pending),
                        centering_group=tuple(str(item["centeringGroup"]) for item in pending),
                        perturbation_id=tuple(str(item["perturbationId"]) for item in pending),
                        intervention_ids=tuple(tuple(item["interventionIds"]) for item in pending),
                        readout_ids=tuple(tuple(item["readoutIds"]) for item in pending),
                        parameters=prediction.parameters.detach().clone(),
                        likelihood_type=prediction.likelihood_type.detach().clone(),
                        query_mask=prediction.query_mask.detach().clone(),
                    )
                    pending = []
                if pending:
                    batch = molecular_query.materialize(pending)
                    prediction = model(batch.world)
                    yield PredictionBatch(
                        profile_id=tuple(str(item["profileId"]) for item in pending),
                        source_id=tuple(str(item["sourceId"]) for item in pending),
                        species_taxon=tuple(int(item["speciesTaxon"]) for item in pending),
                        centering_group=tuple(str(item["centeringGroup"]) for item in pending),
                        perturbation_id=tuple(str(item["perturbationId"]) for item in pending),
                        intervention_ids=tuple(tuple(item["interventionIds"]) for item in pending),
                        readout_ids=tuple(tuple(item["readoutIds"]) for item in pending),
                        parameters=prediction.parameters.detach().clone(),
                        likelihood_type=prediction.likelihood_type.detach().clone(),
                        query_mask=prediction.query_mask.detach().clone(),
                    )
    finally:
        model.train(was_training)


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


def _corpus_identity(corpus: CorpusIndex) -> dict[str, Any]:
    return {
        "datasetId": corpus.dataset_id,
        "version": corpus.version,
        "role": corpus.role,
        "contentDigest": corpus.content_digest,
        "trajectoryGeneCount": len(corpus.trajectory_genes),
        "trajectoryGeneSetSha256": _canonical_sha256(
            sorted(corpus.trajectory_genes)
        ),
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
