"""Shard-streaming maximum-likelihood and molecular-reward training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn

from architecture import SpeciesAwareWorldModel, WorldBatch, WorldConfig


REQUIRED_ARRAYS = {
    "record_id",
    "source_id",
    "perturbation_id",
    "context_features",
    "context_mask",
    "action_features",
    "action_covariates",
    "action_curies",
    "action_mask",
    "query_features",
    "query_mask",
    "readout_type",
    "species_features",
    "species_taxon",
    "target",
    "target_mask",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _document_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _relative_file(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("corpus paths must be non-empty strings")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe corpus path: {value!r}")
    path = root.joinpath(*relative.parts)
    if not path.is_file():
        raise ValueError(f"missing corpus file: {value}")
    return path


@dataclass(frozen=True)
class CorpusIndex:
    root: Path
    role: str
    dataset_id: str
    version: str
    entity_feature_dim: int
    species_feature_dim: int
    action_covariate_dim: int
    readout_types: tuple[str, ...]
    species_taxa: tuple[int, ...]
    species_feature_vectors: dict[int, tuple[float, ...]]
    trajectory_genes: frozenset[str]
    shards: tuple[dict[str, object], ...]
    identity: dict[str, Any]

    @classmethod
    def load(cls, root: str | Path, role: str) -> "CorpusIndex":
        root = Path(root).resolve()
        manifest_path = root / "corpus.json"
        if not manifest_path.is_file():
            raise ValueError("snapshot must contain corpus.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "slp.corpus/v1" or manifest.get("role") != role:
            raise ValueError(f"expected an slp.corpus/v1 {role!r} snapshot")
        if manifest.get("benchmarkLabelsPresent") is not False:
            raise ValueError("benchmark labels are forbidden in a world-model corpus")
        dataset_id = manifest.get("datasetId")
        version = manifest.get("version")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("datasetId must be non-empty")
        if not isinstance(version, str) or not version:
            raise ValueError("version must be non-empty")
        feature_dim = manifest.get("entityFeatureDim")
        species_dim = manifest.get("speciesFeatureDim")
        covariate_dim = manifest.get("actionCovariateDim")
        readouts = manifest.get("readoutTypes")
        if not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("entityFeatureDim must be positive")
        if not isinstance(species_dim, int) or species_dim <= 0:
            raise ValueError("speciesFeatureDim must be positive")
        if not isinstance(covariate_dim, int) or covariate_dim <= 0:
            raise ValueError("actionCovariateDim must be positive")
        if not isinstance(readouts, list) or not readouts or any(not item for item in readouts):
            raise ValueError("readoutTypes must be a non-empty string list")
        if len(readouts) != len(set(readouts)):
            raise ValueError("readoutTypes must be unique")
        taxa = manifest.get("speciesTaxa")
        if (
            not isinstance(taxa, list)
            or not taxa
            or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in taxa)
            or len(taxa) != len(set(taxa))
        ):
            raise ValueError("speciesTaxa must contain unique positive taxonomy IDs")
        raw_vectors = manifest.get("speciesFeatureVectors")
        if not isinstance(raw_vectors, dict) or set(raw_vectors) != {str(item) for item in taxa}:
            raise ValueError("speciesFeatureVectors must define every declared taxon exactly once")
        species_vectors: dict[int, tuple[float, ...]] = {}
        for taxon in taxa:
            vector = raw_vectors[str(taxon)]
            if (
                not isinstance(vector, list)
                or len(vector) != species_dim
                or any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in vector)
                or not np.isfinite(np.asarray(vector, dtype=np.float64)).all()
            ):
                raise ValueError("species feature vectors must be finite and match speciesFeatureDim")
            species_vectors[taxon] = tuple(float(item) for item in vector)
        gene_path = _relative_file(root, manifest.get("trajectoryGenes"))
        genes = frozenset(
            line.strip()
            for line in gene_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if not genes or any(":" not in gene for gene in genes):
            raise ValueError("trajectory genes must be stable CURIE identifiers")
        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError("corpus must contain at least one shard")
        verified_shards: list[dict[str, object]] = []
        seen_paths: set[str] = set()
        for shard in shards:
            if not isinstance(shard, dict) or set(shard) != {"path", "sha256", "records"}:
                raise ValueError("each shard requires only path, sha256, and records")
            path_value = shard["path"]
            if not isinstance(path_value, str) or path_value in seen_paths:
                raise ValueError("shard paths must be unique strings")
            seen_paths.add(path_value)
            path = _relative_file(root, path_value)
            expected_digest = shard["sha256"]
            actual_digest = _sha256(path)
            if (
                not isinstance(expected_digest, str)
                or len(expected_digest) != 64
                or actual_digest != expected_digest
            ):
                raise ValueError(f"shard digest mismatch: {path_value}")
            records = shard["records"]
            if not isinstance(records, int) or isinstance(records, bool) or records <= 0:
                raise ValueError("shard records must be positive integers")
            verified_shards.append(
                {"path": path_value, "sha256": actual_digest, "records": records}
            )
        identity: dict[str, Any] = {
            "datasetId": dataset_id,
            "version": version,
            "role": role,
            "manifestSha256": _sha256(manifest_path),
            "trajectoryGenesSha256": _sha256(gene_path),
            "shards": verified_shards,
        }
        identity["contentDigest"] = _document_digest(identity)
        return cls(
            root=root,
            role=role,
            dataset_id=dataset_id,
            version=version,
            entity_feature_dim=feature_dim,
            species_feature_dim=species_dim,
            action_covariate_dim=covariate_dim,
            readout_types=tuple(readouts),
            species_taxa=tuple(taxa),
            species_feature_vectors=species_vectors,
            trajectory_genes=genes,
            shards=tuple(verified_shards),
            identity=identity,
        )

    def shard_path(self, shard: dict[str, object]) -> Path:
        return _relative_file(self.root, shard["path"])


def _validate_compatible(*corpora: CorpusIndex) -> None:
    first = corpora[0]
    for corpus in corpora[1:]:
        if (
            corpus.entity_feature_dim != first.entity_feature_dim
            or corpus.species_feature_dim != first.species_feature_dim
            or corpus.action_covariate_dim != first.action_covariate_dim
            or corpus.readout_types != first.readout_types
        ):
            raise ValueError("corpus feature and readout contracts do not match")


def _load_shard(corpus: CorpusIndex, shard: dict[str, object]) -> dict[str, np.ndarray]:
    path = corpus.shard_path(shard)
    if _sha256(path) != shard["sha256"]:
        raise ValueError(f"shard digest drifted after admission: {shard['path']}")
    with np.load(path, allow_pickle=False) as source:
        missing = sorted(REQUIRED_ARRAYS - set(source.files))
        if missing:
            raise ValueError(f"shard is missing arrays: {', '.join(missing)}")
        arrays = {name: source[name] for name in REQUIRED_ARRAYS}
    records = int(shard["records"])
    if any(array.shape[0] != records for array in arrays.values()):
        raise ValueError("shard record count does not match corpus.json")
    _validate_shard_arrays(corpus, arrays, records)
    return arrays


def _validate_shard_arrays(
    corpus: CorpusIndex,
    arrays: dict[str, np.ndarray],
    records: int,
) -> None:
    context = arrays["context_features"]
    action = arrays["action_features"]
    query = arrays["query_features"]
    if context.ndim != 3 or context.shape[2] != corpus.entity_feature_dim:
        raise ValueError("context_features must have shape [records, context, entityFeatureDim]")
    if action.ndim != 3 or action.shape[2] != corpus.entity_feature_dim:
        raise ValueError("action_features must have shape [records, actions, entityFeatureDim]")
    if query.ndim != 3 or query.shape[2] != corpus.entity_feature_dim:
        raise ValueError("query_features must have shape [records, queries, entityFeatureDim]")
    expected_shapes = {
        "record_id": (records,),
        "source_id": (records,),
        "perturbation_id": (records,),
        "context_mask": context.shape[:2],
        "action_covariates": (*action.shape[:2], corpus.action_covariate_dim),
        "action_curies": action.shape[:2],
        "action_mask": action.shape[:2],
        "query_mask": query.shape[:2],
        "readout_type": query.shape[:2],
        "species_features": (records, corpus.species_feature_dim),
        "species_taxon": (records,),
        "target": query.shape[:2],
        "target_mask": query.shape[:2],
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise ValueError(f"{name} has shape {arrays[name].shape}, expected {expected}")
    for name in ("context_mask", "action_mask", "query_mask", "target_mask"):
        if arrays[name].dtype.kind != "b":
            raise ValueError(f"{name} must have boolean dtype")
    for name in (
        "context_features",
        "action_features",
        "action_covariates",
        "query_features",
        "species_features",
        "target",
    ):
        value = arrays[name]
        if value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite floating-point values")
    for name in ("species_taxon", "readout_type"):
        if arrays[name].dtype.kind not in "iu":
            raise ValueError(f"{name} must have integer dtype")
    for name in ("record_id", "source_id", "perturbation_id", "action_curies"):
        if arrays[name].dtype.kind not in "US":
            raise ValueError(f"{name} must use a fixed-width string dtype")
    query_mask = arrays["query_mask"]
    target_mask = arrays["target_mask"]
    if (
        not np.all(arrays["context_mask"].any(axis=1))
        or not np.all(arrays["action_mask"].any(axis=1))
        or not np.all(query_mask.any(axis=1))
    ):
        raise ValueError("every record requires at least one context, action, and query token")
    if np.any(target_mask & ~query_mask):
        raise ValueError("target_mask cannot observe a padded query")
    active_readouts = arrays["readout_type"][query_mask]
    if np.any(active_readouts < 0) or np.any(active_readouts >= len(corpus.readout_types)):
        raise ValueError("active readout_type values are outside the declared vocabulary")
    taxa = arrays["species_taxon"].astype(np.int64, copy=False)
    if not set(taxa.tolist()).issubset(set(corpus.species_taxa)):
        raise ValueError("species_taxon contains a taxon absent from corpus.json")
    for row, taxon in enumerate(taxa.tolist()):
        expected = np.asarray(corpus.species_feature_vectors[int(taxon)], dtype=np.float64)
        if not np.allclose(arrays["species_features"][row], expected, rtol=0.0, atol=1e-6):
            raise ValueError("species_features do not match species_taxon and corpus.json")
    for name in ("record_id", "source_id", "perturbation_id"):
        values = np.char.strip(arrays[name].astype(str))
        if np.any(values == ""):
            raise ValueError(f"{name} values must be non-empty")
    action_ids = np.char.strip(arrays["action_curies"].astype(str))
    action_mask = arrays["action_mask"]
    if np.any(action_ids[~action_mask] != ""):
        raise ValueError("padded action_curies must be empty")
    active_actions = action_ids[action_mask]
    if np.any(active_actions == "") or any(":" not in item for item in active_actions.tolist()):
        raise ValueError("active action_curies must be stable CURIE identifiers")


def _validate_corpus(corpus: CorpusIndex) -> None:
    record_ids: set[str] = set()
    action_ids: set[str] = set()
    observed_targets = 0
    row_taxa: set[int] = set()
    targets_by_taxon = {taxon: 0 for taxon in corpus.species_taxa}
    for shard in corpus.shards:
        arrays = _load_shard(corpus, shard)
        shard_records = arrays["record_id"].astype(str).tolist()
        duplicate = record_ids.intersection(shard_records)
        if duplicate or len(shard_records) != len(set(shard_records)):
            raise ValueError("record_id values must be unique across a corpus")
        record_ids.update(shard_records)
        action_ids.update(
            np.char.strip(arrays["action_curies"].astype(str))[arrays["action_mask"]].tolist()
        )
        observed_mask = arrays["target_mask"] & arrays["query_mask"]
        observed_targets += int(np.count_nonzero(observed_mask))
        for row, taxon in enumerate(arrays["species_taxon"].astype(np.int64).tolist()):
            row_taxa.add(int(taxon))
            targets_by_taxon[int(taxon)] += int(np.count_nonzero(observed_mask[row]))
    if action_ids != set(corpus.trajectory_genes):
        missing = sorted(set(corpus.trajectory_genes) - action_ids)
        undeclared = sorted(action_ids - set(corpus.trajectory_genes))
        raise ValueError(
            "trajectoryGenes does not exactly match shard action_curies "
            f"(missing={missing[:5]}, undeclared={undeclared[:5]})"
        )
    if observed_targets == 0:
        raise ValueError("corpus has no observed molecular targets")
    if row_taxa != set(corpus.species_taxa):
        raise ValueError("every declared species taxon must occur in the corpus")
    if any(count == 0 for count in targets_by_taxon.values()):
        raise ValueError("every represented species requires an observed molecular target")


def validate_audit_binding(
    corpora: dict[str, CorpusIndex],
    audit: object,
) -> None:
    if (
        not isinstance(audit, dict)
        or audit.get("schema") != "slp.corpus-audit/v1"
        or audit.get("auditPassed") is not True
        or audit.get("strictInterventionIsolation") is not True
    ):
        raise ValueError("a passing strict corpus audit is required before training")
    datasets = audit.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(corpora):
        raise ValueError("corpus audit does not attest the exact training inputs")
    for name, corpus in corpora.items():
        attested = datasets.get(name)
        if not isinstance(attested, dict):
            raise ValueError(f"corpus audit is missing {name}")
        for field, expected in corpus.identity.items():
            if attested.get(field) != expected:
                raise ValueError(f"corpus audit identity mismatch for {name}.{field}")


def iter_batches(
    corpus: CorpusIndex,
    batch_size: int,
    *,
    seed: int,
    shuffle: bool,
) -> Iterator[tuple[WorldBatch, torch.Tensor, torch.Tensor, torch.Tensor]]:
    generator = np.random.default_rng(seed)
    shard_order = generator.permutation(len(corpus.shards)) if shuffle else np.arange(len(corpus.shards))
    for shard_id in shard_order:
        arrays = _load_shard(corpus, corpus.shards[int(shard_id)])
        records = len(arrays["target"])
        order = generator.permutation(records) if shuffle else np.arange(records)
        for start in range(0, records, batch_size):
            index = order[start : start + batch_size]
            batch = WorldBatch(
                context_features=torch.as_tensor(arrays["context_features"][index], dtype=torch.float32),
                context_mask=torch.as_tensor(arrays["context_mask"][index], dtype=torch.bool),
                action_features=torch.as_tensor(arrays["action_features"][index], dtype=torch.float32),
                action_covariates=torch.as_tensor(
                    arrays["action_covariates"][index], dtype=torch.float32
                ),
                action_mask=torch.as_tensor(arrays["action_mask"][index], dtype=torch.bool),
                query_features=torch.as_tensor(arrays["query_features"][index], dtype=torch.float32),
                query_mask=torch.as_tensor(arrays["query_mask"][index], dtype=torch.bool),
                readout_type=torch.as_tensor(arrays["readout_type"][index], dtype=torch.long),
                species_features=torch.as_tensor(
                    arrays["species_features"][index], dtype=torch.float32
                ),
            )
            yield (
                batch,
                torch.as_tensor(arrays["target"][index], dtype=torch.float32),
                torch.as_tensor(arrays["target_mask"][index], dtype=torch.bool),
                torch.as_tensor(arrays["species_taxon"][index], dtype=torch.long),
            )


def _to_device(batch: WorldBatch, device: torch.device) -> WorldBatch:
    return WorldBatch(**{name: value.to(device) for name, value in batch.__dict__.items()})


def _gaussian_nll(
    mean: torch.Tensor,
    log_scale: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return (
        0.5 * ((target - mean) * torch.exp(-log_scale)).square()
        + log_scale
        + 0.5 * math.log(2.0 * math.pi)
    )


def _linear_features(batch: WorldBatch, readout_types: int) -> torch.Tensor:
    context_weight = batch.context_mask.to(batch.context_features.dtype).unsqueeze(-1)
    action_weight = batch.action_mask.to(batch.action_features.dtype).unsqueeze(-1)
    context = (batch.context_features * context_weight).sum(1) / context_weight.sum(1).clamp_min(1)
    actions = (batch.action_features * action_weight).sum(1) / action_weight.sum(1).clamp_min(1)
    covariates = (batch.action_covariates * action_weight).sum(1) / action_weight.sum(1).clamp_min(1)
    fixed = torch.cat((context, actions, covariates, batch.species_features), dim=-1)
    fixed = fixed[:, None, :].expand(-1, batch.query_features.shape[1], -1)
    readout = torch.nn.functional.one_hot(
        batch.readout_type.clamp(0, readout_types - 1), num_classes=readout_types
    ).to(batch.query_features.dtype)
    intercept = torch.ones(
        (*batch.query_features.shape[:2], 1),
        dtype=batch.query_features.dtype,
        device=batch.query_features.device,
    )
    return torch.cat((fixed, batch.query_features, readout, intercept), dim=-1)


@dataclass(frozen=True)
class Baselines:
    mean: torch.Tensor
    mean_log_scale: torch.Tensor
    linear_weight: torch.Tensor
    linear_log_scale: torch.Tensor


def fit_baselines(corpus: CorpusIndex, batch_size: int, ridge: float = 1.0) -> Baselines:
    count = torch.zeros(len(corpus.readout_types), dtype=torch.float64)
    total = torch.zeros_like(count)
    square = torch.zeros_like(count)
    linear_xtx: torch.Tensor | None = None
    linear_xty: torch.Tensor | None = None
    for batch, target, target_mask, _species in iter_batches(
        corpus, batch_size, seed=0, shuffle=False
    ):
        mask = target_mask & batch.query_mask
        types = batch.readout_type[mask]
        values = target[mask].double()
        count.scatter_add_(0, types, torch.ones_like(values))
        total.scatter_add_(0, types, values)
        square.scatter_add_(0, types, values.square())
        features = _linear_features(batch, len(corpus.readout_types))[mask].double()
        if linear_xtx is None:
            linear_xtx = torch.zeros(
                (features.shape[1], features.shape[1]), dtype=torch.float64
            )
            linear_xty = torch.zeros(features.shape[1], dtype=torch.float64)
        linear_xtx.add_(features.T @ features)
        linear_xty.add_(features.T @ values)
    if (count == 0).any():
        raise ValueError("every declared readout type needs at least one pretraining target")
    mean = total / count
    variance = (square / count - mean.square()).clamp_min(1e-4)
    assert linear_xtx is not None and linear_xty is not None
    penalty = torch.eye(linear_xtx.shape[0], dtype=torch.float64) * ridge
    penalty[-1, -1] = 0.0
    linear_weight = torch.linalg.solve(linear_xtx + penalty, linear_xty)
    linear_square = torch.zeros_like(count)
    linear_count = torch.zeros_like(count)
    for batch, target, target_mask, _species in iter_batches(
        corpus, batch_size, seed=0, shuffle=False
    ):
        mask = target_mask & batch.query_mask
        types = batch.readout_type[mask]
        prediction = _linear_features(batch, len(corpus.readout_types))[mask].double() @ linear_weight
        residual_square = (target[mask].double() - prediction).square()
        linear_count.scatter_add_(0, types, torch.ones_like(residual_square))
        linear_square.scatter_add_(0, types, residual_square)
    linear_variance = (linear_square / linear_count).clamp_min(1e-4)
    return Baselines(
        mean=mean.float(),
        mean_log_scale=variance.sqrt().log().float(),
        linear_weight=linear_weight.float(),
        linear_log_scale=linear_variance.sqrt().log().float(),
    )


@dataclass
class Moments:
    count: int = 0
    sum_prediction: float = 0.0
    sum_target: float = 0.0
    sum_prediction_square: float = 0.0
    sum_target_square: float = 0.0
    sum_product: float = 0.0
    model_nll: float = 0.0
    mean_baseline_nll: float = 0.0
    linear_baseline_nll: float = 0.0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        model_nll: torch.Tensor,
        mean_baseline_nll: torch.Tensor,
        linear_baseline_nll: torch.Tensor,
    ) -> None:
        x = prediction.double()
        y = target.double()
        self.count += x.numel()
        self.sum_prediction += x.sum().item()
        self.sum_target += y.sum().item()
        self.sum_prediction_square += x.square().sum().item()
        self.sum_target_square += y.square().sum().item()
        self.sum_product += (x * y).sum().item()
        self.model_nll += model_nll.double().sum().item()
        self.mean_baseline_nll += mean_baseline_nll.double().sum().item()
        self.linear_baseline_nll += linear_baseline_nll.double().sum().item()

    @property
    def pearson(self) -> float:
        numerator = self.count * self.sum_product - self.sum_prediction * self.sum_target
        left = self.count * self.sum_prediction_square - self.sum_prediction**2
        right = self.count * self.sum_target_square - self.sum_target**2
        denominator = math.sqrt(max(left * right, 0.0))
        return numerator / denominator if denominator else 0.0

    @property
    def improvement(self) -> float:
        if not self.count:
            return 0.0
        model = self.model_nll / self.count
        baseline = min(self.mean_baseline_nll, self.linear_baseline_nll) / self.count
        return (baseline - model) / max(abs(baseline), 1e-8)

    @property
    def nll_delta(self) -> float:
        if not self.count:
            return 0.0
        return self.strongest_baseline_nll - self.model_nll / self.count

    @property
    def strongest_baseline_nll(self) -> float:
        return min(self.mean_baseline_nll, self.linear_baseline_nll) / self.count


@torch.no_grad()
def evaluate(
    model: SpeciesAwareWorldModel,
    corpus: CorpusIndex,
    baselines: Baselines,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    overall = Moments()
    by_species: dict[int, Moments] = {}
    for batch, target, target_mask, taxon in iter_batches(
        corpus, batch_size, seed=0, shuffle=False
    ):
        batch = _to_device(batch, device)
        target = target.to(device)
        mask = target_mask.to(device) & batch.query_mask
        prediction = model(batch)
        model_loss = _gaussian_nll(prediction.mean, prediction.log_scale, target)
        means = baselines.mean.to(device)[batch.readout_type]
        scales = baselines.mean_log_scale.to(device)[batch.readout_type]
        mean_baseline_loss = _gaussian_nll(means, scales, target)
        linear_mean = _linear_features(batch, len(corpus.readout_types)) @ baselines.linear_weight.to(device)
        linear_scale = baselines.linear_log_scale.to(device)[batch.readout_type]
        linear_baseline_loss = _gaussian_nll(linear_mean, linear_scale, target)
        overall.update(
            prediction.mean[mask].cpu(),
            target[mask].cpu(),
            model_loss[mask].cpu(),
            mean_baseline_loss[mask].cpu(),
            linear_baseline_loss[mask].cpu(),
        )
        for species in taxon.unique().tolist():
            row_mask = taxon == species
            cell_mask = mask.cpu() & row_mask[:, None]
            by_species.setdefault(int(species), Moments()).update(
                prediction.mean.cpu()[cell_mask],
                target.cpu()[cell_mask],
                model_loss.cpu()[cell_mask],
                mean_baseline_loss.cpu()[cell_mask],
                linear_baseline_loss.cpu()[cell_mask],
            )
    if not overall.count:
        raise ValueError("molecular validation corpus has no observed targets")
    return {
        "nll": overall.model_nll / overall.count,
        "baselineNll": overall.strongest_baseline_nll,
        "meanBaselineNll": overall.mean_baseline_nll / overall.count,
        "linearBaselineNll": overall.linear_baseline_nll / overall.count,
        "nllDelta": overall.nll_delta,
        "nllImprovement": overall.improvement,
        "effectPearson": overall.pearson,
        "minimumSpeciesNllDelta": min(item.nll_delta for item in by_species.values()),
        "minimumSpeciesNllImprovement": min(item.improvement for item in by_species.values()),
        "species": {
            str(taxon): {
                "targets": item.count,
                "nll": item.model_nll / item.count,
                "baselineNll": item.strongest_baseline_nll,
                "meanBaselineNll": item.mean_baseline_nll / item.count,
                "linearBaselineNll": item.linear_baseline_nll / item.count,
                "nllDelta": item.nll_delta,
                "nllImprovement": item.improvement,
                "effectPearson": item.pearson,
            }
            for taxon, item in sorted(by_species.items())
        },
    }


def train_world(
    roots: dict[str, str | Path],
    config: dict[str, object],
    audit: object,
) -> tuple[SpeciesAwareWorldModel, dict[str, object], Baselines]:
    pretrain = CorpusIndex.load(roots["pretrain"], "pretrain")
    validation = CorpusIndex.load(roots["molecularValidation"], "molecular-validation")
    reward = CorpusIndex.load(roots["molecularReward"], "molecular-reward")
    _validate_compatible(pretrain, validation, reward)
    corpora = {
        "pretrain": pretrain,
        "molecularValidation": validation,
        "molecularReward": reward,
    }
    validate_audit_binding(corpora, audit)
    for corpus in corpora.values():
        _validate_corpus(corpus)
    reinforcement_epochs = int(config.get("reinforcementEpochs", 0))
    if reinforcement_epochs:
        raise ValueError(
            "molecular reinforcement is disabled until a matched deterministic "
            "continuation and per-source preservation gate are implemented"
        )
    seed = int(config.get("seed", 111))
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_config = WorldConfig(
        entity_feature_dim=pretrain.entity_feature_dim,
        species_feature_dim=pretrain.species_feature_dim,
        readout_types=len(pretrain.readout_types),
        action_covariate_dim=pretrain.action_covariate_dim,
        d_model=int(config.get("dModel", 256)),
        nhead=int(config.get("nhead", 8)),
        encoder_layers=int(config.get("encoderLayers", 4)),
        decoder_layers=int(config.get("decoderLayers", 2)),
        ffn_multiplier=int(config.get("ffnMultiplier", 4)),
        dropout=float(config.get("dropout", 0.1)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeciesAwareWorldModel(model_config).to(device)
    batch_size = int(config.get("batchSize", 32))
    baselines = fit_baselines(pretrain, batch_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learningRate", 3e-4)),
        weight_decay=1e-2,
    )
    history: list[dict[str, object]] = []
    initial_metrics = evaluate(model, validation, baselines, batch_size, device)
    history.append({"phase": "initial", "epoch": 0, **initial_metrics})
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, object] | None = None
    for epoch in range(int(config.get("epochs", 4))):
        model.train()
        for batch, target, target_mask, _species in iter_batches(
            pretrain, batch_size, seed=seed + epoch, shuffle=True
        ):
            batch = _to_device(batch, device)
            target = target.to(device)
            mask = target_mask.to(device) & batch.query_mask
            if not mask.any():
                continue
            prediction = model(batch)
            loss = _gaussian_nll(prediction.mean, prediction.log_scale, target)[mask].mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metrics = evaluate(model, validation, baselines, batch_size, device)
        history.append({"phase": "pretrain", "epoch": epoch + 1, **metrics})
        if best_metrics is None or float(metrics["nll"]) < float(best_metrics["nll"]):
            best_metrics = metrics
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state)

    pre_rl_metrics = best_metrics
    pre_rl_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    rl_retained = False
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("reinforcementLearningRate", 2e-5)),
        weight_decay=1e-2,
    )
    anchor_weight = float(config.get("reinforcementAnchorWeight", 0.1))
    for epoch in range(reinforcement_epochs):
        model.train()
        for batch, target, target_mask, _species in iter_batches(
            reward, batch_size, seed=seed + 1000 + epoch, shuffle=True
        ):
            batch = _to_device(batch, device)
            target = target.to(device)
            mask = target_mask.to(device) & batch.query_mask
            if not mask.any():
                continue
            prediction = model(batch)
            distribution = torch.distributions.Normal(prediction.mean, prediction.scale)
            sample = distribution.sample()
            advantage = (
                -(sample - target).abs() + (prediction.mean.detach() - target).abs()
            ).detach()
            policy_loss = -(advantage * distribution.log_prob(sample))[mask].mean()
            anchor = _gaussian_nll(prediction.mean, prediction.log_scale, target)[mask].mean()
            loss = policy_loss + anchor_weight * anchor
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metrics = evaluate(model, validation, baselines, batch_size, device)
        history.append({"phase": "reinforcement", "epoch": epoch + 1, **metrics})
        if (
            float(metrics["nll"]) < float(best_metrics["nll"])
            and float(metrics["minimumSpeciesNllImprovement"])
            >= float(pre_rl_metrics["minimumSpeciesNllImprovement"])
        ):
            best_metrics = metrics
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            rl_retained = True
    if rl_retained:
        model.load_state_dict(best_state)
    else:
        model.load_state_dict(pre_rl_state)
        best_metrics = pre_rl_metrics
    report = {
        "schema": "slp.training-report/v1.1",
        "seed": seed,
        "modelConfig": model.config.as_dict(),
        "readoutTypes": list(pretrain.readout_types),
        "corpora": {name: corpus.identity for name, corpus in corpora.items()},
        "parameterCount": model.count_parameters(),
        "reinforcementRetained": rl_retained,
        "selected": best_metrics,
        "history": history,
    }
    return model.cpu(), report, baselines
