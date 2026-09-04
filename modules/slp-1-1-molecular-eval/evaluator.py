from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "slp.molecular-evaluation/v1"
REFERENCE_ROLE = "molecular-reference"
PREDICTION_ROLE = "molecular-validation-predictions"
REPORT_SCHEMA = "slp.molecular-evaluation-report/v2"
SYSTEMA_DOI = "10.1038/s41587-025-02777-8"
PROFILE_GATE_SCHEMA = "slp.molecular-profile-gate/v1"
MINIMUM_PERTURBED_CENTROID_PEARSON = 0.10
MINIMUM_SPECIES_PERTURBED_CENTROID_PEARSON = 0.0
CENTRAL_50_NORMAL_Z = 0.6744897501960817
CENTRAL_90_NORMAL_Z = 1.6448536269514722
HEX_DIGITS = frozenset("0123456789abcdef")
MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "datasetId",
        "version",
        "role",
        "labelClass",
        "benchmarkLabelsPresent",
        "valueSpace",
        "speciesTaxa",
        "sourceIds",
        "sourceSnapshotSha256",
        "shards",
        "modelCheckpointSha256",
        "referenceManifestSha256",
    }
)
RECORD_FIELDS = frozenset(
    {
        "speciesTaxon",
        "sourceId",
        "centeringGroup",
        "perturbationId",
        "interventionIds",
        "readoutIds",
        "target",
        "predictionMean",
        "predictionLogScale",
    }
)


class MolecularEvaluationError(ValueError):
    """Raised when evaluation evidence is incomplete, mutable, or invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in HEX_DIGITS for character in value)
    )


def _stable_id(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MolecularEvaluationError(f"{field_name} must be a non-empty string")
    if ":" not in value:
        raise MolecularEvaluationError(f"{field_name} must be a stable namespaced identifier")
    return value


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MolecularEvaluationError(f"{field_name} must be a non-empty string")
    return value


def _number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MolecularEvaluationError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MolecularEvaluationError(f"{field_name} must be finite")
    return result


def _resolve_shard(root: Path, relative_path: object) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise MolecularEvaluationError("shard path must be a non-empty string")
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise MolecularEvaluationError("shard paths must be relative and may not escape the snapshot")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise MolecularEvaluationError("shard path escapes the snapshot") from error
    if not resolved.is_file():
        raise MolecularEvaluationError(f"missing evaluation shard: {relative_path}")
    if resolved.suffix not in {".jsonl", ".gz"}:
        raise MolecularEvaluationError("evaluation shards must end in .jsonl or .jsonl.gz")
    return resolved


def resolve_literal_omf_artifact(value: object, name: str) -> tuple[str, str]:
    """Require the object OMF creates from a literal, admission-pinned artifact digest."""
    if not isinstance(value, dict):
        raise MolecularEvaluationError(f"{name} must be a materialized OMF artifact object")
    if value.get("kind") != "artifact":
        raise MolecularEvaluationError(f"{name} must be a literal OMF artifact input")
    artifacts = value.get("artifacts")
    paths = value.get("paths")
    path = value.get("path")
    if not isinstance(artifacts, dict) or set(artifacts) != {"payload"}:
        raise MolecularEvaluationError(f"{name}.artifacts must contain only payload")
    digest = artifacts["payload"]
    if not isinstance(digest, str) or not digest.startswith("sha256:") or not _is_digest(
        digest.removeprefix("sha256:")
    ):
        raise MolecularEvaluationError(f"{name} payload must be a SHA-256 artifact manifest")
    if value.get("resource") != f"artifact:{digest}":
        raise MolecularEvaluationError(f"{name} resource does not match its artifact manifest")
    if (
        not isinstance(path, str)
        or not path
        or not isinstance(paths, dict)
        or paths.get("payload") != path
    ):
        raise MolecularEvaluationError(f"{name} materialized payload path is inconsistent")
    return path, digest


@dataclass(frozen=True)
class SnapshotManifest:
    root: Path
    role: str
    dataset_id: str
    version: str
    value_space: str
    species_taxa: frozenset[int]
    source_ids: frozenset[str]
    source_snapshot_sha256: str
    manifest_sha256: str
    shards: tuple[dict[str, object], ...]
    model_checkpoint_sha256: str | None = None
    reference_manifest_sha256: str | None = None

    @classmethod
    def load(cls, root: str | Path, expected_role: str) -> SnapshotManifest:
        snapshot_root = Path(root)
        manifest_path = snapshot_root / "evaluation.json"
        if not manifest_path.is_file():
            raise MolecularEvaluationError(f"missing evaluation manifest: {manifest_path}")
        manifest_bytes = manifest_path.read_bytes()
        try:
            raw = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MolecularEvaluationError(f"invalid evaluation manifest: {manifest_path}") from error
        if not isinstance(raw, dict):
            raise MolecularEvaluationError("evaluation manifest must be a JSON object")
        unexpected_manifest_fields = sorted(set(raw) - MANIFEST_FIELDS)
        if unexpected_manifest_fields:
            raise MolecularEvaluationError(
                "unexpected evaluation manifest fields: "
                + ", ".join(unexpected_manifest_fields)
            )
        if raw.get("schema") != SCHEMA:
            raise MolecularEvaluationError(f"evaluation manifest schema must be {SCHEMA}")
        if raw.get("role") != expected_role:
            raise MolecularEvaluationError(f"evaluation manifest role must be {expected_role}")
        if raw.get("labelClass") != "molecular":
            raise MolecularEvaluationError("only molecular evaluation labels are permitted")
        if raw.get("benchmarkLabelsPresent") is not False:
            raise MolecularEvaluationError("benchmark-bearing evaluation input is forbidden")
        species_raw = raw.get("speciesTaxa")
        if (
            not isinstance(species_raw, list)
            or not species_raw
            or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in species_raw)
        ):
            raise MolecularEvaluationError("speciesTaxa must contain positive NCBI taxonomy IDs")
        species_taxa = frozenset(species_raw)
        if len(species_taxa) != len(species_raw):
            raise MolecularEvaluationError("speciesTaxa may not contain duplicates")
        sources_raw = raw.get("sourceIds")
        if not isinstance(sources_raw, list) or not sources_raw:
            raise MolecularEvaluationError("sourceIds must be a non-empty list")
        source_ids = frozenset(_stable_id(item, "sourceIds[]") for item in sources_raw)
        if len(source_ids) != len(sources_raw):
            raise MolecularEvaluationError("sourceIds may not contain duplicates")
        source_snapshot_sha256 = raw.get("sourceSnapshotSha256")
        if not _is_digest(source_snapshot_sha256):
            raise MolecularEvaluationError("sourceSnapshotSha256 must be a lowercase SHA-256")
        shards_raw = raw.get("shards")
        if not isinstance(shards_raw, list) or not shards_raw:
            raise MolecularEvaluationError("evaluation manifest must declare at least one shard")
        shards: list[dict[str, object]] = []
        seen_paths: set[str] = set()
        for shard in shards_raw:
            if not isinstance(shard, dict):
                raise MolecularEvaluationError("shard declarations must be objects")
            unexpected_shard_fields = sorted(set(shard) - {"path", "sha256", "records"})
            if unexpected_shard_fields:
                raise MolecularEvaluationError(
                    "unexpected shard fields: " + ", ".join(unexpected_shard_fields)
                )
            path_value = shard.get("path")
            if not isinstance(path_value, str) or path_value in seen_paths:
                raise MolecularEvaluationError("shard paths must be unique strings")
            seen_paths.add(path_value)
            path = _resolve_shard(snapshot_root, path_value)
            expected_digest = shard.get("sha256")
            if not _is_digest(expected_digest):
                raise MolecularEvaluationError("shard sha256 must be a lowercase SHA-256")
            if _sha256(path) != expected_digest:
                raise MolecularEvaluationError(f"digest mismatch for evaluation shard: {path_value}")
            records = shard.get("records")
            if isinstance(records, bool) or not isinstance(records, int) or records <= 0:
                raise MolecularEvaluationError("shard records must be a positive integer")
            shards.append({"path": path_value, "sha256": expected_digest, "records": records})
        model_digest: str | None = None
        reference_digest: str | None = None
        if expected_role == PREDICTION_ROLE:
            model_digest = raw.get("modelCheckpointSha256")
            reference_digest = raw.get("referenceManifestSha256")
            if not _is_digest(model_digest):
                raise MolecularEvaluationError("modelCheckpointSha256 must be a lowercase SHA-256")
            if not _is_digest(reference_digest):
                raise MolecularEvaluationError("referenceManifestSha256 must be a lowercase SHA-256")
        elif "modelCheckpointSha256" in raw or "referenceManifestSha256" in raw:
            raise MolecularEvaluationError(
                "molecular-reference manifests may not declare prediction provenance"
            )
        return cls(
            root=snapshot_root,
            role=expected_role,
            dataset_id=_string(raw.get("datasetId"), "datasetId"),
            version=_string(raw.get("version"), "version"),
            value_space=_string(raw.get("valueSpace"), "valueSpace"),
            species_taxa=species_taxa,
            source_ids=source_ids,
            source_snapshot_sha256=source_snapshot_sha256,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            shards=tuple(shards),
            model_checkpoint_sha256=model_digest,
            reference_manifest_sha256=reference_digest,
        )

    def records(self, max_line_bytes: int) -> Iterator[tuple[str, int, dict[str, object]]]:
        for shard in self.shards:
            relative_path = str(shard["path"])
            path = _resolve_shard(self.root, relative_path)
            opener = gzip.open if path.suffix == ".gz" else open
            seen = 0
            with opener(path, "rb") as source:
                for line_number, line in enumerate(source, start=1):
                    if len(line) > max_line_bytes:
                        raise MolecularEvaluationError(
                            f"{relative_path}:{line_number} exceeds maxLineBytes"
                        )
                    if not line.strip():
                        raise MolecularEvaluationError(
                            f"{relative_path}:{line_number} is an empty record"
                        )
                    try:
                        record = json.loads(line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise MolecularEvaluationError(
                            f"invalid JSON in {relative_path}:{line_number}"
                        ) from error
                    if not isinstance(record, dict):
                        raise MolecularEvaluationError(
                            f"{relative_path}:{line_number} must be a JSON object"
                        )
                    seen += 1
                    yield relative_path, line_number, record
            if seen != int(shard["records"]):
                raise MolecularEvaluationError(
                    f"record count mismatch for {relative_path}: expected {shard['records']}, got {seen}"
                )


ProfileKey = tuple[int, str, str, str]
GroupKey = tuple[int, str, str]


@dataclass
class MeanValue:
    total: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        self.total += value
        self.count += 1

    @property
    def mean(self) -> float:
        return self.total / self.count


@dataclass
class Profile:
    intervention_ids: tuple[str, ...]
    target: dict[str, MeanValue] = field(default_factory=dict)
    prediction: dict[str, MeanValue] = field(default_factory=dict)

    def add(self, readout: str, target: float, prediction: float | None) -> None:
        self.target.setdefault(readout, MeanValue()).add(target)
        if prediction is not None:
            self.prediction.setdefault(readout, MeanValue()).add(prediction)

    def target_means(self) -> dict[str, float]:
        return {key: value.mean for key, value in self.target.items()}

    def prediction_means(self) -> dict[str, float]:
        return {key: value.mean for key, value in self.prediction.items()}


@dataclass
class ScalarMoments:
    count: int = 0
    sum_prediction: float = 0.0
    sum_target: float = 0.0
    sum_prediction_square: float = 0.0
    sum_target_square: float = 0.0
    sum_product: float = 0.0
    sum_absolute_error: float = 0.0
    sum_square_error: float = 0.0
    sum_gaussian_nll: float = 0.0
    central_50_covered: int = 0
    central_90_covered: int = 0
    sum_central_50_interval_width: float = 0.0
    sum_central_90_interval_width: float = 0.0

    def add(self, prediction: float, target: float, log_scale: float) -> None:
        error = prediction - target
        scale = math.exp(log_scale)
        self.count += 1
        self.sum_prediction += prediction
        self.sum_target += target
        self.sum_prediction_square += prediction * prediction
        self.sum_target_square += target * target
        self.sum_product += prediction * target
        self.sum_absolute_error += abs(error)
        self.sum_square_error += error * error
        self.sum_gaussian_nll += (
            0.5 * (error * math.exp(-log_scale)) ** 2
            + log_scale
            + 0.5 * math.log(2.0 * math.pi)
        )
        absolute_error = abs(error)
        central_50_radius = CENTRAL_50_NORMAL_Z * scale
        central_90_radius = CENTRAL_90_NORMAL_Z * scale
        self.central_50_covered += int(absolute_error <= central_50_radius)
        self.central_90_covered += int(absolute_error <= central_90_radius)
        self.sum_central_50_interval_width += 2.0 * central_50_radius
        self.sum_central_90_interval_width += 2.0 * central_90_radius

    def merge(self, other: ScalarMoments) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def pearson(self) -> float | None:
        if self.count < 2:
            return None
        numerator = self.count * self.sum_product - self.sum_prediction * self.sum_target
        left = self.count * self.sum_prediction_square - self.sum_prediction**2
        right = self.count * self.sum_target_square - self.sum_target**2
        denominator = math.sqrt(max(left * right, 0.0))
        return numerator / denominator if denominator > 1e-15 else None

    def report(self) -> dict[str, object]:
        if not self.count:
            raise MolecularEvaluationError("a metric group has no observed molecular targets")
        pearson = self.pearson()
        return {
            "targets": self.count,
            "rmse": math.sqrt(self.sum_square_error / self.count),
            "mae": self.sum_absolute_error / self.count,
            "pearson": pearson if pearson is not None else 0.0,
            "pearsonDefined": pearson is not None,
            "gaussianNll": self.sum_gaussian_nll / self.count,
        }

    def gaussian_calibration_report(self) -> dict[str, object]:
        if not self.count:
            raise MolecularEvaluationError("a metric group has no observed molecular targets")
        return {
            "targets": self.count,
            "central50": {
                "nominalCoverage": 0.50,
                "z": CENTRAL_50_NORMAL_Z,
                "empiricalCoverage": self.central_50_covered / self.count,
                "meanIntervalWidth": self.sum_central_50_interval_width / self.count,
            },
            "central90": {
                "nominalCoverage": 0.90,
                "z": CENTRAL_90_NORMAL_Z,
                "empiricalCoverage": self.central_90_covered / self.count,
                "meanIntervalWidth": self.sum_central_90_interval_width / self.count,
            },
        }


@dataclass(frozen=True)
class ProfileMetric:
    key: ProfileKey
    readouts: int
    ordinary_pearson: float | None
    perturbed_centroid_pearson: float | None
    perturbed_centroid_cosine: float | None
    centroid_accuracy: float | None = None


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_norm = sum((x - left_mean) ** 2 for x in left)
    right_norm = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(max(left_norm * right_norm, 0.0))
    return numerator / denominator if denominator > 1e-15 else None


def _cosine(left: list[float], right: list[float]) -> float | None:
    numerator = sum(x * y for x, y in zip(left, right))
    denominator = math.sqrt(sum(x * x for x in left) * sum(y * y for y in right))
    return numerator / denominator if denominator > 1e-15 else None


def _conservative_macro_mean(values: Iterable[float | None]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(value if value is not None else 0.0 for value in materialized) / len(materialized)


def _parse_record(
    manifest: SnapshotManifest,
    record: dict[str, object],
    location: str,
    prediction: bool,
    maximum_absolute_log_scale: float,
) -> tuple[ProfileKey, tuple[str, ...], list[tuple[str, float, float | None, float | None]]]:
    unexpected_record_fields = sorted(set(record) - RECORD_FIELDS)
    if unexpected_record_fields:
        raise MolecularEvaluationError(
            f"{location}: unexpected record fields: " + ", ".join(unexpected_record_fields)
        )
    taxon = record.get("speciesTaxon")
    if isinstance(taxon, bool) or not isinstance(taxon, int) or taxon not in manifest.species_taxa:
        raise MolecularEvaluationError(f"{location}: speciesTaxon is not declared by the manifest")
    source = _stable_id(record.get("sourceId"), f"{location}.sourceId")
    if source not in manifest.source_ids:
        raise MolecularEvaluationError(f"{location}: sourceId is not declared by the manifest")
    centering_group = _string(record.get("centeringGroup"), f"{location}.centeringGroup")
    perturbation_id = _stable_id(record.get("perturbationId"), f"{location}.perturbationId")
    interventions_raw = record.get("interventionIds")
    if not isinstance(interventions_raw, list) or not interventions_raw:
        raise MolecularEvaluationError(f"{location}.interventionIds must be a non-empty list")
    intervention_ids = tuple(
        _stable_id(item, f"{location}.interventionIds[]") for item in interventions_raw
    )
    if len(set(intervention_ids)) != len(intervention_ids):
        raise MolecularEvaluationError(f"{location}.interventionIds contains duplicates")
    readouts_raw = record.get("readoutIds")
    targets_raw = record.get("target")
    if not isinstance(readouts_raw, list) or not readouts_raw:
        raise MolecularEvaluationError(f"{location}.readoutIds must be a non-empty list")
    if not isinstance(targets_raw, list) or len(targets_raw) != len(readouts_raw):
        raise MolecularEvaluationError(f"{location}.target must align with readoutIds")
    means_raw = record.get("predictionMean") if prediction else None
    scales_raw = record.get("predictionLogScale") if prediction else None
    if not prediction and (
        "predictionMean" in record or "predictionLogScale" in record
    ):
        raise MolecularEvaluationError(
            f"{location}: molecular-reference records may not contain predictions"
        )
    if prediction and (
        not isinstance(means_raw, list)
        or not isinstance(scales_raw, list)
        or len(means_raw) != len(readouts_raw)
        or len(scales_raw) != len(readouts_raw)
    ):
        raise MolecularEvaluationError(
            f"{location}.predictionMean and predictionLogScale must align with readoutIds"
        )
    parsed: list[tuple[str, float, float | None, float | None]] = []
    seen_readouts: set[str] = set()
    for index, readout_raw in enumerate(readouts_raw):
        readout = _stable_id(readout_raw, f"{location}.readoutIds[{index}]")
        if readout in seen_readouts:
            raise MolecularEvaluationError(f"{location}.readoutIds contains duplicates")
        seen_readouts.add(readout)
        target_raw = targets_raw[index]
        if target_raw is None:
            continue
        target = _number(target_raw, f"{location}.target[{index}]")
        mean: float | None = None
        log_scale: float | None = None
        if prediction:
            assert isinstance(means_raw, list) and isinstance(scales_raw, list)
            mean = _number(means_raw[index], f"{location}.predictionMean[{index}]")
            log_scale = _number(scales_raw[index], f"{location}.predictionLogScale[{index}]")
            if abs(log_scale) > maximum_absolute_log_scale:
                raise MolecularEvaluationError(
                    f"{location}.predictionLogScale[{index}] exceeds the declared bound"
                )
        parsed.append((readout, target, mean, log_scale))
    if not parsed:
        raise MolecularEvaluationError(f"{location} contains no observed molecular targets")
    return (taxon, source, centering_group, perturbation_id), intervention_ids, parsed


def _load_profiles(
    manifest: SnapshotManifest,
    *,
    prediction: bool,
    max_line_bytes: int,
    maximum_absolute_log_scale: float,
) -> tuple[
    dict[ProfileKey, Profile],
    dict[str, ScalarMoments],
    dict[int, ScalarMoments],
    dict[tuple[int, str], ScalarMoments],
    int,
]:
    profiles: dict[ProfileKey, Profile] = {}
    source_moments: dict[str, ScalarMoments] = defaultdict(ScalarMoments)
    species_moments: dict[int, ScalarMoments] = defaultdict(ScalarMoments)
    species_source_moments: dict[tuple[int, str], ScalarMoments] = defaultdict(ScalarMoments)
    records = 0
    for shard_path, line_number, record in manifest.records(max_line_bytes):
        location = f"{shard_path}:{line_number}"
        key, intervention_ids, observations = _parse_record(
            manifest,
            record,
            location,
            prediction,
            maximum_absolute_log_scale,
        )
        existing = profiles.get(key)
        if existing is None:
            existing = Profile(intervention_ids=intervention_ids)
            profiles[key] = existing
        elif existing.intervention_ids != intervention_ids:
            raise MolecularEvaluationError(
                f"{location}: interventionIds changed within one perturbation profile"
            )
        for readout, target, mean, log_scale in observations:
            existing.add(readout, target, mean)
            if prediction:
                assert mean is not None and log_scale is not None
                moments = ScalarMoments()
                moments.add(mean, target, log_scale)
                taxon, source, _centering_group, _perturbation = key
                source_moments[source].merge(moments)
                species_moments[taxon].merge(moments)
                species_source_moments[(taxon, source)].merge(moments)
        records += 1
    return profiles, source_moments, species_moments, species_source_moments, records


def _perturbed_references(
    reference_profiles: dict[ProfileKey, Profile],
    minimum_reference_perturbations: int,
) -> tuple[dict[GroupKey, dict[str, float]], dict[GroupKey, int]]:
    totals: dict[GroupKey, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[GroupKey, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    perturbation_counts: dict[GroupKey, int] = defaultdict(int)
    for (taxon, source, centering_group, _perturbation), profile in reference_profiles.items():
        group = (taxon, source, centering_group)
        perturbation_counts[group] += 1
        for readout, value in profile.target_means().items():
            totals[group][readout] += value
            counts[group][readout] += 1
    references: dict[GroupKey, dict[str, float]] = {}
    for group, readout_totals in totals.items():
        references[group] = {
            readout: total / counts[group][readout]
            for readout, total in readout_totals.items()
            if counts[group][readout] >= minimum_reference_perturbations
        }
    return references, dict(perturbation_counts)


def _profile_metrics(
    evaluation_profiles: dict[ProfileKey, Profile],
    references: dict[GroupKey, dict[str, float]],
    minimum_profile_readouts: int,
) -> list[ProfileMetric]:
    metrics: list[ProfileMetric] = []
    for key, profile in evaluation_profiles.items():
        group = key[:3]
        if group not in references:
            raise MolecularEvaluationError(f"no molecular reference group for {group}")
        target = profile.target_means()
        prediction = profile.prediction_means()
        readouts = sorted(set(target) & set(prediction) & set(references[group]))
        if len(readouts) < minimum_profile_readouts:
            continue
        target_values = [target[readout] for readout in readouts]
        prediction_values = [prediction[readout] for readout in readouts]
        target_shift = [
            target[readout] - references[group][readout] for readout in readouts
        ]
        prediction_shift = [
            prediction[readout] - references[group][readout] for readout in readouts
        ]
        metrics.append(
            ProfileMetric(
                key=key,
                readouts=len(readouts),
                ordinary_pearson=_pearson(prediction_values, target_values),
                perturbed_centroid_pearson=_pearson(prediction_shift, target_shift),
                perturbed_centroid_cosine=_cosine(prediction_shift, target_shift),
            )
        )
    return metrics


def _with_centroid_accuracy(
    profile_metrics: list[ProfileMetric],
    evaluation_profiles: dict[ProfileKey, Profile],
    minimum_profile_readouts: int,
) -> list[ProfileMetric]:
    grouped: dict[GroupKey, list[ProfileMetric]] = defaultdict(list)
    for metric in profile_metrics:
        grouped[metric.key[:3]].append(metric)
    accuracies: dict[ProfileKey, float] = {}
    for group_metrics in grouped.values():
        if len(group_metrics) < 2:
            continue
        common: set[str] | None = None
        for metric in group_metrics:
            profile = evaluation_profiles[metric.key]
            available = set(profile.target) & set(profile.prediction)
            common = available if common is None else common & available
        common_readouts = sorted(common or ())
        if len(common_readouts) < minimum_profile_readouts:
            continue
        targets = {
            metric.key: evaluation_profiles[metric.key].target_means() for metric in group_metrics
        }
        predictions = {
            metric.key: evaluation_profiles[metric.key].prediction_means()
            for metric in group_metrics
        }
        for metric in group_metrics:
            prediction = predictions[metric.key]
            correct = targets[metric.key]
            correct_distance = sum(
                (prediction[readout] - correct[readout]) ** 2 for readout in common_readouts
            )
            wins = 0
            comparisons = 0
            for alternative in group_metrics:
                if alternative.key == metric.key:
                    continue
                alternative_target = targets[alternative.key]
                alternative_distance = sum(
                    (prediction[readout] - alternative_target[readout]) ** 2
                    for readout in common_readouts
                )
                wins += int(correct_distance < alternative_distance)
                comparisons += 1
            accuracies[metric.key] = wins / comparisons
    return [
        ProfileMetric(
            key=metric.key,
            readouts=metric.readouts,
            ordinary_pearson=metric.ordinary_pearson,
            perturbed_centroid_pearson=metric.perturbed_centroid_pearson,
            perturbed_centroid_cosine=metric.perturbed_centroid_cosine,
            centroid_accuracy=accuracies.get(metric.key),
        )
        for metric in profile_metrics
    ]


def _profile_report(metrics: list[ProfileMetric]) -> dict[str, object]:
    return {
        "profiles": len(metrics),
        "profileReadouts": sum(metric.readouts for metric in metrics),
        "ordinaryPearsonProfiles": sum(metric.ordinary_pearson is not None for metric in metrics),
        "ordinaryPearsonUndefinedProfiles": sum(
            metric.ordinary_pearson is None for metric in metrics
        ),
        "ordinaryProfilePearson": _conservative_macro_mean(
            metric.ordinary_pearson for metric in metrics
        ),
        "perturbedCentroidPearsonProfiles": sum(
            metric.perturbed_centroid_pearson is not None for metric in metrics
        ),
        "perturbedCentroidPearsonUndefinedProfiles": sum(
            metric.perturbed_centroid_pearson is None for metric in metrics
        ),
        "perturbedCentroidPearson": _conservative_macro_mean(
            metric.perturbed_centroid_pearson for metric in metrics
        ),
        "perturbedCentroidCosineProfiles": sum(
            metric.perturbed_centroid_cosine is not None for metric in metrics
        ),
        "perturbedCentroidCosineUndefinedProfiles": sum(
            metric.perturbed_centroid_cosine is None for metric in metrics
        ),
        "perturbedCentroidCosine": _conservative_macro_mean(
            metric.perturbed_centroid_cosine for metric in metrics
        ),
        "centroidAccuracyProfiles": sum(metric.centroid_accuracy is not None for metric in metrics),
        "centroidAccuracyUndefinedProfiles": sum(
            metric.centroid_accuracy is None for metric in metrics
        ),
        "centroidAccuracyCommonPanel": _conservative_macro_mean(
            metric.centroid_accuracy for metric in metrics
        ),
    }


def _combined_report(moments: ScalarMoments, metrics: list[ProfileMetric]) -> dict[str, object]:
    return {
        "ordinary": moments.report(),
        "gaussianCalibration": moments.gaussian_calibration_report(),
        "perturbationSpecific": _profile_report(metrics),
    }


def molecular_profile_decision(report: dict[str, object]) -> dict[str, object]:
    """Apply the frozen profile-only gate without claiming full model advancement."""
    overall = report["overall"]
    audit = report["audit"]
    species = report["species"]
    sources = report["sources"]
    assert isinstance(overall, dict)
    assert isinstance(audit, dict)
    assert isinstance(species, dict)
    assert isinstance(sources, dict)
    specific = overall["perturbationSpecific"]
    assert isinstance(specific, dict)
    species_pearson = {
        str(taxon): float(value["perturbationSpecific"]["perturbedCentroidPearson"])
        for taxon, value in species.items()
    }
    source_profiles = {
        str(source): int(value["perturbationSpecific"]["profiles"])
        for source, value in sources.items()
    }
    species_profiles = {
        str(taxon): int(value["perturbationSpecific"]["profiles"])
        for taxon, value in species.items()
    }
    minimum_species = min(species_pearson.values())
    checks = {
        "zeroBenchmarkLabelRecords": int(audit["benchmarkLabelRecords"]) == 0,
        "zeroHeldInterventionOverlap": int(audit["heldInterventionOverlap"]) == 0,
        "overallPerturbedCentroidPearson": (
            float(specific["perturbedCentroidPearson"])
            >= MINIMUM_PERTURBED_CENTROID_PEARSON
        ),
        "minimumSpeciesPerturbedCentroidPearson": (
            minimum_species >= MINIMUM_SPECIES_PERTURBED_CENTROID_PEARSON
        ),
        "everySpeciesHasEligibleProfiles": all(value > 0 for value in species_profiles.values()),
        "everySourceHasEligibleProfiles": all(value > 0 for value in source_profiles.values()),
    }
    return {
        "schema": PROFILE_GATE_SCHEMA,
        "scope": "molecular-profile-evaluation-only",
        "passed": all(checks.values()),
        "compatibilityPassed": True,
        "compatibilityScope": "immutable molecular reference/prediction artifact contract",
        "thresholds": {
            "minimumPerturbedCentroidPearson": MINIMUM_PERTURBED_CENTROID_PEARSON,
            "minimumSpeciesPerturbedCentroidPearson": (
                MINIMUM_SPECIES_PERTURBED_CENTROID_PEARSON
            ),
        },
        "observed": {
            "perturbedCentroidPearson": specific["perturbedCentroidPearson"],
            "minimumSpeciesPerturbedCentroidPearson": minimum_species,
            "speciesProfiles": species_profiles,
            "sourceProfiles": source_profiles,
        },
        "checks": checks,
        "doesNotEstablish": [
            "the separate Gaussian-NLL improvement gate",
            "checkpoint selection eligibility",
            "synthetic-lethality benchmark performance",
            "portable inference or release compatibility",
        ],
    }


def evaluate_molecular_predictions(
    reference_root: str | Path,
    prediction_root: str | Path,
    *,
    minimum_reference_perturbations: int = 2,
    minimum_profile_readouts: int = 2,
    max_line_bytes: int = 16 * 1024 * 1024,
    maximum_absolute_log_scale: float = 20.0,
) -> dict[str, object]:
    if minimum_reference_perturbations < 2:
        raise MolecularEvaluationError("minimumReferencePerturbations must be at least 2")
    if minimum_profile_readouts < 2:
        raise MolecularEvaluationError("minimumProfileReadouts must be at least 2")
    if max_line_bytes < 1024:
        raise MolecularEvaluationError("maxLineBytes must be at least 1024")
    if maximum_absolute_log_scale <= 0:
        raise MolecularEvaluationError("maximumAbsoluteLogScale must be positive")
    reference_manifest = SnapshotManifest.load(reference_root, REFERENCE_ROLE)
    prediction_manifest = SnapshotManifest.load(prediction_root, PREDICTION_ROLE)
    if prediction_manifest.reference_manifest_sha256 != reference_manifest.manifest_sha256:
        raise MolecularEvaluationError(
            "prediction input is not bound to the supplied molecular-reference manifest"
        )
    if prediction_manifest.value_space != reference_manifest.value_space:
        raise MolecularEvaluationError("reference and prediction valueSpace values do not match")
    reference_profiles, _, _, _, reference_records = _load_profiles(
        reference_manifest,
        prediction=False,
        max_line_bytes=max_line_bytes,
        maximum_absolute_log_scale=maximum_absolute_log_scale,
    )
    (
        evaluation_profiles,
        source_moments,
        species_moments,
        species_source_moments,
        prediction_records,
    ) = _load_profiles(
        prediction_manifest,
        prediction=True,
        max_line_bytes=max_line_bytes,
        maximum_absolute_log_scale=maximum_absolute_log_scale,
    )
    for manifest, profiles in (
        (reference_manifest, reference_profiles),
        (prediction_manifest, evaluation_profiles),
    ):
        observed_species = frozenset(key[0] for key in profiles)
        observed_sources = frozenset(key[1] for key in profiles)
        if observed_species != manifest.species_taxa:
            raise MolecularEvaluationError(
                f"{manifest.role} declared speciesTaxa do not exactly match its records"
            )
        if observed_sources != manifest.source_ids:
            raise MolecularEvaluationError(
                f"{manifest.role} declared sourceIds do not exactly match its records"
            )
    reference_interventions = {
        (key[0], intervention)
        for key, profile in reference_profiles.items()
        for intervention in profile.intervention_ids
    }
    evaluation_interventions = {
        (key[0], intervention)
        for key, profile in evaluation_profiles.items()
        for intervention in profile.intervention_ids
    }
    overlap = sorted(reference_interventions & evaluation_interventions)
    if overlap:
        formatted = ", ".join(f"{taxon}:{identifier}" for taxon, identifier in overlap[:10])
        raise MolecularEvaluationError(
            f"held intervention leakage between reference and validation predictions: {formatted}"
        )
    references, reference_perturbation_counts = _perturbed_references(
        reference_profiles, minimum_reference_perturbations
    )
    metrics = _with_centroid_accuracy(
        _profile_metrics(evaluation_profiles, references, minimum_profile_readouts),
        evaluation_profiles,
        minimum_profile_readouts,
    )
    if not metrics:
        raise MolecularEvaluationError("no validation profile meets the perturbation-specific contract")
    overall_moments = ScalarMoments()
    for moments in source_moments.values():
        overall_moments.merge(moments)
    def selected(
        *, taxon: int | None = None, source: str | None = None
    ) -> list[ProfileMetric]:
        return [
            metric
            for metric in metrics
            if (taxon is None or metric.key[0] == taxon)
            and (source is None or metric.key[1] == source)
        ]

    overall = _combined_report(overall_moments, metrics)
    species = {
        str(taxon): _combined_report(moments, selected(taxon=taxon))
        for taxon, moments in sorted(species_moments.items())
    }
    sources = {
        source: _combined_report(moments, selected(source=source))
        for source, moments in sorted(source_moments.items())
    }
    species_sources = {
        f"{taxon}|{source}": _combined_report(
            moments, selected(taxon=taxon, source=source)
        )
        for (taxon, source), moments in sorted(species_source_moments.items())
    }
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "method": {
            "name": "training-perturbed-centroid sparse-profile evaluation",
            "class": "Systema-inspired",
            "referenceDefinition": (
                "For each species, source, centering group, and readout, average each "
                "training perturbation centroid with equal perturbation weight, then subtract "
                "that reference from validation truth and prediction profiles."
            ),
            "centroidAccuracyDefinition": (
                "Strict Euclidean-distance rank accuracy against other validation perturbation "
                "centroids on the common observed readout panel."
            ),
            "gaussianScaleDefinition": (
                "Each observed target is scored under Normal(predictionMean, "
                "exp(predictionLogScale)^2); predictionLogScale is the natural logarithm "
                "of the standard deviation in the declared valueSpace."
            ),
            "gaussianIntervalDefinition": (
                "Central 50% and 90% intervals use exact standard-Normal z values and "
                "closed, inclusive endpoints. Empirical coverage and mean full interval "
                "width are diagnostic only and do not enter any decision check."
            ),
            "citationDoi": SYSTEMA_DOI,
            "deviations": [
                "Supports sparse, multi-modal molecular readout profiles rather than only dense single-cell expression.",
                "Centroid accuracy is restricted to the common observed readout panel within each centering group.",
                "Undefined per-profile correlations, cosines, and centroid accuracies contribute zero to macro means and are counted explicitly.",
            ],
        },
        "inputs": {
            "reference": {
                "datasetId": reference_manifest.dataset_id,
                "version": reference_manifest.version,
                "manifestSha256": reference_manifest.manifest_sha256,
                "sourceSnapshotSha256": reference_manifest.source_snapshot_sha256,
                "records": reference_records,
            },
            "predictions": {
                "datasetId": prediction_manifest.dataset_id,
                "version": prediction_manifest.version,
                "manifestSha256": prediction_manifest.manifest_sha256,
                "sourceSnapshotSha256": prediction_manifest.source_snapshot_sha256,
                "modelCheckpointSha256": prediction_manifest.model_checkpoint_sha256,
                "records": prediction_records,
            },
            "valueSpace": prediction_manifest.value_space,
        },
        "audit": {
            "benchmarkLabelRecords": 0,
            "heldInterventionOverlap": 0,
            "speciesTaxa": sorted(prediction_manifest.species_taxa),
            "sourceIds": sorted(prediction_manifest.source_ids),
            "minimumReferencePerturbations": minimum_reference_perturbations,
            "minimumProfileReadouts": minimum_profile_readouts,
            "referencePerturbationsByGroup": {
                "|".join(map(str, group)): count
                for group, count in sorted(reference_perturbation_counts.items())
            },
        },
        "overall": overall,
        "species": species,
        "sources": sources,
        "speciesSources": species_sources,
    }
    report["decision"] = molecular_profile_decision(report)
    return report
