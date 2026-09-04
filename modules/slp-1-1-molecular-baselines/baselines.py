"""Deterministic molecular baselines over explicit, sparse profile identities.

The contract intentionally does not infer biological context or basal controls
from action emptiness, field names, or centering labels.  It accepts only the
versioned manifest and record shapes documented beside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable, Iterator, Mapping, Sequence


SNAPSHOT_SCHEMA = "slp.molecular-baseline-snapshot/v1"
RECORD_SCHEMA = "slp.molecular-baseline-profile/v1"
PREDICTION_SCHEMA = "slp.molecular-baseline-predictions/v1"
PREDICTION_RECORD_SCHEMA = "slp.molecular-baseline-prediction/v1"
REPORT_SCHEMA = "slp.molecular-baseline-report/v1"
TRAINING_ROLE = "molecular-baseline-training"
REFERENCE_ROLE = "molecular-baseline-reference"
TASKS = {
    "intervention-gene-cold",
    "perturbation-context-cold-with-basal-access",
    "double-cold",
}
BASELINES = ("context-only", "txpert-mean-additive")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
CURIE = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*:[^\s]+$")
RESOURCE_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9.-]{0,61}[a-z0-9])?$")


class MolecularBaselineError(ValueError):
    """Raised when an input is unsafe or cannot define the frozen baseline."""


@dataclass(frozen=True)
class Limits:
    max_shards: int = 256
    max_records: int = 2_000_000
    max_readouts_per_record: int = 100_000
    max_line_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in ("max_shards", "max_records", "max_readouts_per_record", "max_line_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise MolecularBaselineError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class PinnedDataset:
    path: Path
    resource: str
    revision: str
    manifest_digest: str


@dataclass(frozen=True)
class Profile:
    species_taxon: int
    source_id: str
    context_id: str
    role: str
    perturbation_id: str | None
    intervention_ids: tuple[str, ...]
    readout_ids: tuple[str, ...]
    values: tuple[float | None, ...]

    @property
    def key(self) -> tuple[int, str, str, str, str]:
        return (
            self.species_taxon,
            self.source_id,
            self.context_id,
            self.role,
            self.perturbation_id or "",
        )

    @property
    def context_key(self) -> tuple[int, str, str]:
        return self.species_taxon, self.source_id, self.context_id

    @property
    def species_source(self) -> tuple[int, str]:
        return self.species_taxon, self.source_id

    def observed(self) -> dict[str, float]:
        return {
            readout: value
            for readout, value in zip(self.readout_ids, self.values)
            if value is not None
        }


@dataclass(frozen=True)
class Snapshot:
    root: Path
    manifest: Mapping[str, object]
    manifest_sha256: str
    profiles: tuple[Profile, ...]

    @property
    def task_name(self) -> str:
        return str(self.manifest["taskName"])

    @property
    def value_space(self) -> str:
        return str(self.manifest["valueSpace"])


def _exact_object(value: object, expected: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise MolecularBaselineError(f"{label} must be a JSON object")
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise MolecularBaselineError(
            f"{label} fields do not match the contract; missing={missing}, extra={extra}"
        )
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MolecularBaselineError(f"{label} must be a non-empty trimmed string")
    return value


def _curie(value: object, label: str) -> str:
    result = _string(value, label)
    if CURIE.fullmatch(result) is None:
        raise MolecularBaselineError(f"{label} must be a stable CURIE")
    return result


def _sha256(value: object, label: str, *, prefix: bool = False) -> str:
    result = _string(value, label)
    raw = result.removeprefix("sha256:") if prefix else result
    if (prefix and not result.startswith("sha256:")) or SHA256.fullmatch(raw) is None:
        form = "sha256:<hex>" if prefix else "64 lowercase hexadecimal characters"
        raise MolecularBaselineError(f"{label} must be {form}")
    return raw


def _sorted_curies(value: object, label: str, *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise MolecularBaselineError(f"{label} must be {qualifier} of stable CURIEs")
    result = tuple(_curie(item, f"{label}[]") for item in value)
    if len(set(result)) != len(result):
        raise MolecularBaselineError(f"{label} contains duplicates")
    if tuple(sorted(result)) != result:
        raise MolecularBaselineError(f"{label} must be lexically sorted")
    return result


def resolve_pinned_dataset(value: object, input_name: str) -> PinnedDataset:
    """Resolve OMF's exact copied, revision-pinned DatasetSnapshot shape."""

    item = _exact_object(
        value, {"resource", "mode", "path", "manifestDigest"}, input_name
    )
    if item["mode"] != "copy":
        raise MolecularBaselineError(
            f"{input_name} must be an immutable copied DatasetSnapshot"
        )
    manifest_digest = _sha256(
        item["manifestDigest"], f"{input_name}.manifestDigest", prefix=True
    )
    resource = _string(item["resource"], f"{input_name}.resource")
    if not resource.startswith("omf://"):
        raise MolecularBaselineError(
            f"{input_name}.resource must be an OMF DatasetSnapshot URI"
        )
    identity, separator, revision_raw = resource.removeprefix("omf://").rpartition("@")
    if not separator:
        raise MolecularBaselineError(
            f"{input_name}.resource must contain an admission-pinned revision"
        )
    revision = _sha256(revision_raw, f"{input_name}.resource revision", prefix=True)
    parts = identity.split("/")
    if (
        len(parts) < 3
        or parts[-2] != "datasetsnapshot"
        or RESOURCE_NAME.fullmatch(parts[-1]) is None
        or any(
            not part or part in {".", ".."} or any(char.isspace() for char in part)
            for part in parts
        )
    ):
        raise MolecularBaselineError(
            f"{input_name}.resource kind must be DatasetSnapshot with a valid resource name"
        )
    path_value = _string(item["path"], f"{input_name}.path")
    requested = Path(path_value).absolute()
    if any(candidate.is_symlink() for candidate in (requested, *requested.parents)):
        raise MolecularBaselineError(f"{input_name}.path must not traverse a symlink")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise MolecularBaselineError(f"{input_name}.path does not exist") from error
    if not resolved.is_dir():
        raise MolecularBaselineError(
            f"{input_name}.path must materialize a DatasetSnapshot directory"
        )
    if (
        resolved.name != parts[-1]
        or resolved.parent.name != input_name
        or resolved.parent.parent.name != "inputs"
    ):
        raise MolecularBaselineError(
            f"{input_name}.path is inconsistent with its input name and DatasetSnapshot resource"
        )
    return PinnedDataset(resolved, resource, revision, manifest_digest)


def _safe_file(root: Path, value: object, label: str) -> Path:
    relative = _string(value, label)
    candidate_relative = Path(relative)
    if (
        candidate_relative.is_absolute()
        or ".." in candidate_relative.parts
        or candidate_relative.suffix != ".jsonl"
    ):
        raise MolecularBaselineError(f"{label} must be a relative .jsonl path without '..'")
    candidate = root.joinpath(candidate_relative)
    if candidate.is_symlink():
        raise MolecularBaselineError(f"{label} must not identify a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise MolecularBaselineError(f"{label} does not exist") from error
    if not resolved.is_file() or root not in resolved.parents:
        raise MolecularBaselineError(f"{label} must remain within the snapshot")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile(value: object, location: str, limits: Limits) -> Profile:
    item = _exact_object(
        value,
        {
            "schema",
            "speciesTaxon",
            "sourceId",
            "contextId",
            "recordRole",
            "perturbationId",
            "interventionIds",
            "readoutIds",
            "values",
        },
        location,
    )
    if item["schema"] != RECORD_SCHEMA:
        raise MolecularBaselineError(f"{location}.schema must be {RECORD_SCHEMA}")
    taxon = item["speciesTaxon"]
    if not isinstance(taxon, int) or isinstance(taxon, bool) or taxon < 1:
        raise MolecularBaselineError(f"{location}.speciesTaxon must be a positive integer")
    source = _curie(item["sourceId"], f"{location}.sourceId")
    context = _curie(item["contextId"], f"{location}.contextId")
    role = item["recordRole"]
    if role not in {"basal-control", "perturbation-outcome"}:
        raise MolecularBaselineError(
            f"{location}.recordRole must explicitly be basal-control or perturbation-outcome"
        )
    interventions = _sorted_curies(
        item["interventionIds"], f"{location}.interventionIds", allow_empty=role == "basal-control"
    )
    perturbation_raw = item["perturbationId"]
    if role == "basal-control":
        if perturbation_raw is not None or interventions:
            raise MolecularBaselineError(
                f"{location}: basal-control requires null perturbationId and empty interventionIds"
            )
        perturbation = None
    else:
        perturbation = _curie(perturbation_raw, f"{location}.perturbationId")
        if not interventions:
            raise MolecularBaselineError(
                f"{location}: perturbation-outcome requires explicit interventionIds"
            )
    readouts = _sorted_curies(item["readoutIds"], f"{location}.readoutIds", allow_empty=False)
    if len(readouts) > limits.max_readouts_per_record:
        raise MolecularBaselineError(f"{location} exceeds maxReadoutsPerRecord")
    raw_values = item["values"]
    if not isinstance(raw_values, list) or len(raw_values) != len(readouts):
        raise MolecularBaselineError(f"{location}.values must align exactly with readoutIds")
    values: list[float | None] = []
    for index, raw in enumerate(raw_values):
        if raw is None:
            values.append(None)
        elif isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(raw):
            values.append(float(raw))
        else:
            raise MolecularBaselineError(f"{location}.values[{index}] must be finite or null")
    if all(value is None for value in values):
        raise MolecularBaselineError(f"{location} must contain at least one observed value")
    return Profile(
        species_taxon=taxon,
        source_id=source,
        context_id=context,
        role=role,
        perturbation_id=perturbation,
        intervention_ids=interventions,
        readout_ids=readouts,
        values=tuple(values),
    )


def _read_jsonl(path: Path, limits: Limits, label: str) -> Iterator[object]:
    with path.open("rb") as handle:
        line_number = 0
        while True:
            raw = handle.readline(limits.max_line_bytes + 1)
            if not raw:
                break
            line_number += 1
            if len(raw) > limits.max_line_bytes:
                raise MolecularBaselineError(f"{label}:{line_number} exceeds maxLineBytes")
            if not raw.endswith(b"\n") and handle.peek(1):
                raise MolecularBaselineError(f"{label}:{line_number} exceeds maxLineBytes")
            if not raw.strip():
                raise MolecularBaselineError(f"{label}:{line_number} is blank")
            try:
                yield json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise MolecularBaselineError(f"{label}:{line_number} is not valid JSON") from error


def load_snapshot(root_value: str | Path, expected_role: str, limits: Limits) -> Snapshot:
    root = Path(root_value).resolve(strict=True)
    manifest_path = root / "baseline.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise MolecularBaselineError("snapshot requires a regular baseline.json manifest")
    manifest_bytes = manifest_path.read_bytes()
    try:
        raw_manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MolecularBaselineError("baseline.json is not valid JSON") from error
    common = {
        "schema",
        "datasetId",
        "version",
        "role",
        "taskName",
        "labelClass",
        "benchmarkLabelsPresent",
        "valueSpace",
        "speciesTaxa",
        "sourceIds",
        "recordsEncoding",
        "profileLevel",
        "aggregationProtocolSha256",
        "shards",
    }
    expected = common | ({"pairedTrainingManifestSha256"} if expected_role == REFERENCE_ROLE else set())
    manifest = _exact_object(raw_manifest, expected, "baseline.json")
    if manifest["schema"] != SNAPSHOT_SCHEMA:
        raise MolecularBaselineError(f"baseline.json.schema must be {SNAPSHOT_SCHEMA}")
    _curie(manifest["datasetId"], "baseline.json.datasetId")
    _string(manifest["version"], "baseline.json.version")
    if manifest["role"] != expected_role:
        raise MolecularBaselineError(f"baseline.json.role must be {expected_role}")
    if manifest["taskName"] not in TASKS:
        raise MolecularBaselineError("baseline.json.taskName is not a frozen protocol task")
    if manifest["labelClass"] != "molecular" or manifest["benchmarkLabelsPresent"] is not False:
        raise MolecularBaselineError("snapshot must contain molecular values and no benchmark labels")
    _curie(manifest["valueSpace"], "baseline.json.valueSpace")
    if manifest["recordsEncoding"] != "identity-keyed-sparse-jsonl-v1":
        raise MolecularBaselineError("baseline.json.recordsEncoding is not supported")
    if manifest["profileLevel"] != "context-perturbation-centroid-v1":
        raise MolecularBaselineError(
            "baseline.json.profileLevel must be context-perturbation-centroid-v1"
        )
    _sha256(
        manifest["aggregationProtocolSha256"],
        "baseline.json.aggregationProtocolSha256",
    )
    taxa_raw = manifest["speciesTaxa"]
    if (
        not isinstance(taxa_raw, list)
        or not taxa_raw
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in taxa_raw)
        or taxa_raw != sorted(set(taxa_raw))
    ):
        raise MolecularBaselineError("baseline.json.speciesTaxa must be sorted unique positive integers")
    sources = _sorted_curies(manifest["sourceIds"], "baseline.json.sourceIds", allow_empty=False)
    if expected_role == REFERENCE_ROLE:
        _sha256(
            manifest["pairedTrainingManifestSha256"],
            "baseline.json.pairedTrainingManifestSha256",
        )
    shards = manifest["shards"]
    if not isinstance(shards, list) or not shards or len(shards) > limits.max_shards:
        raise MolecularBaselineError("baseline.json.shards must be a non-empty bounded list")
    profiles: list[Profile] = []
    seen_shard_paths: set[str] = set()
    for shard_index, raw_shard in enumerate(shards):
        shard = _exact_object(raw_shard, {"path", "sha256", "records"}, f"shards[{shard_index}]")
        path_text = _string(shard["path"], f"shards[{shard_index}].path")
        if path_text in seen_shard_paths:
            raise MolecularBaselineError("baseline.json.shards contains a duplicate path")
        seen_shard_paths.add(path_text)
        path = _safe_file(root, path_text, f"shards[{shard_index}].path")
        expected_digest = _sha256(shard["sha256"], f"shards[{shard_index}].sha256")
        if _file_sha256(path) != expected_digest:
            raise MolecularBaselineError(f"shards[{shard_index}] checksum mismatch")
        count = shard["records"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise MolecularBaselineError(f"shards[{shard_index}].records must be positive")
        parsed = [
            _profile(value, f"{path_text}:{line_number}", limits)
            for line_number, value in enumerate(_read_jsonl(path, limits, path_text), 1)
        ]
        if len(parsed) != count:
            raise MolecularBaselineError(f"shards[{shard_index}].records does not match the shard")
        profiles.extend(parsed)
        if len(profiles) > limits.max_records:
            raise MolecularBaselineError("snapshot exceeds maxRecords")
    seen_keys: set[tuple[int, str, str, str, str]] = set()
    context_owner: dict[str, tuple[int, str]] = {}
    perturbation_definition: dict[str, tuple[int, tuple[str, ...]]] = {}
    intervention_taxon: dict[str, int] = {}
    for profile in profiles:
        if profile.key in seen_keys:
            if profile.role == "basal-control":
                raise MolecularBaselineError(
                    f"ambiguous basal controls for context {profile.context_key}"
                )
            raise MolecularBaselineError(f"duplicate profile identity key: {profile.key}")
        seen_keys.add(profile.key)
        owner = (profile.species_taxon, profile.source_id)
        previous_owner = context_owner.setdefault(profile.context_id, owner)
        if previous_owner != owner:
            raise MolecularBaselineError(
                f"contextId {profile.context_id} has a source/species mismatch"
            )
        if profile.perturbation_id is not None:
            definition = (profile.species_taxon, profile.intervention_ids)
            previous_definition = perturbation_definition.setdefault(
                profile.perturbation_id, definition
            )
            if previous_definition != definition:
                raise MolecularBaselineError(
                    f"perturbationId {profile.perturbation_id} has inconsistent interventions/species"
                )
        for intervention in profile.intervention_ids:
            previous_taxon = intervention_taxon.setdefault(intervention, profile.species_taxon)
            if previous_taxon != profile.species_taxon:
                raise MolecularBaselineError(
                    f"interventionId {intervention} has a species mismatch"
                )
    actual_taxa = sorted({profile.species_taxon for profile in profiles})
    actual_sources = tuple(sorted({profile.source_id for profile in profiles}))
    if actual_taxa != taxa_raw or actual_sources != sources:
        raise MolecularBaselineError(
            "manifest speciesTaxa/sourceIds must exactly describe all profile records"
        )
    return Snapshot(
        root=root,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        profiles=tuple(sorted(profiles, key=lambda item: item.key)),
    )


def _basal_by_context(snapshot: Snapshot, label: str) -> dict[tuple[int, str, str], Profile]:
    result: dict[tuple[int, str, str], Profile] = {}
    for profile in snapshot.profiles:
        if profile.role != "basal-control":
            continue
        if profile.context_key in result:
            raise MolecularBaselineError(
                f"{label} has ambiguous basal controls for context {profile.context_key}"
            )
        result[profile.context_key] = profile
    outcomes = [profile for profile in snapshot.profiles if profile.role == "perturbation-outcome"]
    if not outcomes:
        raise MolecularBaselineError(f"{label} has no perturbation-outcome records")
    for profile in outcomes:
        if profile.context_key not in result:
            raise MolecularBaselineError(
                f"{label} outcome {profile.key} has no explicit matched basal-control"
            )
    return result


def _validate_pair(training: Snapshot, reference: Snapshot) -> None:
    if training.task_name != reference.task_name:
        raise MolecularBaselineError("training/reference taskName mismatch")
    if training.value_space != reference.value_space:
        raise MolecularBaselineError("training/reference valueSpace mismatch")
    if (
        training.manifest["aggregationProtocolSha256"]
        != reference.manifest["aggregationProtocolSha256"]
    ):
        raise MolecularBaselineError("training/reference aggregation protocol mismatch")
    if reference.manifest["pairedTrainingManifestSha256"] != training.manifest_sha256:
        raise MolecularBaselineError("reference is not pinned to this training manifest")
    training_keys = {profile.key for profile in training.profiles}
    overlap = training_keys & {profile.key for profile in reference.profiles}
    if overlap:
        raise MolecularBaselineError(f"training/reference contain duplicate identity keys: {min(overlap)}")
    context_owner: dict[str, tuple[int, str]] = {}
    perturbation_definition: dict[str, tuple[int, tuple[str, ...]]] = {}
    intervention_taxon: dict[str, int] = {}
    for profile in (*training.profiles, *reference.profiles):
        owner = (profile.species_taxon, profile.source_id)
        previous_owner = context_owner.setdefault(profile.context_id, owner)
        if previous_owner != owner:
            raise MolecularBaselineError(
                f"contextId {profile.context_id} has a cross-snapshot source/species mismatch"
            )
        if profile.perturbation_id is not None:
            definition = (profile.species_taxon, profile.intervention_ids)
            previous_definition = perturbation_definition.setdefault(
                profile.perturbation_id, definition
            )
            if previous_definition != definition:
                raise MolecularBaselineError(
                    f"perturbationId {profile.perturbation_id} has a cross-snapshot intervention/species mismatch"
                )
        for intervention in profile.intervention_ids:
            previous_taxon = intervention_taxon.setdefault(
                intervention, profile.species_taxon
            )
            if previous_taxon != profile.species_taxon:
                raise MolecularBaselineError(
                    f"interventionId {intervention} has a cross-snapshot species mismatch"
                )
    training_outcomes = [
        profile for profile in training.profiles if profile.role == "perturbation-outcome"
    ]
    reference_outcomes = [
        profile for profile in reference.profiles if profile.role == "perturbation-outcome"
    ]
    training_pairs = {profile.species_source for profile in training_outcomes}
    missing_pairs = sorted(
        {profile.species_source for profile in reference_outcomes} - training_pairs
    )
    if missing_pairs:
        raise MolecularBaselineError(
            f"reference source/species pairs lack fitting outcomes: {missing_pairs}"
        )
    task = reference.task_name
    train_interventions = {
        intervention
        for profile in training_outcomes
        for intervention in profile.intervention_ids
    }
    reference_interventions = {
        intervention
        for profile in reference_outcomes
        for intervention in profile.intervention_ids
    }
    if task in {"intervention-gene-cold", "double-cold"}:
        leaked = sorted(train_interventions & reference_interventions)
        if leaked:
            raise MolecularBaselineError(
                f"intervention-gene-cold leakage into fitting outcomes: {leaked}"
            )
    elif task == "perturbation-context-cold-with-basal-access":
        training_availability = {
            (profile.species_taxon, profile.source_id, intervention)
            for profile in training_outcomes
            for intervention in profile.intervention_ids
        }
        reference_requirements = {
            (profile.species_taxon, profile.source_id, intervention)
            for profile in reference_outcomes
            for intervention in profile.intervention_ids
        }
        unavailable = sorted(reference_requirements - training_availability)
        if unavailable:
            raise MolecularBaselineError(
                "perturbation-context-cold requires quantitative fitting outcomes "
                "for every reference intervention in other contexts within its "
                f"species/source stratum: {unavailable}"
            )
    if task in {"perturbation-context-cold-with-basal-access", "double-cold"}:
        leaked_contexts = sorted(
            {profile.context_key for profile in training_outcomes}
            & {profile.context_key for profile in reference_outcomes}
        )
        if leaked_contexts:
            raise MolecularBaselineError(
                f"perturbation-context-cold leakage into fitting outcomes: {leaked_contexts}"
            )


def _mean(values: Iterable[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise MolecularBaselineError("internal error: cannot average an empty collection")
    result = math.fsum(ordered) / len(ordered)
    if not math.isfinite(result):
        raise MolecularBaselineError("a fitted baseline mean is non-finite")
    return result


def _append(table: dict[tuple[object, ...], list[float]], key: tuple[object, ...], value: float) -> None:
    table.setdefault(key, []).append(value)


def _fit_effects(
    training: Snapshot, basal: Mapping[tuple[int, str, str], Profile]
) -> tuple[
    dict[tuple[object, ...], float],
    dict[tuple[object, ...], float],
    dict[tuple[object, ...], float],
    int,
]:
    exact_values: dict[tuple[object, ...], list[float]] = {}
    single_values: dict[tuple[object, ...], list[float]] = {}
    global_values: dict[tuple[object, ...], list[float]] = {}
    effect_values = 0
    for profile in training.profiles:
        if profile.role != "perturbation-outcome":
            continue
        basal_values = basal[profile.context_key].observed()
        overlap = 0
        for readout, outcome in profile.observed().items():
            if readout not in basal_values:
                continue
            effect = outcome - basal_values[readout]
            if not math.isfinite(effect):
                raise MolecularBaselineError("a derived training perturbation effect is non-finite")
            prefix = (profile.species_taxon, profile.source_id)
            _append(exact_values, (*prefix, profile.intervention_ids, readout), effect)
            _append(global_values, (*prefix, readout), effect)
            if len(profile.intervention_ids) == 1:
                _append(single_values, (*prefix, profile.intervention_ids[0], readout), effect)
            overlap += 1
            effect_values += 1
        if overlap == 0:
            raise MolecularBaselineError(
                f"training outcome {profile.key} has no observed readout shared with its basal control"
            )
    return (
        {key: _mean(values) for key, values in exact_values.items()},
        {key: _mean(values) for key, values in single_values.items()},
        {key: _mean(values) for key, values in global_values.items()},
        effect_values,
    )


def _prediction_rows(
    reference: Snapshot,
    reference_basal: Mapping[tuple[int, str, str], Profile],
    exact: Mapping[tuple[object, ...], float],
    single: Mapping[tuple[object, ...], float],
    global_mean: Mapping[tuple[object, ...], float],
    baseline: str,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    rows: list[dict[str, object]] = []
    observed_targets = 0
    predicted_targets = 0
    exact_effects = 0
    additive_effects = 0
    global_fallback_effects = 0
    for profile in reference.profiles:
        if profile.role != "perturbation-outcome":
            continue
        basal_values = reference_basal[profile.context_key].observed()
        predictions: list[float | None] = []
        for readout, target in zip(profile.readout_ids, profile.values):
            if target is None:
                predictions.append(None)
                continue
            observed_targets += 1
            basal_value = basal_values.get(readout)
            if basal_value is None:
                predictions.append(None)
                continue
            if baseline == "context-only":
                prediction = basal_value
            else:
                prefix = (profile.species_taxon, profile.source_id)
                exact_key = (*prefix, profile.intervention_ids, readout)
                if exact_key in exact:
                    prediction = basal_value + exact[exact_key]
                    exact_effects += 1
                else:
                    effects: list[float] = []
                    fallback_count = 0
                    for intervention in profile.intervention_ids:
                        single_key = (*prefix, intervention, readout)
                        global_key = (*prefix, readout)
                        if single_key in single:
                            effects.append(single[single_key])
                        elif global_key in global_mean:
                            effects.append(global_mean[global_key])
                            fallback_count += 1
                        else:
                            effects = []
                            break
                    if not effects:
                        predictions.append(None)
                        continue
                    prediction = basal_value + math.fsum(effects)
                    additive_effects += 1
                    global_fallback_effects += fallback_count
            if not math.isfinite(prediction):
                raise MolecularBaselineError("a generated prediction is non-finite")
            predictions.append(prediction)
            predicted_targets += 1
        rows.append(
            {
                "schema": PREDICTION_RECORD_SCHEMA,
                "speciesTaxon": profile.species_taxon,
                "sourceId": profile.source_id,
                "contextId": profile.context_id,
                "perturbationId": profile.perturbation_id,
                "interventionIds": list(profile.intervention_ids),
                "readoutIds": list(profile.readout_ids),
                "predictionMean": predictions,
            }
        )
    if not rows:
        raise MolecularBaselineError("reference has no perturbation outcomes to predict")
    return rows, {
        "profiles": len(rows),
        "observedReferenceValues": observed_targets,
        "predictedValues": predicted_targets,
        "missingPredictions": observed_targets - predicted_targets,
        "exactEffectPredictions": exact_effects,
        "additiveEffectPredictions": additive_effects,
        "globalFallbackComponents": global_fallback_effects,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_prediction_artifact(
    destination: Path,
    baseline: str,
    rows: Sequence[Mapping[str, object]],
    training: Snapshot,
    reference: Snapshot,
) -> str:
    destination.mkdir(parents=True, exist_ok=False)
    shard = destination / "predictions-000.jsonl"
    with shard.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )
    manifest = {
        "schema": PREDICTION_SCHEMA,
        "baselineName": baseline,
        "baselineDefinitionVersion": "1.0.0",
        "role": "molecular-baseline-point-predictions",
        "taskName": reference.task_name,
        "labelClass": "molecular",
        "benchmarkLabelsPresent": False,
        "valueSpace": reference.value_space,
        "speciesTaxa": sorted({row["speciesTaxon"] for row in rows}),
        "sourceIds": sorted({row["sourceId"] for row in rows}),
        "profileLevel": reference.manifest["profileLevel"],
        "aggregationProtocolSha256": reference.manifest[
            "aggregationProtocolSha256"
        ],
        "trainingManifestSha256": training.manifest_sha256,
        "referenceManifestSha256": reference.manifest_sha256,
        "uncertainty": {
            "status": "contract-blocked",
            "reasonCode": "prediction-log-scale-not-defined",
            "detail": "The frozen point baselines do not define a probabilistic scale; no residual scale is inferred.",
        },
        "shards": [
            {
                "path": shard.name,
                "sha256": _file_sha256(shard),
                "records": len(rows),
            }
        ],
    }
    manifest_path = destination / "predictions.json"
    _write_json(manifest_path, manifest)
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def build_baselines(
    training_root: str | Path,
    reference_root: str | Path,
    output_root: str | Path,
    limits: Limits | None = None,
) -> dict[str, object]:
    """Validate paired snapshots and write both frozen point baselines."""

    limits = limits or Limits()
    training = load_snapshot(training_root, TRAINING_ROLE, limits)
    reference = load_snapshot(reference_root, REFERENCE_ROLE, limits)
    _validate_pair(training, reference)
    training_basal = _basal_by_context(training, "training")
    reference_basal = _basal_by_context(reference, "reference")
    exact, single, global_mean, effect_values = _fit_effects(training, training_basal)
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=False)
    summaries: dict[str, dict[str, int]] = {}
    manifest_digests: dict[str, str] = {}
    for baseline in BASELINES:
        rows, summary = _prediction_rows(
            reference, reference_basal, exact, single, global_mean, baseline
        )
        artifact_name = baseline
        digest = _write_prediction_artifact(
            destination / artifact_name, baseline, rows, training, reference
        )
        summaries[baseline] = summary
        manifest_digests[baseline] = digest
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "protocolResource": "evaluations/slp-1-1-molecular-comparison-protocol-v1.yaml",
        "taskName": reference.task_name,
        "profileLevel": reference.manifest["profileLevel"],
        "aggregationProtocolSha256": reference.manifest[
            "aggregationProtocolSha256"
        ],
        "training": {
            "manifestSha256": training.manifest_sha256,
            "profiles": len(training.profiles),
            "fittedEffectValues": effect_values,
        },
        "reference": {
            "manifestSha256": reference.manifest_sha256,
            "profiles": len(reference.profiles),
        },
        "baselines": {
            baseline: {
                **summaries[baseline],
                "predictionManifestSha256": manifest_digests[baseline],
            }
            for baseline in BASELINES
        },
        "evaluationCompatibility": {
            "targetSchema": "slp.molecular-evaluation/v1",
            "status": "contract-blocked",
            "reasonCode": "prediction-log-scale-not-defined",
            "detail": "Point predictions omit predictionLogScale; a probabilistic scale must be frozen before conversion.",
        },
        "featureBilinearRidge": {
            "status": "protocol-required-contract-blocked",
            "reasonCode": "feature-vectors-absent",
            "detail": "The v1 baseline snapshots carry identity and sparse values, not query/action feature vectors.",
        },
        "missingness": {
            "semantics": "Null reference values remain null; an observed value receives null when its explicit basal or fitting effect is unavailable.",
            "implicitZero": False,
        },
    }
    _write_json(destination / "baseline-report.json", report)
    return report
