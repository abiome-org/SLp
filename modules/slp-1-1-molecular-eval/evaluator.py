"""Fail-closed evaluation of frozen, target-free molecular predictions.

The producer-facing prediction artifact never contains held targets or an
observed-value mask. Held truth, fitting-only centering profiles, a strict
corpus audit, and the outcome-blind held roster are independently admitted,
copied OMF DatasetSnapshots consumed only by this evaluator.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import tarfile
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

SCHEMA = "slp.molecular-evaluation/v2"
CENTERING_ROLE = "molecular-centering-reference"
TRUTH_ROLE = "molecular-validation-truth"
PREDICTION_ROLE = "molecular-validation-predictions"
QUERY_SCHEMA = "slp.molecular-query-manifest/v1"
REPORT_SCHEMA = "slp.molecular-evaluation-report/v3"
DIAGNOSTIC_SCHEMA = "slp.molecular-profile-diagnostic/v3"
CORPUS_AUDIT_SCHEMA = "slp.corpus-audit/v1.2"
ROSTER_SCHEMA = "slp.held-intervention-roster-report/v1"
ROSTER_ASSIGNMENT_DOMAIN = b"slp-1.1-yeast-global-held-v1\x00"
YEAST_TAXON = 4932
MAX_MANIFEST_BYTES = 1_048_576
MAX_SHARDS = 128
MAX_RECORDS = 2_000_000
MAX_SHARD_BYTES = 8 * 1024 * 1024 * 1024
MAX_TOTAL_SHARD_BYTES = 64 * 1024 * 1024 * 1024
MAX_AUXILIARY_DOCUMENT_BYTES = 16 * 1024 * 1024
MAX_ROSTER_RECORDS = 200_000
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024 * 1024
MAX_PREDICTION_BUNDLE_BYTES = MAX_TOTAL_SHARD_BYTES + MAX_MANIFEST_BYTES + 1024 * 1024
MAX_ABSOLUTE_TARGET = 1_000_000_000_000.0
SYSTEMA_DOI = "10.1038/s41587-025-02777-8"
MINIMUM_PERTURBED_CENTROID_PEARSON = 0.10
MINIMUM_SPECIES_PERTURBED_CENTROID_PEARSON = 0.0
CENTRAL_50_NORMAL_Z = 0.6744897501960817
CENTRAL_90_NORMAL_Z = 1.6448536269514722
HEX_DIGITS = frozenset("0123456789abcdef")
SGD_INTERVENTION = re.compile(r"^SGD:S[0-9]{9}$")

COMMON_MANIFEST_FIELDS = {
    "schema", "datasetId", "version", "role", "labelClass",
    "benchmarkLabelsPresent", "valueSpace", "speciesTaxa", "sourceIds", "shards",
}
ROLE_MANIFEST_FIELDS = {
    CENTERING_ROLE: COMMON_MANIFEST_FIELDS
    | {"fittingOnly"},
    TRUTH_ROLE: COMMON_MANIFEST_FIELDS
    | {"evaluatorOnly", "queryResource", "queryDatasetManifestDigest", "queryManifestSha256"},
    PREDICTION_ROLE: COMMON_MANIFEST_FIELDS
    | {"modelCheckpointContentSha256", "queryResource", "queryDatasetManifestDigest",
       "queryManifestSha256", "targetValuesPresent", "observedMaskPresent"},
}
COMMON_RECORD_FIELDS = {
    "profileId", "speciesTaxon", "sourceId", "centeringGroup",
    "perturbationId", "interventionIds", "readoutIds", "distributionTypes",
}
ROLE_RECORD_FIELDS = {
    CENTERING_ROLE: COMMON_RECORD_FIELDS | {"target"},
    TRUTH_ROLE: COMMON_RECORD_FIELDS | {"target"},
    PREDICTION_ROLE: COMMON_RECORD_FIELDS | {"predictionParameters"},
}
QUERY_MANIFEST_FIELDS = {
    "schema", "datasetId", "version", "role", "labelClass", "targetValuesPresent",
    "observedMaskPresent", "valueSpace", "speciesTaxa", "sourceIds", "shards",
}
CORPUS_AUDIT_FIELDS = {
    "schema", "rewardEnabled", "auditPassed", "strictInterventionIsolation",
    "leakageViolations", "leakedTrajectoryGenes", "benchmarkLabelRecords",
    "omfPriorAdmissionRequired", "datasets", "heldRoster",
}
CORPUS_DATASET_FIELDS = {
    "resource", "revision", "manifestDigest", "datasetId", "version", "role",
    "corpusManifestSha256", "contentDigest", "trajectoryGenesSha256",
    "trajectoryGeneSetSha256", "trajectoryGeneCount", "records", "targetValues",
    "modalities", "sourceIds", "speciesTaxa",
}
AUDIT_DATASET_ROLES = {
    "pretrain": "pretrain",
    "molecularValidation": "molecular-validation",
    "molecularFinal": "molecular-final",
}
AUDIT_ROSTER_FIELDS = {
    "resource", "revision", "manifestDigest", "rosterSha256", "coverageSha256",
    "assignmentDomainHex", "bucketRule", "identityMappingId", "identityMappingSha256",
    "sourceInventories", "intersectionSize", "pretrainGeneCount",
    "validationGeneSetSha256", "validationGeneCount", "finalGeneSetSha256",
    "finalGeneCount", "unionGeneSetSha256", "unionGeneCount",
}
AUDIT_SOURCE_INVENTORY_FIELDS = {
    "resource", "revision", "artifactManifestDigest", "sourceId", "sourceRelease",
    "identityMappingId", "identityMappingSha256", "manifestSha256", "records",
    "duplicateRecords", "uniqueInterventions", "qcPassing", "qcFailed",
    "intersectionCoverage",
}
ROSTER_COVERAGE_FIELDS = {
    "schema", "assignment", "sourceCount", "identityMapping", "minimumIntersectionSize",
    "intersectionSize", "roleCounts", "rejectionCounts", "rosterPath", "rosterSha256",
    "sources",
}


class MolecularEvaluationError(ValueError):
    """Raised when evaluation evidence is incomplete, mutable, or invalid."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bounded_bytes(path: Path, maximum: int, label: str) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise MolecularEvaluationError(f"could not stat {label}") from error
    if size > maximum:
        raise MolecularEvaluationError(f"{label} exceeds its byte bound")
    return path.read_bytes()


def _require_exact_files(root: Path, expected: set[str], label: str) -> None:
    actual: set[str] = set()
    for entry in root.rglob("*"):
        if entry.is_symlink():
            raise MolecularEvaluationError(f"{label} must not contain symlinks")
        if entry.is_file():
            actual.add(entry.relative_to(root).as_posix())
        elif not entry.is_dir():
            raise MolecularEvaluationError(f"{label} contains a non-regular entry")
    if actual != expected:
        raise MolecularEvaluationError(
            f"{label} file set mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _bounded_lines(path: Path, maximum: int, label: str, *, compressed: bool = False) -> Iterator[tuple[int, bytes]]:
    opener = gzip.open if compressed else open
    try:
        with opener(path, "rb") as source:
            line_number = 0
            while True:
                line = source.readline(maximum + 1)
                if not line:
                    break
                line_number += 1
                if len(line) > maximum:
                    raise MolecularEvaluationError(f"{label}:{line_number} exceeds maxLineBytes")
                if not line.strip():
                    raise MolecularEvaluationError(f"{label}:{line_number} is empty")
                yield line_number, line
    except OSError as error:
        raise MolecularEvaluationError(f"could not read {label}") from error


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in HEX_DIGITS for c in value)


def _pinned_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or not _is_digest(value[7:]):
        raise MolecularEvaluationError(f"{name} must be an exact sha256:<lowercase hex> digest")
    return value


def _dataset_resource(value: object, name: str) -> str:
    resource = _string(value, name)
    if not resource.startswith("omf://") or "/datasetsnapshot/" not in resource:
        raise MolecularEvaluationError(f"{name} must be an OMF DatasetSnapshot resource URI")
    identity, separator, revision = resource.rpartition("@")
    if not separator or not identity.split("/")[-1]:
        raise MolecularEvaluationError(f"{name} must contain an exact resource revision")
    _pinned_digest(revision, f"{name} revision")
    return resource


def _stable_id(value: object, name: str) -> str:
    if (
        not isinstance(value, str) or not value or value != value.strip()
        or ":" not in value or any(character.isspace() for character in value)
        or not all(value.partition(":")[index] for index in (0, 2))
    ):
        raise MolecularEvaluationError(f"{name} must be a trimmed stable namespaced identifier")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MolecularEvaluationError(f"{name} must be a non-empty trimmed string")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MolecularEvaluationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MolecularEvaluationError(f"{name} must be finite")
    return result


def _logaddexp(left: float, right: float) -> float:
    maximum = max(left, right)
    return maximum + math.log1p(math.exp(min(left, right) - maximum))


def _strict_fields(value: object, expected: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise MolecularEvaluationError(f"{label} must be a JSON object")
    missing, extra = sorted(expected - set(value)), sorted(set(value) - expected)
    if missing or extra:
        raise MolecularEvaluationError(
            f"{label} fields do not match the contract; missing={missing}, extra={extra}"
        )
    return value


def _relative_file(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise MolecularEvaluationError(f"{label} must be a non-empty string")
    portable = PurePosixPath(relative)
    if (portable.is_absolute() or relative != portable.as_posix() or "\\" in relative
            or ":" in relative or any(part in {"", ".", ".."} for part in portable.parts)):
        raise MolecularEvaluationError(f"{label} must be a canonical relative POSIX path")
    cursor = root
    for part in portable.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise MolecularEvaluationError(f"{label} must not contain a symlink")
    try:
        path = root.joinpath(*portable.parts).resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as error:
        raise MolecularEvaluationError(f"{label} escapes or is missing from its artifact") from error
    if not path.is_file():
        raise MolecularEvaluationError(f"{label} must be a regular file")
    return path


def _artifact_path(value: str | Path, expected_name: str) -> Path:
    requested = Path(value)
    if requested.is_symlink():
        raise MolecularEvaluationError(f"{expected_name} artifact path must not be a symlink")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise MolecularEvaluationError(f"{expected_name} artifact is missing") from error
    if resolved.is_file():
        if resolved.name != expected_name:
            raise MolecularEvaluationError(f"artifact payload must be named {expected_name}")
        return resolved
    if not resolved.is_dir():
        raise MolecularEvaluationError(f"{expected_name} artifact must be a file or directory")
    return _relative_file(resolved, expected_name, expected_name)


def resolve_literal_omf_artifact(value: object, name: str) -> tuple[str, str]:
    """Require the exact object OMF creates for an admission-pinned artifact."""
    expected = {"resource", "kind", "artifacts", "paths", "path"}
    if not isinstance(value, dict) or set(value) != expected:
        raise MolecularEvaluationError(f"{name} must be an exact materialized OMF artifact object")
    if value.get("kind") != "artifact":
        raise MolecularEvaluationError(f"{name} must be a literal OMF artifact input")
    artifacts, paths, path = value.get("artifacts"), value.get("paths"), value.get("path")
    if not isinstance(artifacts, dict) or set(artifacts) != {"payload"}:
        raise MolecularEvaluationError(f"{name}.artifacts must contain only payload")
    digest = artifacts["payload"]
    if (not isinstance(digest, str) or not digest.startswith("sha256:")
            or not _is_digest(digest.removeprefix("sha256:"))):
        raise MolecularEvaluationError(f"{name} payload must be a SHA-256 artifact manifest")
    if value.get("resource") != f"artifact:{digest}":
        raise MolecularEvaluationError(f"{name} resource does not match its artifact manifest")
    if not isinstance(path, str) or not path or not isinstance(paths, dict) or paths != {"payload": path}:
        raise MolecularEvaluationError(f"{name} materialized payload path is inconsistent")
    return path, digest


@dataclass(frozen=True)
class PinnedCheckpointInput:
    path: str
    resource: str
    checkpoint_artifact_digest: str
    content_sha256: str


def resolve_pinned_checkpoint_input(value: object, name: str = "modelCheckpoint") -> PinnedCheckpointInput:
    """Validate and hash the exact file-valued OMF artifact containing checkpoint bytes."""
    path_value, artifact_digest = resolve_literal_omf_artifact(value, name)
    requested = Path(path_value)
    if requested.is_symlink():
        raise MolecularEvaluationError(f"{name} payload must not be a symlink")
    try:
        path = requested.resolve(strict=True)
    except OSError as error:
        raise MolecularEvaluationError(f"{name} payload is missing") from error
    if not path.is_file() or path.name != "payload" or path.parent.name != "payload":
        raise MolecularEvaluationError(
            f"{name} must use OMF file artifact semantics .../payload/payload"
        )
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise MolecularEvaluationError(f"{name} checkpoint byte size is outside bounds")
    return PinnedCheckpointInput(
        str(path), f"artifact:{artifact_digest}", artifact_digest, _sha256(path)
    )


@dataclass(frozen=True)
class PinnedDatasetInput:
    input_name: str
    path: str
    resource: str
    revision: str
    manifest_digest: str


def resolve_pinned_dataset_input(
    value: object,
    name: str,
    *,
    expected_resource: str | None = None,
    expected_manifest_digest: str | None = None,
) -> PinnedDatasetInput:
    """Validate OMF's exact path-bound materialized DatasetSnapshot object."""
    expected = {"resource", "mode", "path", "manifestDigest"}
    if not isinstance(value, dict) or set(value) != expected:
        raise MolecularEvaluationError(f"{name} must be an exact materialized DatasetSnapshot")
    resource = _dataset_resource(value["resource"], f"{name}.resource")
    manifest_digest = _pinned_digest(value["manifestDigest"], f"{name}.manifestDigest")
    if expected_resource is not None and resource != expected_resource:
        raise MolecularEvaluationError(f"{name}.resource does not match the frozen workload URI")
    if expected_manifest_digest is not None and manifest_digest != expected_manifest_digest:
        raise MolecularEvaluationError(
            f"{name}.manifestDigest does not match the frozen workload digest"
        )
    if value["mode"] != "copy":
        raise MolecularEvaluationError(f"{name} must be an immutable copied DatasetSnapshot")
    path_value = _string(value["path"], f"{name}.path")
    requested = Path(path_value)
    if requested.is_symlink():
        raise MolecularEvaluationError(f"{name}.path must not be a symlink")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise MolecularEvaluationError(f"{name}.path is missing") from error
    if not resolved.is_dir():
        raise MolecularEvaluationError(f"{name}.path must be a directory")
    resource_name = resource.rpartition("/datasetsnapshot/")[2].partition("@")[0]
    if (
        resolved.name != resource_name
        or resolved.parent.name != name
        or resolved.parent.parent.name != "inputs"
    ):
        raise MolecularEvaluationError(
            f"{name}.path is inconsistent with its input name and resource name"
        )
    return PinnedDatasetInput(
        name, str(resolved), resource, resource.rpartition("@")[2], manifest_digest
    )


def resolve_pinned_query_input(
    value: object,
    name: str = "molecularQuery",
    *,
    expected_resource: str | None = None,
    expected_manifest_digest: str | None = None,
) -> PinnedDatasetInput:
    return resolve_pinned_dataset_input(
        value,
        name,
        expected_resource=expected_resource,
        expected_manifest_digest=expected_manifest_digest,
    )


@dataclass(frozen=True)
class SnapshotManifest:
    root: Path
    role: str
    dataset_id: str
    version: str
    value_space: str
    species_taxa: frozenset[int]
    source_ids: frozenset[str]
    manifest_sha256: str
    shards: tuple[dict[str, object], ...]
    model_checkpoint_content_sha256: str | None = None
    query_resource: str | None = None
    query_dataset_manifest_digest: str | None = None
    query_manifest_sha256: str | None = None
    archive_path: Path | None = None

    @classmethod
    def load(cls, root: str | Path, expected_role: str) -> "SnapshotManifest":
        requested = Path(root)
        if requested.is_symlink():
            raise MolecularEvaluationError("snapshot root must not be a symlink")
        try:
            snapshot_root = requested.resolve(strict=True)
        except OSError as error:
            raise MolecularEvaluationError("snapshot root is missing") from error
        if snapshot_root.is_file():
            if expected_role != PREDICTION_ROLE:
                raise MolecularEvaluationError(
                    "only target-free predictions may use the file-valued tar transport"
                )
            return cls._load_prediction_tar(snapshot_root)
        if not snapshot_root.is_dir():
            raise MolecularEvaluationError("molecular profile artifact must be a directory")
        manifest_path = _relative_file(snapshot_root, "evaluation.json", "manifest")
        manifest_bytes = _bounded_bytes(manifest_path, MAX_MANIFEST_BYTES, "evaluation.json")
        try:
            raw = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MolecularEvaluationError("evaluation.json is not valid UTF-8 JSON") from error
        _strict_fields(raw, ROLE_MANIFEST_FIELDS[expected_role], f"{expected_role} manifest")
        assert isinstance(raw, dict)
        if raw["schema"] != SCHEMA or raw["role"] != expected_role:
            raise MolecularEvaluationError(f"manifest must be {SCHEMA} role {expected_role}")
        expected_label = "none" if expected_role == PREDICTION_ROLE else "molecular"
        if raw["labelClass"] != expected_label:
            raise MolecularEvaluationError(f"{expected_role} labelClass must be {expected_label}")
        if raw["benchmarkLabelsPresent"] is not False:
            raise MolecularEvaluationError("benchmark-bearing evaluation input is forbidden")
        species = raw["speciesTaxa"]
        if (not isinstance(species, list) or not species or any(type(x) is not int or x <= 0 for x in species)
                or len(species) != len(set(species))):
            raise MolecularEvaluationError("speciesTaxa must contain unique positive taxonomy IDs")
        if species != [YEAST_TAXON]:
            raise MolecularEvaluationError(
                "molecular evaluator v2 roster contract is yeast-only (NCBI taxon 4932); "
                "mixed-species evaluation requires a new roster schema"
            )
        sources = raw["sourceIds"]
        if not isinstance(sources, list) or not sources:
            raise MolecularEvaluationError("sourceIds must be a non-empty list")
        source_ids = frozenset(_stable_id(x, "sourceIds[]") for x in sources)
        if len(source_ids) != len(sources):
            raise MolecularEvaluationError("sourceIds may not contain duplicates")
        raw_shards = raw["shards"]
        if not isinstance(raw_shards, list) or not raw_shards or len(raw_shards) > MAX_SHARDS:
            raise MolecularEvaluationError("evaluation manifest shard count is outside bounds")
        shards, seen, total_bytes, total_records = [], set(), 0, 0
        for index, item in enumerate(raw_shards):
            shard = _strict_fields(item, {"path", "sha256", "bytes", "records"}, f"shards[{index}]")
            path_value = shard["path"]
            if not isinstance(path_value, str) or path_value in seen:
                raise MolecularEvaluationError("shard paths must be unique strings")
            seen.add(path_value)
            path = _relative_file(snapshot_root, path_value, f"shards[{index}].path")
            if not (path_value.endswith(".jsonl") or path_value.endswith(".jsonl.gz")):
                raise MolecularEvaluationError("shards must end in .jsonl or .jsonl.gz")
            if not _is_digest(shard["sha256"]) or _sha256(path) != shard["sha256"]:
                raise MolecularEvaluationError(f"digest mismatch for evaluation shard: {path_value}")
            if type(shard["bytes"]) is not int or shard["bytes"] <= 0 or shard["bytes"] > MAX_SHARD_BYTES:
                raise MolecularEvaluationError("shard bytes must be a positive bounded integer")
            if path.stat().st_size != shard["bytes"]:
                raise MolecularEvaluationError(f"byte count mismatch for evaluation shard: {path_value}")
            if type(shard["records"]) is not int or shard["records"] <= 0:
                raise MolecularEvaluationError("shard records must be a positive integer")
            total_bytes += shard["bytes"]
            total_records += shard["records"]
            if total_bytes > MAX_TOTAL_SHARD_BYTES or total_records > MAX_RECORDS:
                raise MolecularEvaluationError("declared molecular snapshot size exceeds bounds")
            shards.append(dict(shard))
        _require_exact_files(snapshot_root, {"evaluation.json", *seen}, "molecular snapshot")
        model_digest = query_resource = None
        query_dataset_digest = query_digest = None
        if expected_role == CENTERING_ROLE:
            if raw["fittingOnly"] is not True:
                raise MolecularEvaluationError("centering reference must be fitting-only")
        elif expected_role == TRUTH_ROLE:
            if raw["evaluatorOnly"] is not True:
                raise MolecularEvaluationError("held truth must be evaluator-only molecularValidation")
            query_resource = _dataset_resource(raw["queryResource"], "queryResource")
            query_dataset_digest = _pinned_digest(
                raw["queryDatasetManifestDigest"], "queryDatasetManifestDigest"
            )
            query_digest = raw["queryManifestSha256"]
            if not _is_digest(query_digest):
                raise MolecularEvaluationError("truth query digest must be lowercase SHA-256")
        else:
            if raw["targetValuesPresent"] is not False or raw["observedMaskPresent"] is not False:
                raise MolecularEvaluationError("prediction manifest must attest no targets or observed mask")
            model_digest, query_digest = raw["modelCheckpointContentSha256"], raw["queryManifestSha256"]
            query_resource = _dataset_resource(raw["queryResource"], "queryResource")
            query_dataset_digest = _pinned_digest(
                raw["queryDatasetManifestDigest"], "queryDatasetManifestDigest"
            )
            if not _is_digest(model_digest) or not _is_digest(query_digest):
                raise MolecularEvaluationError("prediction provenance digests must be lowercase SHA-256")
        return cls(snapshot_root, expected_role, _string(raw["datasetId"], "datasetId"),
                   _string(raw["version"], "version"), _string(raw["valueSpace"], "valueSpace"),
                   frozenset(species), source_ids, hashlib.sha256(manifest_bytes).hexdigest(),
                   tuple(shards), model_digest,
                   query_resource, query_dataset_digest, query_digest)

    @classmethod
    def _load_prediction_tar(cls, path: Path) -> "SnapshotManifest":
        """Read the exact two-file, deterministic tar emitted by the trainer."""

        size = path.stat().st_size
        if size <= 0 or size > MAX_PREDICTION_BUNDLE_BYTES:
            raise MolecularEvaluationError("prediction tar byte size is outside bounds")
        try:
            with tarfile.open(path, mode="r:") as archive:
                members = archive.getmembers()
                if [member.name for member in members] != [
                    "evaluation.json", "profiles-000.jsonl"
                ]:
                    raise MolecularEvaluationError(
                        "prediction tar must contain exactly evaluation.json then profiles-000.jsonl"
                    )
                for member in members:
                    if (
                        not member.isreg()
                        or member.pax_headers
                        or member.mtime != 0
                        or member.uid != 0
                        or member.gid != 0
                        or member.uname != ""
                        or member.gname != ""
                        or member.mode != 0o644
                    ):
                        raise MolecularEvaluationError(
                            "prediction tar member metadata is not canonical"
                        )
                manifest_member, records_member = members
                if manifest_member.size <= 0 or manifest_member.size > MAX_MANIFEST_BYTES:
                    raise MolecularEvaluationError("prediction manifest byte size is outside bounds")
                manifest_stream = archive.extractfile(manifest_member)
                records_stream = archive.extractfile(records_member)
                if manifest_stream is None or records_stream is None:
                    raise MolecularEvaluationError("prediction tar member is unreadable")
                manifest_bytes = manifest_stream.read(MAX_MANIFEST_BYTES + 1)
                if len(manifest_bytes) != manifest_member.size:
                    raise MolecularEvaluationError("prediction manifest byte count drifted")
                records_digest = hashlib.sha256()
                records_bytes = 0
                while block := records_stream.read(1024 * 1024):
                    records_bytes += len(block)
                    if records_bytes > MAX_TOTAL_SHARD_BYTES:
                        raise MolecularEvaluationError("prediction records exceed aggregate bound")
                    records_digest.update(block)
        except (OSError, tarfile.TarError) as error:
            raise MolecularEvaluationError("prediction artifact is not a valid uncompressed tar") from error
        try:
            raw = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MolecularEvaluationError("prediction evaluation.json is not valid UTF-8 JSON") from error
        _strict_fields(raw, ROLE_MANIFEST_FIELDS[PREDICTION_ROLE], "prediction manifest")
        assert isinstance(raw, dict)
        if (
            raw["schema"] != SCHEMA
            or raw["role"] != PREDICTION_ROLE
            or raw["labelClass"] != "none"
            or raw["benchmarkLabelsPresent"] is not False
            or raw["targetValuesPresent"] is not False
            or raw["observedMaskPresent"] is not False
        ):
            raise MolecularEvaluationError("prediction tar is not structurally target-free")
        species = raw["speciesTaxa"]
        if species != [YEAST_TAXON]:
            raise MolecularEvaluationError("prediction tar v1 must be yeast-only")
        sources = raw["sourceIds"]
        if not isinstance(sources, list) or not sources:
            raise MolecularEvaluationError("prediction sourceIds must be non-empty")
        source_ids = frozenset(_stable_id(item, "prediction sourceIds[]") for item in sources)
        if len(source_ids) != len(sources):
            raise MolecularEvaluationError("prediction sourceIds may not contain duplicates")
        shards = raw["shards"]
        if not isinstance(shards, list) or len(shards) != 1:
            raise MolecularEvaluationError("prediction tar must declare exactly one record member")
        shard = _strict_fields(
            shards[0], {"path", "sha256", "bytes", "records"}, "prediction shard"
        )
        if (
            shard["path"] != "profiles-000.jsonl"
            or not _is_digest(shard["sha256"])
            or shard["sha256"] != records_digest.hexdigest()
            or type(shard["bytes"]) is not int
            or shard["bytes"] != records_bytes
            or records_member.size != records_bytes
            or type(shard["records"]) is not int
            or shard["records"] <= 0
            or shard["records"] > MAX_RECORDS
        ):
            raise MolecularEvaluationError("prediction tar record member binding is invalid")
        model_digest = raw["modelCheckpointContentSha256"]
        query_digest = raw["queryManifestSha256"]
        if not _is_digest(model_digest) or not _is_digest(query_digest):
            raise MolecularEvaluationError("prediction provenance digests must be lowercase SHA-256")
        query_resource = _dataset_resource(raw["queryResource"], "queryResource")
        query_dataset_digest = _pinned_digest(
            raw["queryDatasetManifestDigest"], "queryDatasetManifestDigest"
        )
        return cls(
            path.parent,
            PREDICTION_ROLE,
            _string(raw["datasetId"], "datasetId"),
            _string(raw["version"], "version"),
            _string(raw["valueSpace"], "valueSpace"),
            frozenset(species),
            source_ids,
            hashlib.sha256(manifest_bytes).hexdigest(),
            (dict(shard),),
            model_digest,
            query_resource,
            query_dataset_digest,
            query_digest,
            path,
        )

    def records(self, max_line_bytes: int) -> Iterator[tuple[str, int, dict[str, object]]]:
        if self.archive_path is not None:
            shard = self.shards[0]
            name = str(shard["path"])
            try:
                with tarfile.open(self.archive_path, mode="r:") as archive:
                    source = archive.extractfile(name)
                    if source is None:
                        raise MolecularEvaluationError("prediction record member is unreadable")
                    count = 0
                    while True:
                        line = source.readline(max_line_bytes + 1)
                        if not line:
                            break
                        count += 1
                        if len(line) > max_line_bytes or not line.strip():
                            raise MolecularEvaluationError(
                                f"{name}:{count} is empty or exceeds maxLineBytes"
                            )
                        try:
                            record = json.loads(line)
                        except (UnicodeDecodeError, json.JSONDecodeError) as error:
                            raise MolecularEvaluationError(
                                f"invalid JSON in {name}:{count}"
                            ) from error
                        if not isinstance(record, dict):
                            raise MolecularEvaluationError(f"{name}:{count} must be an object")
                        yield name, count, record
                    if count != shard["records"]:
                        raise MolecularEvaluationError(f"record count mismatch for {name}")
            except (OSError, tarfile.TarError) as error:
                raise MolecularEvaluationError("could not stream prediction tar") from error
            return
        for shard in self.shards:
            name = str(shard["path"])
            path = _relative_file(self.root, name, "shard path")
            count = 0
            for line_number, line in _bounded_lines(
                path, max_line_bytes, name, compressed=name.endswith(".gz")
            ):
                try:
                    record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise MolecularEvaluationError(f"invalid JSON in {name}:{line_number}") from error
                if not isinstance(record, dict):
                    raise MolecularEvaluationError(f"{name}:{line_number} must be an object")
                count += 1
                yield name, line_number, record
            if count != shard["records"]:
                raise MolecularEvaluationError(f"record count mismatch for {name}")


ProfileKey = tuple[int, str, str, str, str]
GroupKey = tuple[int, str, str]


def canonical_perturbation_id(intervention_ids: tuple[str, ...]) -> str:
    return "PERTURBATION:" + hashlib.sha256(_canonical_bytes(list(intervention_ids))).hexdigest()


def canonical_profile_id(
    taxon: int, source: str, centering_group: str, perturbation_id: str
) -> str:
    identity = {
        "speciesTaxon": taxon,
        "sourceId": source,
        "centeringGroup": centering_group,
        "perturbationId": perturbation_id,
    }
    return "PROFILE:" + hashlib.sha256(_canonical_bytes(identity)).hexdigest()


@dataclass
class Profile:
    key: ProfileKey
    intervention_ids: tuple[str, ...]
    readout_ids: tuple[str, ...]
    distribution_types: tuple[str, ...]
    target: dict[str, float] = field(default_factory=dict)
    prediction: dict[str, float] = field(default_factory=dict)
    log_scale: dict[str, float] = field(default_factory=dict)
    primary_parameter: dict[str, float] = field(default_factory=dict)

    def query_document(self) -> dict[str, object]:
        taxon, source, group, perturbation, profile_id = self.key
        return {"profileId": profile_id, "speciesTaxon": taxon, "sourceId": source,
                "centeringGroup": group, "perturbationId": perturbation,
                "interventionIds": list(self.intervention_ids), "readoutIds": list(self.readout_ids),
                "distributionTypes": list(self.distribution_types)}


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
    sum_nll: float = 0.0
    gaussian_targets: int = 0
    negative_binomial_targets: int = 0
    central_50_covered: int = 0
    central_90_covered: int = 0
    sum_central_50_interval_width: float = 0.0
    sum_central_90_interval_width: float = 0.0

    def add(
        self,
        prediction: float,
        target: float,
        distribution: str,
        primary_parameter: float,
        auxiliary_parameter: float,
    ) -> None:
        error = prediction - target
        self.count += 1
        self.sum_prediction += prediction
        self.sum_target += target
        self.sum_prediction_square += prediction * prediction
        self.sum_target_square += target * target
        self.sum_product += prediction * target
        self.sum_absolute_error += abs(error)
        self.sum_square_error += error * error
        if distribution == "gaussian":
            scale = math.exp(auxiliary_parameter)
            self.gaussian_targets += 1
            self.sum_nll += (
                0.5 * (error * math.exp(-auxiliary_parameter)) ** 2
                + auxiliary_parameter
                + 0.5 * math.log(2 * math.pi)
            )
            self.central_50_covered += int(abs(error) <= CENTRAL_50_NORMAL_Z * scale)
            self.central_90_covered += int(abs(error) <= CENTRAL_90_NORMAL_Z * scale)
            self.sum_central_50_interval_width += 2 * CENTRAL_50_NORMAL_Z * scale
            self.sum_central_90_interval_width += 2 * CENTRAL_90_NORMAL_Z * scale
        else:
            self.negative_binomial_targets += 1
            inverse_dispersion = math.exp(auxiliary_parameter)
            log_total = _logaddexp(auxiliary_parameter, primary_parameter)
            log_probability = (
                math.lgamma(target + inverse_dispersion)
                - math.lgamma(inverse_dispersion)
                - math.lgamma(target + 1.0)
                + inverse_dispersion * (auxiliary_parameter - log_total)
                + target * (primary_parameter - log_total)
            )
            self.sum_nll -= log_probability

    def merge(self, other: "ScalarMoments") -> None:
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
            raise MolecularEvaluationError("metric group has no observed held truth")
        pearson = self.pearson()
        return {"targets": self.count, "rmse": math.sqrt(self.sum_square_error / self.count),
                "mae": self.sum_absolute_error / self.count,
                "pearson": pearson if pearson is not None else 0.0,
                "pearsonDefined": pearson is not None,
                "meanNll": self.sum_nll / self.count,
                "distributionTargets": {"gaussian": self.gaussian_targets,
                    "negative-binomial": self.negative_binomial_targets}}

    def gaussian_calibration_report(self) -> dict[str, object]:
        if not self.gaussian_targets:
            return {"targets": 0,
                    "central50": {"nominalCoverage": 0.5, "z": CENTRAL_50_NORMAL_Z,
                        "empiricalCoverage": None, "meanIntervalWidth": None},
                    "central90": {"nominalCoverage": 0.9, "z": CENTRAL_90_NORMAL_Z,
                        "empiricalCoverage": None, "meanIntervalWidth": None}}
        return {"targets": self.gaussian_targets,
                "central50": {"nominalCoverage": 0.5, "z": CENTRAL_50_NORMAL_Z,
                    "empiricalCoverage": self.central_50_covered / self.gaussian_targets,
                    "meanIntervalWidth": self.sum_central_50_interval_width / self.gaussian_targets},
                "central90": {"nominalCoverage": 0.9, "z": CENTRAL_90_NORMAL_Z,
                    "empiricalCoverage": self.central_90_covered / self.gaussian_targets,
                    "meanIntervalWidth": self.sum_central_90_interval_width / self.gaussian_targets}}


@dataclass(frozen=True)
class ProfileMetric:
    key: ProfileKey
    readouts: int
    ordinary_pearson: float | None
    perturbed_centroid_pearson: float | None
    perturbed_centroid_cosine: float | None
    centroid_accuracy: float | None = None


def _parse_profile(manifest: SnapshotManifest, record: dict[str, object], location: str,
                   maximum_absolute_log_scale: float) -> Profile:
    _strict_fields(record, ROLE_RECORD_FIELDS[manifest.role], f"{location} {manifest.role} record")
    taxon = record["speciesTaxon"]
    if type(taxon) is not int or taxon not in manifest.species_taxa:
        raise MolecularEvaluationError(f"{location}: speciesTaxon is not declared")
    source = _stable_id(record["sourceId"], f"{location}.sourceId")
    if source not in manifest.source_ids:
        raise MolecularEvaluationError(f"{location}: sourceId is not declared")
    group = _string(record["centeringGroup"], f"{location}.centeringGroup")
    perturbation = _stable_id(record["perturbationId"], f"{location}.perturbationId")
    profile_id = _stable_id(record["profileId"], f"{location}.profileId")
    raw_interventions, raw_readouts = record["interventionIds"], record["readoutIds"]
    raw_distributions = record["distributionTypes"]
    if not isinstance(raw_interventions, list) or not raw_interventions:
        raise MolecularEvaluationError(f"{location}.interventionIds must be non-empty")
    if not isinstance(raw_readouts, list) or not raw_readouts:
        raise MolecularEvaluationError(f"{location}.readoutIds must be non-empty")
    if not isinstance(raw_distributions, list) or len(raw_distributions) != len(raw_readouts):
        raise MolecularEvaluationError(f"{location}.distributionTypes must align with readoutIds")
    interventions = tuple(_stable_id(x, f"{location}.interventionIds[]") for x in raw_interventions)
    readouts = tuple(_stable_id(x, f"{location}.readoutIds[]") for x in raw_readouts)
    distributions = tuple(raw_distributions)
    if len(interventions) != len(set(interventions)) or len(readouts) != len(set(readouts)):
        raise MolecularEvaluationError(f"{location}: interventionIds/readoutIds contain duplicates")
    if interventions != tuple(sorted(interventions)):
        raise MolecularEvaluationError(f"{location}.interventionIds must be canonical sorted IDs")
    if any(item not in {"gaussian", "negative-binomial"} for item in distributions):
        raise MolecularEvaluationError(f"{location}.distributionTypes contains an unsupported type")
    if perturbation != canonical_perturbation_id(interventions):
        raise MolecularEvaluationError(f"{location}.perturbationId is not derived from interventionIds")
    if profile_id != canonical_profile_id(taxon, source, group, perturbation):
        raise MolecularEvaluationError(f"{location}.profileId is not the canonical natural profile key")
    profile = Profile(
        (taxon, source, group, perturbation, profile_id), interventions, readouts, distributions
    )
    if manifest.role in {CENTERING_ROLE, TRUTH_ROLE}:
        target = record["target"]
        if not isinstance(target, list) or len(target) != len(readouts):
            raise MolecularEvaluationError(f"{location}.target must align with readoutIds")
        for index, (readout, value) in enumerate(zip(readouts, target)):
            if value is not None:
                parsed = _number(value, f"{location}.target[{index}]")
                if abs(parsed) > MAX_ABSOLUTE_TARGET:
                    raise MolecularEvaluationError(f"{location}.target[{index}] exceeds bound")
                if distributions[index] == "negative-binomial" and (
                    parsed < 0 or parsed != math.floor(parsed)
                ):
                    raise MolecularEvaluationError(
                        f"{location}.target[{index}] must be a non-negative integer count"
                    )
                profile.target[readout] = parsed
        if manifest.role == CENTERING_ROLE and not profile.target:
            raise MolecularEvaluationError(f"{location}: centering record has no fitting targets")
    else:
        parameters = record["predictionParameters"]
        if not isinstance(parameters, list) or len(parameters) != len(readouts):
            raise MolecularEvaluationError(
                f"{location}: predictions must cover the full preregistered readout panel"
            )
        for index, (readout, distribution) in enumerate(zip(readouts, distributions)):
            parameter = parameters[index]
            expected = {"mean", "logScale"} if distribution == "gaussian" else {
                "logMean", "logInverseDispersion"
            }
            parameter = _strict_fields(parameter, expected, f"{location}.predictionParameters[{index}]")
            if distribution == "gaussian":
                mean = _number(parameter["mean"], f"{location}.predictionParameters[{index}].mean")
                auxiliary = _number(parameter["logScale"], f"{location}.predictionParameters[{index}].logScale")
                expectation = mean
            else:
                mean = _number(parameter["logMean"], f"{location}.predictionParameters[{index}].logMean")
                auxiliary = _number(parameter["logInverseDispersion"], f"{location}.predictionParameters[{index}].logInverseDispersion")
                expectation = math.exp(mean)
            if (
                abs(auxiliary) > maximum_absolute_log_scale
                or (distribution == "negative-binomial" and abs(mean) > maximum_absolute_log_scale)
                or abs(expectation) > MAX_ABSOLUTE_TARGET
            ):
                raise MolecularEvaluationError(f"{location}.predictionParameters[{index}] exceeds bound")
            profile.prediction[readout], profile.log_scale[readout] = expectation, auxiliary
            profile.primary_parameter[readout] = mean
    return profile


def _load_profiles(manifest: SnapshotManifest, max_line_bytes: int,
                   maximum_absolute_log_scale: float) -> tuple[dict[str, Profile], int]:
    profiles: dict[str, Profile] = {}
    count = 0
    for shard, line, record in manifest.records(max_line_bytes):
        profile = _parse_profile(manifest, record, f"{shard}:{line}", maximum_absolute_log_scale)
        profile_id = profile.key[4]
        if profile_id in profiles:
            raise MolecularEvaluationError(f"duplicate profileId {profile_id}")
        profiles[profile_id], count = profile, count + 1
    if frozenset(x.key[0] for x in profiles.values()) != manifest.species_taxa:
        raise MolecularEvaluationError(f"{manifest.role} speciesTaxa do not exactly match records")
    if frozenset(x.key[1] for x in profiles.values()) != manifest.source_ids:
        raise MolecularEvaluationError(f"{manifest.role} sourceIds do not exactly match records")
    return profiles, count


def _load_query_snapshot(
    pinned: PinnedDatasetInput, max_line_bytes: int
) -> tuple[dict[str, Profile], dict[str, object], str, int]:
    root = Path(pinned.path)
    manifest_path = _relative_file(root, "query.json", "query manifest")
    try:
        raw = json.loads(_bounded_bytes(manifest_path, MAX_MANIFEST_BYTES, "query.json"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MolecularEvaluationError("query.json is not valid UTF-8 JSON") from error
    manifest = _strict_fields(raw, QUERY_MANIFEST_FIELDS, "query manifest")
    if (manifest["schema"] != QUERY_SCHEMA or manifest["role"] != "molecular-validation-query"
            or manifest["labelClass"] != "none" or manifest["targetValuesPresent"] is not False
            or manifest["observedMaskPresent"] is not False):
        raise MolecularEvaluationError("query manifest is not a target-free molecularValidation query")
    species, sources = manifest["speciesTaxa"], manifest["sourceIds"]
    if (not isinstance(species, list) or not species or any(type(x) is not int or x <= 0 for x in species)
            or len(species) != len(set(species))):
        raise MolecularEvaluationError("query speciesTaxa are invalid")
    if species != [YEAST_TAXON]:
        raise MolecularEvaluationError(
            "molecular query must be yeast-only under the v1 held-roster contract"
        )
    if not isinstance(sources, list) or not sources:
        raise MolecularEvaluationError("query sourceIds are invalid")
    source_ids = frozenset(_stable_id(x, "query sourceIds[]") for x in sources)
    if len(source_ids) != len(sources):
        raise MolecularEvaluationError("query sourceIds contain duplicates")
    raw_shards = manifest["shards"]
    if not isinstance(raw_shards, list) or not raw_shards or len(raw_shards) > MAX_SHARDS:
        raise MolecularEvaluationError("query manifest shard count is outside bounds")
    pseudo = SnapshotManifest(
        root=root, role=PREDICTION_ROLE, dataset_id=_string(manifest["datasetId"], "query datasetId"),
        version=_string(manifest["version"], "query version"),
        value_space=_string(manifest["valueSpace"], "query valueSpace"),
        species_taxa=frozenset(species), source_ids=source_ids,
        manifest_sha256=_sha256(manifest_path), shards=(),
    )
    profiles: dict[str, Profile] = {}
    total_records = 0
    total_bytes = 0
    seen_paths: set[str] = set()
    for shard_index, raw_shard in enumerate(raw_shards):
        shard = _strict_fields(raw_shard, {"path", "sha256", "bytes", "records"}, f"query shards[{shard_index}]")
        name = shard["path"]
        if not isinstance(name, str) or name in seen_paths or not (name.endswith(".jsonl") or name.endswith(".jsonl.gz")):
            raise MolecularEvaluationError("query shard paths must be unique JSONL paths")
        seen_paths.add(name)
        path = _relative_file(root, name, "query shard path")
        if not _is_digest(shard["sha256"]) or _sha256(path) != shard["sha256"]:
            raise MolecularEvaluationError(f"query shard digest mismatch: {name}")
        if type(shard["bytes"]) is not int or shard["bytes"] <= 0 or shard["bytes"] > MAX_SHARD_BYTES:
            raise MolecularEvaluationError("query shard bytes must be a positive bounded integer")
        if path.stat().st_size != shard["bytes"]:
            raise MolecularEvaluationError(f"query shard byte count mismatch: {name}")
        if type(shard["records"]) is not int or shard["records"] <= 0:
            raise MolecularEvaluationError("query shard record count must be positive")
        total_bytes += shard["bytes"]
        if total_bytes > MAX_TOTAL_SHARD_BYTES or total_records + shard["records"] > MAX_RECORDS:
            raise MolecularEvaluationError("declared query size exceeds bounds")
        seen_records = 0
        for line_number, line in _bounded_lines(
            path, max_line_bytes, name, compressed=name.endswith(".gz")
        ):
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise MolecularEvaluationError(f"invalid query JSON at {name}:{line_number}") from error
            _strict_fields(record, COMMON_RECORD_FIELDS, f"{name}:{line_number} query record")
            assert isinstance(record, dict)
            readouts, distributions = record.get("readoutIds"), record.get("distributionTypes")
            if not isinstance(readouts, list) or not isinstance(distributions, list):
                raise MolecularEvaluationError("query readoutIds/distributionTypes must be lists")
            parameters = [
                ({"mean": 0.0, "logScale": 0.0} if distribution == "gaussian"
                 else {"logMean": 0.0, "logInverseDispersion": 0.0})
                for distribution in distributions
            ]
            augmented = {**record, "predictionParameters": parameters}
            profile = _parse_profile(pseudo, augmented, f"{name}:{line_number}", 1.0)
            profile_id = profile.key[4]
            if profile_id in profiles:
                raise MolecularEvaluationError(f"duplicate query profileId {profile_id}")
            profiles[profile_id] = profile
            seen_records += 1
        if seen_records != shard["records"]:
            raise MolecularEvaluationError(f"query record count mismatch for {name}")
        total_records += seen_records
    _require_exact_files(root, {"query.json", *seen_paths}, "molecular query")
    if frozenset(x.key[0] for x in profiles.values()) != pseudo.species_taxa:
        raise MolecularEvaluationError("query speciesTaxa do not exactly match records")
    if frozenset(x.key[1] for x in profiles.values()) != pseudo.source_ids:
        raise MolecularEvaluationError("query sourceIds do not exactly match records")
    return profiles, manifest, _sha256(manifest_path), total_records


def _assert_query_join(query: dict[str, Profile], observed: dict[str, Profile], label: str) -> None:
    missing, extra = sorted(set(query) - set(observed)), sorted(set(observed) - set(query))
    if missing or extra:
        raise MolecularEvaluationError(
            f"{label}/query profile join mismatch; missing={missing[:10]}, extra={extra[:10]}"
        )
    for profile_id in sorted(query):
        if query[profile_id].query_document() != observed[profile_id].query_document():
            raise MolecularEvaluationError(
                f"{label}/query identity, intervention, or readout panel mismatch for {profile_id}"
            )


def _exact_join(truth: dict[str, Profile], predictions: dict[str, Profile]) -> dict[str, Profile]:
    missing, extra = sorted(set(truth) - set(predictions)), sorted(set(predictions) - set(truth))
    if missing or extra:
        raise MolecularEvaluationError(
            f"prediction/truth profile join mismatch; missing={missing[:10]}, extra={extra[:10]}"
        )
    for profile_id in sorted(truth):
        if truth[profile_id].query_document() != predictions[profile_id].query_document():
            raise MolecularEvaluationError(
                f"prediction/truth identity, intervention, or readout panel mismatch for {profile_id}"
            )
        truth[profile_id].prediction = predictions[profile_id].prediction
        truth[profile_id].log_scale = predictions[profile_id].log_scale
        truth[profile_id].primary_parameter = predictions[profile_id].primary_parameter
    return truth


def _load_json_dataset(
    pinned: PinnedDatasetInput, filename: str, expected_files: set[str], label: str
) -> tuple[dict[str, object], str]:
    root = Path(pinned.path)
    _require_exact_files(root, expected_files, label)
    document = _relative_file(root, filename, label)
    try:
        value = json.loads(_bounded_bytes(document, MAX_AUXILIARY_DOCUMENT_BYTES, label))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MolecularEvaluationError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise MolecularEvaluationError(f"{label} must be a JSON object")
    return value, _sha256(document)


def _load_corpus_audit(pinned: PinnedDatasetInput) -> tuple[dict[str, object], str]:
    audit, digest = _load_json_dataset(
        pinned, "corpus-audit.json", {"corpus-audit.json"}, "corpus audit snapshot"
    )
    _strict_fields(audit, CORPUS_AUDIT_FIELDS, "corpus audit")
    if (
        audit["schema"] != CORPUS_AUDIT_SCHEMA
        or audit["rewardEnabled"] is not False
        or audit["strictInterventionIsolation"] is not True
        or audit["auditPassed"] is not True
        or audit["omfPriorAdmissionRequired"] is not True
        or type(audit["leakageViolations"]) is not int
        or audit["leakageViolations"] != 0
        or audit["leakedTrajectoryGenes"] != []
        or type(audit["benchmarkLabelRecords"]) is not int
        or audit["benchmarkLabelRecords"] != 0
    ):
        raise MolecularEvaluationError("a passing admitted strict zero-leakage corpus audit is required")
    datasets = audit["datasets"]
    if not isinstance(datasets, dict) or set(datasets) != set(AUDIT_DATASET_ROLES):
        raise MolecularEvaluationError("corpus audit must attest exactly all three governed corpora")
    for name, identity in datasets.items():
        identity = _strict_fields(identity, CORPUS_DATASET_FIELDS, f"corpus audit dataset {name}")
        resource = _dataset_resource(identity["resource"], f"audit datasets.{name}.resource")
        if identity["revision"] != resource.rpartition("@")[2]:
            raise MolecularEvaluationError(f"audit dataset {name} revision is inconsistent")
        _pinned_digest(identity["manifestDigest"], f"audit datasets.{name}.manifestDigest")
        _stable_id(identity["datasetId"], f"audit datasets.{name}.datasetId")
        _string(identity["version"], f"audit datasets.{name}.version")
        if identity["role"] != AUDIT_DATASET_ROLES[name]:
            raise MolecularEvaluationError(f"audit dataset {name} role is inconsistent")
        for field_name in (
            "corpusManifestSha256", "contentDigest", "trajectoryGenesSha256",
            "trajectoryGeneSetSha256",
        ):
            if not _is_digest(identity[field_name]):
                raise MolecularEvaluationError(f"invalid audit digest {name}.{field_name}")
        for field_name in ("trajectoryGeneCount", "records", "targetValues"):
            minimum = 0 if field_name == "trajectoryGeneCount" else 1
            if type(identity[field_name]) is not int or identity[field_name] < minimum:
                raise MolecularEvaluationError(f"invalid audit count {name}.{field_name}")
        for field_name in ("modalities", "sourceIds"):
            values = identity[field_name]
            if (
                not isinstance(values, list) or not values
                or len(values) != len(set(values))
                or any(_stable_id(value, f"audit datasets.{name}.{field_name}[]") != value for value in values)
            ):
                raise MolecularEvaluationError(f"audit dataset {name} {field_name} is invalid")
        if identity["speciesTaxa"] != [YEAST_TAXON]:
            raise MolecularEvaluationError("audit v1.2 molecular boundary is yeast-only")
    roster = _strict_fields(audit["heldRoster"], AUDIT_ROSTER_FIELDS, "audit heldRoster")
    resource = _dataset_resource(roster["resource"], "audit heldRoster.resource")
    if roster["revision"] != resource.rpartition("@")[2]:
        raise MolecularEvaluationError("audit heldRoster revision is inconsistent")
    _pinned_digest(roster["manifestDigest"], "audit heldRoster.manifestDigest")
    for field_name in (
        "rosterSha256", "coverageSha256", "identityMappingSha256",
        "validationGeneSetSha256", "finalGeneSetSha256", "unionGeneSetSha256",
    ):
        if not _is_digest(roster[field_name]):
            raise MolecularEvaluationError(f"invalid audit heldRoster digest {field_name}")
    for field_name in (
        "intersectionSize", "pretrainGeneCount", "validationGeneCount",
        "finalGeneCount", "unionGeneCount",
    ):
        if type(roster[field_name]) is not int or roster[field_name] < 0:
            raise MolecularEvaluationError(f"invalid audit heldRoster count {field_name}")
    sources = roster["sourceInventories"]
    if not isinstance(sources, list) or len(sources) < 2:
        raise MolecularEvaluationError("audit heldRoster requires at least two protected inventories")
    source_ids: list[str] = []
    source_resources: list[str] = []
    for index, source in enumerate(sources):
        source = _strict_fields(
            source, AUDIT_SOURCE_INVENTORY_FIELDS,
            f"audit heldRoster sourceInventories[{index}]",
        )
        resource = _dataset_resource(
            source["resource"], f"audit heldRoster sourceInventories[{index}].resource"
        )
        if source["revision"] != resource.rpartition("@")[2]:
            raise MolecularEvaluationError("audit protected-inventory revision is inconsistent")
        _pinned_digest(
            source["artifactManifestDigest"],
            f"audit heldRoster sourceInventories[{index}].artifactManifestDigest",
        )
        source_ids.append(_stable_id(source["sourceId"], "audit protected-inventory sourceId"))
        source_resources.append(resource)
        _string(source["sourceRelease"], "audit protected-inventory sourceRelease")
        _stable_id(source["identityMappingId"], "audit protected-inventory identityMappingId")
        for field_name in ("identityMappingSha256", "manifestSha256"):
            if not _is_digest(source[field_name]):
                raise MolecularEvaluationError(
                    f"invalid audit protected-inventory digest {field_name}"
                )
        for field_name in (
            "records", "duplicateRecords", "uniqueInterventions", "qcPassing",
            "qcFailed", "intersectionCoverage",
        ):
            if type(source[field_name]) is not int or source[field_name] < 0:
                raise MolecularEvaluationError("invalid audit protected-inventory count")
        if (
            source["uniqueInterventions"] != source["qcPassing"] + source["qcFailed"]
            or source["records"] != source["uniqueInterventions"] + source["duplicateRecords"]
            or source["intersectionCoverage"] != roster["intersectionSize"]
            or source["identityMappingId"] != roster["identityMappingId"]
            or source["identityMappingSha256"] != roster["identityMappingSha256"]
        ):
            raise MolecularEvaluationError("audit protected-inventory binding is inconsistent")
    if (
        source_ids != sorted(source_ids) or len(source_ids) != len(set(source_ids))
        or len(source_resources) != len(set(source_resources))
    ):
        raise MolecularEvaluationError("audit protected inventories must be unique and sorted")
    return audit, digest


def _role_from_digest(digest: str) -> tuple[str, int]:
    bucket = int(digest[:16], 16) % 100
    if bucket <= 9:
        return "molecular-final", bucket
    if bucket <= 29:
        return "molecular-validation", bucket
    return "pretrain", bucket


def _gene_set_sha256(identifiers: Iterable[str]) -> str:
    return hashlib.sha256(_canonical_bytes(sorted(identifiers))).hexdigest()


def _load_held_roster(
    pinned: PinnedDatasetInput,
) -> tuple[dict[str, str], dict[str, object], str, str]:
    root = Path(pinned.path)
    expected_files = {"held-intervention-roster.tsv", "coverage.json"}
    coverage, coverage_digest = _load_json_dataset(
        pinned, "coverage.json", expected_files, "held roster snapshot"
    )
    coverage = _strict_fields(coverage, ROSTER_COVERAGE_FIELDS, "held roster coverage")
    if coverage["schema"] != ROSTER_SCHEMA or coverage["rosterPath"] != "held-intervention-roster.tsv":
        raise MolecularEvaluationError("held roster coverage schema/path is invalid")
    for field_name in ("sourceCount", "minimumIntersectionSize", "intersectionSize"):
        if type(coverage[field_name]) is not int or coverage[field_name] < 1:
            raise MolecularEvaluationError(f"held roster {field_name} is invalid")
    if coverage["intersectionSize"] < coverage["minimumIntersectionSize"]:
        raise MolecularEvaluationError("held roster intersection is below its frozen minimum")
    assignment = _strict_fields(
        coverage["assignment"], {"domainHex", "digest", "bucketRule", "roles"},
        "held roster assignment",
    )
    if (
        assignment["domainHex"] != ROSTER_ASSIGNMENT_DOMAIN.hex()
        or assignment["digest"] != "sha256"
        or assignment["bucketRule"] != "int(first-16-lowercase-hex,16) mod 100"
        or assignment["roles"] != {
            "0-9": "molecular-final", "10-29": "molecular-validation", "30-99": "pretrain"
        }
    ):
        raise MolecularEvaluationError("held roster assignment contract has drifted")
    mapping = _strict_fields(coverage["identityMapping"], {"id", "sha256"}, "identityMapping")
    _stable_id(mapping["id"], "identityMapping.id")
    if not _is_digest(mapping["sha256"]):
        raise MolecularEvaluationError("identityMapping.sha256 is invalid")
    sources = coverage["sources"]
    if not isinstance(sources, list) or len(sources) < 2 or coverage["sourceCount"] != len(sources):
        raise MolecularEvaluationError("held roster requires complete multi-source coverage")
    source_ids: list[str] = []
    compact_sources: list[dict[str, object]] = []
    excluded_ids: set[str] = set()
    rejection_totals = {
        "qcFailed": 0, "notPassingAllProtectedSources": 0,
        "identicalDuplicatesCollapsed": 0,
    }
    source_fields = AUDIT_SOURCE_INVENTORY_FIELDS | {"exclusions"}
    for index, source in enumerate(sources):
        source = _strict_fields(source, source_fields, f"held roster sources[{index}]")
        resource = _dataset_resource(
            source["resource"], f"held roster sources[{index}].resource"
        )
        if source["revision"] != resource.rpartition("@")[2]:
            raise MolecularEvaluationError("held roster source DatasetSnapshot revision mismatch")
        _pinned_digest(
            source["artifactManifestDigest"],
            f"held roster sources[{index}].artifactManifestDigest",
        )
        source_ids.append(_stable_id(source["sourceId"], "held roster sourceId"))
        _string(source["sourceRelease"], "held roster sourceRelease")
        if source["identityMappingId"] != mapping["id"] or source["identityMappingSha256"] != mapping["sha256"]:
            raise MolecularEvaluationError("held roster source identity mapping mismatch")
        if not _is_digest(source["manifestSha256"]):
            raise MolecularEvaluationError("held roster source manifest digest is invalid")
        for field_name in (
            "records", "duplicateRecords", "uniqueInterventions", "qcPassing", "qcFailed",
            "intersectionCoverage",
        ):
            if type(source[field_name]) is not int or source[field_name] < 0:
                raise MolecularEvaluationError("held roster source count is invalid")
        if (
            source["uniqueInterventions"] != source["qcPassing"] + source["qcFailed"]
            or source["records"] != source["uniqueInterventions"] + source["duplicateRecords"]
        ):
            raise MolecularEvaluationError("held roster source counts are internally inconsistent")
        if source["intersectionCoverage"] != coverage["intersectionSize"]:
            raise MolecularEvaluationError("held roster source does not cover the complete intersection")
        if not isinstance(source["exclusions"], list):
            raise MolecularEvaluationError("held roster source exclusions must be explicit")
        exclusions: list[tuple[str, str]] = []
        for exclusion_index, exclusion in enumerate(source["exclusions"]):
            exclusion = _strict_fields(
                exclusion, {"interventionId", "reason"},
                f"held roster sources[{index}].exclusions[{exclusion_index}]",
            )
            identifier = _string(exclusion["interventionId"], "coverage exclusion interventionId")
            if SGD_INTERVENTION.fullmatch(identifier) is None or exclusion["reason"] not in {
                "qc-failed", "not-qc-passing-in-all-protected-sources"
            }:
                raise MolecularEvaluationError("held roster coverage exclusion is invalid")
            exclusions.append((identifier, exclusion["reason"]))
            excluded_ids.add(identifier)
        if exclusions != sorted(set(exclusions)) or len(exclusions) != source["uniqueInterventions"] - coverage["intersectionSize"]:
            raise MolecularEvaluationError("held roster exclusions do not explain the source intersection")
        rejection_totals["qcFailed"] += sum(reason == "qc-failed" for _, reason in exclusions)
        rejection_totals["notPassingAllProtectedSources"] += sum(
            reason == "not-qc-passing-in-all-protected-sources" for _, reason in exclusions
        )
        rejection_totals["identicalDuplicatesCollapsed"] += source["duplicateRecords"]
        compact_sources.append({key: source[key] for key in source_fields - {"exclusions"}})
    if source_ids != sorted(source_ids) or len(source_ids) != len(set(source_ids)):
        raise MolecularEvaluationError("held roster sources must be unique and sorted")
    if coverage["rejectionCounts"] != rejection_totals:
        raise MolecularEvaluationError("held roster rejectionCounts do not match full coverage")
    roster_path = _relative_file(root, "held-intervention-roster.tsv", "held roster")
    roles: dict[str, str] = {}
    previous = ""
    for line_number, raw in _bounded_lines(roster_path, 4096, "held-intervention-roster.tsv"):
        if not raw.endswith(b"\n"):
            raise MolecularEvaluationError(f"held roster line {line_number} lacks canonical newline")
        try:
            columns = raw[:-1].decode("ascii").split("\t")
        except UnicodeDecodeError as error:
            raise MolecularEvaluationError("held roster must be canonical ASCII") from error
        if len(columns) != 3:
            raise MolecularEvaluationError("held roster rows require id, role, and digest")
        identifier, claimed_role, digest = columns
        if SGD_INTERVENTION.fullmatch(identifier) is None:
            raise MolecularEvaluationError("held roster ID must be a canonical yeast SGD CURIE")
        expected = hashlib.sha256(ROSTER_ASSIGNMENT_DOMAIN + identifier.encode("ascii")).hexdigest()
        derived_role, _bucket = _role_from_digest(expected)
        if digest != expected or claimed_role != derived_role:
            raise MolecularEvaluationError("held roster bucket/role assignment mismatch")
        if identifier <= previous or identifier in roles:
            raise MolecularEvaluationError("held roster IDs must be unique and sorted")
        roles[identifier], previous = claimed_role, identifier
        if len(roles) > MAX_ROSTER_RECORDS:
            raise MolecularEvaluationError("held roster exceeds record bound")
    if not roles or coverage["intersectionSize"] != len(roles):
        raise MolecularEvaluationError("held roster does not cover the declared intersection")
    if excluded_ids & set(roles):
        raise MolecularEvaluationError("held roster exclusion also occurs in the admitted intersection")
    role_counts = {role: sum(value == role for value in roles.values()) for role in (
        "pretrain", "molecular-validation", "molecular-final"
    )}
    if coverage["roleCounts"] != role_counts:
        raise MolecularEvaluationError("held roster roleCounts do not match recomputed assignments")
    roster_digest = _sha256(roster_path)
    if coverage["rosterSha256"] != roster_digest:
        raise MolecularEvaluationError("held roster content digest mismatch")
    coverage["validatedSourceInventories"] = compact_sources
    return roles, coverage, roster_digest, coverage_digest


def _validate_isolation(centering: dict[str, Profile], truth: dict[str, Profile],
                        roster: dict[str, str]) -> None:
    truth_ids = {x for profile in truth.values() for x in profile.intervention_ids}
    wrong = sorted(x for x in truth_ids if roster.get(x) != "molecular-validation")
    if wrong:
        raise MolecularEvaluationError("held truth does not match frozen held roster: " + ", ".join(wrong[:10]))
    centering_ids = {x for profile in centering.values() for x in profile.intervention_ids}
    held = sorted(x for x in centering_ids if roster.get(x) in {"molecular-validation", "molecular-final"})
    if held:
        raise MolecularEvaluationError("fitting centering contains held-roster interventions: " + ", ".join(held[:10]))


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    lm, rm = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((x - lm) * (y - rm) for x, y in zip(left, right))
    denominator = math.sqrt(sum((x - lm) ** 2 for x in left) * sum((y - rm) ** 2 for y in right))
    return numerator / denominator if denominator > 1e-15 else None


def _cosine(left: list[float], right: list[float]) -> float | None:
    denominator = math.sqrt(sum(x * x for x in left) * sum(y * y for y in right))
    return sum(x * y for x, y in zip(left, right)) / denominator if denominator > 1e-15 else None


def _macro(values: Iterable[float | None]) -> float:
    items = list(values)
    return sum(x if x is not None else 0.0 for x in items) / len(items) if items else 0.0


def _references(profiles: dict[str, Profile], minimum: int) -> tuple[dict[GroupKey, dict[str, float]], dict[GroupKey, int]]:
    # Collapse replicates to one centroid per perturbation first. This prevents
    # high-replicate interventions from dominating the fitting reference.
    perturbation_totals: dict[tuple[GroupKey, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    perturbation_readout_counts: dict[tuple[GroupKey, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    perturbations: dict[GroupKey, set[str]] = defaultdict(set)
    for profile in profiles.values():
        group, perturbation = profile.key[:3], profile.key[3]
        perturbations[group].add(perturbation)
        for readout, value in profile.target.items():
            perturbation_totals[(group, perturbation)][readout] += value
            perturbation_readout_counts[(group, perturbation)][readout] += 1
    totals: dict[GroupKey, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[GroupKey, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (group, perturbation), readout_totals in perturbation_totals.items():
        for readout, total in readout_totals.items():
            totals[group][readout] += total / perturbation_readout_counts[(group, perturbation)][readout]
            counts[group][readout] += 1
    references = {group: {r: v / counts[group][r] for r, v in values.items()
                          if counts[group][r] >= minimum} for group, values in totals.items()}
    return references, {group: len(values) for group, values in perturbations.items()}


def _profile_metrics(profiles: dict[str, Profile], references: dict[GroupKey, dict[str, float]],
                     minimum: int) -> list[ProfileMetric]:
    metrics = []
    for profile in profiles.values():
        group = profile.key[:3]
        if group not in references:
            raise MolecularEvaluationError(f"no fitting centering reference for {group}")
        readouts = sorted(set(profile.target) & set(references[group]))
        if len(readouts) < minimum:
            raise MolecularEvaluationError(
                f"held profile {profile.key[4]} has fewer than {minimum} observed scoreable readouts"
            )
        target = [profile.target[x] for x in readouts]
        prediction = [profile.prediction[x] for x in readouts]
        target_shift = [profile.target[x] - references[group][x] for x in readouts]
        prediction_shift = [profile.prediction[x] - references[group][x] for x in readouts]
        metrics.append(ProfileMetric(profile.key, len(readouts), _pearson(prediction, target),
                                     _pearson(prediction_shift, target_shift),
                                     _cosine(prediction_shift, target_shift)))
    return metrics


def _validate_full_centering_support(
    query: dict[str, Profile],
    centering: dict[str, Profile],
    references: dict[GroupKey, dict[str, float]],
) -> None:
    support_types: dict[tuple[GroupKey, str], set[str]] = defaultdict(set)
    for profile in centering.values():
        group = profile.key[:3]
        for readout, distribution in zip(profile.readout_ids, profile.distribution_types):
            if readout in profile.target:
                support_types[(group, readout)].add(distribution)
    for profile in query.values():
        group = profile.key[:3]
        missing = sorted(set(profile.readout_ids) - set(references.get(group, {})))
        if missing:
            raise MolecularEvaluationError(
                f"query profile {profile.key[4]} lacks full fitting centering support: "
                + ", ".join(missing[:10])
            )
        for readout, distribution in zip(profile.readout_ids, profile.distribution_types):
            observed = support_types.get((group, readout), set())
            if observed != {distribution}:
                raise MolecularEvaluationError(
                    f"centering distribution mismatch for {profile.key[4]} readout {readout}"
                )


def _with_accuracy(metrics: list[ProfileMetric], profiles: dict[str, Profile], minimum: int) -> list[ProfileMetric]:
    by_key = {x.key: x for x in profiles.values()}
    groups: dict[GroupKey, list[ProfileMetric]] = defaultdict(list)
    for metric in metrics:
        groups[metric.key[:3]].append(metric)
    accuracies: dict[ProfileKey, float] = {}
    for group_metrics in groups.values():
        if len(group_metrics) < 2:
            continue
        readouts = sorted(set.intersection(*(set(by_key[x.key].target) for x in group_metrics)))
        if len(readouts) < minimum:
            continue
        for metric in group_metrics:
            prediction = by_key[metric.key].prediction
            correct = sum((prediction[r] - by_key[metric.key].target[r]) ** 2 for r in readouts)
            others = [x for x in group_metrics if x.key != metric.key]
            wins = sum(correct < sum((prediction[r] - by_key[x.key].target[r]) ** 2 for r in readouts)
                       for x in others)
            accuracies[metric.key] = wins / len(others)
    return [ProfileMetric(x.key, x.readouts, x.ordinary_pearson, x.perturbed_centroid_pearson,
                          x.perturbed_centroid_cosine, accuracies.get(x.key)) for x in metrics]


def _profile_report(metrics: list[ProfileMetric]) -> dict[str, object]:
    return {"profiles": len(metrics), "profileReadouts": sum(x.readouts for x in metrics),
        "ordinaryPearsonProfiles": sum(x.ordinary_pearson is not None for x in metrics),
        "ordinaryPearsonUndefinedProfiles": sum(x.ordinary_pearson is None for x in metrics),
        "ordinaryProfilePearson": _macro(x.ordinary_pearson for x in metrics),
        "perturbedCentroidPearsonProfiles": sum(x.perturbed_centroid_pearson is not None for x in metrics),
        "perturbedCentroidPearsonUndefinedProfiles": sum(x.perturbed_centroid_pearson is None for x in metrics),
        "perturbedCentroidPearson": _macro(x.perturbed_centroid_pearson for x in metrics),
        "perturbedCentroidCosineProfiles": sum(x.perturbed_centroid_cosine is not None for x in metrics),
        "perturbedCentroidCosineUndefinedProfiles": sum(x.perturbed_centroid_cosine is None for x in metrics),
        "perturbedCentroidCosine": _macro(x.perturbed_centroid_cosine for x in metrics),
        "centroidAccuracyProfiles": sum(x.centroid_accuracy is not None for x in metrics),
        "centroidAccuracyUndefinedProfiles": sum(x.centroid_accuracy is None for x in metrics),
        "centroidAccuracyCommonPanel": _macro(x.centroid_accuracy for x in metrics)}


def _combined(moments: ScalarMoments, metrics: list[ProfileMetric]) -> dict[str, object]:
    return {"ordinary": moments.report(), "gaussianCalibration": moments.gaussian_calibration_report(),
            "perturbationSpecific": _profile_report(metrics)}


def molecular_profile_decision(report: dict[str, object]) -> dict[str, object]:
    specific = report["overall"]["perturbationSpecific"]
    species = report["species"]
    sources = report["sources"]
    audit = report["audit"]
    species_pearson = {str(k): float(v["perturbationSpecific"]["perturbedCentroidPearson"])
                       for k, v in species.items()}
    species_profiles = {str(k): int(v["perturbationSpecific"]["profiles"]) for k, v in species.items()}
    source_profiles = {str(k): int(v["perturbationSpecific"]["profiles"]) for k, v in sources.items()}
    minimum_species = min(species_pearson.values())
    correlations_defined = all(
        group["ordinary"]["pearsonDefined"] is True
        and group["perturbationSpecific"]["ordinaryPearsonUndefinedProfiles"] == 0
        and group["perturbationSpecific"]["perturbedCentroidPearsonUndefinedProfiles"] == 0
        for group in [*species.values(), *sources.values()]
    )
    checks = {
        "strictCorpusAuditPassed": audit["strictCorpusAuditPassed"] is True,
        "heldRosterValidationMatch": audit["heldRosterValidationMatch"] is True,
        "exactTargetFreeQueryManifest": audit["exactTargetFreeQueryManifest"] is True,
        "exactProfilePanelJoin": audit["exactProfilePanelJoin"] is True,
        "zeroBenchmarkLabelRecords": audit["benchmarkLabelRecords"] == 0,
        "zeroCenteringHeldInterventionOverlap": audit["centeringHeldInterventionOverlap"] == 0,
        "overallPerturbedCentroidPearson": specific["perturbedCentroidPearson"] >= MINIMUM_PERTURBED_CENTROID_PEARSON,
        "minimumSpeciesPerturbedCentroidPearson": minimum_species >= MINIMUM_SPECIES_PERTURBED_CENTROID_PEARSON,
        "everySpeciesHasEligibleProfiles": all(x > 0 for x in species_profiles.values()),
        "everySourceHasEligibleProfiles": all(x > 0 for x in source_profiles.values()),
        "everySpeciesAndSourceCorrelationDefined": correlations_defined,
    }
    return {"schema": DIAGNOSTIC_SCHEMA, "scope": "diagnostic-only; not MODEL_CARD advancement",
            "diagnosticPassed": all(checks.values()), "compatibilityPassed": True,
            "compatibilityScope": "five independently admitted DatasetSnapshots, one frozen prediction artifact, and the exact checkpoint bytes",
            "thresholds": {"minimumPerturbedCentroidPearson": MINIMUM_PERTURBED_CENTROID_PEARSON,
                           "minimumSpeciesPerturbedCentroidPearson": MINIMUM_SPECIES_PERTURBED_CENTROID_PEARSON},
            "observed": {"perturbedCentroidPearson": specific["perturbedCentroidPearson"],
                         "minimumSpeciesPerturbedCentroidPearson": minimum_species,
                         "speciesProfiles": species_profiles, "sourceProfiles": source_profiles},
            "checks": checks,
            "doesNotEstablish": ["the required NLL improvement against frozen baselines",
                                 "checkpoint selection eligibility", "synthetic-lethality benchmark performance",
                                 "portable inference or release compatibility"]}


def evaluate_molecular_predictions(
    centering_input: object,
    prediction_root: str | Path,
    truth_input: object,
    query_input: object,
    corpus_audit_input: object,
    held_roster_input: object,
    checkpoint_input: object,
    *,
    minimum_reference_perturbations: int = 2,
    minimum_profile_readouts: int = 2,
    max_line_bytes: int = 16 * 1024 * 1024,
    maximum_absolute_log_scale: float = 20.0,
) -> dict[str, object]:
    if minimum_reference_perturbations < 2 or minimum_profile_readouts < 2:
        raise MolecularEvaluationError("minimum reference perturbations/readouts must be at least 2")
    if not 1024 <= max_line_bytes <= 16 * 1024 * 1024 or maximum_absolute_log_scale <= 0:
        raise MolecularEvaluationError("invalid evaluator bounds")

    def dataset(value: object, name: str) -> PinnedDatasetInput:
        return value if isinstance(value, PinnedDatasetInput) else resolve_pinned_dataset_input(value, name)

    pinned_centering = dataset(centering_input, "molecularCenteringReference")
    pinned_truth = dataset(truth_input, "molecularTruth")
    pinned_query = dataset(query_input, "molecularQuery")
    pinned_audit = dataset(corpus_audit_input, "corpusAudit")
    pinned_roster = dataset(held_roster_input, "heldRoster")
    pinned_checkpoint = (
        checkpoint_input if isinstance(checkpoint_input, PinnedCheckpointInput)
        else resolve_pinned_checkpoint_input(checkpoint_input)
    )
    dataset_resources = {
        item.resource for item in (
            pinned_centering, pinned_truth, pinned_query, pinned_audit, pinned_roster
        )
    }
    if len(dataset_resources) != 5:
        raise MolecularEvaluationError("all five evaluator DatasetSnapshots must be independently pinned")

    centering_manifest = SnapshotManifest.load(pinned_centering.path, CENTERING_ROLE)
    prediction_manifest = SnapshotManifest.load(prediction_root, PREDICTION_ROLE)
    truth_manifest = SnapshotManifest.load(pinned_truth.path, TRUTH_ROLE)
    query, query_manifest, query_manifest_sha256, query_records = _load_query_snapshot(
        pinned_query, max_line_bytes
    )
    if prediction_manifest.model_checkpoint_content_sha256 != pinned_checkpoint.content_sha256:
        raise MolecularEvaluationError("prediction manifest does not bind the supplied checkpoint bytes")
    if len({centering_manifest.value_space, prediction_manifest.value_space, truth_manifest.value_space}) != 1:
        raise MolecularEvaluationError("all molecular valueSpace values must match")
    if (
        prediction_manifest.species_taxa != truth_manifest.species_taxa
        or prediction_manifest.source_ids != truth_manifest.source_ids
    ):
        raise MolecularEvaluationError("prediction and truth species/source declarations must match")
    if (
        prediction_manifest.query_resource != pinned_query.resource
        or truth_manifest.query_resource != pinned_query.resource
        or prediction_manifest.query_dataset_manifest_digest != pinned_query.manifest_digest
        or truth_manifest.query_dataset_manifest_digest != pinned_query.manifest_digest
        or prediction_manifest.query_manifest_sha256 != query_manifest_sha256
        or truth_manifest.query_manifest_sha256 != query_manifest_sha256
    ):
        raise MolecularEvaluationError("prediction/truth query binding is stale or does not match the admitted query")
    if prediction_manifest.value_space != query_manifest["valueSpace"]:
        raise MolecularEvaluationError("query and molecular artifact valueSpace values must match")
    if (
        prediction_manifest.dataset_id != query_manifest["datasetId"]
        or prediction_manifest.version != query_manifest["version"]
    ):
        raise MolecularEvaluationError("prediction dataset identity does not match its query")
    if prediction_manifest.species_taxa != frozenset(query_manifest["speciesTaxa"]):
        raise MolecularEvaluationError("query and molecular artifact speciesTaxa must match")
    if prediction_manifest.source_ids != frozenset(query_manifest["sourceIds"]):
        raise MolecularEvaluationError("query and molecular artifact sourceIds must match")

    centering, centering_records = _load_profiles(
        centering_manifest, max_line_bytes, maximum_absolute_log_scale
    )
    predictions, prediction_records = _load_profiles(
        prediction_manifest, max_line_bytes, maximum_absolute_log_scale
    )
    truth, truth_records = _load_profiles(truth_manifest, max_line_bytes, maximum_absolute_log_scale)
    _assert_query_join(query, predictions, "prediction")
    _assert_query_join(query, truth, "truth")
    joined = _exact_join(truth, predictions)

    corpus_audit, corpus_audit_digest = _load_corpus_audit(pinned_audit)
    roster, roster_coverage, roster_digest, roster_coverage_digest = _load_held_roster(
        pinned_roster
    )
    audit_roster = corpus_audit["heldRoster"]
    assert isinstance(audit_roster, dict)
    roster_sets = {
        role: {identifier for identifier, assigned in roster.items() if assigned == role}
        for role in ("pretrain", "molecular-validation", "molecular-final")
    }
    held_union = roster_sets["molecular-validation"] | roster_sets["molecular-final"]
    roster_bindings = {
        "resource": pinned_roster.resource,
        "revision": pinned_roster.revision,
        "manifestDigest": pinned_roster.manifest_digest,
        "rosterSha256": roster_digest,
        "coverageSha256": roster_coverage_digest,
        "assignmentDomainHex": ROSTER_ASSIGNMENT_DOMAIN.hex(),
        "bucketRule": "int(first-16-lowercase-hex,16) mod 100",
        "identityMappingId": roster_coverage["identityMapping"]["id"],
        "identityMappingSha256": roster_coverage["identityMapping"]["sha256"],
        "intersectionSize": len(roster),
        "pretrainGeneCount": len(roster_sets["pretrain"]),
        "validationGeneSetSha256": _gene_set_sha256(roster_sets["molecular-validation"]),
        "validationGeneCount": len(roster_sets["molecular-validation"]),
        "finalGeneSetSha256": _gene_set_sha256(roster_sets["molecular-final"]),
        "finalGeneCount": len(roster_sets["molecular-final"]),
        "unionGeneSetSha256": _gene_set_sha256(held_union),
        "unionGeneCount": len(held_union),
    }
    for field_name, expected in roster_bindings.items():
        if audit_roster[field_name] != expected:
            raise MolecularEvaluationError(f"corpus audit held-roster binding mismatch: {field_name}")
    if audit_roster["sourceInventories"] != roster_coverage["validatedSourceInventories"]:
        raise MolecularEvaluationError("corpus audit held-roster source inventory bindings mismatch")
    validation_identity = corpus_audit["datasets"]["molecularValidation"]
    assert isinstance(validation_identity, dict)
    if (
        set(validation_identity["speciesTaxa"]) != set(prediction_manifest.species_taxa)
        or set(validation_identity["sourceIds"]) != set(prediction_manifest.source_ids)
    ):
        raise MolecularEvaluationError("query species/sources do not match audited molecularValidation")
    query_interventions = {
        identifier for profile in query.values() for identifier in profile.intervention_ids
    }
    if (
        validation_identity["trajectoryGeneCount"] != len(query_interventions)
        or validation_identity["trajectoryGeneSetSha256"] != _gene_set_sha256(query_interventions)
    ):
        raise MolecularEvaluationError(
            "target-free query intervention domain does not match audited molecularValidation"
        )
    _validate_isolation(centering, truth, roster)

    references, reference_counts = _references(centering, minimum_reference_perturbations)
    _validate_full_centering_support(query, centering, references)
    metrics = _with_accuracy(
        _profile_metrics(joined, references, minimum_profile_readouts),
        joined,
        minimum_profile_readouts,
    )
    if not metrics:
        raise MolecularEvaluationError("no held profile meets perturbation-specific contract")
    source_moments: dict[str, ScalarMoments] = defaultdict(ScalarMoments)
    species_moments: dict[int, ScalarMoments] = defaultdict(ScalarMoments)
    pair_moments: dict[tuple[int, str], ScalarMoments] = defaultdict(ScalarMoments)
    nulls = 0
    for profile in joined.values():
        taxon, source = profile.key[:2]
        nulls += len(profile.readout_ids) - len(profile.target)
        distribution_by_readout = dict(zip(profile.readout_ids, profile.distribution_types))
        for readout, target in profile.target.items():
            moment = ScalarMoments()
            moment.add(
                profile.prediction[readout], target, distribution_by_readout[readout],
                profile.primary_parameter[readout], profile.log_scale[readout],
            )
            source_moments[source].merge(moment)
            species_moments[taxon].merge(moment)
            pair_moments[(taxon, source)].merge(moment)
    if (
        set(source_moments) != set(prediction_manifest.source_ids)
        or set(species_moments) != set(prediction_manifest.species_taxa)
    ):
        raise MolecularEvaluationError("every declared species and source requires observed held truth")
    overall_moments = ScalarMoments()
    for item in source_moments.values():
        overall_moments.merge(item)

    def selected(taxon: int | None = None, source: str | None = None) -> list[ProfileMetric]:
        return [
            metric for metric in metrics
            if (taxon is None or metric.key[0] == taxon)
            and (source is None or metric.key[1] == source)
        ]

    overall = _combined(overall_moments, metrics)
    species = {str(key): _combined(value, selected(taxon=key)) for key, value in sorted(species_moments.items())}
    sources = {key: _combined(value, selected(source=key)) for key, value in sorted(source_moments.items())}
    pairs = {f"{key[0]}|{key[1]}": _combined(value, selected(*key)) for key, value in sorted(pair_moments.items())}
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "method": {
            "name": "fitting-centroid target-separated typed-distribution evaluation",
            "class": "Systema-inspired", "citationDoi": SYSTEMA_DOI,
            "referenceDefinition": "Every preregistered readout is centered using fitting-only perturbation centroids. Isolation is independently established by the admitted audit and frozen roster, never inferred from centering overlap alone.",
            "queryBindingDefinition": "Prediction and evaluator-only truth exact-join the same canonical target-free query identities, interventions, distribution types, and readout panels.",
            "likelihoodDefinition": "Non-null truth is scored under the declared Gaussian or negative-binomial distribution. Null truth is not scored; predictions still cover the entire preregistered panel.",
        },
        "inputs": {
            "centeringReference": {
                "resource": pinned_centering.resource, "revision": pinned_centering.revision,
                "datasetManifestDigest": pinned_centering.manifest_digest,
                "datasetId": centering_manifest.dataset_id, "version": centering_manifest.version,
                "manifestSha256": centering_manifest.manifest_sha256, "records": centering_records,
            },
            "predictions": {
                "datasetId": prediction_manifest.dataset_id, "version": prediction_manifest.version,
                "manifestSha256": prediction_manifest.manifest_sha256,
                "modelCheckpointContentSha256": prediction_manifest.model_checkpoint_content_sha256,
                "queryResource": prediction_manifest.query_resource,
                "queryDatasetManifestDigest": prediction_manifest.query_dataset_manifest_digest,
                "queryManifestSha256": query_manifest_sha256, "records": prediction_records,
            },
            "heldTruth": {
                "resource": pinned_truth.resource, "revision": pinned_truth.revision,
                "datasetManifestDigest": pinned_truth.manifest_digest,
                "datasetId": truth_manifest.dataset_id, "version": truth_manifest.version,
                "manifestSha256": truth_manifest.manifest_sha256,
                "queryResource": truth_manifest.query_resource,
                "queryDatasetManifestDigest": truth_manifest.query_dataset_manifest_digest,
                "queryManifestSha256": truth_manifest.query_manifest_sha256,
                "records": truth_records,
            },
            "molecularQuery": {
                "datasetId": query_manifest["datasetId"], "version": query_manifest["version"],
                "resource": pinned_query.resource, "revision": pinned_query.revision,
                "datasetManifestDigest": pinned_query.manifest_digest,
                "queryManifestSha256": query_manifest_sha256, "records": query_records,
            },
            "corpusAudit": {
                "resource": pinned_audit.resource, "revision": pinned_audit.revision,
                "datasetManifestDigest": pinned_audit.manifest_digest,
                "contentSha256": corpus_audit_digest,
            },
            "heldRoster": {
                "resource": pinned_roster.resource, "revision": pinned_roster.revision,
                "datasetManifestDigest": pinned_roster.manifest_digest,
                "rosterSha256": roster_digest, "coverageSha256": roster_coverage_digest,
            },
            "modelCheckpoint": {
                "resource": pinned_checkpoint.resource,
                "checkpointArtifactDigest": pinned_checkpoint.checkpoint_artifact_digest,
                "contentSha256": pinned_checkpoint.content_sha256,
            },
            "valueSpace": prediction_manifest.value_space,
        },
        "audit": {
            "strictCorpusAuditPassed": True, "heldRosterValidationMatch": True,
            "exactTargetFreeQueryManifest": True, "exactProfilePanelJoin": True,
            "benchmarkLabelRecords": 0, "centeringHeldInterventionOverlap": 0,
            "speciesTaxa": sorted(prediction_manifest.species_taxa),
            "sourceIds": sorted(prediction_manifest.source_ids),
            "nullTruthValuesNotScored": nulls,
            "predictionsCoverPreregisteredValues": sum(len(item.readout_ids) for item in joined.values()),
            "minimumReferencePerturbations": minimum_reference_perturbations,
            "minimumProfileReadouts": minimum_profile_readouts,
            "referencePerturbationsByGroup": {
                "|".join(map(str, key)): value for key, value in sorted(reference_counts.items())
            },
            "interventionIsolationEvidence": [
                "admitted-strict-corpus-audit-datasetsnapshot",
                "admitted-frozen-held-roster-datasetsnapshot",
            ],
        },
        "overall": overall, "species": species, "sources": sources, "speciesSources": pairs,
    }
    report["diagnostic"] = molecular_profile_decision(report)
    return report
