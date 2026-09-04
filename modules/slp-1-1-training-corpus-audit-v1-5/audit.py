"""Fail-closed v1.5 clean-training corpus and held-boundary audit.

Identity-bearing NPZ arrays are inspected with the standard library so a
trajectory manifest cannot conceal a record-level intervention.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import tarfile
import tempfile
import zipfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

CORPUS_SCHEMA = "slp.corpus/v1.2"
AUDIT_SCHEMA = "slp.corpus-audit/v1.5"
ROSTER_SCHEMA = "slp.held-intervention-roster-report/v1"
INVENTORY_SCHEMA = "slp.intervention-identity-inventory/v1"
INVENTORY_RECORD_SCHEMA = "slp.intervention-identity-record/v1"
TRAJECTORY_INTERVENTION_SCHEMA = "slp.trajectory-intervention/v1"
COMPOSITION_AUDIT_SCHEMA = "slp.proteome-corpus-compose-audit/v1"
ASSIGNMENT_DOMAIN = b"slp-1.1-yeast-global-held-v1\x00"
ASSIGNMENT_DOMAIN_HEX = ASSIGNMENT_DOMAIN.hex()
BUCKET_RULE = "int(first-16-lowercase-hex,16) mod 100"
TRAINING_INPUT_NAME = "pretrain"
CORPUS_FIELDS = {
    "schema", "datasetId", "version", "role", "labelClass",
    "benchmarkLabelsPresent", "rewardEnabled", "identityKey", "rights",
    "modalities", "sources", "sampling", "inputs", "counts",
    "species", "featurePack", "entityTypes", "contextTypes", "actionTypes",
    "covariates", "readoutTypes", "entityDictionary", "queryDictionary",
    "queryPanels", "trajectoryInterventions", "normalization", "bounds", "shards",
}
CORPUS_INPUT_FIELDS = {"observations", "staticFeatures", "heldInterventionRoster"}
CORPUS_LINEAGE_FIELDS = {"datasetSnapshot", "semanticSha256", "files"}
DATASET_SNAPSHOT_FIELDS = {
    "resource", "revision", "outerManifestDigest", "treeDigest",
}
FEATURE_PACK_FIELDS = {
    "schema", "revision", "sha256", "entityFeatureDim", "speciesFeatureDim",
    "blocks",
}
FEATURE_BLOCK_FIELDS = {
    "id", "offset", "dimension", "datasetSnapshot", "semanticSha256",
    "entityKeySetSha256", "files",
}
LINEAGE_FILE_FIELDS = {"path", "sha256", "bytes"}
CORPUS_COUNT_FIELDS = {
    "entities", "featureRows", "contexts", "queries", "panels",
    "trajectoryInterventions", "records", "targetValues", "shards",
}
REFERENCE_FIELDS = {"path", "sha256", "bytes", "count"}
SHARD_FIELDS = {"path", "sha256", "bytes", "records", "targetValues"}
COVERAGE_FIELDS = {
    "schema", "assignment", "sourceCount", "identityMapping",
    "minimumIntersectionSize", "intersectionSize", "roleCounts",
    "rejectionCounts", "rosterPath", "rosterSha256", "sources",
}
SOURCE_COVERAGE_FIELDS = {
    "sourceId", "sourceRelease", "identityMappingId", "identityMappingSha256",
    "manifestSha256", "records", "duplicateRecords", "uniqueInterventions",
    "qcPassing", "qcFailed", "intersectionCoverage", "exclusions",
}
INVENTORY_FIELDS = {
    "schema", "sourceId", "sourceRelease", "ncbiTaxon", "stableIdNamespace",
    "identityMappingId", "identityMappingSha256", "inventoryFormat", "files",
}
INVENTORY_RECORD_FIELDS = {"schema", "interventionId", "ncbiTaxon", "qcPassing"}
INVENTORY_FILE_FIELDS = {"path", "sha256", "records"}
ENTITY_MEMBERS = {
    "entity_taxon", "entity_id", "entity_type", "entity_feature_value",
    "entity_feature_present",
}
QUERY_MEMBERS = {"query_entity_index", "query_readout_index"}
PANEL_MEMBERS = {"panel_id", "panel_indptr", "panel_query_index"}
ENTITY_TYPES = [
    "slp-entity-type:gene",
    "slp-entity-type:protein",
    "slp-entity-type:context",
]
QUANTITATIVE_SHARD_MEMBERS = {
    "record_id", "observation_unit_id", "source_index", "replicate_id",
    "perturbation_id", "species_taxon", "species_feature_value",
    "species_feature_present", "context_entity_index", "context_type",
    "context_mask", "context_covariate_value", "context_covariate_present",
    "record_covariate_value", "record_covariate_present", "action_entity_index",
    "action_type", "action_mask", "action_covariate_value",
    "action_covariate_present", "observation_covariate_value",
    "observation_covariate_present", "query_panel_index", "target_indptr",
    "target_query_index", "target_value",
}
COMPOSITION_AUDIT_FIELDS = {
    "schema", "archive", "corpusManifestSha256", "inputs", "counts",
    "identity", "featurePackSha256", "featurePreservation",
    "targetPreservation", "leakage", "formats", "limitations",
}
COMPOSITION_IDENTITY_FIELDS = {
    "key", "corpusEntityKeySetSha256", "featureEntityKeySetSha256",
    "contextEntity",
}
FEATURE_PRESERVATION_FIELDS = {
    "rows", "dimension", "sourceValueBytesSha256",
    "composedValueBytesSha256", "sourcePresentBytesSha256",
    "composedPresentBytesSha256", "byteExact",
}
TARGET_PRESERVATION_FIELDS = {
    "dtype", "values", "sourceBytesSha256", "composedBytesSha256", "byteExact",
}
COMPOSITION_LIMITATIONS = [
    "this is a fitting-only yeast proteome corpus, not molecular validation or final-holdout evidence",
    "the 21-dimensional hand-designed sequence block is a weak static baseline, not a frontier representation",
    "composition preserves quantitative targets but does not train or evaluate a world model",
]
CURIE = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*:[^\s]+$")
SGD_CURIE = re.compile(r"^SGD:S[0-9]{9}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RESOURCE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
INPUT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
SOURCE_ID = re.compile(r"^[^\s:]+:[^\s:]+$")
MUTABLE_VERSION = re.compile(
    r"(?:^|[-_.:/])(latest|current|head|main|master|nightly)(?:$|[-_.:/])",
    re.IGNORECASE,
)


class CorpusAuditError(ValueError):
    """Raised whenever a passing leakage attestation cannot be proven."""


@dataclass(frozen=True)
class AuditBounds:
    max_manifest_bytes: int = 4 * 1024 * 1024
    max_coverage_bytes: int = 64 * 1024 * 1024
    max_files_per_corpus: int = 256
    max_file_bytes: int = 8 * 1024 * 1024 * 1024
    max_total_bytes_per_corpus: int = 64 * 1024 * 1024 * 1024
    max_records_per_corpus: int = 20_000_000
    max_trajectory_interventions: int = 2_000_000
    max_line_bytes: int = 4_096
    max_sources: int = 4_096
    max_species: int = 128
    max_entities: int = 2_000_000
    max_identity_array_bytes: int = 2 * 1024 * 1024 * 1024
    max_roster_records: int = 2_000_000
    max_coverage_exclusions: int = 8_000_000
    max_npz_members: int = 256
    max_inventory_files_per_source: int = 32
    max_inventory_records_per_source: int = 2_000_000

    def __post_init__(self) -> None:
        limits = {
            "maxManifestBytes": (self.max_manifest_bytes, 1_024, 64 * 1024 * 1024),
            "maxCoverageBytes": (self.max_coverage_bytes, 1_024, 512 * 1024 * 1024),
            "maxFilesPerCorpus": (self.max_files_per_corpus, 5, 4_096),
            "maxFileBytes": (self.max_file_bytes, 1, 128 * 1024**3),
            "maxTotalBytesPerCorpus": (self.max_total_bytes_per_corpus, 1, 512 * 1024**3),
            "maxRecordsPerCorpus": (self.max_records_per_corpus, 1, 1_000_000_000),
            "maxTrajectoryInterventions": (
                self.max_trajectory_interventions, 1, 20_000_000
            ),
            "maxLineBytes": (self.max_line_bytes, 64, 1_048_576),
            "maxSources": (self.max_sources, 1, 100_000),
            "maxSpecies": (self.max_species, 1, 10_000),
            "maxEntities": (self.max_entities, 1, 20_000_000),
            "maxIdentityArrayBytes": (self.max_identity_array_bytes, 1_024, 64 * 1024**3),
            "maxRosterRecords": (self.max_roster_records, 1, 20_000_000),
            "maxCoverageExclusions": (self.max_coverage_exclusions, 0, 100_000_000),
            "maxNpzMembers": (self.max_npz_members, 8, 4_096),
            "maxInventoryFilesPerSource": (
                self.max_inventory_files_per_source, 1, 256
            ),
            "maxInventoryRecordsPerSource": (
                self.max_inventory_records_per_source, 1, 20_000_000
            ),
        }
        for name, (value, minimum, maximum) in limits.items():
            if type(value) is not int or not minimum <= value <= maximum:
                raise CorpusAuditError(f"{name} must be an integer in [{minimum}, {maximum}]")


@dataclass(frozen=True)
class DatasetInput:
    input_name: str
    root: Path
    resource: str
    revision: str
    manifest_digest: str


@dataclass(frozen=True)
class FileReference:
    path: str
    sha256: str
    bytes: int
    count: int


@dataclass(frozen=True)
class Corpus:
    input: DatasetInput
    identity: dict[str, Any]
    trajectory_interventions: frozenset[tuple[int, str]]
    active_actions: frozenset[tuple[int, str]]
    composition_facts: dict[str, Any]


@dataclass(frozen=True)
class ProtectedInventory:
    input: DatasetInput
    source_id: str
    source_release: str
    identity_mapping_id: str
    identity_mapping_sha256: str
    manifest_sha256: str
    records: int
    duplicate_records: int
    qc_passing: frozenset[str]
    qc_failed: frozenset[str]


@dataclass(frozen=True)
class NpySpec:
    descr: str
    shape: tuple[int, ...]
    count: int
    item_size: int
    header_bytes: int


def canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _composite_key(ncbi_taxon: int, entity_id: str) -> tuple[int, str]:
    return ncbi_taxon, entity_id


def _composite_object(key: tuple[int, str]) -> dict[str, object]:
    return {"ncbiTaxon": key[0], "entityId": key[1]}


def _composite_set_sha256(values: set[tuple[int, str]] | frozenset[tuple[int, str]]) -> str:
    rows = [_composite_object(item) for item in sorted(values)]
    return canonical_sha256(rows)


def _perturbation_id(actions: set[tuple[int, str]]) -> str:
    rows = [_composite_object(item) for item in sorted(actions)]
    return "slp-perturbation:sha256-" + canonical_sha256(rows)


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CorpusAuditError(f"duplicate JSON member: {key}")
        value[key] = item
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise CorpusAuditError(f"could not read {path.name}") from error
    return digest.hexdigest()


def _strict_dict(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        actual = set(value) if isinstance(value, dict) else set()
        raise CorpusAuditError(
            f"{label} fields mismatch; missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CorpusAuditError(f"{label} must be a non-empty trimmed string")
    return value


def _curie(value: object, label: str) -> str:
    value = _string(value, label)
    if CURIE.fullmatch(value) is None:
        raise CorpusAuditError(f"{label} must be a stable namespace-bearing CURIE")
    return value


def _raw_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise CorpusAuditError(f"{label} must be a lowercase SHA-256")
    return value


def _prefixed_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise CorpusAuditError(f"{label} must be an exact sha256:<lowercase hex> digest")
    _raw_sha(value[7:], label)
    return value


def _positive_int(value: object, label: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise CorpusAuditError(f"{label} must be a positive bounded integer")
    return value


def _nonnegative_int(value: object, label: str, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise CorpusAuditError(f"{label} must be a non-negative bounded integer")
    return value


def _dataset_resource(value: object, label: str) -> tuple[str, str]:
    resource = _string(value, label)
    if not resource.startswith("omf://"):
        raise CorpusAuditError(f"{label} must be an OMF DatasetSnapshot resource URI")
    identity, separator, revision = resource.removeprefix("omf://").rpartition("@")
    if not separator:
        raise CorpusAuditError(f"{label} must contain an exact resource revision")
    revision = _prefixed_sha(revision, f"{label} revision")
    parts = identity.split("/")
    if (
        len(parts) < 3
        or parts[-2] != "datasetsnapshot"
        or RESOURCE_NAME.fullmatch(parts[-1] if parts else "") is None
        or any(not part or part in {".", ".."} or any(c.isspace() for c in part) for part in parts)
    ):
        raise CorpusAuditError(f"{label} must identify a named DatasetSnapshot")
    return parts[-1], revision


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        try:
            if candidate.is_symlink():
                raise CorpusAuditError(f"{label} must not contain a symlink")
        except OSError as error:
            raise CorpusAuditError(f"could not inspect {label}") from error


def resolve_dataset_input(value: object, input_name: str) -> DatasetInput:
    if INPUT_NAME.fullmatch(input_name) is None:
        raise CorpusAuditError("input name is not canonical")
    value = _strict_dict(
        value, {"resource", "mode", "path", "manifestDigest"}, f"{input_name} input"
    )
    resource_name, revision = _dataset_resource(value["resource"], f"{input_name}.resource")
    if value["mode"] != "copy":
        raise CorpusAuditError(f"{input_name} must be an immutable copied DatasetSnapshot")
    manifest_digest = _prefixed_sha(value["manifestDigest"], f"{input_name}.manifestDigest")
    requested = Path(_string(value["path"], f"{input_name}.path"))
    _reject_symlink_components(requested, f"{input_name}.path")
    try:
        root = requested.resolve(strict=True)
    except OSError as error:
        raise CorpusAuditError(f"{input_name}.path does not exist") from error
    if not root.is_dir():
        raise CorpusAuditError(f"{input_name}.path must materialize a directory")
    if root.name != resource_name or root.parent.name != input_name or root.parent.parent.name != "inputs":
        raise CorpusAuditError(
            f"{input_name}.path is inconsistent with its input name and resource name"
        )
    return DatasetInput(input_name, root, str(value["resource"]), revision, manifest_digest)


def _relative_path(value: object, label: str) -> str:
    relative = _string(value, label)
    portable = PurePosixPath(relative)
    if (
        portable.is_absolute()
        or portable.as_posix() != relative
        or "\\" in relative
        or ":" in relative
        or any(part in {"", ".", ".."} for part in portable.parts)
    ):
        raise CorpusAuditError(f"{label} must be a canonical relative POSIX path")
    return relative


def _regular_file(root: Path, relative: object, label: str, maximum: int) -> Path:
    relative = _relative_path(relative, label)
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise CorpusAuditError(f"{label} must not contain a symlink")
    try:
        path = cursor.resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as error:
        raise CorpusAuditError(f"{label} resolves outside or is missing") from error
    if not path.is_file():
        raise CorpusAuditError(f"{label} must be a regular file")
    if path.stat().st_size > maximum:
        raise CorpusAuditError(f"{label} exceeds its byte bound")
    return path


def _directory_entries(root: Path, maximum_entries: int) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for entry_number, path in enumerate(root.rglob("*"), 1):
        if entry_number > maximum_entries:
            raise CorpusAuditError("snapshot directory entry count exceeds its exact bound")
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise CorpusAuditError(f"snapshot contains a symlink: {relative}")
        if path.is_file():
            files.add(relative)
        elif path.is_dir():
            directories.add(relative)
        else:
            raise CorpusAuditError(f"snapshot contains a non-regular entry: {relative}")
    return files, directories


def _exact_file_set(root: Path, expected_files: set[str], label: str) -> None:
    expected_directories: set[str] = set()
    for item in expected_files:
        parent = PurePosixPath(item).parent
        while parent.as_posix() != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    actual_files, actual_directories = _directory_entries(
        root, len(expected_files) + len(expected_directories)
    )
    if actual_files != expected_files or actual_directories != expected_directories:
        raise CorpusAuditError(
            f"{label} file set mismatch; missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}, "
            f"extraDirectories={sorted(actual_directories - expected_directories)}"
        )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusAuditError(f"{label} must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise CorpusAuditError(f"{label} must be a JSON object")
    return value


def _unique_curies(value: object, label: str, *, sorted_required: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise CorpusAuditError(f"{label} must be a non-empty CURIE list")
    items = tuple(_curie(item, label) for item in value)
    if len(items) != len(set(items)):
        raise CorpusAuditError(f"{label} must contain unique identifiers")
    if sorted_required and list(items) != sorted(items):
        raise CorpusAuditError(f"{label} must be deterministically sorted")
    return items


def _file_reference(
    value: object, label: str, bounds: AuditBounds, *, allow_zero: bool = False
) -> FileReference:
    value = _strict_dict(value, REFERENCE_FIELDS, label)
    count = _nonnegative_int(value["count"], f"{label}.count", bounds.max_entities)
    if count == 0 and not allow_zero:
        raise CorpusAuditError(f"{label}.count must be positive")
    return FileReference(
        _relative_path(value["path"], f"{label}.path"),
        _raw_sha(value["sha256"], f"{label}.sha256"),
        _nonnegative_int(value["bytes"], f"{label}.bytes", bounds.max_file_bytes),
        count,
    )


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for item in shape:
        if type(item) is not int or item < 0:
            raise CorpusAuditError("NPY shape must contain non-negative integers")
        result *= item
    return result


def _npy_item_size(descr: str) -> int:
    match = re.fullmatch(r"([<|>])([Uuifb])(\d+)", descr)
    if match is None:
        raise CorpusAuditError(f"unsupported identity NPY dtype: {descr}")
    endian, kind, width_text = match.groups()
    width = int(width_text)
    if width < 1 or (kind == "U" and endian != "<"):
        raise CorpusAuditError(f"unsupported identity NPY dtype: {descr}")
    if kind == "U":
        return width * 4
    if kind == "b" and width != 1:
        raise CorpusAuditError("boolean NPY dtype must have width one")
    if kind in {"i", "u"} and width not in {1, 2, 4, 8}:
        raise CorpusAuditError("integer NPY dtype has an unsupported width")
    if kind == "f" and width not in {4, 8}:
        raise CorpusAuditError("floating-point NPY dtype has an unsupported width")
    return width


def _parse_npy_header(
    stream: Any, member_size: int, label: str, bounds: AuditBounds
) -> NpySpec:
    if stream.read(6) != b"\x93NUMPY":
        raise CorpusAuditError(f"{label} is not an NPY array")
    version = stream.read(2)
    if len(version) != 2 or version[0] not in {1, 2, 3}:
        raise CorpusAuditError(f"{label} uses an unsupported NPY version")
    size_bytes = 2 if version[0] == 1 else 4
    encoded_length = stream.read(size_bytes)
    if len(encoded_length) != size_bytes:
        raise CorpusAuditError(f"{label} has a truncated NPY header")
    header_length = int.from_bytes(encoded_length, "little")
    if header_length > 65_536:
        raise CorpusAuditError(f"{label} NPY header exceeds 64 KiB")
    header = stream.read(header_length)
    if len(header) != header_length:
        raise CorpusAuditError(f"{label} has a truncated NPY header")
    try:
        document = ast.literal_eval(
            header.decode("latin1" if version[0] < 3 else "utf-8").strip()
        )
    except (UnicodeDecodeError, SyntaxError, ValueError) as error:
        raise CorpusAuditError(f"{label} NPY header is invalid") from error
    document = _strict_dict(
        document, {"descr", "fortran_order", "shape"}, f"{label} NPY header"
    )
    if document["fortran_order"] is not False:
        raise CorpusAuditError(f"{label} must use C-order storage")
    if not isinstance(document["descr"], str) or not isinstance(document["shape"], tuple):
        raise CorpusAuditError(f"{label} has invalid NPY dtype or shape")
    shape = tuple(document["shape"])
    count = _product(shape)
    item_size = _npy_item_size(document["descr"])
    header_bytes = 6 + 2 + size_bytes + header_length
    payload_bytes = count * item_size
    if payload_bytes > bounds.max_identity_array_bytes:
        raise CorpusAuditError(f"{label} exceeds maxIdentityArrayBytes")
    if header_bytes + payload_bytes != member_size:
        raise CorpusAuditError(f"{label} NPY payload length does not match its header")
    return NpySpec(document["descr"], shape, count, item_size, header_bytes)


def _member_map(
    archive: zipfile.ZipFile,
    label: str,
    bounds: AuditBounds,
    expected_keys: set[str] | None = None,
) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if not infos or len(infos) > bounds.max_npz_members:
        raise CorpusAuditError(f"{label} NPZ member count is outside bounds")
    result: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        name = info.filename
        if info.is_dir() or PurePosixPath(name).name != name or not name.endswith(".npy"):
            raise CorpusAuditError(f"{label} contains an unsafe or non-NPY member")
        key = name[:-4]
        if key in result:
            raise CorpusAuditError(f"{label} contains a duplicate array member")
        if info.file_size > bounds.max_identity_array_bytes + 65_536:
            raise CorpusAuditError(f"{label}.{key} exceeds maxIdentityArrayBytes")
        result[key] = info
    if expected_keys is not None and set(result) != expected_keys:
        raise CorpusAuditError(
            f"{label} NPZ arrays mismatch; missing={sorted(expected_keys - set(result))}, "
            f"extra={sorted(set(result) - expected_keys)}"
        )
    return result


def _array_spec(
    archive: zipfile.ZipFile,
    members: Mapping[str, zipfile.ZipInfo],
    key: str,
    label: str,
    bounds: AuditBounds,
) -> NpySpec:
    if key not in members:
        raise CorpusAuditError(f"{label} is missing required identity array {key}")
    try:
        with archive.open(members[key]) as stream:
            return _parse_npy_header(stream, members[key].file_size, f"{label}.{key}", bounds)
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise CorpusAuditError(f"could not inspect {label}.{key}") from error


def _iter_array_values(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    expected: NpySpec,
    label: str,
    bounds: AuditBounds,
) -> Iterator[object]:
    match = re.fullmatch(r"([<|>])([Uuifb])(\d+)", expected.descr)
    assert match is not None
    endian, kind, width_text = match.groups()
    width = int(width_text)
    remaining = expected.count
    consumed_bytes = 0
    # Close the ZipExtFile before yielding a value.  Besides bounding memory,
    # this guarantees that a fail-closed validation exception cannot retain a
    # Windows lock on the audited NPZ through its traceback.
    batch_items = max(1, (16 * 1024 * 1024) // expected.item_size)
    while remaining:
        batch = min(remaining, batch_items)
        with archive.open(info) as stream:
            actual = _parse_npy_header(stream, info.file_size, label, bounds)
            if actual != expected:
                raise CorpusAuditError(f"{label} header changed during inspection")
            if consumed_bytes:
                stream.seek(actual.header_bytes + consumed_bytes)
            payload = stream.read(batch * actual.item_size)
            if len(payload) != batch * actual.item_size:
                raise CorpusAuditError(f"{label} has a truncated payload")
            if remaining == batch and stream.read(1):
                raise CorpusAuditError(f"{label} contains trailing NPY bytes")
        consumed_bytes += len(payload)
        remaining -= batch
        if kind == "U":
            for offset in range(0, len(payload), actual.item_size):
                try:
                    yield payload[offset : offset + actual.item_size].decode(
                        "utf-32-le"
                    ).rstrip("\x00")
                except UnicodeDecodeError as error:
                    raise CorpusAuditError(f"{label} contains invalid Unicode") from error
        elif kind == "b":
            for value in payload:
                if value not in {0, 1}:
                    raise CorpusAuditError(f"{label} contains a non-boolean byte")
                yield bool(value)
        else:
            prefix = "<" if endian in {"<", "|"} else ">"
            codes = {
                ("i", 1): "b", ("u", 1): "B", ("i", 2): "h", ("u", 2): "H",
                ("i", 4): "i", ("u", 4): "I", ("i", 8): "q", ("u", 8): "Q",
                ("f", 4): "f", ("f", 8): "d",
            }
            for (value,) in struct.iter_unpack(prefix + codes[(kind, width)], payload):
                yield value


def _read_entity_dictionary(
    path: Path,
    count: int,
    declared_species: frozenset[int],
    entity_feature_dim: int,
    blocks: list[dict[str, Any]],
    bounds: AuditBounds,
) -> tuple[
    tuple[tuple[int, str], ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[bool, ...],
    tuple[str, ...],
    str,
    str,
]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = _member_map(
                archive, "entityDictionary", bounds, ENTITY_MEMBERS
            )
            id_spec = _array_spec(archive, members, "entity_id", "entityDictionary", bounds)
            taxon_spec = _array_spec(
                archive, members, "entity_taxon", "entityDictionary", bounds
            )
            type_spec = _array_spec(
                archive, members, "entity_type", "entityDictionary", bounds
            )
            value_spec = _array_spec(
                archive, members, "entity_feature_value", "entityDictionary", bounds
            )
            present_spec = _array_spec(
                archive, members, "entity_feature_present", "entityDictionary", bounds
            )
            if id_spec.shape != (count,) or taxon_spec.shape != (count,):
                raise CorpusAuditError("entity dictionary array shapes must match declared count")
            if not id_spec.descr.startswith("<U") or taxon_spec.descr[1:2] not in {"i", "u"}:
                raise CorpusAuditError("entity dictionary identity arrays use invalid dtypes")
            if type_spec.shape != (count,) or type_spec.descr != "<i8":
                raise CorpusAuditError("entity_type must be a count-aligned int64 array")
            expected_feature_shape = (count, entity_feature_dim)
            if (
                value_spec.shape != expected_feature_shape
                or value_spec.descr != "<f4"
                or present_spec.shape != expected_feature_shape
                or present_spec.descr != "|b1"
            ):
                raise CorpusAuditError(
                    "entity feature arrays must match the declared feature dimension"
                )
            identifiers = tuple(
                _iter_array_values(
                    archive, members["entity_id"], id_spec,
                    "entityDictionary.entity_id", bounds,
                )
            )
            taxa = tuple(
                _iter_array_values(
                    archive, members["entity_taxon"], taxon_spec,
                    "entityDictionary.entity_taxon", bounds,
                )
            )
            entity_types = tuple(
                int(item)
                for item in _iter_array_values(
                    archive, members["entity_type"], type_spec,
                    "entityDictionary.entity_type", bounds,
                )
            )
            values = _iter_array_values(
                archive, members["entity_feature_value"], value_spec,
                "entityDictionary.entity_feature_value", bounds,
            )
            present = _iter_array_values(
                archive, members["entity_feature_present"], present_spec,
                "entityDictionary.entity_feature_present", bounds,
            )
            row_is_feature: list[bool] = []
            block_rows: list[set[int]] = [set() for _ in blocks]
            feature_value_digest = hashlib.sha256()
            feature_present_digest = hashlib.sha256()
            for row in range(count):
                row_flags: list[bool] = []
                row_value_bytes = bytearray()
                for column in range(entity_feature_dim):
                    value = float(next(values))
                    flag = bool(next(present))
                    if not math.isfinite(value):
                        raise CorpusAuditError("entity feature values must be finite")
                    if not flag and value != 0.0:
                        raise CorpusAuditError("absent entity features must store exact zero")
                    row_flags.append(flag)
                    row_value_bytes.extend(struct.pack("<f", value))
                if any(row_flags) and not all(row_flags):
                    raise CorpusAuditError(
                        "entity feature rows must be entirely present or entirely absent"
                    )
                row_is_feature.append(all(row_flags))
                if all(row_flags):
                    feature_value_digest.update(row_value_bytes)
                    feature_present_digest.update(b"\x01" * entity_feature_dim)
                for block_index, block in enumerate(blocks):
                    start = int(block["offset"])
                    stop = start + int(block["dimension"])
                    flags = row_flags[start:stop]
                    if any(flags) and not all(flags):
                        raise CorpusAuditError(
                            "feature-block presence must be complete within each row"
                        )
                    if all(flags):
                        block_rows[block_index].add(row)
    except zipfile.BadZipFile as error:
        raise CorpusAuditError("entityDictionary must be a valid NPZ archive") from error
    parsed_ids = tuple(_curie(item, "entity_id") for item in identifiers)
    parsed_taxa = tuple(int(item) for item in taxa)
    if any(type(item) is not int or item <= 0 for item in taxa):
        raise CorpusAuditError("entity_taxon must contain positive integers")
    if any(item not in declared_species for item in parsed_taxa):
        raise CorpusAuditError("entity taxon must be declared")
    keys = tuple(
        _composite_key(taxon, identifier)
        for identifier, taxon in zip(parsed_ids, parsed_taxa, strict=True)
    )
    if len(keys) != len(set(keys)):
        raise CorpusAuditError("composite entity keys must be unique")
    if list(keys) != sorted(keys):
        raise CorpusAuditError(
            "composite entity keys must be ordered by ncbiTaxon then entityId"
        )
    if any(index < 0 for index in entity_types):
        raise CorpusAuditError("entity_type values must be non-negative")
    block_hashes = tuple(
        _composite_set_sha256({keys[row] for row in rows}) for rows in block_rows
    )
    return (
        keys,
        parsed_taxa,
        entity_types,
        tuple(row_is_feature),
        block_hashes,
        feature_value_digest.hexdigest(),
        feature_present_digest.hexdigest(),
    )


def _validate_npz_member_set(
    path: Path, label: str, expected_keys: set[str], bounds: AuditBounds
) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            _member_map(archive, label, bounds, expected_keys)
    except zipfile.BadZipFile as error:
        raise CorpusAuditError(f"{label} must be a valid NPZ archive") from error


def _read_query_dictionary(
    path: Path,
    count: int,
    entity_keys: tuple[tuple[int, str], ...],
    readout_count: int,
    bounds: AuditBounds,
) -> tuple[int, ...]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = _member_map(archive, "queryDictionary", bounds, QUERY_MEMBERS)
            entity_spec = _array_spec(
                archive, members, "query_entity_index", "queryDictionary", bounds
            )
            readout_spec = _array_spec(
                archive, members, "query_readout_index", "queryDictionary", bounds
            )
            if (
                entity_spec.shape != (count,)
                or entity_spec.descr != "<i8"
                or readout_spec.shape != (count,)
                or readout_spec.descr != "<i8"
            ):
                raise CorpusAuditError(
                    "query dictionary arrays must be count-aligned int64 indices"
                )
            entity_indices = tuple(
                int(item)
                for item in _iter_array_values(
                    archive, members["query_entity_index"], entity_spec,
                    "queryDictionary.query_entity_index", bounds,
                )
            )
            readout_indices = tuple(
                int(item)
                for item in _iter_array_values(
                    archive, members["query_readout_index"], readout_spec,
                    "queryDictionary.query_readout_index", bounds,
                )
            )
    except zipfile.BadZipFile as error:
        raise CorpusAuditError("queryDictionary must be a valid NPZ archive") from error
    if any(index < 0 or index >= len(entity_keys) for index in entity_indices):
        raise CorpusAuditError("query_entity_index contains an out-of-range index")
    if any(index < 0 or index >= readout_count for index in readout_indices):
        raise CorpusAuditError("query_readout_index contains an out-of-range index")
    query_keys = [
        (entity_keys[entity_index], readout_index)
        for entity_index, readout_index in zip(
            entity_indices, readout_indices, strict=True
        )
    ]
    if query_keys != sorted(set(query_keys)):
        raise CorpusAuditError(
            "query dictionary composite identities must be unique and ordered"
        )
    return entity_indices


def _validate_query_panels(
    path: Path,
    panel_count: int,
    query_count: int,
    max_panel_queries: int,
    bounds: AuditBounds,
) -> tuple[frozenset[int], ...]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = _member_map(archive, "queryPanels", bounds, PANEL_MEMBERS)
            id_spec = _array_spec(
                archive, members, "panel_id", "queryPanels", bounds
            )
            pointer_spec = _array_spec(
                archive, members, "panel_indptr", "queryPanels", bounds
            )
            query_spec = _array_spec(
                archive, members, "panel_query_index", "queryPanels", bounds
            )
            if id_spec.shape != (panel_count,) or not id_spec.descr.startswith("<U"):
                raise CorpusAuditError(
                    "queryPanels.panel_id must be count-aligned Unicode"
                )
            if pointer_spec.shape != (panel_count + 1,) or pointer_spec.descr != "<i8":
                raise CorpusAuditError(
                    "queryPanels.panel_indptr must be a count-aligned int64 array"
                )
            if len(query_spec.shape) != 1 or query_spec.descr != "<i8":
                raise CorpusAuditError(
                    "queryPanels.panel_query_index must be a one-dimensional int64 array"
                )
            panel_ids = tuple(
                _curie(item, "queryPanels.panel_id")
                for item in _iter_array_values(
                    archive, members["panel_id"], id_spec,
                    "queryPanels.panel_id", bounds,
                )
            )
            pointers = tuple(
                int(item)
                for item in _iter_array_values(
                    archive, members["panel_indptr"], pointer_spec,
                    "queryPanels.panel_indptr", bounds,
                )
            )
            query_indices = tuple(
                int(item)
                for item in _iter_array_values(
                    archive, members["panel_query_index"], query_spec,
                    "queryPanels.panel_query_index", bounds,
                )
            )
    except zipfile.BadZipFile as error:
        raise CorpusAuditError("queryPanels must be a valid NPZ archive") from error
    if list(panel_ids) != sorted(set(panel_ids)):
        raise CorpusAuditError("queryPanels.panel_id must be unique and ordered")
    if not pointers or pointers[0] != 0 or pointers[-1] != len(query_indices):
        raise CorpusAuditError("queryPanels.panel_indptr does not span panel_query_index")
    covered: set[int] = set()
    memberships: list[frozenset[int]] = []
    for start, stop in pairwise(pointers):
        if start < 0 or stop < start or stop - start > max_panel_queries:
            raise CorpusAuditError("queryPanels.panel_indptr violates panel bounds")
        panel_queries = query_indices[start:stop]
        if not panel_queries or list(panel_queries) != sorted(set(panel_queries)):
            raise CorpusAuditError(
                "each query panel must contain ordered unique query indices"
            )
        if any(index < 0 or index >= query_count for index in panel_queries):
            raise CorpusAuditError("queryPanels contains an out-of-range query index")
        covered.update(panel_queries)
        memberships.append(frozenset(panel_queries))
    if covered != set(range(query_count)):
        raise CorpusAuditError("queryPanels must cover the query dictionary exactly")
    return tuple(memberships)


def _validate_lineage_file(value: object, label: str, bounds: AuditBounds) -> dict[str, Any]:
    item = _strict_dict(value, LINEAGE_FILE_FIELDS, label)
    return {
        "path": _relative_path(item["path"], f"{label}.path"),
        "sha256": _raw_sha(item["sha256"], f"{label}.sha256"),
        "bytes": _nonnegative_int(item["bytes"], f"{label}.bytes", bounds.max_file_bytes),
    }


def _validate_lineage_files(value: object, label: str, bounds: AuditBounds) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value or len(value) > bounds.max_files_per_corpus:
        raise CorpusAuditError(f"{label} must be a non-empty bounded list")
    files = [
        _validate_lineage_file(item, f"{label}[{index}]", bounds)
        for index, item in enumerate(value)
    ]
    paths = [item["path"] for item in files]
    if paths != sorted(set(paths)):
        raise CorpusAuditError(f"{label} paths must be sorted and unique")
    return files


def _validate_dataset_snapshot(value: object, label: str) -> dict[str, str]:
    item = _strict_dict(value, DATASET_SNAPSHOT_FIELDS, label)
    resource = _string(item["resource"], f"{label}.resource")
    if not resource.startswith("omf://abiome/slp/datasetsnapshot/"):
        raise CorpusAuditError(f"{label}.resource must be an SLp DatasetSnapshot")
    _, resource_revision = _dataset_resource(resource, f"{label}.resource")
    revision = _prefixed_sha(item["revision"], f"{label}.revision")
    if revision != resource_revision:
        raise CorpusAuditError(f"{label}.revision must match its resource URI")
    return {
        "resource": resource,
        "revision": revision,
        "outerManifestDigest": _prefixed_sha(
            item["outerManifestDigest"], f"{label}.outerManifestDigest"
        ),
        "treeDigest": _prefixed_sha(item["treeDigest"], f"{label}.treeDigest"),
    }


def _validate_corpus_inputs(
    value: object, bounds: AuditBounds
) -> dict[str, dict[str, Any]]:
    raw = _strict_dict(value, CORPUS_INPUT_FIELDS, "inputs")
    result: dict[str, dict[str, Any]] = {}
    for name in sorted(CORPUS_INPUT_FIELDS):
        item = _strict_dict(raw[name], CORPUS_LINEAGE_FIELDS, f"inputs.{name}")
        result[name] = {
            "datasetSnapshot": _validate_dataset_snapshot(
                item["datasetSnapshot"], f"inputs.{name}.datasetSnapshot"
            ),
            "semanticSha256": _raw_sha(
                item["semanticSha256"], f"inputs.{name}.semanticSha256"
            ),
            "files": _validate_lineage_files(
                item["files"], f"inputs.{name}.files", bounds
            ),
        }
    return result


def _validate_feature_pack(value: object, bounds: AuditBounds) -> dict[str, Any]:
    pack = _strict_dict(value, FEATURE_PACK_FIELDS, "featurePack")
    if pack["schema"] != "slp.static-feature-pack/v1":
        raise CorpusAuditError("featurePack schema is unsupported")
    _curie(pack["revision"], "featurePack.revision")
    declared_sha = _raw_sha(pack["sha256"], "featurePack.sha256")
    digest_basis = dict(pack)
    del digest_basis["sha256"]
    if canonical_sha256(digest_basis) != declared_sha:
        raise CorpusAuditError("featurePack.sha256 does not match canonical pack content")
    entity_dim = _positive_int(
        pack["entityFeatureDim"], "featurePack.entityFeatureDim", 1_000_000
    )
    _positive_int(
        pack["speciesFeatureDim"], "featurePack.speciesFeatureDim", 1_000_000
    )
    raw_blocks = pack["blocks"]
    if not isinstance(raw_blocks, list) or not raw_blocks or len(raw_blocks) > 1024:
        raise CorpusAuditError("featurePack.blocks must be a non-empty bounded list")
    expected_offset = 0
    block_ids: set[str] = set()
    for index, raw_block in enumerate(raw_blocks):
        label = f"featurePack.blocks[{index}]"
        block = _strict_dict(raw_block, FEATURE_BLOCK_FIELDS, label)
        block_id = _curie(block["id"], f"{label}.id")
        if block_id in block_ids:
            raise CorpusAuditError("featurePack block ids must be unique")
        block_ids.add(block_id)
        offset = _nonnegative_int(block["offset"], f"{label}.offset", entity_dim)
        dimension = _positive_int(block["dimension"], f"{label}.dimension", entity_dim)
        if offset != expected_offset or offset + dimension > entity_dim:
            raise CorpusAuditError("featurePack blocks must be ordered and contiguous")
        expected_offset += dimension
        _validate_dataset_snapshot(block["datasetSnapshot"], f"{label}.datasetSnapshot")
        _raw_sha(block["semanticSha256"], f"{label}.semanticSha256")
        _raw_sha(block["entityKeySetSha256"], f"{label}.entityKeySetSha256")
        _validate_lineage_files(block["files"], f"{label}.files", bounds)
    if expected_offset != entity_dim:
        raise CorpusAuditError("featurePack blocks do not cover entityFeatureDim")
    return dict(pack)


def _audit_shard(
    path: Path,
    records: int,
    target_values: int,
    max_context_tokens: int,
    max_action_tokens: int,
    max_targets_per_record: int,
    source_count: int,
    query_entity_indices: tuple[int, ...],
    declared_species: frozenset[int],
    species_features: Mapping[int, tuple[tuple[float, ...], tuple[bool, ...]]],
    species_feature_dim: int,
    context_type_count: int,
    action_type_count: int,
    record_covariate_dim: int,
    context_covariate_dim: int,
    action_covariate_dim: int,
    observation_covariate_dim: int,
    panel_query_memberships: tuple[frozenset[int], ...],
    entity_keys: tuple[tuple[int, str], ...],
    entity_taxa: tuple[int, ...],
    record_db: sqlite3.Connection,
    target_value_digest: Any,
    bounds: AuditBounds,
) -> tuple[
    set[int], set[int], set[int], set[tuple[int, str]], set[tuple[int, str]]
]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = _member_map(
                archive, path.name, bounds, QUANTITATIVE_SHARD_MEMBERS
            )
            keys = tuple(sorted(QUANTITATIVE_SHARD_MEMBERS))
            specs = {
                key: _array_spec(archive, members, key, path.name, bounds) for key in keys
            }
            for key in (
                "record_id", "observation_unit_id", "replicate_id", "perturbation_id"
            ):
                if specs[key].shape != (records,) or not specs[key].descr.startswith("<U"):
                    raise CorpusAuditError(
                        f"{path.name} {key} must be record-aligned Unicode"
                    )
            for key in ("source_index", "species_taxon", "query_panel_index"):
                if specs[key].shape != (records,) or specs[key].descr != "<i8":
                    raise CorpusAuditError(f"{path.name} {key} must be a record-aligned integer array")
            if (
                specs["species_feature_value"].shape
                != (records, species_feature_dim)
                or specs["species_feature_value"].descr != "<f4"
                or specs["species_feature_present"].shape
                != (records, species_feature_dim)
                or specs["species_feature_present"].descr != "|b1"
            ):
                raise CorpusAuditError(
                    f"{path.name} species feature arrays do not match the manifest"
                )
            action_shape = specs["action_mask"].shape
            context_shape = specs["context_mask"].shape
            if (
                len(context_shape) != 2
                or context_shape[0] != records
                or not 1 <= context_shape[1] <= max_context_tokens
                or specs["context_entity_index"].shape != context_shape
                or specs["context_type"].shape != context_shape
                or specs["context_mask"].descr != "|b1"
                or specs["context_entity_index"].descr != "<i8"
                or specs["context_type"].descr != "<i8"
                or specs["context_covariate_value"].shape
                != (*context_shape, context_covariate_dim)
                or specs["context_covariate_value"].descr != "<f4"
                or specs["context_covariate_present"].shape
                != (*context_shape, context_covariate_dim)
                or specs["context_covariate_present"].descr != "|b1"
            ):
                raise CorpusAuditError(f"{path.name} context identity arrays are misaligned")
            if (
                len(action_shape) != 2
                or action_shape[0] != records
                or not 1 <= action_shape[1] <= max_action_tokens
                or specs["action_entity_index"].shape != action_shape
                or specs["action_type"].shape != action_shape
                or specs["action_mask"].descr != "|b1"
                or specs["action_entity_index"].descr != "<i8"
                or specs["action_type"].descr != "<i8"
                or specs["action_covariate_value"].shape
                != (*action_shape, action_covariate_dim)
                or specs["action_covariate_value"].descr != "<f4"
                or specs["action_covariate_present"].shape
                != (*action_shape, action_covariate_dim)
                or specs["action_covariate_present"].descr != "|b1"
            ):
                raise CorpusAuditError(f"{path.name} action identity arrays are misaligned")
            for axis, dimension in (
                ("record", record_covariate_dim),
                ("observation", observation_covariate_dim),
            ):
                if (
                    specs[f"{axis}_covariate_value"].shape != (records, dimension)
                    or specs[f"{axis}_covariate_value"].descr != "<f4"
                    or specs[f"{axis}_covariate_present"].shape
                    != (records, dimension)
                    or specs[f"{axis}_covariate_present"].descr != "|b1"
                ):
                    raise CorpusAuditError(
                        f"{path.name} {axis} covariates do not match the manifest"
                    )
            if (
                specs["target_indptr"].shape != (records + 1,)
                or specs["target_indptr"].descr != "<i8"
                or specs["target_query_index"].shape != (target_values,)
                or specs["target_query_index"].descr != "<i8"
                or specs["target_value"].shape != (target_values,)
                or specs["target_value"].descr != "<f4"
            ):
                raise CorpusAuditError(f"{path.name} target arrays do not match declared targets")

            record_batch: list[tuple[str]] = []
            try:
                for item in _iter_array_values(
                    archive, members["record_id"], specs["record_id"],
                    f"{path.name}.record_id", bounds,
                ):
                    record_batch.append((_curie(item, "record_id"),))
                    if len(record_batch) == 8_192:
                        record_db.executemany("INSERT INTO record_ids VALUES (?)", record_batch)
                        record_batch.clear()
                if record_batch:
                    record_db.executemany("INSERT INTO record_ids VALUES (?)", record_batch)
            except sqlite3.IntegrityError as error:
                raise CorpusAuditError("record_id values must be unique across a corpus") from error

            for key in ("observation_unit_id", "replicate_id"):
                for item in _iter_array_values(
                    archive, members[key], specs[key], f"{path.name}.{key}", bounds
                ):
                    _string(item, f"{path.name}.{key}")

            source_values: set[int] = set()
            for item in _iter_array_values(
                    archive, members["source_index"], specs["source_index"],
                    f"{path.name}.source_index", bounds,
            ):
                value = int(item)
                if value < 0 or value >= source_count:
                    raise CorpusAuditError(f"{path.name} contains an undeclared source index")
                source_values.add(value)
            species_values: set[int] = set()
            for item in _iter_array_values(
                    archive, members["species_taxon"], specs["species_taxon"],
                    f"{path.name}.species_taxon", bounds,
            ):
                value = int(item)
                if value not in declared_species:
                    raise CorpusAuditError(f"{path.name} contains an undeclared species taxon")
                species_values.add(value)

            feature_taxa = _iter_array_values(
                archive, members["species_taxon"], specs["species_taxon"],
                f"{path.name}.species_taxon", bounds,
            )
            feature_values = _iter_array_values(
                archive, members["species_feature_value"],
                specs["species_feature_value"],
                f"{path.name}.species_feature_value", bounds,
            )
            feature_present = _iter_array_values(
                archive, members["species_feature_present"],
                specs["species_feature_present"],
                f"{path.name}.species_feature_present", bounds,
            )
            for _ in range(records):
                taxon = int(next(feature_taxa))
                expected_values, expected_present = species_features[taxon]
                for column in range(species_feature_dim):
                    value = float(next(feature_values))
                    present = bool(next(feature_present))
                    expected_value = struct.unpack(
                        "<f", struct.pack("<f", expected_values[column])
                    )[0]
                    if (
                        not math.isfinite(value)
                        or (not present and value != 0.0)
                        or value != expected_value
                        or present != expected_present[column]
                    ):
                        raise CorpusAuditError(
                            f"{path.name} species features do not match the declared taxon"
                        )

            for axis in ("record", "observation"):
                covariate_values = _iter_array_values(
                    archive,
                    members[f"{axis}_covariate_value"],
                    specs[f"{axis}_covariate_value"],
                    f"{path.name}.{axis}_covariate_value",
                    bounds,
                )
                covariate_present = _iter_array_values(
                    archive,
                    members[f"{axis}_covariate_present"],
                    specs[f"{axis}_covariate_present"],
                    f"{path.name}.{axis}_covariate_present",
                    bounds,
                )
                for value, present in zip(
                    covariate_values, covariate_present, strict=True
                ):
                    numeric = float(value)
                    if not math.isfinite(numeric) or (not bool(present) and numeric != 0.0):
                        raise CorpusAuditError(
                            f"{path.name} {axis} covariates are non-finite or nonzero when absent"
                        )

            context_indices = _iter_array_values(
                archive, members["context_entity_index"], specs["context_entity_index"],
                f"{path.name}.context_entity_index", bounds,
            )
            context_masks = _iter_array_values(
                archive, members["context_mask"], specs["context_mask"],
                f"{path.name}.context_mask", bounds,
            )
            context_types = _iter_array_values(
                archive, members["context_type"], specs["context_type"],
                f"{path.name}.context_type", bounds,
            )
            context_covariate_values = _iter_array_values(
                archive, members["context_covariate_value"],
                specs["context_covariate_value"],
                f"{path.name}.context_covariate_value", bounds,
            )
            context_covariate_present = _iter_array_values(
                archive, members["context_covariate_present"],
                specs["context_covariate_present"],
                f"{path.name}.context_covariate_present", bounds,
            )
            context_row_taxa = _iter_array_values(
                archive, members["species_taxon"], specs["species_taxon"],
                f"{path.name}.species_taxon", bounds,
            )
            active_contexts: set[int] = set()
            for _ in range(records):
                record_taxon = int(next(context_row_taxa))
                row_contexts: set[int] = set()
                for _ in range(context_shape[1]):
                    index = int(next(context_indices))
                    context_type = int(next(context_types))
                    active = bool(next(context_masks))
                    covariates_valid_for_padding = True
                    for _ in range(context_covariate_dim):
                        value = float(next(context_covariate_values))
                        present = bool(next(context_covariate_present))
                        if not math.isfinite(value) or (not present and value != 0.0):
                            raise CorpusAuditError(
                                f"{path.name} context covariates are non-finite or nonzero when absent"
                            )
                        covariates_valid_for_padding &= not present and value == 0.0
                    if not active:
                        if index != -1 or context_type != -1 or not covariates_valid_for_padding:
                            raise CorpusAuditError(
                                f"{path.name} padded context tokens must use sentinel values"
                            )
                        continue
                    if context_type < 0 or context_type >= context_type_count:
                        raise CorpusAuditError(
                            f"{path.name} context_type is out of range"
                        )
                    if index < 0 or index >= len(entity_keys):
                        raise CorpusAuditError(
                            f"{path.name} context_entity_index is out of range"
                        )
                    if entity_taxa[index] != record_taxon:
                        raise CorpusAuditError(
                            f"{path.name} active context entity taxon does not match its record"
                        )
                    if index in row_contexts:
                        raise CorpusAuditError(
                            f"{path.name} repeats an active context in one record"
                        )
                    row_contexts.add(index)
                    active_contexts.add(index)

            indices = _iter_array_values(
                archive, members["action_entity_index"], specs["action_entity_index"],
                f"{path.name}.action_entity_index", bounds,
            )
            masks = _iter_array_values(
                archive, members["action_mask"], specs["action_mask"],
                f"{path.name}.action_mask", bounds,
            )
            action_types = _iter_array_values(
                archive, members["action_type"], specs["action_type"],
                f"{path.name}.action_type", bounds,
            )
            action_covariate_values = _iter_array_values(
                archive, members["action_covariate_value"],
                specs["action_covariate_value"],
                f"{path.name}.action_covariate_value", bounds,
            )
            action_covariate_present = _iter_array_values(
                archive, members["action_covariate_present"],
                specs["action_covariate_present"],
                f"{path.name}.action_covariate_present", bounds,
            )
            row_taxa = _iter_array_values(
                archive, members["species_taxon"], specs["species_taxon"],
                f"{path.name}.species_taxon", bounds,
            )
            interventions: set[tuple[int, str]] = set()
            active_actions: set[tuple[int, str]] = set()
            perturbation_ids = _iter_array_values(
                archive, members["perturbation_id"], specs["perturbation_id"],
                f"{path.name}.perturbation_id", bounds,
            )
            for _ in range(records):
                record_taxon = int(next(row_taxa))
                row_actions: set[tuple[int, str]] = set()
                for _ in range(action_shape[1]):
                    index = int(next(indices))
                    action_type = int(next(action_types))
                    active = bool(next(masks))
                    covariates_valid_for_padding = True
                    for _ in range(action_covariate_dim):
                        value = float(next(action_covariate_values))
                        present = bool(next(action_covariate_present))
                        if not math.isfinite(value) or (not present and value != 0.0):
                            raise CorpusAuditError(
                                f"{path.name} action covariates are non-finite or nonzero when absent"
                            )
                        covariates_valid_for_padding &= not present and value == 0.0
                    if not active:
                        if index != -1 or action_type != -1 or not covariates_valid_for_padding:
                            raise CorpusAuditError(
                                f"{path.name} padded action tokens must use sentinel values"
                            )
                        continue
                    if action_type < 0 or action_type >= action_type_count:
                        raise CorpusAuditError(
                            f"{path.name} action_type is out of range"
                        )
                    if index < 0 or index >= len(entity_keys):
                        raise CorpusAuditError(f"{path.name} action_entity_index is out of range")
                    entity_taxon = entity_taxa[index]
                    if entity_taxon != record_taxon:
                        raise CorpusAuditError(
                            f"{path.name} active action entity taxon does not match its record"
                        )
                    key = entity_keys[index]
                    if key in row_actions:
                        raise CorpusAuditError(
                            f"{path.name} repeats an active action in one record"
                        )
                    active_actions.add(key)
                    interventions.add(key)
                    row_actions.add(key)
                if not row_actions:
                    raise CorpusAuditError(
                        f"{path.name} records require at least one active action"
                    )
                if next(perturbation_ids) != _perturbation_id(row_actions):
                    raise CorpusAuditError(
                        f"{path.name} perturbation_id does not match active composite actions"
                    )

            pointer_values = _iter_array_values(
                archive, members["target_indptr"], specs["target_indptr"],
                f"{path.name}.target_indptr", bounds,
            )
            previous = int(next(pointer_values))
            if previous != 0:
                raise CorpusAuditError(f"{path.name} targetValues does not match target_indptr")
            target_queries = _iter_array_values(
                archive, members["target_query_index"], specs["target_query_index"],
                f"{path.name}.target_query_index", bounds,
            )
            target_row_taxa = _iter_array_values(
                archive, members["species_taxon"], specs["species_taxon"],
                f"{path.name}.species_taxon", bounds,
            )
            target_panels = _iter_array_values(
                archive, members["query_panel_index"], specs["query_panel_index"],
                f"{path.name}.query_panel_index", bounds,
            )
            for _ in range(records):
                current = int(next(pointer_values))
                if current < previous or current - previous > max_targets_per_record:
                    raise CorpusAuditError(
                        f"{path.name} target_indptr violates maxTargetsPerRecord"
                    )
                record_taxon = int(next(target_row_taxa))
                panel_index = int(next(target_panels))
                if panel_index < 0 or panel_index >= len(panel_query_memberships):
                    raise CorpusAuditError(
                        f"{path.name} query_panel_index is out of range"
                    )
                panel_queries = panel_query_memberships[panel_index]
                row_queries: set[int] = set()
                for _ in range(current - previous):
                    query_index = int(next(target_queries))
                    if query_index < 0 or query_index >= len(query_entity_indices):
                        raise CorpusAuditError(
                            f"{path.name} target query index is out of range"
                        )
                    if query_index not in panel_queries:
                        raise CorpusAuditError(
                            f"{path.name} target query is absent from its selected panel"
                        )
                    entity_index = query_entity_indices[query_index]
                    if entity_taxa[entity_index] != record_taxon:
                        raise CorpusAuditError(
                            f"{path.name} target query entity taxon does not match its record"
                        )
                    if query_index in row_queries:
                        raise CorpusAuditError(
                            f"{path.name} repeats a target query in one record"
                        )
                    row_queries.add(query_index)
                previous = current
            if previous != target_values:
                raise CorpusAuditError(f"{path.name} targetValues does not match target_indptr")
            for item in _iter_array_values(
                archive, members["target_value"], specs["target_value"],
                f"{path.name}.target_value", bounds,
            ):
                value = float(item)
                if not math.isfinite(value):
                    raise CorpusAuditError(f"{path.name} target values must be finite")
                target_value_digest.update(struct.pack("<f", value))
            return (
                source_values,
                species_values,
                active_contexts,
                interventions,
                active_actions,
            )
    except zipfile.BadZipFile as error:
        raise CorpusAuditError(f"{path.name} must be a valid NPZ archive") from error


def _validate_manifest_semantics(
    manifest: dict[str, Any], expected_role: str, bounds: AuditBounds
) -> tuple[str, str, tuple[str, ...], tuple[int, ...], int, int, int]:
    _strict_dict(manifest, CORPUS_FIELDS, "corpus manifest")
    if manifest["schema"] != CORPUS_SCHEMA:
        raise CorpusAuditError(f"corpus schema must be {CORPUS_SCHEMA}")
    dataset_id = _curie(manifest["datasetId"], "datasetId")
    version = _string(manifest["version"], "version")
    if MUTABLE_VERSION.search(version):
        raise CorpusAuditError("corpus version must be immutable")
    if manifest["role"] != expected_role:
        raise CorpusAuditError(f"corpus role must be exactly {expected_role}")
    if manifest["labelClass"] != "molecular":
        raise CorpusAuditError("all governed corpora must contain molecular labels")
    if manifest["benchmarkLabelsPresent"] is not False:
        raise CorpusAuditError("benchmark labels are forbidden")
    if manifest["rewardEnabled"] is not False:
        raise CorpusAuditError("corpus rewardEnabled must be false")
    if manifest["identityKey"] != ["ncbiTaxon", "entityId"]:
        raise CorpusAuditError(
            "corpus identityKey must be [ncbiTaxon, entityId]"
        )
    corpus_inputs = _validate_corpus_inputs(manifest["inputs"], bounds)

    rights = _strict_dict(
        manifest["rights"],
        {"revision", "trainingAllowed", "redistributionAllowed"},
        "rights",
    )
    _curie(rights["revision"], "rights.revision")
    if rights["trainingAllowed"] is not True:
        raise CorpusAuditError("trainingAllowed must be true for all governed corpora")
    if type(rights["redistributionAllowed"]) is not bool:
        raise CorpusAuditError("redistributionAllowed must be boolean")

    _unique_curies(manifest["modalities"], "modalities", sorted_required=True)
    sources_raw = manifest["sources"]
    if not isinstance(sources_raw, list) or not 1 <= len(sources_raw) <= bounds.max_sources:
        raise CorpusAuditError("sources must be a non-empty bounded list")
    sources: list[str] = []
    for item in sources_raw:
        item = _strict_dict(item, {"id"}, "source")
        sources.append(_curie(item["id"], "source.id"))
    if sources != sorted(set(sources)):
        raise CorpusAuditError("source inventory must be unique and deterministically sorted")
    sampling = _strict_dict(manifest["sampling"], {"scheme", "sourceWeights"}, "sampling")
    if sampling["scheme"] != "slp.source-intervention-replicate-record/v1":
        raise CorpusAuditError("sampling scheme is not the frozen v1 contract")
    weights = sampling["sourceWeights"]
    if not isinstance(weights, list) or len(weights) != len(sources):
        raise CorpusAuditError("sourceWeights must align exactly with source inventory")
    if any(
        type(value) not in {int, float} or not math.isfinite(value) or value <= 0
        for value in weights
    ):
        raise CorpusAuditError("sourceWeights must be positive finite numbers")

    feature_pack = _validate_feature_pack(manifest["featurePack"], bounds)
    if (
        len(feature_pack["blocks"]) != 1
        or corpus_inputs["staticFeatures"]["semanticSha256"]
        != feature_pack["blocks"][0]["semanticSha256"]
        or corpus_inputs["staticFeatures"]["datasetSnapshot"]
        != feature_pack["blocks"][0]["datasetSnapshot"]
        or corpus_inputs["staticFeatures"]["files"]
        != feature_pack["blocks"][0]["files"]
    ):
        raise CorpusAuditError(
            "v1.2 staticFeatures must exactly bind its one feature block lineage"
        )
    species_feature_dim = _positive_int(
        feature_pack["speciesFeatureDim"], "speciesFeatureDim", 1_000_000
    )
    species_raw = manifest["species"]
    if not isinstance(species_raw, list) or not 1 <= len(species_raw) <= bounds.max_species:
        raise CorpusAuditError("species must be a non-empty bounded list")
    taxa: list[int] = []
    for item in species_raw:
        item = _strict_dict(item, {"taxon", "featureValue", "featurePresent"}, "species")
        taxon = _positive_int(item["taxon"], "species.taxon", 2_147_483_647)
        values, present = item["featureValue"], item["featurePresent"]
        if (
            not isinstance(values, list)
            or not isinstance(present, list)
            or len(values) != species_feature_dim
            or len(present) != species_feature_dim
            or any(type(flag) is not bool for flag in present)
            or any(
                type(value) not in {int, float} or not math.isfinite(value)
                for value in values
            )
        ):
            raise CorpusAuditError("species feature vectors must match speciesFeatureDim")
        if any(not flag and float(value) != 0.0 for value, flag in zip(values, present)):
            raise CorpusAuditError("absent species features must store exact zero")
        taxa.append(taxon)
    if taxa != sorted(set(taxa)):
        raise CorpusAuditError("species inventory must be unique and deterministically sorted")

    for name in ("entityTypes", "contextTypes", "actionTypes"):
        _unique_curies(manifest[name], name)
    if manifest["entityTypes"] != ENTITY_TYPES:
        raise CorpusAuditError("entityTypes must match the ordered v1.2 contract")
    covariates = _strict_dict(
        manifest["covariates"],
        {"record", "context", "action", "observation"},
        "covariates",
    )
    for location, declarations in covariates.items():
        if not isinstance(declarations, list):
            raise CorpusAuditError(f"covariates.{location} must be a list")
        seen: set[str] = set()
        for item in declarations:
            item = _strict_dict(
                item, {"id", "unit", "access"}, f"covariates.{location}"
            )
            identifier = _curie(item["id"], "covariate.id")
            _curie(item["unit"], "covariate.unit")
            if item["access"] not in {"world", "likelihood", "audit"} or identifier in seen:
                raise CorpusAuditError(f"covariates.{location} is invalid or duplicated")
            seen.add(identifier)
    readouts = manifest["readoutTypes"]
    if not isinstance(readouts, list) or not readouts:
        raise CorpusAuditError("readoutTypes must be non-empty")
    readout_ids: list[str] = []
    for item in readouts:
        item = _strict_dict(
            item, {"id", "likelihood", "unit", "implicitZero"}, "readoutType"
        )
        readout_ids.append(_curie(item["id"], "readoutType.id"))
        _curie(item["unit"], "readoutType.unit")
        if item["likelihood"] not in {"gaussian", "negative-binomial"}:
            raise CorpusAuditError("readout likelihood is unsupported")
        if type(item["implicitZero"]) is not bool:
            raise CorpusAuditError("readout implicitZero must be boolean")
    if len(readout_ids) != len(set(readout_ids)):
        raise CorpusAuditError("readout ids must be unique")
    normalization = _strict_dict(
        manifest["normalization"], {"id", "valueSpace"}, "normalization"
    )
    _curie(normalization["id"], "normalization.id")
    _curie(normalization["valueSpace"], "normalization.valueSpace")
    corpus_bounds = _strict_dict(
        manifest["bounds"],
        {
            "maxRecordsPerShard", "maxContextTokens", "maxActionTokens",
            "maxPanelQueries", "maxTargetsPerRecord",
        },
        "bounds",
    )
    for name, value in corpus_bounds.items():
        _positive_int(value, f"bounds.{name}", bounds.max_records_per_corpus)
    counts = _strict_dict(manifest["counts"], CORPUS_COUNT_FIELDS, "counts")
    for name, value in counts.items():
        _nonnegative_int(value, f"counts.{name}", bounds.max_records_per_corpus * 1_000)
    return (
        dataset_id,
        version,
        tuple(sources),
        tuple(taxa),
        int(corpus_bounds["maxRecordsPerShard"]),
        int(corpus_bounds["maxActionTokens"]),
        int(corpus_bounds["maxTargetsPerRecord"]),
    )


def _validate_canonical_ustar(
    bundle: Path,
    members: list[tarfile.TarInfo],
) -> None:
    names = [member.name for member in members]
    if names != sorted(names):
        raise CorpusAuditError("corpus tar members must use exact sorted order")
    expected_offset = 0
    try:
        with bundle.open("rb") as stream:
            for member in members:
                if (
                    not member.isreg()
                    or member.type != tarfile.REGTYPE
                    or member.pax_headers
                    or member.offset != expected_offset
                    or member.offset_data != member.offset + tarfile.BLOCKSIZE
                ):
                    raise CorpusAuditError(
                        "corpus tar must contain only canonical regular-file members"
                    )
                expected = tarfile.TarInfo(member.name)
                expected.size = member.size
                expected.mode = 0o644
                expected.mtime = 0
                expected.uid = 0
                expected.gid = 0
                expected.uname = ""
                expected.gname = ""
                expected_header = expected.tobuf(
                    format=tarfile.USTAR_FORMAT,
                    encoding="utf-8",
                    errors="strict",
                )
                stream.seek(member.offset)
                if stream.read(tarfile.BLOCKSIZE) != expected_header:
                    raise CorpusAuditError(
                        "corpus tar member metadata is not canonical USTAR"
                    )
                data_end = member.offset_data + member.size
                expected_offset = (
                    (data_end + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE
                ) * tarfile.BLOCKSIZE
                stream.seek(data_end)
                if any(stream.read(expected_offset - data_end)):
                    raise CorpusAuditError("corpus tar member padding must be zero")
            expected_size = (
                (expected_offset + 2 * tarfile.BLOCKSIZE + tarfile.RECORDSIZE - 1)
                // tarfile.RECORDSIZE
            ) * tarfile.RECORDSIZE
            stream.seek(0, os.SEEK_END)
            actual_size = stream.tell()
            if actual_size != expected_size:
                raise CorpusAuditError("corpus tar block size or trailer is not canonical")
            stream.seek(expected_offset)
            remaining = actual_size - expected_offset
            while remaining:
                block = stream.read(min(1024 * 1024, remaining))
                if not block or any(block):
                    raise CorpusAuditError("corpus tar trailer must contain only zero blocks")
                remaining -= len(block)
    except (OSError, ValueError, tarfile.TarError) as error:
        if isinstance(error, CorpusAuditError):
            raise
        raise CorpusAuditError("could not validate canonical corpus USTAR") from error


@contextmanager
def _materialized_corpus_payload(
    dataset: DatasetInput, bounds: AuditBounds
) -> Iterator[tuple[Path, Path, Path]]:
    files, directories = _directory_entries(
        dataset.root, bounds.max_files_per_corpus * 2
    )
    expected_files = {"corpus-v1-2.tar", "corpus-compose-audit.json"}
    if files != expected_files or directories:
        raise CorpusAuditError(
            "pretrain must contain exactly corpus-v1-2.tar and "
            "corpus-compose-audit.json"
        )
    bundle = _regular_file(
        dataset.root,
        "corpus-v1-2.tar",
        "pretrain corpus tar",
        bounds.max_total_bytes_per_corpus,
    )
    companion = _regular_file(
        dataset.root,
        "corpus-compose-audit.json",
        "pretrain composition audit",
        bounds.max_manifest_bytes,
    )
    with tempfile.TemporaryDirectory(prefix="slp-corpus-audit-bundle-") as temporary:
        root = Path(temporary) / "payload"
        root.mkdir()
        seen: set[str] = set()
        total_bytes = 0
        try:
            with tarfile.open(bundle, mode="r:") as archive:
                members = archive.getmembers()
                if not members or len(members) > bounds.max_files_per_corpus * 2:
                    raise CorpusAuditError("corpus tar has an invalid bounded member count")
                _validate_canonical_ustar(bundle, members)
                for member in members:
                    name = _relative_path(member.name.rstrip("/"), "corpus tar member")
                    if name in seen:
                        raise CorpusAuditError("corpus tar contains duplicate member names")
                    seen.add(name)
                    destination = root.joinpath(*PurePosixPath(name).parts)
                    if not member.isreg():
                        raise CorpusAuditError("corpus tar may contain only regular files")
                    if member.size > bounds.max_file_bytes:
                        raise CorpusAuditError("corpus tar member exceeds maxFileBytes")
                    total_bytes += member.size
                    if total_bytes > bounds.max_total_bytes_per_corpus:
                        raise CorpusAuditError("corpus tar exceeds maxTotalBytesPerCorpus")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        raise CorpusAuditError("corpus tar member path collides")
                    source = archive.extractfile(member)
                    if source is None:
                        raise CorpusAuditError("could not read regular corpus tar member")
                    with source, destination.open("xb") as output:
                        remaining = member.size
                        while remaining:
                            block = source.read(min(1024 * 1024, remaining))
                            if not block:
                                raise CorpusAuditError("corpus tar member is truncated")
                            output.write(block)
                            remaining -= len(block)
                        if source.read(1):
                            raise CorpusAuditError("corpus tar member has trailing bytes")
        except (OSError, tarfile.TarError) as error:
            raise CorpusAuditError("pretrain corpus tar is invalid") from error
        payload = root / "composite-corpus"
        if not (payload / "corpus.json").is_file():
            raise CorpusAuditError(
                "corpus tar must contain composite-corpus/corpus.json"
            )
        outer_files, outer_directories = _directory_entries(
            root, bounds.max_files_per_corpus * 2
        )
        if (
            not outer_files
            or any(not name.startswith("composite-corpus/") for name in outer_files)
            or "composite-corpus" not in outer_directories
        ):
            raise CorpusAuditError("corpus tar has files outside composite-corpus")
        yield payload, bundle, companion


def _load_corpus_root(
    dataset: DatasetInput,
    expected_role: str,
    bounds: AuditBounds,
    root: Path,
) -> Corpus:
    manifest_path = _regular_file(
        root,
        "corpus.json",
        f"{dataset.input_name}.corpus.json",
        bounds.max_manifest_bytes,
    )
    manifest = _read_json(manifest_path, f"{dataset.input_name}.corpus.json")
    (
        dataset_id,
        version,
        sources,
        taxa,
        max_records_per_shard,
        max_action_tokens,
        max_targets_per_record,
    ) = (
        _validate_manifest_semantics(manifest, expected_role, bounds)
    )
    entity_ref = _file_reference(manifest["entityDictionary"], "entityDictionary", bounds)
    query_ref = _file_reference(manifest["queryDictionary"], "queryDictionary", bounds)
    panel_ref = _file_reference(manifest["queryPanels"], "queryPanels", bounds)
    trajectory_ref = _file_reference(
        manifest["trajectoryInterventions"],
        "trajectoryInterventions",
        bounds,
        allow_zero=True,
    )
    shards_raw = manifest["shards"]
    if (
        not isinstance(shards_raw, list)
        or not shards_raw
        or len(shards_raw) > bounds.max_files_per_corpus - 5
    ):
        raise CorpusAuditError("shards must be a non-empty bounded list")
    shards: list[dict[str, Any]] = []
    shard_refs: list[FileReference] = []
    total_records = 0
    total_targets = 0
    for index, item in enumerate(shards_raw):
        item = _strict_dict(item, SHARD_FIELDS, f"shards[{index}]")
        records = _positive_int(
            item["records"], f"shards[{index}].records", bounds.max_records_per_corpus
        )
        if records > max_records_per_shard:
            raise CorpusAuditError("shard records exceed corpus maxRecordsPerShard")
        targets = _nonnegative_int(
            item["targetValues"],
            f"shards[{index}].targetValues",
            bounds.max_records_per_corpus * 1_000,
        )
        if targets > records * max_targets_per_record:
            raise CorpusAuditError("shard targetValues exceed maxTargetsPerRecord")
        reference = FileReference(
            _relative_path(item["path"], f"shards[{index}].path"),
            _raw_sha(item["sha256"], f"shards[{index}].sha256"),
            _nonnegative_int(
                item["bytes"], f"shards[{index}].bytes", bounds.max_file_bytes
            ),
            records,
        )
        shards.append(
            {
                "path": reference.path,
                "sha256": reference.sha256,
                "bytes": reference.bytes,
                "records": records,
                "targetValues": targets,
            }
        )
        shard_refs.append(reference)
        total_records += records
        total_targets += targets
    shard_paths = [item["path"] for item in shards]
    if shard_paths != sorted(set(shard_paths)):
        raise CorpusAuditError("shard paths must be unique and deterministically sorted")
    if total_records > bounds.max_records_per_corpus:
        raise CorpusAuditError("corpus records exceed maxRecordsPerCorpus")
    if total_targets == 0:
        raise CorpusAuditError("governed quantitative corpus must contain target values")
    expected_counts = {
        "entities": entity_ref.count,
        "queries": query_ref.count,
        "panels": panel_ref.count,
        "trajectoryInterventions": trajectory_ref.count,
        "records": total_records,
        "targetValues": total_targets,
        "shards": len(shards),
    }
    for name, expected in expected_counts.items():
        if manifest["counts"][name] != expected:
            raise CorpusAuditError(f"counts.{name} does not match audited corpus content")

    references = [entity_ref, query_ref, panel_ref, trajectory_ref, *shard_refs]
    paths = [item.path for item in references]
    if len(paths) != len(set(paths)):
        raise CorpusAuditError("internal corpus file references must be unique")
    expected_files = {"corpus.json", *paths}
    if len(expected_files) > bounds.max_files_per_corpus:
        raise CorpusAuditError("corpus file count exceeds maxFilesPerCorpus")
    _exact_file_set(root, expected_files, dataset.input_name)
    total_bytes = manifest_path.stat().st_size
    materialized: dict[str, Path] = {}
    for reference in references:
        path = _regular_file(
            root, reference.path, reference.path, bounds.max_file_bytes
        )
        total_bytes += path.stat().st_size
        if total_bytes > bounds.max_total_bytes_per_corpus:
            raise CorpusAuditError("corpus exceeds maxTotalBytesPerCorpus")
        if path.stat().st_size != reference.bytes:
            raise CorpusAuditError(f"internal content byte count mismatch: {reference.path}")
        if _sha256(path) != reference.sha256:
            raise CorpusAuditError(f"internal content digest mismatch: {reference.path}")
        materialized[reference.path] = path

    intervention_path = materialized[trajectory_ref.path]
    interventions: set[tuple[int, str]] = set()
    previous_key: tuple[int, str] | None = None
    try:
        with intervention_path.open("rb") as stream:
            line_number = 0
            while True:
                raw = stream.readline(bounds.max_line_bytes + 1)
                if raw == b"":
                    break
                line_number += 1
                if line_number > bounds.max_trajectory_interventions:
                    raise CorpusAuditError(
                        "trajectoryInterventions exceeds maxTrajectoryInterventions"
                    )
                if len(raw) > bounds.max_line_bytes:
                    raise CorpusAuditError(
                        f"trajectoryInterventions line {line_number} exceeds maxLineBytes"
                    )
                if not raw.endswith(b"\n") or raw.endswith(b"\r\n"):
                    raise CorpusAuditError(
                        "trajectoryInterventions must use canonical LF-terminated lines"
                    )
                try:
                    row = json.loads(
                        raw[:-1].decode("utf-8"),
                        object_pairs_hook=_duplicate_rejecting_object,
                    )
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise CorpusAuditError(
                        "trajectoryInterventions must be UTF-8 JSONL"
                    ) from error
                row = _strict_dict(
                    row,
                    {"schema", "ncbiTaxon", "entityId"},
                    f"trajectoryInterventions[{line_number}]",
                )
                if row["schema"] != TRAJECTORY_INTERVENTION_SCHEMA:
                    raise CorpusAuditError(
                        "trajectory intervention schema is unsupported"
                    )
                taxon = _positive_int(
                    row["ncbiTaxon"],
                    "trajectory intervention ncbiTaxon",
                    2_147_483_647,
                )
                if taxon not in taxa:
                    raise CorpusAuditError(
                        "trajectory intervention taxon is not declared"
                    )
                identifier = _curie(
                    row["entityId"], "trajectory intervention entityId"
                )
                canonical = canonical_json_bytes(
                    {
                        "schema": TRAJECTORY_INTERVENTION_SCHEMA,
                        **_composite_object(_composite_key(taxon, identifier)),
                    }
                )
                if raw[:-1] != canonical:
                    raise CorpusAuditError(
                        "trajectoryInterventions rows must be canonical JSON objects"
                    )
                key = _composite_key(taxon, identifier)
                if previous_key is not None and key <= previous_key:
                    raise CorpusAuditError(
                        "trajectoryInterventions must be unique and sorted by composite key"
                    )
                interventions.add(key)
                previous_key = key
    except OSError as error:
        raise CorpusAuditError("could not read trajectoryInterventions") from error
    if len(interventions) != trajectory_ref.count:
        raise CorpusAuditError("trajectoryInterventions count drift")

    (
        entity_keys,
        entity_taxa,
        entity_types,
        entity_feature_rows,
        block_key_hashes,
        feature_value_bytes_sha,
        feature_present_bytes_sha,
    ) = _read_entity_dictionary(
        materialized[entity_ref.path],
        entity_ref.count,
        frozenset(taxa),
        int(manifest["featurePack"]["entityFeatureDim"]),
        manifest["featurePack"]["blocks"],
        bounds,
    )
    entity_key_set_sha = _composite_set_sha256(set(entity_keys))
    if any(index >= len(manifest["entityTypes"]) for index in entity_types):
        raise CorpusAuditError("entity_type contains an undeclared type index")
    for block, actual_hash in zip(
        manifest["featurePack"]["blocks"], block_key_hashes, strict=True
    ):
        if block["entityKeySetSha256"] != actual_hash:
            raise CorpusAuditError(
                "featurePack block entityKeySetSha256 does not match its present rows"
            )
    if manifest["counts"]["featureRows"] != sum(entity_feature_rows):
        raise CorpusAuditError(
            "counts.featureRows does not match complete entity feature rows"
        )
    query_entity_indices = _read_query_dictionary(
        materialized[query_ref.path],
        query_ref.count,
        entity_keys,
        len(manifest["readoutTypes"]),
        bounds,
    )
    panel_query_memberships = _validate_query_panels(
        materialized[panel_ref.path],
        panel_ref.count,
        query_ref.count,
        int(manifest["bounds"]["maxPanelQueries"]),
        bounds,
    )
    if not interventions.issubset(set(entity_keys)):
        raise CorpusAuditError(
            "trajectoryInterventions contains a composite key outside entityDictionary"
        )
    seen_sources: set[int] = set()
    seen_species: set[int] = set()
    active_context_indices: set[int] = set()
    record_interventions: set[tuple[int, str]] = set()
    active_actions: set[tuple[int, str]] = set()
    target_value_digest = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix="slp-corpus-audit-records-") as temporary:
        record_db = sqlite3.connect(Path(temporary) / "record-ids.sqlite3")
        try:
            record_db.execute("CREATE TABLE record_ids (identifier TEXT PRIMARY KEY) WITHOUT ROWID")
            for shard, reference in zip(shards, shard_refs, strict=True):
                (
                    source_values,
                    species_values,
                    shard_contexts,
                    shard_interventions,
                    shard_actions,
                    ) = _audit_shard(
                    materialized[reference.path],
                    shard["records"],
                    shard["targetValues"],
                    int(manifest["bounds"]["maxContextTokens"]),
                    max_action_tokens,
                    max_targets_per_record,
                    len(sources),
                        query_entity_indices,
                        frozenset(taxa),
                        {
                            int(item["taxon"]): (
                                tuple(float(value) for value in item["featureValue"]),
                                tuple(bool(value) for value in item["featurePresent"]),
                            )
                            for item in manifest["species"]
                        },
                        int(manifest["featurePack"]["speciesFeatureDim"]),
                        len(manifest["contextTypes"]),
                        len(manifest["actionTypes"]),
                        len(manifest["covariates"]["record"]),
                        len(manifest["covariates"]["context"]),
                        len(manifest["covariates"]["action"]),
                        len(manifest["covariates"]["observation"]),
                        panel_query_memberships,
                        entity_keys,
                        entity_taxa,
                        record_db,
                        target_value_digest,
                        bounds,
                )
                seen_sources.update(source_values)
                seen_species.update(species_values)
                active_context_indices.update(shard_contexts)
                record_interventions.update(shard_interventions)
                active_actions.update(shard_actions)
        finally:
            record_db.close()
    if seen_sources != set(range(len(sources))):
        raise CorpusAuditError("source inventory does not exactly match record-level source indices")
    if seen_species != set(taxa):
        raise CorpusAuditError("species inventory does not exactly match record-level taxa")
    if manifest["counts"]["contexts"] != len(active_context_indices):
        raise CorpusAuditError(
            "counts.contexts does not match unique active context entities"
        )
    context_type_index = ENTITY_TYPES.index("slp-entity-type:context")
    typed_contexts = {
        index for index, entity_type in enumerate(entity_types)
        if entity_type == context_type_index
    }
    missing_feature_rows = {
        index for index, has_features in enumerate(entity_feature_rows)
        if not has_features
    }
    if (
        typed_contexts != active_context_indices
        or missing_feature_rows != active_context_indices
        or manifest["counts"]["featureRows"]
        + manifest["counts"]["contexts"]
        != entity_ref.count
    ):
        raise CorpusAuditError(
            "context entities must exactly equal active missing-feature rows"
        )
    if record_interventions != interventions:
        raise CorpusAuditError(
            "trajectoryInterventions does not exactly match record-level composite "
            "actions; "
            f"missing={sorted(record_interventions - interventions)}, "
            f"extra={sorted(interventions - record_interventions)}"
        )

    manifest_sha = _sha256(manifest_path)
    content_digest = canonical_sha256(
        {
            "manifestSha256": manifest_sha,
            "files": [
                {"path": reference.path, "sha256": reference.sha256}
                for reference in references
            ],
        }
    )
    identity = {
        "resource": dataset.resource,
        "revision": dataset.revision,
        "manifestDigest": dataset.manifest_digest,
        "datasetId": dataset_id,
        "version": version,
        "role": expected_role,
        "corpusManifestSha256": manifest_sha,
        "contentDigest": content_digest,
        "trajectoryInterventionsSha256": trajectory_ref.sha256,
        "trajectoryInterventionSetSha256": _composite_set_sha256(interventions),
        "trajectoryInterventionCount": len(interventions),
        "entityKeySetSha256": entity_key_set_sha,
        "records": total_records,
        "targetValues": total_targets,
        "entityCount": entity_ref.count,
        "queryCount": query_ref.count,
        "queryPanelCount": panel_ref.count,
        "shardCount": len(shards),
        "featurePack": dict(manifest["featurePack"]),
        "inputs": dict(manifest["inputs"]),
        "counts": dict(manifest["counts"]),
        "identityKey": list(manifest["identityKey"]),
        "normalization": dict(manifest["normalization"]),
        "modalities": list(manifest["modalities"]),
        "sourceIds": list(sources),
        "speciesTaxa": list(taxa),
    }
    composition_facts = {
        "corpusManifestSha256": manifest_sha,
        "inputs": dict(manifest["inputs"]),
        "counts": dict(manifest["counts"]),
        "entityKeySetSha256": entity_key_set_sha,
        "featureEntityKeySetSha256": block_key_hashes[0],
        "contextEntities": [
            _composite_object(entity_keys[index])
            for index in sorted(active_context_indices)
        ],
        "featurePackSha256": manifest["featurePack"]["sha256"],
        "featureValueBytesSha256": feature_value_bytes_sha,
        "featurePresentBytesSha256": feature_present_bytes_sha,
        "targetValueBytesSha256": target_value_digest.hexdigest(),
    }
    return Corpus(
        dataset,
        identity,
        frozenset(interventions),
        frozenset(active_actions),
        composition_facts,
    )


def _read_composition_audit(path: Path, bounds: AuditBounds) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise CorpusAuditError("could not read corpus-compose-audit.json") from error
    if not payload or len(payload) > bounds.max_manifest_bytes:
        raise CorpusAuditError("composition audit is empty or exceeds maxManifestBytes")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CorpusAuditError(f"composition audit contains {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusAuditError("composition audit must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise CorpusAuditError("composition audit must be a JSON object")
    canonical = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if payload != canonical:
        raise CorpusAuditError(
            "corpus-compose-audit.json must use canonical sorted pretty JSON"
        )
    return value


def _validate_composition_audit(
    path: Path,
    archive_path: Path,
    corpus: Corpus,
    bounds: AuditBounds,
) -> tuple[str, str, str]:
    companion = _strict_dict(
        _read_composition_audit(path, bounds),
        COMPOSITION_AUDIT_FIELDS,
        "composition audit",
    )
    facts = corpus.composition_facts
    archive_sha = _sha256(archive_path)
    archive_bytes = archive_path.stat().st_size
    archive = _strict_dict(companion["archive"], {"path", "sha256", "bytes"}, "composition audit archive")
    if (
        companion["schema"] != COMPOSITION_AUDIT_SCHEMA
        or archive["path"] != "corpus-v1-2.tar"
        or _raw_sha(archive["sha256"], "composition audit archive.sha256")
        != archive_sha
        or _positive_int(
            archive["bytes"], "composition audit archive.bytes", bounds.max_total_bytes_per_corpus
        )
        != archive_bytes
        or _raw_sha(
            companion["corpusManifestSha256"],
            "composition audit corpusManifestSha256",
        )
        != facts["corpusManifestSha256"]
    ):
        raise CorpusAuditError("composition audit archive or corpus digest mismatch")

    inputs = _validate_corpus_inputs(companion["inputs"], bounds)
    counts = _strict_dict(companion["counts"], CORPUS_COUNT_FIELDS, "composition audit counts")
    for name, value in counts.items():
        _nonnegative_int(
            value,
            f"composition audit counts.{name}",
            bounds.max_records_per_corpus * 1_000,
        )
    if inputs != facts["inputs"] or counts != facts["counts"]:
        raise CorpusAuditError("composition audit input lineage or counts mismatch")

    identity = _strict_dict(
        companion["identity"], COMPOSITION_IDENTITY_FIELDS, "composition audit identity"
    )
    context = _strict_dict(
        identity["contextEntity"], {"ncbiTaxon", "entityId"},
        "composition audit contextEntity",
    )
    context_key = _composite_key(
        _positive_int(context["ncbiTaxon"], "contextEntity.ncbiTaxon", 2_147_483_647),
        _curie(context["entityId"], "contextEntity.entityId"),
    )
    if (
        identity["key"] != ["ncbiTaxon", "entityId"]
        or _raw_sha(
            identity["corpusEntityKeySetSha256"],
            "composition audit corpusEntityKeySetSha256",
        )
        != facts["entityKeySetSha256"]
        or _raw_sha(
            identity["featureEntityKeySetSha256"],
            "composition audit featureEntityKeySetSha256",
        )
        != facts["featureEntityKeySetSha256"]
        or facts["contextEntities"] != [_composite_object(context_key)]
    ):
        raise CorpusAuditError("composition audit identity mismatch")

    feature_pack_sha = _raw_sha(
        companion["featurePackSha256"], "composition audit featurePackSha256"
    )
    if feature_pack_sha != facts["featurePackSha256"]:
        raise CorpusAuditError("composition audit feature-pack digest mismatch")
    feature = _strict_dict(
        companion["featurePreservation"],
        FEATURE_PRESERVATION_FIELDS,
        "composition audit featurePreservation",
    )
    source_value_sha = _raw_sha(
        feature["sourceValueBytesSha256"], "featurePreservation.sourceValueBytesSha256"
    )
    composed_value_sha = _raw_sha(
        feature["composedValueBytesSha256"],
        "featurePreservation.composedValueBytesSha256",
    )
    source_present_sha = _raw_sha(
        feature["sourcePresentBytesSha256"],
        "featurePreservation.sourcePresentBytesSha256",
    )
    composed_present_sha = _raw_sha(
        feature["composedPresentBytesSha256"],
        "featurePreservation.composedPresentBytesSha256",
    )
    if (
        _positive_int(feature["rows"], "featurePreservation.rows", bounds.max_entities)
        != facts["counts"]["featureRows"]
        or _positive_int(
            feature["dimension"], "featurePreservation.dimension", 1_000_000
        )
        != corpus.identity["featurePack"]["entityFeatureDim"]
        or source_value_sha != composed_value_sha
        or composed_value_sha != facts["featureValueBytesSha256"]
        or source_present_sha != composed_present_sha
        or composed_present_sha != facts["featurePresentBytesSha256"]
        or feature["byteExact"] is not True
    ):
        raise CorpusAuditError("composition audit feature preservation mismatch")

    target = _strict_dict(
        companion["targetPreservation"],
        TARGET_PRESERVATION_FIELDS,
        "composition audit targetPreservation",
    )
    source_target_sha = _raw_sha(
        target["sourceBytesSha256"], "targetPreservation.sourceBytesSha256"
    )
    composed_target_sha = _raw_sha(
        target["composedBytesSha256"], "targetPreservation.composedBytesSha256"
    )
    if (
        target["dtype"] != "little-endian-float32"
        or _positive_int(
            target["values"],
            "targetPreservation.values",
            bounds.max_records_per_corpus * 1_000,
        )
        != facts["counts"]["targetValues"]
        or source_target_sha != composed_target_sha
        or composed_target_sha != facts["targetValueBytesSha256"]
        or target["byteExact"] is not True
    ):
        raise CorpusAuditError("composition audit target preservation mismatch")

    leakage = _strict_dict(
        companion["leakage"],
        {
            "heldRosterChecked", "protectedInterventionOverlap",
            "benchmarkLabelsPresent", "rewardDataPresent",
        },
        "composition audit leakage",
    )
    overlap = _nonnegative_int(
        leakage["protectedInterventionOverlap"],
        "composition audit protectedInterventionOverlap",
        bounds.max_roster_records,
    )
    if (
        leakage["heldRosterChecked"] is not True
        or overlap != 0
        or leakage["benchmarkLabelsPresent"] is not False
        or leakage["rewardDataPresent"] is not False
    ):
        raise CorpusAuditError("composition audit leakage declaration mismatch")
    if companion["formats"] != {
        "archive": "canonical-USTAR",
        "arrays": "deterministic-uncompressed-NPZ",
    }:
        raise CorpusAuditError("composition audit format declaration mismatch")
    if companion["limitations"] != COMPOSITION_LIMITATIONS:
        raise CorpusAuditError("composition audit limitations declaration mismatch")

    companion_sha = _sha256(path)
    package_digest = canonical_sha256(
        {
            "archive": {"path": "corpus-v1-2.tar", "sha256": archive_sha, "bytes": archive_bytes},
            "compositionAudit": {
                "path": "corpus-compose-audit.json",
                "sha256": companion_sha,
                "bytes": path.stat().st_size,
            },
        }
    )
    return archive_sha, companion_sha, package_digest


def load_corpus(dataset: DatasetInput, expected_role: str, bounds: AuditBounds) -> Corpus:
    with _materialized_corpus_payload(dataset, bounds) as materialized:
        root, archive_path, companion_path = materialized
        corpus = _load_corpus_root(dataset, expected_role, bounds, root)
        archive_sha, companion_sha, package_digest = _validate_composition_audit(
            companion_path, archive_path, corpus, bounds
        )
        identity = {
            **corpus.identity,
            "bundleArchiveSha256": archive_sha,
            "compositionAuditSha256": companion_sha,
            "compositionAuditSchema": COMPOSITION_AUDIT_SCHEMA,
            "compositionCompanionValidated": True,
            "sourcePreservationIndependentlyRecomputed": False,
            "packageContentDigest": package_digest,
        }
        return Corpus(
            corpus.input,
            identity,
            corpus.trajectory_interventions,
            corpus.active_actions,
            corpus.composition_facts,
        )


def _role_for_digest(digest: str) -> tuple[str, int]:
    _raw_sha(digest, "roster assignment hash")
    bucket = int(digest[:16], 16) % 100
    if bucket <= 9:
        return "molecular-final", bucket
    if bucket <= 29:
        return "molecular-validation", bucket
    return "pretrain", bucket


def assign_intervention(identifier: str) -> tuple[str, str, int]:
    """Return the frozen roster role, digest, and bucket for one SGD CURIE."""
    if not isinstance(identifier, str) or SGD_CURIE.fullmatch(identifier) is None:
        raise CorpusAuditError("held roster intervention must be a canonical SGD CURIE")
    digest = hashlib.sha256(ASSIGNMENT_DOMAIN + identifier.encode("ascii")).hexdigest()
    role, bucket = _role_for_digest(digest)
    return role, digest, bucket


def load_protected_inventory(
    dataset: DatasetInput, bounds: AuditBounds
) -> ProtectedInventory:
    manifest_path = _regular_file(
        dataset.root,
        "inventory.json",
        f"{dataset.input_name}.inventory.json",
        bounds.max_manifest_bytes,
    )
    manifest = _strict_dict(
        _read_json(manifest_path, f"{dataset.input_name}.inventory.json"),
        INVENTORY_FIELDS,
        "protected inventory manifest",
    )
    if manifest["schema"] != INVENTORY_SCHEMA:
        raise CorpusAuditError("protected inventory schema is unsupported")
    if manifest["inventoryFormat"] != INVENTORY_RECORD_SCHEMA:
        raise CorpusAuditError("protected inventory record format is unsupported")
    source_id = _string(manifest["sourceId"], "protected inventory sourceId")
    if SOURCE_ID.fullmatch(source_id) is None:
        raise CorpusAuditError("protected inventory sourceId is ambiguous")
    source_release = _string(
        manifest["sourceRelease"], "protected inventory sourceRelease"
    )
    if MUTABLE_VERSION.search(source_release):
        raise CorpusAuditError("protected inventory sourceRelease must be immutable")
    if manifest["ncbiTaxon"] != 4932 or manifest["stableIdNamespace"] != "SGD":
        raise CorpusAuditError("protected inventory must retain taxon 4932 and SGD identity")
    mapping_id = _curie(manifest["identityMappingId"], "identityMappingId")
    mapping_sha = _raw_sha(
        manifest["identityMappingSha256"], "identityMappingSha256"
    )
    files = manifest["files"]
    if (
        not isinstance(files, list)
        or not files
        or len(files) > bounds.max_inventory_files_per_source
    ):
        raise CorpusAuditError("protected inventory files are outside bounds")
    parsed_files: list[tuple[str, str, int]] = []
    for index, item in enumerate(files):
        item = _strict_dict(
            item, INVENTORY_FILE_FIELDS, f"protected inventory files[{index}]"
        )
        relative = _relative_path(item["path"], f"protected inventory files[{index}].path")
        if not relative.endswith(".jsonl"):
            raise CorpusAuditError("protected inventory records must use JSONL files")
        digest = _raw_sha(item["sha256"], "protected inventory file sha256")
        records = _positive_int(
            item["records"],
            "protected inventory file records",
            bounds.max_inventory_records_per_source,
        )
        parsed_files.append((relative, digest, records))
    paths = [item[0] for item in parsed_files]
    if paths != sorted(set(paths)):
        raise CorpusAuditError("protected inventory paths must be unique and sorted")
    _exact_file_set(dataset.root, {"inventory.json", *paths}, dataset.input_name)

    seen: dict[str, bool] = {}
    total_records = 0
    for relative, expected_sha, expected_records in parsed_files:
        path = _regular_file(
            dataset.root, relative, relative, bounds.max_file_bytes
        )
        if _sha256(path) != expected_sha:
            raise CorpusAuditError(f"protected inventory digest drift: {relative}")
        actual_records = 0
        try:
            with path.open("rb") as stream:
                while True:
                    raw = stream.readline(bounds.max_line_bytes + 1)
                    if raw == b"":
                        break
                    actual_records += 1
                    total_records += 1
                    if (
                        len(raw) > bounds.max_line_bytes
                        or total_records > bounds.max_inventory_records_per_source
                    ):
                        raise CorpusAuditError("protected inventory record bounds exceeded")
                    if not raw.strip():
                        raise CorpusAuditError("protected inventory contains a blank record")
                    try:
                        record = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise CorpusAuditError("protected inventory record is invalid JSON") from error
                    record = _strict_dict(
                        record, INVENTORY_RECORD_FIELDS, "protected inventory record"
                    )
                    if (
                        record["schema"] != INVENTORY_RECORD_SCHEMA
                        or record["ncbiTaxon"] != 4932
                    ):
                        raise CorpusAuditError("protected inventory record schema/taxon drift")
                    identifier = record["interventionId"]
                    if not isinstance(identifier, str) or SGD_CURIE.fullmatch(identifier) is None:
                        raise CorpusAuditError("protected inventory requires canonical SGD CURIEs")
                    passes = record["qcPassing"]
                    if type(passes) is not bool:
                        raise CorpusAuditError("protected inventory qcPassing must be boolean")
                    previous = seen.get(identifier)
                    if previous is not None and previous != passes:
                        raise CorpusAuditError("protected inventory contains a conflicting duplicate")
                    seen.setdefault(identifier, passes)
        except OSError as error:
            raise CorpusAuditError("could not read protected inventory") from error
        if actual_records != expected_records:
            raise CorpusAuditError("protected inventory record count drift")
    passing = frozenset(identifier for identifier, passes in seen.items() if passes)
    failed = frozenset(identifier for identifier, passes in seen.items() if not passes)
    return ProtectedInventory(
        dataset,
        source_id,
        source_release,
        mapping_id,
        mapping_sha,
        _sha256(manifest_path),
        total_records,
        total_records - len(seen),
        passing,
        failed,
    )


def _expected_coverage_source(
    source: ProtectedInventory, intersection: frozenset[str]
) -> dict[str, Any]:
    exclusions = [
        {"interventionId": identifier, "reason": "qc-failed"}
        for identifier in sorted(source.qc_failed)
    ]
    exclusions.extend(
        {
            "interventionId": identifier,
            "reason": "not-qc-passing-in-all-protected-sources",
        }
        for identifier in sorted(source.qc_passing - intersection)
    )
    exclusions.sort(key=lambda item: (item["interventionId"], item["reason"]))
    return {
        "sourceId": source.source_id,
        "sourceRelease": source.source_release,
        "identityMappingId": source.identity_mapping_id,
        "identityMappingSha256": source.identity_mapping_sha256,
        "manifestSha256": source.manifest_sha256,
        "records": source.records,
        "duplicateRecords": source.duplicate_records,
        "uniqueInterventions": len(source.qc_passing | source.qc_failed),
        "qcPassing": len(source.qc_passing),
        "qcFailed": len(source.qc_failed),
        "intersectionCoverage": len(intersection),
        "exclusions": exclusions,
    }


def _validate_coverage_source(
    value: object,
    roster_ids: frozenset[str],
    intersection_size: int,
    bounds: AuditBounds,
) -> tuple[dict[str, Any], dict[str, int]]:
    source = _strict_dict(value, SOURCE_COVERAGE_FIELDS, "coverage source")
    source_id = _string(source["sourceId"], "coverage sourceId")
    if SOURCE_ID.fullmatch(source_id) is None:
        raise CorpusAuditError("coverage sourceId must be an unambiguous CURIE-like identifier")
    source_release = _string(source["sourceRelease"], "coverage sourceRelease")
    if MUTABLE_VERSION.search(source_release):
        raise CorpusAuditError("coverage sourceRelease must be immutable")
    mapping_id = _curie(source["identityMappingId"], "identityMappingId")
    mapping_sha = _raw_sha(source["identityMappingSha256"], "identityMappingSha256")
    manifest_sha = _raw_sha(source["manifestSha256"], "source manifestSha256")
    records = _nonnegative_int(
        source["records"], "source.records", bounds.max_records_per_corpus
    )
    duplicates = _nonnegative_int(
        source["duplicateRecords"], "source.duplicateRecords", records
    )
    unique = _nonnegative_int(
        source["uniqueInterventions"], "source.uniqueInterventions", bounds.max_roster_records
    )
    passing = _nonnegative_int(source["qcPassing"], "source.qcPassing", unique)
    failed = _nonnegative_int(source["qcFailed"], "source.qcFailed", unique)
    coverage = _nonnegative_int(
        source["intersectionCoverage"], "source.intersectionCoverage", unique
    )
    if unique != passing + failed or records != unique + duplicates or coverage != intersection_size:
        raise CorpusAuditError("coverage source counts are internally inconsistent")
    exclusions = source["exclusions"]
    if not isinstance(exclusions, list) or len(exclusions) > bounds.max_coverage_exclusions:
        raise CorpusAuditError("coverage exclusions exceed their bound")
    parsed_exclusions: list[tuple[str, str]] = []
    for item in exclusions:
        item = _strict_dict(item, {"interventionId", "reason"}, "coverage exclusion")
        identifier = _string(item["interventionId"], "coverage exclusion interventionId")
        if SGD_CURIE.fullmatch(identifier) is None:
            raise CorpusAuditError("coverage exclusion must use a canonical SGD CURIE")
        reason = item["reason"]
        if reason not in {"qc-failed", "not-qc-passing-in-all-protected-sources"}:
            raise CorpusAuditError("coverage exclusion reason is unsupported")
        if identifier in roster_ids:
            raise CorpusAuditError("coverage exclusion cannot also occur in the held roster")
        parsed_exclusions.append((identifier, reason))
    if parsed_exclusions != sorted(set(parsed_exclusions)):
        raise CorpusAuditError("coverage exclusions must be unique and deterministically sorted")
    if len(parsed_exclusions) != unique - intersection_size:
        raise CorpusAuditError("coverage exclusion count does not explain non-intersection identities")
    counts = {
        "qcFailed": sum(reason == "qc-failed" for _, reason in parsed_exclusions),
        "notPassingAllProtectedSources": sum(
            reason == "not-qc-passing-in-all-protected-sources"
            for _, reason in parsed_exclusions
        ),
        "identicalDuplicatesCollapsed": duplicates,
    }
    compact = {
        "sourceId": source_id,
        "sourceRelease": source_release,
        "identityMappingId": mapping_id,
        "identityMappingSha256": mapping_sha,
        "manifestSha256": manifest_sha,
        "records": records,
        "duplicateRecords": duplicates,
        "uniqueInterventions": unique,
        "qcPassing": passing,
        "qcFailed": failed,
        "intersectionCoverage": coverage,
    }
    return compact, counts


def load_held_roster(
    dataset: DatasetInput,
    protected_inventories: tuple[ProtectedInventory, ...],
    bounds: AuditBounds,
) -> tuple[dict[str, Any], dict[str, str]]:
    if len(protected_inventories) < 2:
        raise CorpusAuditError("at least two protected source inventories are required")
    source_ids_from_inputs = [item.source_id for item in protected_inventories]
    if source_ids_from_inputs != sorted(set(source_ids_from_inputs)):
        raise CorpusAuditError("protected source inventories must be unique and source-sorted")
    input_mappings = {
        (item.identity_mapping_id, item.identity_mapping_sha256)
        for item in protected_inventories
    }
    if len(input_mappings) != 1:
        raise CorpusAuditError("protected source inventories must share one identity mapping")
    protected_intersection = frozenset.intersection(
        *(item.qc_passing for item in protected_inventories)
    )
    expected_files = {"coverage.json", "held-intervention-roster.tsv"}
    _exact_file_set(dataset.root, expected_files, "heldRoster")
    roster_path = _regular_file(
        dataset.root,
        "held-intervention-roster.tsv",
        "held roster",
        bounds.max_file_bytes,
    )
    coverage_path = _regular_file(
        dataset.root, "coverage.json", "held roster coverage", bounds.max_coverage_bytes
    )
    assignments: dict[str, str] = {}
    try:
        with roster_path.open("rb") as stream:
            line_number = 0
            while True:
                raw = stream.readline(bounds.max_line_bytes + 1)
                if raw == b"":
                    break
                line_number += 1
                if line_number > bounds.max_roster_records:
                    raise CorpusAuditError("held roster exceeds maxRosterRecords")
                if (
                    len(raw) > bounds.max_line_bytes
                    or not raw.endswith(b"\n")
                    or raw.endswith(b"\r\n")
                ):
                    raise CorpusAuditError(
                        "held roster lines must be bounded canonical LF records"
                    )
                try:
                    fields = raw[:-1].decode("ascii").split("\t")
                except UnicodeDecodeError as error:
                    raise CorpusAuditError("held roster must be ASCII") from error
                if len(fields) != 3:
                    raise CorpusAuditError("held roster records require id, role, and hash")
                identifier, role, digest = fields
                expected_role, expected_digest, _ = assign_intervention(identifier)
                if role != expected_role or digest != expected_digest:
                    raise CorpusAuditError(f"forged or drifted roster role/hash for {identifier}")
                if identifier in assignments:
                    raise CorpusAuditError("held roster identifiers must be unique")
                assignments[identifier] = role
    except OSError as error:
        raise CorpusAuditError("could not read held roster") from error
    if not assignments or list(assignments) != sorted(assignments):
        raise CorpusAuditError("held roster must be non-empty and sorted by SGD CURIE")
    if frozenset(assignments) != protected_intersection:
        raise CorpusAuditError(
            "held roster is not the exact QC-passing protected-source intersection"
        )

    coverage = _strict_dict(
        _read_json(coverage_path, "coverage.json"), COVERAGE_FIELDS, "coverage"
    )
    if coverage["schema"] != ROSTER_SCHEMA:
        raise CorpusAuditError("held roster coverage schema is unsupported")
    assignment_contract = _strict_dict(
        coverage["assignment"],
        {"domainHex", "digest", "bucketRule", "roles"},
        "assignment",
    )
    if (
        assignment_contract["domainHex"] != ASSIGNMENT_DOMAIN_HEX
        or assignment_contract["digest"] != "sha256"
        or assignment_contract["bucketRule"] != BUCKET_RULE
        or assignment_contract["roles"]
        != {
            "0-9": "molecular-final",
            "10-29": "molecular-validation",
            "30-99": "pretrain",
        }
    ):
        raise CorpusAuditError("held roster assignment contract has drifted")
    source_count = _positive_int(coverage["sourceCount"], "sourceCount", bounds.max_sources)
    if source_count < 2 or source_count != len(protected_inventories):
        raise CorpusAuditError("coverage must bind every protected inventory and at least two sources")
    minimum = _positive_int(
        coverage["minimumIntersectionSize"],
        "minimumIntersectionSize",
        bounds.max_roster_records,
    )
    intersection = _positive_int(
        coverage["intersectionSize"], "intersectionSize", bounds.max_roster_records
    )
    if intersection != len(assignments) or intersection < minimum:
        raise CorpusAuditError("held roster intersection count is inconsistent or undersized")
    if coverage["rosterPath"] != "held-intervention-roster.tsv":
        raise CorpusAuditError("coverage rosterPath has drifted")
    roster_sha = _sha256(roster_path)
    if coverage["rosterSha256"] != roster_sha:
        raise CorpusAuditError("held roster digest drift")
    role_counts = _strict_dict(
        coverage["roleCounts"],
        {"pretrain", "molecular-validation", "molecular-final"},
        "roleCounts",
    )
    actual_role_counts = {
        role: sum(value == role for value in assignments.values())
        for role in ("pretrain", "molecular-validation", "molecular-final")
    }
    if (
        actual_role_counts["molecular-validation"] < 1
        or actual_role_counts["molecular-final"] < 1
    ):
        raise CorpusAuditError("held roster requires non-empty validation and final roles")
    if role_counts != actual_role_counts:
        raise CorpusAuditError("held roster role counts have drifted")
    mapping = _strict_dict(
        coverage["identityMapping"], {"id", "sha256"}, "identityMapping"
    )
    mapping_id = _curie(mapping["id"], "identityMapping.id")
    mapping_sha = _raw_sha(mapping["sha256"], "identityMapping.sha256")
    if (mapping_id, mapping_sha) != next(iter(input_mappings)):
        raise CorpusAuditError("coverage identity mapping differs from protected inventories")
    sources = coverage["sources"]
    if not isinstance(sources, list) or len(sources) != source_count:
        raise CorpusAuditError("coverage sources do not match sourceCount")
    compact_sources: list[dict[str, Any]] = []
    aggregate_rejections = {
        "qcFailed": 0,
        "notPassingAllProtectedSources": 0,
        "identicalDuplicatesCollapsed": 0,
    }
    roster_ids = frozenset(assignments)
    expected_sources = [
        _expected_coverage_source(item, protected_intersection)
        for item in protected_inventories
    ]
    if sources != expected_sources:
        raise CorpusAuditError(
            "coverage sources do not exactly reproduce protected inventory contents"
        )
    for source, protected in zip(sources, protected_inventories, strict=True):
        compact, rejected = _validate_coverage_source(
            source, roster_ids, intersection, bounds
        )
        if (
            compact["identityMappingId"] != mapping_id
            or compact["identityMappingSha256"] != mapping_sha
        ):
            raise CorpusAuditError(
                "coverage sources do not share the declared identity mapping"
            )
        compact_sources.append(
            {
                "resource": protected.input.resource,
                "revision": protected.input.revision,
                "artifactManifestDigest": protected.input.manifest_digest,
                **compact,
            }
        )
        for key, value in rejected.items():
            aggregate_rejections[key] += value
    source_ids = [item["sourceId"] for item in compact_sources]
    if source_ids != sorted(set(source_ids)):
        raise CorpusAuditError("coverage sources must be unique and deterministically sorted")
    rejection_counts = _strict_dict(
        coverage["rejectionCounts"], set(aggregate_rejections), "rejectionCounts"
    )
    if rejection_counts != aggregate_rejections:
        raise CorpusAuditError("coverage rejection counts have drifted")

    validation_ids = sorted(
        identifier for identifier, role in assignments.items()
        if role == "molecular-validation"
    )
    final_ids = sorted(
        identifier for identifier, role in assignments.items() if role == "molecular-final"
    )
    validation = {_composite_key(4932, identifier) for identifier in validation_ids}
    final = {_composite_key(4932, identifier) for identifier in final_ids}
    union = validation | final
    identity = {
        "resource": dataset.resource,
        "revision": dataset.revision,
        "manifestDigest": dataset.manifest_digest,
        "rosterSha256": roster_sha,
        "coverageSha256": _sha256(coverage_path),
        "assignmentDomainHex": ASSIGNMENT_DOMAIN_HEX,
        "bucketRule": BUCKET_RULE,
        "identityMappingId": mapping_id,
        "identityMappingSha256": mapping_sha,
        "sourceInventories": compact_sources,
        "intersectionSize": intersection,
        "pretrainInterventionCount": actual_role_counts["pretrain"],
        "validationInterventionSetSha256": _composite_set_sha256(validation),
        "validationInterventionCount": len(validation),
        "finalInterventionSetSha256": _composite_set_sha256(final),
        "finalInterventionCount": len(final),
        "unionInterventionSetSha256": _composite_set_sha256(union),
        "unionInterventionCount": len(union),
    }
    return identity, assignments


def _protected_inventory_identity(source: ProtectedInventory) -> dict[str, Any]:
    return {
        "resource": source.input.resource,
        "revision": source.input.revision,
        "manifestDigest": source.input.manifest_digest,
        "sourceId": source.source_id,
        "sourceRelease": source.source_release,
        "identityMappingId": source.identity_mapping_id,
        "identityMappingSha256": source.identity_mapping_sha256,
        "manifestSha256": source.manifest_sha256,
        "records": source.records,
        "duplicateRecords": source.duplicate_records,
        "uniqueInterventions": len(source.qc_passing | source.qc_failed),
        "qcPassing": len(source.qc_passing),
        "qcFailed": len(source.qc_failed),
    }


def audit_training_corpus(
    pretrain_input: object,
    held_roster_input: object,
    protected_inventory_inputs: Mapping[str, object],
    custodian_attestation_input: object,
    bounds: AuditBounds | None = None,
    *,
    reward_enabled: object,
    recipient_factory_identity: object,
    challenge_nonce: object,
) -> dict[str, Any]:
    """Audit one optimizer corpus without accepting protected quantitative truth."""
    if reward_enabled is not False:
        raise CorpusAuditError(
            "training corpus-audit v1.5 requires rewardEnabled=false; molecular "
            "reward needs a new versioned contract"
        )
    if (
        not isinstance(protected_inventory_inputs, Mapping)
        or not 2 <= len(protected_inventory_inputs) <= 64
    ):
        raise CorpusAuditError(
            "between two and 64 protectedInventory* inputs are required"
        )
    protected_names = sorted(protected_inventory_inputs)
    if (
        any(
            INPUT_NAME.fullmatch(name) is None
            or not name.startswith("protectedInventory")
            or len(name) == len("protectedInventory")
            for name in protected_names
        )
    ):
        raise CorpusAuditError(
            "protected inventory input names must be sorted unique protectedInventory* names"
        )
    bounds = bounds or AuditBounds()

    # Authenticate the small handoff first. The production verifier has a
    # source-pinned trust anchor and offers no runtime override.
    from attestation import (
        assert_authorized_content,
        verify_custodian_authorization,
    )

    actual_safe_inputs = {
        "pretrain": pretrain_input,
        "heldRoster": held_roster_input,
        **protected_inventory_inputs,
    }
    authorization = verify_custodian_authorization(
        custodian_attestation_input,
        actual_safe_inputs,
        recipient_factory_identity=recipient_factory_identity,
        challenge_nonce=challenge_nonce,
    )

    pretrain_dataset = resolve_dataset_input(pretrain_input, "pretrain")
    held_dataset = resolve_dataset_input(held_roster_input, "heldRoster")
    protected_by_name = {
        name: load_protected_inventory(
            resolve_dataset_input(protected_inventory_inputs[name], name), bounds
        )
        for name in protected_names
    }
    protected_inventories = tuple(
        sorted(protected_by_name.values(), key=lambda item: item.source_id)
    )
    pretrain = load_corpus(pretrain_dataset, "pretrain", bounds)
    held_identity, assignments = load_held_roster(
        held_dataset, protected_inventories, bounds
    )
    held_lineage = pretrain.identity["inputs"]["heldInterventionRoster"]
    held_snapshot = held_lineage["datasetSnapshot"]
    if (
        held_snapshot["resource"] != held_dataset.resource
        or held_snapshot["revision"] != held_dataset.revision
        or held_snapshot["outerManifestDigest"] != held_dataset.manifest_digest
        or held_lineage["semanticSha256"] != held_identity["rosterSha256"]
    ):
        raise CorpusAuditError(
            "corpus heldInterventionRoster lineage does not bind the audited roster"
        )
    expected_held_files = [
        {
            "path": name,
            "sha256": _sha256(held_dataset.root / name),
            "bytes": (held_dataset.root / name).stat().st_size,
        }
        for name in sorted({"coverage.json", "held-intervention-roster.tsv"})
    ]
    if held_lineage["files"] != expected_held_files:
        raise CorpusAuditError(
            "corpus heldInterventionRoster files do not match the audited roster"
        )
    held_union = {
        _composite_key(4932, identifier)
        for identifier, role in assignments.items()
        if role in {"molecular-validation", "molecular-final"}
    }
    leaked = sorted(
        (pretrain.trajectory_interventions | pretrain.active_actions) & held_union
    )
    if leaked:
        raise CorpusAuditError(
            "held validation/final interventions occur in quantitative fitting "
            "trajectories; pretrain="
            + ", ".join(canonical_json_bytes(_composite_object(item)).decode() for item in leaked)
        )

    inventory_identities = {
        name: _protected_inventory_identity(protected_by_name[name])
        for name in protected_names
    }
    assert_authorized_content(
        authorization,
        pretrain_identity=pretrain.identity,
        held_roster_identity=held_identity,
        inventory_identities=inventory_identities,
    )

    return {
        "schema": AUDIT_SCHEMA,
        "factoryRole": "training",
        "rewardEnabled": False,
        "protectedTruthInputsPresent": False,
        "custodianSignatureVerified": True,
        "auditPassed": True,
        "strictInterventionIsolation": True,
        "leakageViolations": 0,
        "leakedTrajectoryInterventions": [],
        "benchmarkLabelRecords": 0,
        "omfPriorAdmissionRequired": True,
        "datasets": {"pretrain": pretrain.identity},
        "heldRoster": held_identity,
        "custodianBoundaryAttestation": authorization.report_identity,
    }


def write_training_audit_artifact(
    pretrain_input: object,
    held_roster_input: object,
    protected_inventory_inputs: Mapping[str, object],
    custodian_attestation_input: object,
    destination: str | Path,
    bounds: AuditBounds | None = None,
    *,
    reward_enabled: object,
    recipient_factory_identity: object,
    challenge_nonce: object,
) -> tuple[dict[str, Any], str]:
    audit = audit_training_corpus(
        pretrain_input,
        held_roster_input,
        protected_inventory_inputs,
        custodian_attestation_input,
        bounds,
        reward_enabled=reward_enabled,
        recipient_factory_identity=recipient_factory_identity,
        challenge_nonce=challenge_nonce,
    )
    destination = Path(destination).absolute()
    if destination.exists():
        raise CorpusAuditError("audit destination must not already exist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / destination.name
        staging.mkdir()
        (staging / "corpus-audit.json").write_bytes(
            canonical_json_bytes(audit, newline=True)
        )
        os.replace(staging, destination)
    _exact_file_set(
        destination, {"corpus-audit.json"}, "training corpus audit output"
    )
    return audit, _sha256(destination / "corpus-audit.json")
