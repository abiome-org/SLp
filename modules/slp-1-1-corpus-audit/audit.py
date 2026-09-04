"""Fail-closed v1.2 corpus, roster, provenance, and leakage audit.

Identity-bearing NPZ arrays are inspected with the standard library so a
trajectory manifest cannot conceal a record-level intervention.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import struct
import tempfile
from typing import Any, Iterator, Mapping
import zipfile


CORPUS_SCHEMA = "slp.corpus/v1.1"
AUDIT_SCHEMA = "slp.corpus-audit/v1.2"
ROSTER_SCHEMA = "slp.held-intervention-roster-report/v1"
INVENTORY_SCHEMA = "slp.intervention-identity-inventory/v1"
INVENTORY_RECORD_SCHEMA = "slp.intervention-identity-record/v1"
ASSIGNMENT_DOMAIN = b"slp-1.1-yeast-global-held-v1\x00"
ASSIGNMENT_DOMAIN_HEX = ASSIGNMENT_DOMAIN.hex()
BUCKET_RULE = "int(first-16-lowercase-hex,16) mod 100"
EXPECTED_ROLES = {
    "pretrain": "pretrain",
    "molecularValidation": "molecular-validation",
    "molecularFinal": "molecular-final",
}
CORPUS_FIELDS = {
    "schema", "datasetId", "version", "role", "labelClass",
    "benchmarkLabelsPresent", "rights", "modalities", "sources", "sampling",
    "species", "featurePack", "entityTypes", "contextTypes", "actionTypes",
    "covariates", "readoutTypes", "entityDictionary", "queryDictionary",
    "queryPanels", "trajectoryGenes", "normalization", "bounds", "shards",
}
REFERENCE_FIELDS = {"path", "sha256", "count"}
SHARD_FIELDS = {"path", "sha256", "records", "targetValues"}
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
    "entity_id", "entity_type", "entity_species_taxon", "entity_feature_value",
    "entity_feature_present",
}
QUERY_MEMBERS = {"query_id", "query_entity_index", "query_readout_index"}
PANEL_MEMBERS = {"panel_id", "panel_indptr", "panel_query_index"}
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
    max_trajectory_genes: int = 2_000_000
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
            "maxTrajectoryGenes": (self.max_trajectory_genes, 1, 20_000_000),
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
    count: int


@dataclass(frozen=True)
class Corpus:
    input: DatasetInput
    identity: dict[str, Any]
    trajectory_genes: frozenset[str]
    active_actions: frozenset[str]


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
    with archive.open(info) as stream:
        actual = _parse_npy_header(stream, info.file_size, label, bounds)
        if actual != expected:
            raise CorpusAuditError(f"{label} header changed during inspection")
        match = re.fullmatch(r"([<|>])([Uuifb])(\d+)", actual.descr)
        assert match is not None
        endian, kind, width_text = match.groups()
        width = int(width_text)
        remaining = actual.count
        while remaining:
            batch = min(remaining, 8_192)
            payload = stream.read(batch * actual.item_size)
            if len(payload) != batch * actual.item_size:
                raise CorpusAuditError(f"{label} has a truncated payload")
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
            remaining -= batch
        if stream.read(1):
            raise CorpusAuditError(f"{label} contains trailing NPY bytes")


def _read_entity_dictionary(
    path: Path, count: int, declared_species: frozenset[int], bounds: AuditBounds
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = _member_map(
                archive, "entityDictionary", bounds, ENTITY_MEMBERS
            )
            id_spec = _array_spec(archive, members, "entity_id", "entityDictionary", bounds)
            taxon_spec = _array_spec(
                archive, members, "entity_species_taxon", "entityDictionary", bounds
            )
            if id_spec.shape != (count,) or taxon_spec.shape != (count,):
                raise CorpusAuditError("entity dictionary array shapes must match declared count")
            if not id_spec.descr.startswith("<U") or taxon_spec.descr[1:2] not in {"i", "u"}:
                raise CorpusAuditError("entity dictionary identity arrays use invalid dtypes")
            identifiers = tuple(
                _iter_array_values(
                    archive, members["entity_id"], id_spec,
                    "entityDictionary.entity_id", bounds,
                )
            )
            taxa = tuple(
                _iter_array_values(
                    archive, members["entity_species_taxon"], taxon_spec,
                    "entityDictionary.entity_species_taxon", bounds,
                )
            )
    except zipfile.BadZipFile as error:
        raise CorpusAuditError("entityDictionary must be a valid NPZ archive") from error
    parsed_ids = tuple(_curie(item, "entity_id") for item in identifiers)
    if len(parsed_ids) != len(set(parsed_ids)):
        raise CorpusAuditError("entity_id values must be unique")
    if any(type(item) is not int or item < 0 for item in taxa):
        raise CorpusAuditError("entity_species_taxon must contain non-negative integers")
    if any(item != 0 and item not in declared_species for item in taxa):
        raise CorpusAuditError("entity species taxon must be declared or zero")
    for identifier, taxon in zip(parsed_ids, taxa, strict=True):
        if SGD_CURIE.fullmatch(identifier) is not None and taxon != 4932:
            raise CorpusAuditError("canonical SGD entities must retain NCBI taxon 4932")
    return parsed_ids, tuple(int(item) for item in taxa)


def _validate_npz_member_set(
    path: Path, label: str, expected_keys: set[str], bounds: AuditBounds
) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            _member_map(archive, label, bounds, expected_keys)
    except zipfile.BadZipFile as error:
        raise CorpusAuditError(f"{label} must be a valid NPZ archive") from error


def _audit_shard(
    path: Path,
    records: int,
    target_values: int,
    max_action_tokens: int,
    max_targets_per_record: int,
    source_count: int,
    query_count: int,
    declared_species: frozenset[int],
    entity_ids: tuple[str, ...],
    entity_taxa: tuple[int, ...],
    record_db: sqlite3.Connection,
    bounds: AuditBounds,
) -> tuple[set[int], set[int], set[str], set[str]]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = _member_map(
                archive, path.name, bounds, QUANTITATIVE_SHARD_MEMBERS
            )
            keys = (
                "record_id", "source_index", "species_taxon", "action_entity_index",
                "action_mask", "target_indptr", "target_query_index", "target_value",
            )
            specs = {
                key: _array_spec(archive, members, key, path.name, bounds) for key in keys
            }
            if specs["record_id"].shape != (records,) or not specs["record_id"].descr.startswith("<U"):
                raise CorpusAuditError(f"{path.name} record_id does not match declared records")
            for key in ("source_index", "species_taxon"):
                if specs[key].shape != (records,) or specs[key].descr != "<i8":
                    raise CorpusAuditError(f"{path.name} {key} must be a record-aligned integer array")
            action_shape = specs["action_mask"].shape
            if (
                len(action_shape) != 2
                or action_shape[0] != records
                or not 1 <= action_shape[1] <= max_action_tokens
                or specs["action_entity_index"].shape != action_shape
                or specs["action_mask"].descr != "|b1"
                or specs["action_entity_index"].descr != "<i8"
            ):
                raise CorpusAuditError(f"{path.name} action identity arrays are misaligned")
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

            indices = _iter_array_values(
                archive, members["action_entity_index"], specs["action_entity_index"],
                f"{path.name}.action_entity_index", bounds,
            )
            masks = _iter_array_values(
                archive, members["action_mask"], specs["action_mask"],
                f"{path.name}.action_mask", bounds,
            )
            row_taxa = _iter_array_values(
                archive, members["species_taxon"], specs["species_taxon"],
                f"{path.name}.species_taxon", bounds,
            )
            genes: set[str] = set()
            active_actions: set[str] = set()
            for row in range(records):
                record_taxon = int(next(row_taxa))
                for _ in range(action_shape[1]):
                    index = int(next(indices))
                    active = bool(next(masks))
                    if not active:
                        if index != -1:
                            raise CorpusAuditError(
                                f"{path.name} padded action_entity_index must be -1"
                            )
                        continue
                    if index < 0 or index >= len(entity_ids):
                        raise CorpusAuditError(f"{path.name} action_entity_index is out of range")
                    entity_taxon = entity_taxa[index]
                    if entity_taxon not in {0, record_taxon}:
                        raise CorpusAuditError(
                            f"{path.name} active action entity taxon does not match its record"
                        )
                    identifier = entity_ids[index]
                    active_actions.add(identifier)
                    if entity_taxon != 0:
                        genes.add(identifier)

            pointer_values = _iter_array_values(
                archive, members["target_indptr"], specs["target_indptr"],
                f"{path.name}.target_indptr", bounds,
            )
            previous = int(next(pointer_values))
            if previous != 0:
                raise CorpusAuditError(f"{path.name} targetValues does not match target_indptr")
            for _ in range(records):
                current = int(next(pointer_values))
                if current < previous or current - previous > max_targets_per_record:
                    raise CorpusAuditError(
                        f"{path.name} target_indptr violates maxTargetsPerRecord"
                    )
                previous = current
            if previous != target_values:
                raise CorpusAuditError(f"{path.name} targetValues does not match target_indptr")
            for item in _iter_array_values(
                archive, members["target_query_index"], specs["target_query_index"],
                f"{path.name}.target_query_index", bounds,
            ):
                value = int(item)
                if value < 0 or value >= query_count:
                    raise CorpusAuditError(f"{path.name} target query index is out of range")
            for item in _iter_array_values(
                archive, members["target_value"], specs["target_value"],
                f"{path.name}.target_value", bounds,
            ):
                if not math.isfinite(float(item)):
                    raise CorpusAuditError(f"{path.name} target values must be finite")
            return source_values, species_values, genes, active_actions
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

    feature_pack = _strict_dict(
        manifest["featurePack"],
        {"revision", "sha256", "entityFeatureDim", "speciesFeatureDim"},
        "featurePack",
    )
    _curie(feature_pack["revision"], "featurePack.revision")
    _raw_sha(feature_pack["sha256"], "featurePack.sha256")
    _positive_int(feature_pack["entityFeatureDim"], "entityFeatureDim", 1_000_000)
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
    return (
        dataset_id,
        version,
        tuple(sources),
        tuple(taxa),
        int(corpus_bounds["maxRecordsPerShard"]),
        int(corpus_bounds["maxActionTokens"]),
        int(corpus_bounds["maxTargetsPerRecord"]),
    )


def load_corpus(dataset: DatasetInput, expected_role: str, bounds: AuditBounds) -> Corpus:
    manifest_path = _regular_file(
        dataset.root,
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
        manifest["trajectoryGenes"], "trajectoryGenes", bounds, allow_zero=True
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
            records,
        )
        shards.append(
            {
                "path": reference.path,
                "sha256": reference.sha256,
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

    references = [entity_ref, query_ref, panel_ref, trajectory_ref, *shard_refs]
    paths = [item.path for item in references]
    if len(paths) != len(set(paths)):
        raise CorpusAuditError("internal corpus file references must be unique")
    expected_files = {"corpus.json", *paths}
    if len(expected_files) > bounds.max_files_per_corpus:
        raise CorpusAuditError("corpus file count exceeds maxFilesPerCorpus")
    _exact_file_set(dataset.root, expected_files, dataset.input_name)
    total_bytes = manifest_path.stat().st_size
    materialized: dict[str, Path] = {}
    for reference in references:
        path = _regular_file(
            dataset.root, reference.path, reference.path, bounds.max_file_bytes
        )
        total_bytes += path.stat().st_size
        if total_bytes > bounds.max_total_bytes_per_corpus:
            raise CorpusAuditError("corpus exceeds maxTotalBytesPerCorpus")
        if _sha256(path) != reference.sha256:
            raise CorpusAuditError(f"internal content digest mismatch: {reference.path}")
        materialized[reference.path] = path

    gene_path = materialized[trajectory_ref.path]
    genes: set[str] = set()
    gene_set_digest = hashlib.sha256(b"[")
    previous_gene: str | None = None
    try:
        with gene_path.open("rb") as stream:
            line_number = 0
            while True:
                raw = stream.readline(bounds.max_line_bytes + 1)
                if raw == b"":
                    break
                line_number += 1
                if line_number > bounds.max_trajectory_genes:
                    raise CorpusAuditError("trajectoryGenes exceeds maxTrajectoryGenes")
                if len(raw) > bounds.max_line_bytes:
                    raise CorpusAuditError(
                        f"trajectoryGenes line {line_number} exceeds maxLineBytes"
                    )
                if not raw.endswith(b"\n") or raw.endswith(b"\r\n"):
                    raise CorpusAuditError(
                        "trajectoryGenes must use canonical LF-terminated lines"
                    )
                try:
                    gene = _curie(raw[:-1].decode("utf-8"), "trajectory gene")
                except UnicodeDecodeError as error:
                    raise CorpusAuditError("trajectoryGenes must be UTF-8") from error
                if previous_gene is not None and gene <= previous_gene:
                    raise CorpusAuditError(
                        "trajectoryGenes must be unique and deterministically sorted"
                    )
                if genes:
                    gene_set_digest.update(b",")
                gene_set_digest.update(
                    json.dumps(gene, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
                )
                genes.add(gene)
                previous_gene = gene
    except OSError as error:
        raise CorpusAuditError("could not read trajectoryGenes") from error
    gene_set_digest.update(b"]")
    if len(genes) != trajectory_ref.count:
        raise CorpusAuditError("trajectoryGenes count drift")

    entity_ids, entity_taxa = _read_entity_dictionary(
        materialized[entity_ref.path], entity_ref.count, frozenset(taxa), bounds
    )
    _validate_npz_member_set(
        materialized[query_ref.path], "queryDictionary", QUERY_MEMBERS, bounds
    )
    _validate_npz_member_set(
        materialized[panel_ref.path], "queryPanels", PANEL_MEMBERS, bounds
    )
    if not genes.issubset(set(entity_ids)):
        raise CorpusAuditError("trajectoryGenes contains an identifier outside entityDictionary")
    seen_sources: set[int] = set()
    seen_species: set[int] = set()
    record_genes: set[str] = set()
    active_actions: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="slp-corpus-audit-records-") as temporary:
        record_db = sqlite3.connect(Path(temporary) / "record-ids.sqlite3")
        try:
            record_db.execute("CREATE TABLE record_ids (identifier TEXT PRIMARY KEY) WITHOUT ROWID")
            for shard, reference in zip(shards, shard_refs, strict=True):
                source_values, species_values, action_genes, shard_actions = _audit_shard(
                    materialized[reference.path],
                    shard["records"],
                    shard["targetValues"],
                    max_action_tokens,
                    max_targets_per_record,
                    len(sources),
                    query_ref.count,
                    frozenset(taxa),
                    entity_ids,
                    entity_taxa,
                    record_db,
                    bounds,
                )
                seen_sources.update(source_values)
                seen_species.update(species_values)
                record_genes.update(action_genes)
                active_actions.update(shard_actions)
        finally:
            record_db.close()
    if seen_sources != set(range(len(sources))):
        raise CorpusAuditError("source inventory does not exactly match record-level source indices")
    if seen_species != set(taxa):
        raise CorpusAuditError("species inventory does not exactly match record-level taxa")
    if record_genes != genes:
        raise CorpusAuditError(
            "trajectoryGenes does not exactly match record-level species actions; "
            f"missing={sorted(record_genes - genes)}, "
            f"extra={sorted(genes - record_genes)}"
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
        "trajectoryGenesSha256": trajectory_ref.sha256,
        "trajectoryGeneSetSha256": gene_set_digest.hexdigest(),
        "trajectoryGeneCount": len(genes),
        "records": total_records,
        "targetValues": total_targets,
        "modalities": list(manifest["modalities"]),
        "sourceIds": list(sources),
        "speciesTaxa": list(taxa),
    }
    return Corpus(dataset, identity, frozenset(genes), frozenset(active_actions))


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

    validation = sorted(
        identifier for identifier, role in assignments.items()
        if role == "molecular-validation"
    )
    final = sorted(
        identifier for identifier, role in assignments.items() if role == "molecular-final"
    )
    union = sorted({*validation, *final})
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
        "pretrainGeneCount": actual_role_counts["pretrain"],
        "validationGeneSetSha256": canonical_sha256(validation),
        "validationGeneCount": len(validation),
        "finalGeneSetSha256": canonical_sha256(final),
        "finalGeneCount": len(final),
        "unionGeneSetSha256": canonical_sha256(union),
        "unionGeneCount": len(union),
    }
    return identity, assignments


def audit_corpora(
    corpus_inputs: Mapping[str, object],
    held_roster_input: object,
    protected_inventory_inputs: Mapping[str, object],
    bounds: AuditBounds | None = None,
    *,
    reward_enabled: object,
) -> dict[str, Any]:
    if reward_enabled is not False:
        raise CorpusAuditError(
            "corpus-audit v1.2 requires rewardEnabled=false; molecular reward "
            "needs a new versioned contract"
        )
    bounds = bounds or AuditBounds()
    if set(corpus_inputs) != set(EXPECTED_ROLES):
        raise CorpusAuditError(
            f"corpus inputs must be exactly {sorted(EXPECTED_ROLES)}"
        )
    datasets = {
        name: resolve_dataset_input(corpus_inputs[name], name) for name in EXPECTED_ROLES
    }
    held_dataset = resolve_dataset_input(held_roster_input, "heldRoster")
    if not isinstance(protected_inventory_inputs, Mapping) or len(protected_inventory_inputs) < 2:
        raise CorpusAuditError("at least two protected inventory inputs are required")
    protected_inventories = tuple(
        sorted(
            (
                load_protected_inventory(
                    resolve_dataset_input(value, name), bounds
                )
                for name, value in protected_inventory_inputs.items()
            ),
            key=lambda item: item.source_id,
        )
    )
    corpora = {
        name: load_corpus(datasets[name], role, bounds)
        for name, role in EXPECTED_ROLES.items()
    }
    held_identity, assignments = load_held_roster(
        held_dataset, protected_inventories, bounds
    )

    validation = corpora["molecularValidation"].trajectory_genes
    final = corpora["molecularFinal"].trajectory_genes
    if not validation or not final:
        raise CorpusAuditError("molecularValidation and molecularFinal require trajectory genes")
    overlap = sorted(validation & final)
    if overlap:
        raise CorpusAuditError(
            "molecularValidation and molecularFinal trajectory interventions overlap: "
            + ", ".join(overlap)
        )
    validation_actions = validation | (
        corpora["molecularValidation"].active_actions & set(assignments)
    )
    final_actions = final | (corpora["molecularFinal"].active_actions & set(assignments))
    wrong_validation = sorted(
        gene for gene in validation_actions
        if assignments.get(gene) != "molecular-validation"
    )
    wrong_final = sorted(
        gene for gene in final_actions if assignments.get(gene) != "molecular-final"
    )
    if wrong_validation or wrong_final:
        raise CorpusAuditError(
            "validation/final trajectory intervention has a missing or forged roster role; "
            f"validation={wrong_validation}, final={wrong_final}"
        )
    held_union = {
        gene for gene, role in assignments.items()
        if role in {"molecular-validation", "molecular-final"}
    }
    leaked = sorted(
        (
            corpora["pretrain"].trajectory_genes
            | corpora["pretrain"].active_actions
        )
        & held_union
    )
    if leaked:
        pretrain_leaks = sorted(
            (corpora["pretrain"].trajectory_genes | corpora["pretrain"].active_actions)
            & held_union
        )
        raise CorpusAuditError(
            "held validation/final interventions occur in quantitative fitting trajectories; "
            f"pretrain={pretrain_leaks}"
        )

    return {
        "schema": AUDIT_SCHEMA,
        "rewardEnabled": False,
        "auditPassed": True,
        "strictInterventionIsolation": True,
        "leakageViolations": 0,
        "leakedTrajectoryGenes": [],
        "benchmarkLabelRecords": 0,
        "omfPriorAdmissionRequired": True,
        "datasets": {name: corpora[name].identity for name in EXPECTED_ROLES},
        "heldRoster": held_identity,
    }


def write_audit_artifact(
    corpus_inputs: Mapping[str, object],
    held_roster_input: object,
    protected_inventory_inputs: Mapping[str, object],
    destination: str | Path,
    bounds: AuditBounds | None = None,
    *,
    reward_enabled: object,
) -> tuple[dict[str, Any], str]:
    audit = audit_corpora(
        corpus_inputs,
        held_roster_input,
        protected_inventory_inputs,
        bounds,
        reward_enabled=reward_enabled,
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
    _exact_file_set(destination, {"corpus-audit.json"}, "corpus audit output")
    return audit, _sha256(destination / "corpus-audit.json")
