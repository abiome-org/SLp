"""Outcome-blind genotype identity extraction from one pinned atlas summary."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Iterable, Iterator, Mapping


NCBI_TAXON = 4932
SOURCE_ID = "zenodo:14062629"
SOURCE_RELEASE = "10.5281/zenodo.14062629"
RAW_DATASET_RESOURCE = (
    "omf://abiome/slp/datasetsnapshot/slp-1-1-atlas-genotype-summary-raw-v1"
    "@sha256:c7cad889b43f293fe5b59e3fd2486f5dabf0b0b362964968eb4858f6917268ef"
)
RAW_DATASET_MANIFEST_DIGEST = (
    "sha256:97df177ff586d3409d6348926562c5ba4c4943ab4789b1823d059bf6c708fa31"
)
IDENTITY_MAPPING_ID = "slp-sgd-map:2026-08-28-object-set-v1"
IDENTITY_MAPPING_SHA256 = (
    "6fd789df6099b78a8842baa8f1d20ab0a3fe77f27ce512ee783444eb2627ef2a"
)
MAPPING_MANIFEST_SHA256 = (
    "570557ab1201913a18de9790f8adc5ee2e3cb56c6bb0e8d588fe43660c0214e1"
)
RDATA_VERSION = "1.1.0"
RAW_FILE_NAME = "ptb_summary.Rdata"

INTERVENTION_INVENTORY_SCHEMA = "slp.intervention-identity-inventory/v1"
INTERVENTION_RECORD_SCHEMA = "slp.intervention-identity-record/v1"
EVIDENCE_MANIFEST_SCHEMA = "slp.atlas-genotype-evidence-inventory/v1"
EVIDENCE_RECORD_SCHEMA = "slp.atlas-genotype-identity-evidence/v1"
QUARANTINE_RECORD_SCHEMA = "slp.atlas-genotype-identity-quarantine/v1"
AUDIT_SCHEMA = "slp.atlas-genotype-identity-audit/v1"

SGD_CURIE = re.compile(r"^SGD:S[0-9]{9}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MD5 = re.compile(r"^[0-9a-f]{32}$")
RESOURCE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
INPUT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")

FRAME_COLUMNS = (
    "assignment_consensus2",
    "cell_number",
    "avg_lvscoreFU2",
    "var_lvscoreFU2",
    "std_lvscoreFU2",
    "avg_lvscore_scaledFU2",
    "var_lvscore_scaledFU2",
    "sd_lvscore_scaledFU2",
    "Stucked",
)
PHENOTYPE_COLUMNS = FRAME_COLUMNS[2:]
CONDITIONS = ("control", "nacl")
MAPPING_SCHEMAS = {
    "current": "slp.sgd-current-orf/v1",
    "retired": "slp.sgd-retired-quarantine/v1",
    "manifest": "slp.sgd-stable-id-mapping/v1",
}


class AtlasInventoryError(ValueError):
    """Raised when the identity-only atlas boundary cannot be proven."""


@dataclass(frozen=True)
class FileSpec:
    name: str
    bytes: int
    sha256: str
    md5: str

    def __post_init__(self) -> None:
        if Path(self.name).name != self.name or not self.name:
            raise AtlasInventoryError("raw file name must be one exact basename")
        if not isinstance(self.bytes, int) or isinstance(self.bytes, bool) or self.bytes < 1:
            raise AtlasInventoryError("raw file byte count must be positive")
        if not isinstance(self.sha256, str) or SHA256.fullmatch(self.sha256) is None:
            raise AtlasInventoryError("raw file SHA-256 must be lowercase hexadecimal")
        if not isinstance(self.md5, str) or MD5.fullmatch(self.md5) is None:
            raise AtlasInventoryError("raw upstream MD5 must be lowercase hexadecimal")


PINNED_RAW_FILE = FileSpec(
    RAW_FILE_NAME,
    345_032,
    "01c2d54ac838179be29694ed300cb17edac47dd4db23a4018407546e0651b165",
    "5b04718fa4b2025c4ec2a464ffed6ab1",
)

MAPPING_ARTIFACT_DIGESTS = {
    "sgdCurrentOrfs": "sha256:e67f0e8773feae108ecdb687139885e01ca972ff4aec95cd1358b33db1ea1192",
    "sgdExternalRelations": "sha256:75e0fef99bbae3bb4e4dc3e2f24cfd0ab62919c0e6e3e321e8d82f3bd557f4da",
    "sgdRetiredQuarantine": "sha256:07ea82f877224496c24effc2aa2a2b684c01b85017e616a70f003a5363f6925f",
    "sgdMappingManifest": "sha256:c74ea81ce604357b998e5f09130dff85bf8a7a26504b9b2426f8038608c52d9c",
}

EXPECTED_COUNTS = {
    "controlRows": 3_207,
    "controlNonWildType": 3_206,
    "naclRows": 3_204,
    "naclNonWildType": 3_203,
    "allAssignmentIntersectionIncludingWildType": 3_152,
    "allAssignmentUnionIncludingWildType": 3_259,
    "candidateNonWildTypeIntersection": 3_151,
    "candidateNonWildTypeUnion": 3_258,
    "exactCurrentCandidateAssignments": 2_941,
    "uniqueCurrentInterventions": 2_941,
    "retiredOrMergedCandidateAssignments": 19,
    "unmatchedCandidateAssignments": 191,
    "ambiguousCurrentCandidateAssignments": 0,
}


@dataclass(frozen=True)
class Bounds:
    max_frame_rows: int = 4_000
    max_assignment_bytes: int = 256
    max_mapping_records: int = 300_000
    max_mapping_artifact_bytes: int = 134_217_728
    max_line_bytes: int = 2_097_152
    max_evidence_records: int = 4_000

    def __post_init__(self) -> None:
        for name, value, minimum, maximum in (
            ("maxFrameRows", self.max_frame_rows, 1, 100_000),
            ("maxAssignmentBytes", self.max_assignment_bytes, 4, 4_096),
            ("maxMappingRecords", self.max_mapping_records, 1, 1_000_000),
            ("maxMappingArtifactBytes", self.max_mapping_artifact_bytes, 1, 536_870_912),
            ("maxLineBytes", self.max_line_bytes, 128, 16_777_216),
            ("maxEvidenceRecords", self.max_evidence_records, 1, 100_000),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
                raise AtlasInventoryError(f"{name} must be an integer in [{minimum}, {maximum}]")


@dataclass(frozen=True)
class PinnedDataset:
    path: Path
    resource: str
    revision: str
    manifest_digest: str


@dataclass(frozen=True)
class LiteralArtifact:
    path: Path
    artifact_manifest_digest: str


@dataclass(frozen=True)
class SourceProvenance:
    resource: str
    revision: str
    manifest_digest: str
    mapping_artifacts: Mapping[str, str]


@dataclass(frozen=True)
class ConditionIdentity:
    assignments: frozenset[str]
    rows: int
    non_wild_type: int
    minimum_cell_number: int
    maximum_cell_number: int


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise AtlasInventoryError(f"{label} must be a non-empty trimmed string")
    return value


def _digest(value: object, label: str, *, prefixed: bool) -> str:
    if not isinstance(value, str):
        raise AtlasInventoryError(f"{label} must be a SHA-256")
    body = value.removeprefix("sha256:") if prefixed else value
    if prefixed and not value.startswith("sha256:"):
        raise AtlasInventoryError(f"{label} must use the sha256: prefix")
    if SHA256.fullmatch(body) is None:
        raise AtlasInventoryError(f"{label} must be a lowercase SHA-256")
    return value


def _hash_file(path: Path) -> tuple[str, str]:
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(chunk)
                md5.update(chunk)
    except OSError as error:
        raise AtlasInventoryError(f"could not read {path.name}") from error
    return sha.hexdigest(), md5.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise AtlasInventoryError(f"could not read {path.name}") from error
    return digest.hexdigest()


def _resolved_path(value: object, label: str, *, directory: bool) -> Path:
    requested = value if isinstance(value, Path) else Path(_nonempty(value, label))
    cursor = requested.absolute()
    for _ in range(3):
        if cursor.is_symlink():
            raise AtlasInventoryError(f"{label} materialization must not contain a symlink")
        cursor = cursor.parent
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise AtlasInventoryError(f"{label} does not exist") from error
    if directory != resolved.is_dir() or (not directory and not resolved.is_file()):
        kind = "directory" if directory else "regular file"
        raise AtlasInventoryError(f"{label} must be a {kind}")
    return resolved


def _resource_name(value: object, label: str) -> tuple[str, str]:
    resource = _nonempty(value, label)
    if not resource.startswith("omf://"):
        raise AtlasInventoryError(f"{label} must be an OMF DatasetSnapshot URI")
    identity, separator, revision = resource.removeprefix("omf://").rpartition("@")
    if not separator:
        raise AtlasInventoryError(f"{label} must carry an exact revision")
    _digest(revision, f"{label} revision", prefixed=True)
    parts = identity.split("/")
    if (
        len(parts) < 3
        or parts[-2] != "datasetsnapshot"
        or RESOURCE_NAME.fullmatch(parts[-1]) is None
        or any(not item or item in {".", ".."} or any(char.isspace() for char in item) for item in parts)
    ):
        raise AtlasInventoryError(f"{label} must identify a DatasetSnapshot")
    return parts[-1], revision


def resolve_pinned_raw_dataset(value: object, input_name: str = "rawAtlasSummary") -> PinnedDataset:
    if INPUT_NAME.fullmatch(input_name) is None or not isinstance(value, dict):
        raise AtlasInventoryError(f"{input_name} must be a materialized DatasetSnapshot")
    if set(value) != {"resource", "mode", "path", "manifestDigest"}:
        raise AtlasInventoryError(f"{input_name} has a spoofed DatasetSnapshot shape")
    resource_name, revision = _resource_name(value["resource"], f"{input_name}.resource")
    if value["resource"] != RAW_DATASET_RESOURCE:
        raise AtlasInventoryError(f"{input_name}.resource is not the admitted raw atlas revision")
    if value["mode"] != "copy":
        raise AtlasInventoryError(f"{input_name} must be copied, not mutable")
    manifest_digest = _digest(value["manifestDigest"], f"{input_name}.manifestDigest", prefixed=True)
    if manifest_digest != RAW_DATASET_MANIFEST_DIGEST:
        raise AtlasInventoryError(f"{input_name}.manifestDigest is not the admitted outer manifest")
    root = _resolved_path(value["path"], f"{input_name}.path", directory=True)
    if root.name != resource_name or root.parent.name != input_name or root.parent.parent.name != "inputs":
        raise AtlasInventoryError(f"{input_name}.path is inconsistent with OMF materialization")
    return PinnedDataset(root, str(value["resource"]), revision, manifest_digest)


def resolve_literal_artifact(
    value: object,
    input_name: str,
    expected_manifest_digest: str,
) -> LiteralArtifact:
    if INPUT_NAME.fullmatch(input_name) is None or not isinstance(value, dict):
        raise AtlasInventoryError(f"{input_name} must be a literal OMF artifact")
    if set(value) != {"resource", "kind", "artifacts", "paths", "path"}:
        raise AtlasInventoryError(f"{input_name} has a spoofed artifact shape")
    if value["kind"] != "artifact":
        raise AtlasInventoryError(f"{input_name} kind must be artifact")
    artifacts = value["artifacts"]
    paths = value["paths"]
    if not isinstance(artifacts, dict) or set(artifacts) != {"payload"}:
        raise AtlasInventoryError(f"{input_name}.artifacts must contain only payload")
    if not isinstance(paths, dict) or set(paths) != {"payload"}:
        raise AtlasInventoryError(f"{input_name}.paths must contain only payload")
    digest = _digest(artifacts["payload"], f"{input_name} artifact", prefixed=True)
    if digest != expected_manifest_digest:
        raise AtlasInventoryError(f"{input_name} does not match its pinned artifact")
    if value["resource"] != f"artifact:{digest}":
        raise AtlasInventoryError(f"{input_name}.resource does not match its payload")
    if paths["payload"] != value["path"]:
        raise AtlasInventoryError(f"{input_name}.path is inconsistent with paths.payload")
    path = _resolved_path(value["path"], f"{input_name}.path", directory=False)
    if (
        path.name != "payload"
        or path.parent.name != "payload"
        or path.parent.parent.name != input_name
        or path.parent.parent.parent.name != "inputs"
    ):
        raise AtlasInventoryError(f"{input_name}.path is inconsistent with OMF materialization")
    return LiteralArtifact(path, digest)


def verify_raw_snapshot(root_value: str | Path, spec: FileSpec = PINNED_RAW_FILE) -> Path:
    root = _resolved_path(root_value, "raw atlas summary snapshot", directory=True)
    children = list(root.iterdir())
    if len(children) != 1 or children[0].name != spec.name:
        raise AtlasInventoryError("raw atlas snapshot must contain exactly ptb_summary.Rdata")
    path = children[0]
    if path.is_symlink() or not path.is_file():
        raise AtlasInventoryError("ptb_summary.Rdata must be one regular non-symlink file")
    if path.stat().st_size != spec.bytes:
        raise AtlasInventoryError("ptb_summary.Rdata byte-count drift")
    sha, upstream_md5 = _hash_file(path)
    if sha != spec.sha256:
        raise AtlasInventoryError("ptb_summary.Rdata SHA-256 drift")
    if upstream_md5 != spec.md5:
        raise AtlasInventoryError("ptb_summary.Rdata upstream MD5 drift")
    return path


def _jsonl(path: Path, max_line_bytes: int) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        with path.open("rb") as stream:
            line_number = 0
            while True:
                raw = stream.readline(max_line_bytes + 1)
                if not raw:
                    break
                line_number += 1
                if len(raw) > max_line_bytes:
                    raise AtlasInventoryError(f"{path.name}:{line_number} exceeds maxLineBytes")
                if not raw.endswith(b"\n") or raw in {b"\n", b"\r\n"}:
                    raise AtlasInventoryError(f"{path.name}:{line_number} is not canonical JSONL")
                try:
                    record = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise AtlasInventoryError(f"{path.name}:{line_number} is invalid JSON") from error
                if not isinstance(record, dict):
                    raise AtlasInventoryError(f"{path.name}:{line_number} must be an object")
                yield line_number, record
    except OSError as error:
        raise AtlasInventoryError(f"could not read {path.name}") from error


def _read_json(path: Path, maximum_bytes: int, label: str) -> dict[str, Any]:
    if path.stat().st_size > maximum_bytes:
        raise AtlasInventoryError(f"{label} exceeds its byte bound")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AtlasInventoryError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise AtlasInventoryError(f"{label} must be a JSON object")
    return value


def _mapping_output_spec(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    basis = manifest.get("digestBasis")
    outputs = basis.get("outputFiles") if isinstance(basis, dict) else None
    if not isinstance(outputs, list):
        raise AtlasInventoryError("mapping manifest lacks digestBasis.outputFiles")
    matched = [item for item in outputs if isinstance(item, dict) and item.get("name") == name]
    if len(matched) != 1:
        raise AtlasInventoryError(f"mapping manifest must name {name} exactly once")
    item = matched[0]
    if set(item) != {"name", "records", "bytes", "sha256"}:
        raise AtlasInventoryError(f"mapping output contract drift for {name}")
    if not isinstance(item["records"], int) or isinstance(item["records"], bool) or item["records"] < 0:
        raise AtlasInventoryError(f"mapping output record count is invalid for {name}")
    if not isinstance(item["bytes"], int) or isinstance(item["bytes"], bool) or item["bytes"] < 0:
        raise AtlasInventoryError(f"mapping output byte count is invalid for {name}")
    _digest(item["sha256"], f"mapping output SHA-256 for {name}", prefixed=False)
    return item


def validate_mapping_manifest(
    path: Path,
    mapping_files: Mapping[str, Path],
    *,
    expected_manifest_sha256: str,
    expected_mapping_id: str = IDENTITY_MAPPING_ID,
    expected_mapping_sha256: str = IDENTITY_MAPPING_SHA256,
    max_mapping_records: int = 300_000,
    max_mapping_artifact_bytes: int = 134_217_728,
) -> dict[str, Any]:
    if _sha256(path) != expected_manifest_sha256:
        raise AtlasInventoryError("mapping manifest content digest drift")
    manifest = _read_json(path, 1_048_576, "mapping manifest")
    if manifest.get("schema") != MAPPING_SCHEMAS["manifest"]:
        raise AtlasInventoryError("mapping manifest schema drift")
    if manifest.get("identityMappingId") != expected_mapping_id:
        raise AtlasInventoryError("mapping release identity drift")
    if manifest.get("identityMappingSha256") != expected_mapping_sha256:
        raise AtlasInventoryError("canonical mapping digest drift")
    if manifest.get("ncbiTaxon") != NCBI_TAXON:
        raise AtlasInventoryError("mapping manifest taxon drift")
    basis = manifest.get("digestBasis")
    if not isinstance(basis, dict):
        raise AtlasInventoryError("mapping manifest digest basis is missing")
    computed = hashlib.sha256((canonical_json(basis) + "\n").encode("utf-8")).hexdigest()
    if computed != expected_mapping_sha256:
        raise AtlasInventoryError("mapping digest basis does not reproduce its identity digest")
    for output_name, input_name in (
        ("current-orfs.jsonl", "sgdCurrentOrfs"),
        ("external-accessions.jsonl", "sgdExternalRelations"),
        ("retired-merged-quarantine.jsonl", "sgdRetiredQuarantine"),
    ):
        output = _mapping_output_spec(manifest, output_name)
        if output["records"] > max_mapping_records:
            raise AtlasInventoryError(f"{input_name} exceeds maxMappingRecords")
        if output["bytes"] > max_mapping_artifact_bytes:
            raise AtlasInventoryError(f"{input_name} exceeds maxMappingArtifactBytes")
        actual = mapping_files[input_name]
        if actual.stat().st_size != output["bytes"] or _sha256(actual) != output["sha256"]:
            raise AtlasInventoryError(f"{input_name} does not match the mapping manifest")
    return manifest


def load_current_orfs(path: Path, expected_records: int, bounds: Bounds) -> dict[str, tuple[str, ...]]:
    by_systematic: dict[str, set[str]] = defaultdict(set)
    seen_curies: set[str] = set()
    records = 0
    for line_number, record in _jsonl(path, bounds.max_line_bytes):
        records += 1
        if records > bounds.max_mapping_records:
            raise AtlasInventoryError("current ORF mapping exceeds maxMappingRecords")
        if record.get("schema") != MAPPING_SCHEMAS["current"] or record.get("ncbiTaxon") != NCBI_TAXON:
            raise AtlasInventoryError(f"current ORF mapping contract drift at line {line_number}")
        curie = record.get("canonicalSgdCurie")
        systematic = record.get("systematicName")
        if not isinstance(curie, str) or SGD_CURIE.fullmatch(curie) is None:
            raise AtlasInventoryError(f"invalid current ORF CURIE at line {line_number}")
        if not isinstance(systematic, str) or not systematic or systematic != systematic.strip():
            raise AtlasInventoryError(f"invalid exact systematic name at line {line_number}")
        display = record.get("displayMetadata")
        if not isinstance(display, dict) or display.get("resolvesIdentity") is not False:
            raise AtlasInventoryError("display metadata must remain non-resolving")
        if record.get("secondaryIdentifiersResolve") is not False:
            raise AtlasInventoryError("secondary identifiers must remain non-resolving")
        if curie in seen_curies:
            raise AtlasInventoryError(f"duplicate current ORF CURIE: {curie}")
        seen_curies.add(curie)
        by_systematic[systematic].add(curie)
    if records != expected_records:
        raise AtlasInventoryError("current ORF mapping record-count drift")
    return {name: tuple(sorted(curies)) for name, curies in sorted(by_systematic.items())}


def load_retired_systematics(path: Path, expected_records: int, bounds: Bounds) -> frozenset[str]:
    names: set[str] = set()
    records = 0
    for line_number, record in _jsonl(path, bounds.max_line_bytes):
        records += 1
        if records > bounds.max_mapping_records:
            raise AtlasInventoryError("retired mapping exceeds maxMappingRecords")
        if record.get("schema") != MAPPING_SCHEMAS["retired"]:
            raise AtlasInventoryError(f"retired mapping schema drift at line {line_number}")
        if record.get("automaticRedirectAllowed") is not False:
            raise AtlasInventoryError("retired mapping must prohibit automatic redirects")
        if record.get("recordKind") == "retired-or-merged":
            if record.get("ncbiTaxon") != NCBI_TAXON:
                raise AtlasInventoryError("retired mapping taxon drift")
            systematic = record.get("systematicName")
            if not isinstance(systematic, str) or not systematic or systematic != systematic.strip():
                raise AtlasInventoryError("retired systematic name is invalid")
            names.add(systematic)
        elif record.get("recordKind") != "malformed-source-row":
            raise AtlasInventoryError("unknown retired quarantine recordKind")
    if records != expected_records:
        raise AtlasInventoryError("retired mapping record-count drift")
    return frozenset(names)


def load_rdata(path: Path) -> tuple[object, str]:
    try:
        import rdata
    except ImportError as error:
        raise AtlasInventoryError("rdata==1.1.0 is required by the locked runtime") from error
    if getattr(rdata, "__version__", None) != RDATA_VERSION:
        raise AtlasInventoryError("rdata runtime version drift")
    try:
        return rdata.read_rda(path), RDATA_VERSION
    except Exception as error:
        raise AtlasInventoryError("could not parse the pinned RData summary") from error


def _frame_column_names(frame: object, condition: str) -> tuple[str, ...]:
    columns = getattr(frame, "columns", None)
    try:
        values = tuple(columns)
    except TypeError as error:
        raise AtlasInventoryError(f"ptbs.{condition} is not a dataframe") from error
    if any(not isinstance(value, str) for value in values):
        raise AtlasInventoryError(f"ptbs.{condition} column names must be strings")
    names = tuple(str(value) for value in values)
    if names != FRAME_COLUMNS:
        raise AtlasInventoryError(f"ptbs.{condition} must have the exact nine-column contract")
    return names


def _series_values(frame: object, name: str, condition: str) -> list[object]:
    try:
        series = frame[name]  # type: ignore[index]
        values = series.tolist()
    except (KeyError, TypeError, AttributeError) as error:
        raise AtlasInventoryError(f"ptbs.{condition}.{name} is not a usable column") from error
    if not isinstance(values, list):
        raise AtlasInventoryError(f"ptbs.{condition}.{name} did not produce a value list")
    return values


def extract_condition_identities(parsed: object, bounds: Bounds) -> dict[str, ConditionIdentity]:
    if not isinstance(parsed, dict) or set(parsed) != {"ptbs"}:
        raise AtlasInventoryError("RData root must contain exactly the ptbs object")
    ptbs = parsed["ptbs"]
    if not isinstance(ptbs, dict) or set(ptbs) != set(CONDITIONS):
        raise AtlasInventoryError("ptbs must contain exactly control and nacl")
    result: dict[str, ConditionIdentity] = {}
    for condition in CONDITIONS:
        frame = ptbs[condition]
        _frame_column_names(frame, condition)
        shape = getattr(frame, "shape", None)
        if not isinstance(shape, tuple) or len(shape) != 2 or shape[1] != len(FRAME_COLUMNS):
            raise AtlasInventoryError(f"ptbs.{condition} shape is invalid")
        if not isinstance(shape[0], Integral) or isinstance(shape[0], bool):
            raise AtlasInventoryError(f"ptbs.{condition} row count is invalid")
        rows = int(shape[0])
        if rows < 1 or rows > bounds.max_frame_rows:
            raise AtlasInventoryError(f"ptbs.{condition} exceeds maxFrameRows")
        assignments_raw = _series_values(frame, "assignment_consensus2", condition)
        cell_numbers_raw = _series_values(frame, "cell_number", condition)
        if len(assignments_raw) != rows or len(cell_numbers_raw) != rows:
            raise AtlasInventoryError(f"ptbs.{condition} selected-column length drift")
        assignments: list[str] = []
        for value in assignments_raw:
            if not isinstance(value, str) or not value or value != value.strip():
                raise AtlasInventoryError(f"ptbs.{condition} assignment is null or noncanonical")
            if len(value.encode("utf-8")) > bounds.max_assignment_bytes:
                raise AtlasInventoryError(f"ptbs.{condition} assignment exceeds maxAssignmentBytes")
            assignments.append(str(value))
        if len(assignments) != len(set(assignments)):
            raise AtlasInventoryError(f"ptbs.{condition} assignment identifiers must be unique")
        cell_numbers: list[int] = []
        for value in cell_numbers_raw:
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise AtlasInventoryError(f"ptbs.{condition} cell_number must be a non-null integer")
            number = int(value)
            if number <= 5:
                raise AtlasInventoryError(f"ptbs.{condition} cell_number must be greater than five")
            cell_numbers.append(number)
        if assignments.count("WT") != 1:
            raise AtlasInventoryError(f"ptbs.{condition} must contain exactly one WT assignment")
        non_wild_type = [value for value in assignments if value != "WT"]
        if any(not value.startswith("bc-") or len(value) == 3 for value in non_wild_type):
            raise AtlasInventoryError(f"ptbs.{condition} non-WT assignments must use literal bc-")
        result[condition] = ConditionIdentity(
            assignments=frozenset(assignments),
            rows=rows,
            non_wild_type=len(non_wild_type),
            minimum_cell_number=min(cell_numbers),
            maximum_cell_number=max(cell_numbers),
        )
    return result


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, records: Iterable[object]) -> tuple[int, str, int]:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(canonical_json(record) + "\n")
            count += 1
    return count, _sha256(path), path.stat().st_size


def _strict_counts(actual: Mapping[str, int], expected: Mapping[str, int] | None) -> None:
    if expected is None:
        return
    for name, value in expected.items():
        if actual.get(name) != value:
            raise AtlasInventoryError(
                f"pinned atlas count drift for {name}: expected {value}, got {actual.get(name)}"
            )


def build_inventory(
    raw_root: str | Path,
    mapping_paths: Mapping[str, Path],
    destination: str | Path,
    bounds: Bounds,
    *,
    raw_spec: FileSpec = PINNED_RAW_FILE,
    mapping_artifact_digests: Mapping[str, str] = MAPPING_ARTIFACT_DIGESTS,
    expected_mapping_manifest_sha256: str = MAPPING_MANIFEST_SHA256,
    expected_mapping_id: str = IDENTITY_MAPPING_ID,
    expected_mapping_sha256: str = IDENTITY_MAPPING_SHA256,
    expected_counts: Mapping[str, int] | None = EXPECTED_COUNTS,
    provenance: SourceProvenance | None = None,
    rdata_loader: Callable[[Path], tuple[object, str]] = load_rdata,
) -> dict[str, Any]:
    required_mapping_inputs = set(MAPPING_ARTIFACT_DIGESTS)
    if set(mapping_paths) != required_mapping_inputs or set(mapping_artifact_digests) != required_mapping_inputs:
        raise AtlasInventoryError("exactly four pinned SGD mapping artifacts are required")
    for name, digest in mapping_artifact_digests.items():
        _digest(digest, f"{name} artifact manifest digest", prefixed=True)
    if provenance is not None and dict(provenance.mapping_artifacts) != dict(mapping_artifact_digests):
        raise AtlasInventoryError("resolved SGD artifacts differ from the pinned digest set")

    raw_path = verify_raw_snapshot(raw_root, raw_spec)
    manifest = validate_mapping_manifest(
        mapping_paths["sgdMappingManifest"],
        mapping_paths,
        expected_manifest_sha256=expected_mapping_manifest_sha256,
        expected_mapping_id=expected_mapping_id,
        expected_mapping_sha256=expected_mapping_sha256,
        max_mapping_records=bounds.max_mapping_records,
        max_mapping_artifact_bytes=bounds.max_mapping_artifact_bytes,
    )
    output_specs = {
        name: _mapping_output_spec(manifest, name)
        for name in (
            "current-orfs.jsonl",
            "external-accessions.jsonl",
            "retired-merged-quarantine.jsonl",
        )
    }
    systematic_map = load_current_orfs(
        mapping_paths["sgdCurrentOrfs"], output_specs["current-orfs.jsonl"]["records"], bounds
    )
    retired_names = load_retired_systematics(
        mapping_paths["sgdRetiredQuarantine"],
        output_specs["retired-merged-quarantine.jsonl"]["records"],
        bounds,
    )
    overlap = sorted(set(systematic_map) & retired_names)
    if overlap:
        raise AtlasInventoryError(
            "mapping artifacts classify systematic names as both current and retired: "
            f"{overlap[:3]}"
        )

    parsed, parser_version = rdata_loader(raw_path)
    if parser_version != RDATA_VERSION:
        raise AtlasInventoryError("rdata loader did not attest version 1.1.0")
    conditions = extract_condition_identities(parsed, bounds)
    control = conditions["control"].assignments
    nacl = conditions["nacl"].assignments
    all_intersection = control & nacl
    all_union = control | nacl
    candidate_assignments = sorted(all_intersection - {"WT"})
    non_wild_union = all_union - {"WT"}
    if len(candidate_assignments) > bounds.max_evidence_records:
        raise AtlasInventoryError("candidate intersection exceeds maxEvidenceRecords")

    evidence: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    mapped_curies: list[str] = []
    mapping_classes: Counter[str] = Counter()
    for assignment in candidate_assignments:
        systematic = assignment.removeprefix("bc-")
        candidates = systematic_map.get(systematic, ())
        if len(candidates) == 1:
            mapping_class = "current-exact"
            mapped_curies.append(candidates[0])
        elif len(candidates) > 1:
            mapping_class = "ambiguous-exact-current"
        elif systematic in retired_names:
            mapping_class = "retired-or-merged-exact"
        else:
            mapping_class = "unmatched-exact-current"
        mapping_classes[mapping_class] += 1
        evidence_record = {
            "schema": EVIDENCE_RECORD_SCHEMA,
            "sourceAssignment": assignment,
            "literalPrefixRemoved": "bc-",
            "exactSystematicName": systematic,
            "ncbiTaxon": NCBI_TAXON,
            "presentInConditions": ["control", "nacl"],
            "mappingClass": mapping_class,
            "currentSgdCuries": list(candidates),
            "caseNormalization": "none",
            "displayMetadataUsedForLookup": False,
            "automaticRedirectAllowed": False,
        }
        evidence.append(evidence_record)
        if mapping_class != "current-exact":
            quarantine.append(
                {
                    "schema": QUARANTINE_RECORD_SCHEMA,
                    "sourceAssignment": assignment,
                    "exactSystematicName": systematic,
                    "ncbiTaxon": NCBI_TAXON,
                    "reason": mapping_class,
                    "currentSgdCuries": list(candidates),
                    "retiredArtifactExactMatch": systematic in retired_names,
                    "automaticRedirectAllowed": False,
                }
            )

    unique_curies = sorted(set(mapped_curies))
    counts = {
        "controlRows": conditions["control"].rows,
        "controlNonWildType": conditions["control"].non_wild_type,
        "naclRows": conditions["nacl"].rows,
        "naclNonWildType": conditions["nacl"].non_wild_type,
        "allAssignmentIntersectionIncludingWildType": len(all_intersection),
        "allAssignmentUnionIncludingWildType": len(all_union),
        "candidateNonWildTypeIntersection": len(candidate_assignments),
        "candidateNonWildTypeUnion": len(non_wild_union),
        "exactCurrentCandidateAssignments": mapping_classes["current-exact"],
        "uniqueCurrentInterventions": len(unique_curies),
        "retiredOrMergedCandidateAssignments": mapping_classes["retired-or-merged-exact"],
        "unmatchedCandidateAssignments": mapping_classes["unmatched-exact-current"],
        "ambiguousCurrentCandidateAssignments": mapping_classes["ambiguous-exact-current"],
    }
    _strict_counts(counts, expected_counts)

    destination_path = Path(destination).resolve()
    if destination_path.exists() or destination_path.is_symlink():
        raise AtlasInventoryError("destination must not already exist")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination_path.name}-", dir=destination_path.parent) as temporary:
        staging = Path(temporary) / destination_path.name
        inventory_root = staging / "intervention-inventory"
        evidence_root = staging / "identity-evidence"
        inventory_root.mkdir(parents=True)
        evidence_root.mkdir()

        inventory_records = [
            {
                "schema": INTERVENTION_RECORD_SCHEMA,
                "interventionId": curie,
                "ncbiTaxon": NCBI_TAXON,
                "qcPassing": True,
            }
            for curie in unique_curies
        ]
        inventory_count, inventory_sha, inventory_bytes = _write_jsonl(
            inventory_root / "interventions.jsonl", inventory_records
        )
        inventory_manifest = {
            "schema": INTERVENTION_INVENTORY_SCHEMA,
            "sourceId": SOURCE_ID,
            "sourceRelease": SOURCE_RELEASE,
            "ncbiTaxon": NCBI_TAXON,
            "stableIdNamespace": "SGD",
            "identityMappingId": expected_mapping_id,
            "identityMappingSha256": expected_mapping_sha256,
            "inventoryFormat": INTERVENTION_RECORD_SCHEMA,
            "files": [
                {
                    "path": "interventions.jsonl",
                    "sha256": inventory_sha,
                    "records": inventory_count,
                }
            ],
        }
        _write_json(inventory_root / "inventory.json", inventory_manifest)
        inventory_manifest_sha = _sha256(inventory_root / "inventory.json")

        evidence_count, evidence_sha, evidence_bytes = _write_jsonl(
            evidence_root / "evidence.jsonl", evidence
        )
        quarantine_count, quarantine_sha, quarantine_bytes = _write_jsonl(
            evidence_root / "quarantine.jsonl", quarantine
        )
        evidence_manifest = {
            "schema": EVIDENCE_MANIFEST_SCHEMA,
            "sourceId": SOURCE_ID,
            "sourceRelease": SOURCE_RELEASE,
            "ncbiTaxon": NCBI_TAXON,
            "identityMappingId": expected_mapping_id,
            "identityMappingSha256": expected_mapping_sha256,
            "files": [
                {
                    "path": "evidence.jsonl",
                    "schema": EVIDENCE_RECORD_SCHEMA,
                    "sha256": evidence_sha,
                    "records": evidence_count,
                },
                {
                    "path": "quarantine.jsonl",
                    "schema": QUARANTINE_RECORD_SCHEMA,
                    "sha256": quarantine_sha,
                    "records": quarantine_count,
                },
            ],
        }
        _write_json(evidence_root / "manifest.json", evidence_manifest)
        evidence_manifest_sha = _sha256(evidence_root / "manifest.json")

        audit = {
            "schema": AUDIT_SCHEMA,
            "source": {
                "id": SOURCE_ID,
                "release": SOURCE_RELEASE,
                "ncbiTaxon": NCBI_TAXON,
                "rawFile": {
                    "name": raw_spec.name,
                    "bytes": raw_spec.bytes,
                    "sha256": raw_spec.sha256,
                    "upstreamMd5": raw_spec.md5,
                },
                "dataset": (
                    {
                        "resource": provenance.resource,
                        "revision": provenance.revision,
                        "manifestDigest": provenance.manifest_digest,
                    }
                    if provenance is not None
                    else None
                ),
            },
            "identityMapping": {
                "id": expected_mapping_id,
                "sha256": expected_mapping_sha256,
                "mappingManifestContentSha256": expected_mapping_manifest_sha256,
                "artifactManifestDigests": dict(sorted(mapping_artifact_digests.items())),
            },
            "rdataContract": {
                "parser": "rdata",
                "parserVersion": parser_version,
                "rootObject": "ptbs",
                "conditionObjects": list(CONDITIONS),
                "exactColumns": list(FRAME_COLUMNS),
                "adapterAccessedColumns": ["assignment_consensus2", "cell_number"],
            },
            "phenotypeBoundary": {
                "phenotypeColumnsPresent": list(PHENOTYPE_COLUMNS),
                "frameConvertedByRdata": True,
                "phenotypeValuesReadByAdapter": False,
                "phenotypeValuesInspectedByAdapter": False,
                "phenotypeValuesUsed": False,
                "phenotypeFieldsEmitted": False,
            },
            "mappingPolicy": {
                "candidateDefinition": "exact-non-WT-assignment-intersection-control-and-nacl",
                "prefixRemoval": "strip-one-literal-bc-prefix-only",
                "systematicNameLookup": "exact-case-sensitive-current-ORF-only",
                "caseNormalization": "none",
                "displaySymbolLookup": False,
                "retiredClassificationSource": "pinned-retired-quarantine-artifact-only",
                "retiredRedirects": False,
                "ambiguousCurrentLookup": "quarantine",
                "qcPassingMeaning": "passes-source-identity-and-cell-count-admissibility-only",
            },
            "conditionCellCountBounds": {
                condition: {
                    "minimum": conditions[condition].minimum_cell_number,
                    "maximum": conditions[condition].maximum_cell_number,
                    "requiredExclusiveMinimum": 5,
                }
                for condition in CONDITIONS
            },
            "counts": counts,
            "mappingClassCounts": dict(sorted(mapping_classes.items())),
            "outputs": {
                "interventionInventory": {
                    "manifestSha256": inventory_manifest_sha,
                    "recordsSha256": inventory_sha,
                    "records": inventory_count,
                    "bytes": inventory_bytes,
                },
                "identityEvidence": {
                    "manifestSha256": evidence_manifest_sha,
                    "evidenceSha256": evidence_sha,
                    "evidenceRecords": evidence_count,
                    "evidenceBytes": evidence_bytes,
                    "quarantineSha256": quarantine_sha,
                    "quarantineRecords": quarantine_count,
                    "quarantineBytes": quarantine_bytes,
                },
            },
            "limitations": [
                "identity-only summary adapter; no transcriptomic matrix is admitted",
                "rdata converts the frame, but phenotype columns are never inspected or used by the adapter",
                "retired and merged identifiers remain quarantined without redirects",
            ],
        }
        _write_json(staging / "audit.json", audit)
        audit_sha = _sha256(staging / "audit.json")
        staging.replace(destination_path)

    return {
        "audit": audit,
        "auditSha256": audit_sha,
        "inventoryManifestSha256": inventory_manifest_sha,
        "evidenceManifestSha256": evidence_manifest_sha,
        **counts,
    }
