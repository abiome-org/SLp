"""Outcome-blind identity normalization for the pinned yeast proteome source.

The quantitative matrix is content-hashed, but only its header and first
``Protein.Group`` field are decoded.  Numeric matrix fields are never parsed,
decoded, transformed, or emitted by this module.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Iterable, Iterator, Mapping


NCBI_TAXON = 4932
SOURCE_ID = "mendeley:w8jtmnszd9.2"
SOURCE_RELEASE = "10.17632/w8jtmnszd9.2"
IDENTITY_MAPPING_ID = "slp-sgd-map:2026-08-28-object-set-v1"
IDENTITY_MAPPING_SHA256 = (
    "6fd789df6099b78a8842baa8f1d20ab0a3fe77f27ce512ee783444eb2627ef2a"
)
MAPPING_MANIFEST_SHA256 = (
    "570557ab1201913a18de9790f8adc5ee2e3cb56c6bb0e8d588fe43660c0214e1"
)

INTERVENTION_INVENTORY_SCHEMA = "slp.intervention-identity-inventory/v1"
INTERVENTION_RECORD_SCHEMA = "slp.intervention-identity-record/v1"
PROTEIN_INVENTORY_SCHEMA = "slp.proteome-protein-relation-inventory/v1"
PROTEIN_RECORD_SCHEMA = "slp.proteome-protein-relation/v1"
AUDIT_SCHEMA = "slp.proteome-identity-audit/v1"
QUARANTINE_SCHEMA = "slp.proteome-intervention-quarantine/v1"

SGD_CURIE = re.compile(r"^SGD:S[0-9]{9}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RESOURCE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
INPUT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
MAPPING_SCHEMAS = {
    "current": "slp.sgd-current-orf/v1",
    "external": "slp.sgd-external-accession-relation/v1",
    "retired": "slp.sgd-retired-quarantine/v1",
    "manifest": "slp.sgd-stable-id-mapping/v1",
}


class ProteomeInventoryError(ValueError):
    """Raised when provenance or identity normalization cannot be trusted."""


@dataclass(frozen=True)
class FileSpec:
    name: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.name or Path(self.name).name != self.name:
            raise ProteomeInventoryError("pinned raw file names must be basenames")
        if not isinstance(self.bytes, int) or isinstance(self.bytes, bool) or self.bytes < 0:
            raise ProteomeInventoryError(f"invalid byte count for {self.name}")
        if not isinstance(self.sha256, str) or SHA256.fullmatch(self.sha256) is None:
            raise ProteomeInventoryError(f"invalid SHA-256 for {self.name}")


PINNED_RAW_FILES = (
    FileSpec(
        "yeast5k_noimpute_wide.csv",
        167_754_298,
        "69a9df05b6db011f595a4e0b3ce25c1cc247f22cbdd066c79e6da9a706aa1df9",
    ),
    FileSpec(
        "yeast5k_metadata.csv",
        377_047,
        "48864282c82d516ae929dc87aff7fae9e05e9b922e316c001f3d29dce0ff878b",
    ),
    FileSpec(
        "Detection_of_KO_proteins.csv",
        30_260,
        "ca7c8f2ac33272df3763807add7b8982b8a8b52d4276bd929a61ecf19e0ae405",
    ),
    FileSpec(
        "summary_fileupload.pdf",
        65_558,
        "4078289dc86dd6b526d9b0c963e6df61d53acdfdf6260abdeae307588623f828",
    ),
)

MAPPING_ARTIFACT_DIGESTS = {
    "sgdCurrentOrfs": "sha256:e67f0e8773feae108ecdb687139885e01ca972ff4aec95cd1358b33db1ea1192",
    "sgdExternalRelations": "sha256:75e0fef99bbae3bb4e4dc3e2f24cfd0ab62919c0e6e3e321e8d82f3bd557f4da",
    "sgdRetiredQuarantine": "sha256:07ea82f877224496c24effc2aa2a2b684c01b85017e616a70f003a5363f6925f",
    "sgdMappingManifest": "sha256:c74ea81ce604357b998e5f09130dff85bf8a7a26504b9b2426f8038608c52d9c",
}

EXPECTED_COUNTS = {
    "metadataRows": 5_476,
    "knockoutRows": 4_699,
    "controlRows": 388,
    "analyticalQcRows": 389,
    "eligibleKnockoutRows": 4_623,
    "eligibleInterventions": 4_476,
    "quarantineRows": 76,
    "quarantineUniqueRawIds": 74,
    "retiredOrMergedRows": 36,
    "unmatchedRows": 40,
    "proteinRecords": 1_850,
    "oneToOneProteinRelations": 1_845,
    "oneToManyProteinRelations": 5,
}

EXPECTED_ONE_TO_MANY = {
    "UniProtKB:P02309": ("SGD:S000000213", "SGD:S000004975"),
    "UniProtKB:P02994": ("SGD:S000000322", "SGD:S000006284"),
    "UniProtKB:P10081": ("SGD:S000001767", "SGD:S000003674"),
    "UniProtKB:P32324": ("SGD:S000002793", "SGD:S000005659"),
    "UniProtKB:P61830": ("SGD:S000000214", "SGD:S000004976"),
}

METADATA_COLUMNS = (
    "Filename",
    "Injection nr",
    "Well nr (counted row-wise)",
    "Plate (batch) nr",
    "sampletype",
    "ORF",
)


@dataclass(frozen=True)
class Bounds:
    max_metadata_rows: int = 6_000
    max_protein_rows: int = 2_000
    max_mapping_records: int = 300_000
    max_line_bytes: int = 2_097_152
    max_quarantine_rows: int = 512

    def __post_init__(self) -> None:
        for name, value, minimum, maximum in (
            ("maxMetadataRows", self.max_metadata_rows, 1, 100_000),
            ("maxProteinRows", self.max_protein_rows, 1, 100_000),
            ("maxMappingRecords", self.max_mapping_records, 1, 1_000_000),
            ("maxLineBytes", self.max_line_bytes, 128, 16_777_216),
            ("maxQuarantineRows", self.max_quarantine_rows, 1, 100_000),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
                raise ProteomeInventoryError(f"{name} must be an integer in [{minimum}, {maximum}]")


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


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ProteomeInventoryError(f"could not read {path.name}") from error
    return digest.hexdigest()


def _digest(value: object, label: str, *, prefixed: bool) -> str:
    if not isinstance(value, str):
        raise ProteomeInventoryError(f"{label} must be a SHA-256")
    body = value.removeprefix("sha256:") if prefixed else value
    if prefixed and not value.startswith("sha256:"):
        raise ProteomeInventoryError(f"{label} must use the sha256: prefix")
    if SHA256.fullmatch(body) is None:
        raise ProteomeInventoryError(f"{label} must be a lowercase SHA-256")
    return value


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ProteomeInventoryError(f"{label} must be a non-empty trimmed string")
    return value


def _resource_name(value: object, label: str) -> tuple[str, str]:
    resource = _nonempty(value, label)
    if not resource.startswith("omf://"):
        raise ProteomeInventoryError(f"{label} must be an OMF DatasetSnapshot URI")
    identity, separator, revision = resource.removeprefix("omf://").rpartition("@")
    if not separator:
        raise ProteomeInventoryError(f"{label} must carry an exact revision")
    _digest(revision, f"{label} revision", prefixed=True)
    parts = identity.split("/")
    if (
        len(parts) < 3
        or parts[-2] != "datasetsnapshot"
        or RESOURCE_NAME.fullmatch(parts[-1]) is None
        or any(not item or item in {".", ".."} or any(char.isspace() for char in item) for item in parts)
    ):
        raise ProteomeInventoryError(f"{label} must identify a DatasetSnapshot")
    return parts[-1], revision


def _resolved_path(value: object, label: str, *, directory: bool) -> Path:
    requested = value if isinstance(value, Path) else Path(_nonempty(value, label))
    cursor = requested.absolute()
    for _ in range(3):
        if cursor.is_symlink():
            raise ProteomeInventoryError(f"{label} materialization must not contain a symlink")
        cursor = cursor.parent
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise ProteomeInventoryError(f"{label} does not exist") from error
    if directory != resolved.is_dir() or (not directory and not resolved.is_file()):
        kind = "directory" if directory else "regular file"
        raise ProteomeInventoryError(f"{label} must be a {kind}")
    return resolved


def resolve_pinned_raw_dataset(value: object, input_name: str = "rawProteome") -> PinnedDataset:
    """Accept only OMF's literal copy-materialized DatasetSnapshot shape."""
    if INPUT_NAME.fullmatch(input_name) is None or not isinstance(value, dict):
        raise ProteomeInventoryError(f"{input_name} must be a materialized DatasetSnapshot")
    if set(value) != {"resource", "mode", "path", "manifestDigest"}:
        raise ProteomeInventoryError(f"{input_name} has a spoofed DatasetSnapshot shape")
    resource_name, revision = _resource_name(value["resource"], f"{input_name}.resource")
    if value["mode"] != "copy":
        raise ProteomeInventoryError(f"{input_name} must be copied, not mutable")
    manifest_digest = _digest(
        value["manifestDigest"], f"{input_name}.manifestDigest", prefixed=True
    )
    root = _resolved_path(value["path"], f"{input_name}.path", directory=True)
    if root.name != resource_name or root.parent.name != input_name or root.parent.parent.name != "inputs":
        raise ProteomeInventoryError(f"{input_name}.path is inconsistent with OMF materialization")
    return PinnedDataset(root, str(value["resource"]), revision, manifest_digest)


def resolve_literal_artifact(
    value: object,
    input_name: str,
    expected_manifest_digest: str,
) -> LiteralArtifact:
    """Accept only one exact, literal, file-valued OMF artifact input."""
    if INPUT_NAME.fullmatch(input_name) is None or not isinstance(value, dict):
        raise ProteomeInventoryError(f"{input_name} must be a literal OMF artifact")
    if set(value) != {"resource", "kind", "artifacts", "paths", "path"}:
        raise ProteomeInventoryError(f"{input_name} has a spoofed artifact shape")
    if value["kind"] != "artifact":
        raise ProteomeInventoryError(f"{input_name} kind must be artifact")
    artifacts = value["artifacts"]
    paths = value["paths"]
    if not isinstance(artifacts, dict) or set(artifacts) != {"payload"}:
        raise ProteomeInventoryError(f"{input_name}.artifacts must contain only payload")
    if not isinstance(paths, dict) or set(paths) != {"payload"}:
        raise ProteomeInventoryError(f"{input_name}.paths must contain only payload")
    digest = _digest(artifacts["payload"], f"{input_name} artifact", prefixed=True)
    if digest != expected_manifest_digest:
        raise ProteomeInventoryError(f"{input_name} does not match its pinned artifact")
    if value["resource"] != f"artifact:{digest}":
        raise ProteomeInventoryError(f"{input_name}.resource does not match its payload")
    if paths["payload"] != value["path"]:
        raise ProteomeInventoryError(f"{input_name}.path is inconsistent with paths.payload")
    path = _resolved_path(value["path"], f"{input_name}.path", directory=False)
    if path.name != "payload" or path.parent.name != input_name or path.parent.parent.name != "inputs":
        raise ProteomeInventoryError(f"{input_name}.path is inconsistent with OMF materialization")
    return LiteralArtifact(path, digest)


def _relative_regular_file(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if (
        relative != posix.as_posix()
        or posix.is_absolute()
        or "\\" in relative
        or ":" in relative
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ProteomeInventoryError("raw file path is not canonical")
    cursor = root
    for part in posix.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ProteomeInventoryError(f"raw file must not be a symlink: {relative}")
    try:
        resolved = cursor.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ProteomeInventoryError(f"raw file is missing or escapes its snapshot: {relative}") from error
    if not resolved.is_file():
        raise ProteomeInventoryError(f"raw file is not regular: {relative}")
    return resolved


def verify_raw_snapshot(root_value: str | Path, specs: Iterable[FileSpec]) -> dict[str, Path]:
    root = _resolved_path(root_value, "raw snapshot", directory=True)
    if root.is_symlink():
        raise ProteomeInventoryError("raw snapshot must not be a symlink")
    specs = tuple(specs)
    specs_by_name = {item.name: item for item in specs}
    if len(specs_by_name) != len(specs):
        raise ProteomeInventoryError("raw file specifications contain duplicates")
    actual = {item.name for item in root.iterdir()}
    if actual != set(specs_by_name):
        raise ProteomeInventoryError("raw snapshot file set differs from the pinned allowlist")
    paths: dict[str, Path] = {}
    for name, spec in sorted(specs_by_name.items()):
        path = _relative_regular_file(root, name)
        if path.stat().st_size != spec.bytes:
            raise ProteomeInventoryError(f"raw file byte drift: {name}")
        if _sha256(path) != spec.sha256:
            raise ProteomeInventoryError(f"raw file digest drift: {name}")
        paths[name] = path
    return paths


def _read_json(path: Path, maximum_bytes: int, label: str) -> dict[str, Any]:
    if path.stat().st_size > maximum_bytes:
        raise ProteomeInventoryError(f"{label} exceeds its byte bound")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProteomeInventoryError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ProteomeInventoryError(f"{label} must be a JSON object")
    return value


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
                    raise ProteomeInventoryError(f"{path.name}:{line_number} exceeds maxLineBytes")
                if not raw.endswith(b"\n") or raw in {b"\n", b"\r\n"}:
                    raise ProteomeInventoryError(f"{path.name}:{line_number} is not canonical JSONL")
                try:
                    record = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ProteomeInventoryError(f"{path.name}:{line_number} is invalid JSON") from error
                if not isinstance(record, dict):
                    raise ProteomeInventoryError(f"{path.name}:{line_number} must be an object")
                yield line_number, record
    except OSError as error:
        raise ProteomeInventoryError(f"could not read {path.name}") from error


def _mapping_output_spec(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    basis = manifest.get("digestBasis")
    outputs = basis.get("outputFiles") if isinstance(basis, dict) else None
    if not isinstance(outputs, list):
        raise ProteomeInventoryError("mapping manifest lacks digestBasis.outputFiles")
    matched = [item for item in outputs if isinstance(item, dict) and item.get("name") == name]
    if len(matched) != 1:
        raise ProteomeInventoryError(f"mapping manifest must name {name} exactly once")
    item = matched[0]
    if set(item) != {"name", "records", "bytes", "sha256"}:
        raise ProteomeInventoryError(f"mapping output contract drift for {name}")
    if not isinstance(item["records"], int) or isinstance(item["records"], bool) or item["records"] < 0:
        raise ProteomeInventoryError(f"mapping output record count is invalid for {name}")
    if not isinstance(item["bytes"], int) or isinstance(item["bytes"], bool) or item["bytes"] < 0:
        raise ProteomeInventoryError(f"mapping output byte count is invalid for {name}")
    _digest(item["sha256"], f"mapping output SHA-256 for {name}", prefixed=False)
    return item


def validate_mapping_manifest(
    path: Path,
    mapping_files: Mapping[str, Path],
    *,
    expected_manifest_sha256: str,
    expected_mapping_id: str = IDENTITY_MAPPING_ID,
    expected_mapping_sha256: str = IDENTITY_MAPPING_SHA256,
) -> dict[str, Any]:
    if _sha256(path) != expected_manifest_sha256:
        raise ProteomeInventoryError("mapping manifest content digest drift")
    manifest = _read_json(path, 1_048_576, "mapping manifest")
    if manifest.get("schema") != MAPPING_SCHEMAS["manifest"]:
        raise ProteomeInventoryError("mapping manifest schema drift")
    if manifest.get("identityMappingId") != expected_mapping_id:
        raise ProteomeInventoryError("mapping release identity drift")
    if manifest.get("identityMappingSha256") != expected_mapping_sha256:
        raise ProteomeInventoryError("canonical mapping digest drift")
    if manifest.get("ncbiTaxon") != NCBI_TAXON:
        raise ProteomeInventoryError("mapping manifest taxon drift")
    basis = manifest.get("digestBasis")
    if not isinstance(basis, dict):
        raise ProteomeInventoryError("mapping manifest digest basis is missing")
    computed = hashlib.sha256((canonical_json(basis) + "\n").encode("utf-8")).hexdigest()
    if computed != expected_mapping_sha256:
        raise ProteomeInventoryError("mapping digest basis does not reproduce its identity digest")
    for output_name, input_name in (
        ("current-orfs.jsonl", "sgdCurrentOrfs"),
        ("external-accessions.jsonl", "sgdExternalRelations"),
        ("retired-merged-quarantine.jsonl", "sgdRetiredQuarantine"),
    ):
        spec = _mapping_output_spec(manifest, output_name)
        actual = mapping_files[input_name]
        if actual.stat().st_size != spec["bytes"] or _sha256(actual) != spec["sha256"]:
            raise ProteomeInventoryError(f"{input_name} does not match the mapping manifest")
    return manifest


def load_current_orfs(path: Path, expected_records: int, bounds: Bounds) -> dict[str, tuple[str, ...]]:
    by_systematic: dict[str, set[str]] = defaultdict(set)
    seen_curies: set[str] = set()
    records = 0
    for line_number, record in _jsonl(path, bounds.max_line_bytes):
        records += 1
        if records > bounds.max_mapping_records:
            raise ProteomeInventoryError("current ORF mapping exceeds maxMappingRecords")
        if record.get("schema") != MAPPING_SCHEMAS["current"] or record.get("ncbiTaxon") != NCBI_TAXON:
            raise ProteomeInventoryError(f"current ORF mapping contract drift at line {line_number}")
        curie = record.get("canonicalSgdCurie")
        systematic = record.get("systematicName")
        if not isinstance(curie, str) or SGD_CURIE.fullmatch(curie) is None:
            raise ProteomeInventoryError(f"invalid current ORF CURIE at line {line_number}")
        if not isinstance(systematic, str) or not systematic or systematic != systematic.strip():
            raise ProteomeInventoryError(f"invalid exact systematic name at line {line_number}")
        display = record.get("displayMetadata")
        if not isinstance(display, dict) or display.get("resolvesIdentity") is not False:
            raise ProteomeInventoryError("display metadata must remain non-resolving")
        if record.get("secondaryIdentifiersResolve") is not False:
            raise ProteomeInventoryError("secondary SGD identifiers must remain non-resolving")
        if curie in seen_curies:
            raise ProteomeInventoryError(f"duplicate current ORF CURIE: {curie}")
        seen_curies.add(curie)
        by_systematic[systematic].add(curie)
    if records != expected_records:
        raise ProteomeInventoryError("current ORF mapping record-count drift")
    return {key: tuple(sorted(values)) for key, values in sorted(by_systematic.items())}


def load_retired_systematics(path: Path, expected_records: int, bounds: Bounds) -> frozenset[str]:
    names: set[str] = set()
    records = 0
    for line_number, record in _jsonl(path, bounds.max_line_bytes):
        records += 1
        if records > bounds.max_mapping_records:
            raise ProteomeInventoryError("retired mapping exceeds maxMappingRecords")
        if record.get("schema") != MAPPING_SCHEMAS["retired"]:
            raise ProteomeInventoryError(f"retired mapping schema drift at line {line_number}")
        if record.get("automaticRedirectAllowed") is not False:
            raise ProteomeInventoryError("retired mapping must prohibit automatic redirects")
        if record.get("recordKind") == "retired-or-merged":
            if record.get("ncbiTaxon") != NCBI_TAXON:
                raise ProteomeInventoryError("retired mapping taxon drift")
            systematic = record.get("systematicName")
            if not isinstance(systematic, str) or not systematic or systematic != systematic.strip():
                raise ProteomeInventoryError("retired systematic name is invalid")
            names.add(systematic)
        elif record.get("recordKind") != "malformed-source-row":
            raise ProteomeInventoryError("unknown retired quarantine recordKind")
    if records != expected_records:
        raise ProteomeInventoryError("retired mapping record-count drift")
    return frozenset(names)


def _parse_csv_text(path: Path, bounds: Bounds) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != METADATA_COLUMNS:
                raise ProteomeInventoryError("proteome metadata column contract drift")
            rows: list[dict[str, str]] = []
            for row in reader:
                if len(rows) >= bounds.max_metadata_rows:
                    raise ProteomeInventoryError("proteome metadata exceeds maxMetadataRows")
                if None in row or set(row) != set(METADATA_COLUMNS):
                    raise ProteomeInventoryError("proteome metadata row shape drift")
                normalized: dict[str, str] = {}
                for field in METADATA_COLUMNS:
                    value = row[field]
                    if not isinstance(value, str) or value != value.strip():
                        raise ProteomeInventoryError("proteome metadata values must be exact and trimmed")
                    normalized[field] = value
                rows.append(normalized)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ProteomeInventoryError("could not parse proteome identity metadata") from error
    filenames = tuple(row["Filename"] for row in rows)
    if any(not item for item in filenames) or len(filenames) != len(set(filenames)):
        raise ProteomeInventoryError("proteome sample filenames must be nonempty and unique")
    return filenames, rows


def _decode_first_csv_field(raw: bytes, line_number: int) -> str:
    line = raw.removesuffix(b"\n").removesuffix(b"\r")
    if not line:
        raise ProteomeInventoryError(f"matrix line {line_number} is blank")
    if line.startswith(b'"'):
        output = bytearray()
        index = 1
        while index < len(line):
            if line[index] == 34:
                if index + 1 < len(line) and line[index + 1] == 34:
                    output.append(34)
                    index += 2
                    continue
                if index + 1 >= len(line) or line[index + 1] != 44:
                    raise ProteomeInventoryError(f"matrix first field is malformed at line {line_number}")
                field = bytes(output)
                break
            output.append(line[index])
            index += 1
        else:
            raise ProteomeInventoryError(f"matrix first field is unterminated at line {line_number}")
    else:
        field, separator, _tail = line.partition(b",")
        if not separator:
            raise ProteomeInventoryError(f"matrix row lacks value fields at line {line_number}")
    try:
        decoded = field.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProteomeInventoryError(f"matrix identity is not UTF-8 at line {line_number}") from error
    return _nonempty(decoded, f"matrix identity line {line_number}")


def scan_matrix_identities(
    path: Path,
    expected_sample_headers: tuple[str, ...],
    bounds: Bounds,
) -> tuple[str, ...]:
    """Decode only the CSV header and first field; never decode matrix values."""
    protein_ids: list[str] = []
    try:
        with path.open("rb") as stream:
            header_raw = stream.readline(bounds.max_line_bytes + 1)
            if len(header_raw) > bounds.max_line_bytes or not header_raw.endswith(b"\n"):
                raise ProteomeInventoryError("matrix header exceeds maxLineBytes or lacks LF")
            try:
                header = next(csv.reader(io.StringIO(header_raw.decode("utf-8"))))
            except (UnicodeDecodeError, csv.Error, StopIteration) as error:
                raise ProteomeInventoryError("matrix header is not valid UTF-8 CSV") from error
            if not header or header[0] != "Protein.Group":
                raise ProteomeInventoryError("matrix first column must be Protein.Group")
            if tuple(header[1:]) != expected_sample_headers:
                raise ProteomeInventoryError("matrix sample header order differs from metadata")
            line_number = 1
            while True:
                raw = stream.readline(bounds.max_line_bytes + 1)
                if not raw:
                    break
                line_number += 1
                if len(raw) > bounds.max_line_bytes or not raw.endswith(b"\n"):
                    raise ProteomeInventoryError(f"matrix line {line_number} exceeds bound or lacks LF")
                if len(protein_ids) >= bounds.max_protein_rows:
                    raise ProteomeInventoryError("matrix exceeds maxProteinRows")
                protein_ids.append(_decode_first_csv_field(raw, line_number))
    except OSError as error:
        raise ProteomeInventoryError("could not scan matrix identities") from error
    if len(protein_ids) != len(set(protein_ids)):
        raise ProteomeInventoryError("Protein.Group identities must be unique")
    return tuple(protein_ids)


def load_protein_relations(
    path: Path,
    expected_records: int,
    protein_accessions: frozenset[str],
    current_curies: frozenset[str],
    bounds: Bounds,
) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    matches: dict[str, tuple[str, ...]] = {}
    mapping_records = 0
    non_current_targets = 0
    for line_number, record in _jsonl(path, bounds.max_line_bytes):
        mapping_records += 1
        if mapping_records > bounds.max_mapping_records:
            raise ProteomeInventoryError("external mapping exceeds maxMappingRecords")
        if record.get("schema") != MAPPING_SCHEMAS["external"] or record.get("ncbiTaxon") != NCBI_TAXON:
            raise ProteomeInventoryError(f"external mapping contract drift at line {line_number}")
        typed = record.get("typedAccession")
        targets = record.get("targets")
        if not isinstance(typed, dict) or not isinstance(targets, list):
            raise ProteomeInventoryError("external mapping lacks typed accession or targets")
        if typed.get("caseNormalization") != "none" or typed.get("namespaceInferred") is not False:
            raise ProteomeInventoryError("external mapping must retain exact typed identity")
        if record.get("relationOnly") is not True or record.get("targetCount") != len(targets):
            raise ProteomeInventoryError("external relation cardinality drift")
        accession = typed.get("value")
        if (
            typed.get("source") != "UniProtKB"
            or typed.get("type") != "UniProtKB ID"
            or accession not in protein_accessions
        ):
            continue
        if accession in matches:
            raise ProteomeInventoryError(f"duplicate typed UniProt relation: {accession}")
        current: set[str] = set()
        for target in targets:
            if not isinstance(target, dict):
                raise ProteomeInventoryError("external mapping target must be an object")
            curie = target.get("canonicalSgdCurie")
            status = target.get("targetStatus")
            if not isinstance(curie, str) or SGD_CURIE.fullmatch(curie) is None:
                raise ProteomeInventoryError("external mapping target CURIE is invalid")
            if status == "current-orf":
                if curie not in current_curies:
                    raise ProteomeInventoryError("external current-orf target is absent from current map")
                current.add(curie)
            else:
                non_current_targets += 1
        if not current:
            raise ProteomeInventoryError(f"protein accession has no exact current ORF relation: {accession}")
        matches[str(accession)] = tuple(sorted(current))
    if mapping_records != expected_records:
        raise ProteomeInventoryError("external mapping record-count drift")
    missing = sorted(protein_accessions - set(matches))
    if missing:
        raise ProteomeInventoryError(f"protein accessions lack exact typed UniProt relations: {missing[:3]}")
    return matches, {"mappingRecords": mapping_records, "nonCurrentTargetsExcluded": non_current_targets}


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
            raise ProteomeInventoryError(
                f"pinned source count drift for {name}: expected {value}, got {actual.get(name)}"
            )


def build_inventory(
    raw_root: str | Path,
    mapping_paths: Mapping[str, Path],
    destination: str | Path,
    bounds: Bounds,
    *,
    raw_specs: Iterable[FileSpec] = PINNED_RAW_FILES,
    mapping_artifact_digests: Mapping[str, str] = MAPPING_ARTIFACT_DIGESTS,
    expected_mapping_manifest_sha256: str = MAPPING_MANIFEST_SHA256,
    expected_mapping_id: str = IDENTITY_MAPPING_ID,
    expected_mapping_sha256: str = IDENTITY_MAPPING_SHA256,
    expected_counts: Mapping[str, int] | None = EXPECTED_COUNTS,
    expected_one_to_many: Mapping[str, tuple[str, ...]] | None = EXPECTED_ONE_TO_MANY,
    provenance: SourceProvenance | None = None,
) -> dict[str, Any]:
    """Build deterministic identity-only inventories from immutable inputs."""
    raw_specs = tuple(raw_specs)
    required_mapping_inputs = set(MAPPING_ARTIFACT_DIGESTS)
    if set(mapping_paths) != required_mapping_inputs or set(mapping_artifact_digests) != required_mapping_inputs:
        raise ProteomeInventoryError("exactly four pinned SGD mapping artifacts are required")
    for name, digest in mapping_artifact_digests.items():
        _digest(digest, f"{name} artifact manifest digest", prefixed=True)
    if provenance is not None and dict(provenance.mapping_artifacts) != dict(mapping_artifact_digests):
        raise ProteomeInventoryError("resolved SGD artifacts differ from the pinned digest set")
    raw_files = verify_raw_snapshot(raw_root, raw_specs)
    manifest = validate_mapping_manifest(
        mapping_paths["sgdMappingManifest"],
        mapping_paths,
        expected_manifest_sha256=expected_mapping_manifest_sha256,
        expected_mapping_id=expected_mapping_id,
        expected_mapping_sha256=expected_mapping_sha256,
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
    current_curies = frozenset(curie for values in systematic_map.values() for curie in values)
    retired_names = load_retired_systematics(
        mapping_paths["sgdRetiredQuarantine"],
        output_specs["retired-merged-quarantine.jsonl"]["records"],
        bounds,
    )
    overlapping_systematics = sorted(set(systematic_map) & retired_names)
    if overlapping_systematics:
        raise ProteomeInventoryError(
            "mapping artifacts classify systematic names as both current and retired: "
            f"{overlapping_systematics[:3]}"
        )
    sample_headers, metadata = _parse_csv_text(raw_files["yeast5k_metadata.csv"], bounds)
    proteins = scan_matrix_identities(
        raw_files["yeast5k_noimpute_wide.csv"], sample_headers, bounds
    )

    eligible_records: list[str] = []
    eligible: set[str] = set()
    quarantine: list[dict[str, Any]] = []
    sample_types: Counter[str] = Counter()
    for source_row, row in enumerate(metadata, start=2):
        sample_type = row["sampletype"]
        if sample_type not in {"ko", "HIS3", "qc"}:
            raise ProteomeInventoryError(f"unsupported sampletype at metadata row {source_row}")
        sample_types[sample_type] += 1
        if sample_type != "ko":
            continue
        raw_id = _nonempty(row["ORF"], f"metadata ORF row {source_row}")
        candidates = systematic_map.get(raw_id, ())
        if len(candidates) == 1:
            eligible_records.append(candidates[0])
            eligible.add(candidates[0])
            continue
        if len(candidates) > 1:
            reason = "ambiguous-exact-current-systematic-name"
        elif raw_id in retired_names:
            reason = "retired-or-merged-exact-systematic-name"
        elif raw_id != raw_id.upper():
            reason = "mixed-case-not-current-exact"
        else:
            reason = "unmatched-exact-systematic-name"
        if len(quarantine) >= bounds.max_quarantine_rows:
            raise ProteomeInventoryError("intervention quarantine exceeds maxQuarantineRows")
        quarantine.append(
            {
                "schema": QUARANTINE_SCHEMA,
                "sourceRow": source_row,
                "sampleId": row["Filename"],
                "rawInterventionId": raw_id,
                "ncbiTaxon": NCBI_TAXON,
                "reason": reason,
                "exactCaseSensitiveLookup": True,
                "caseNormalizedForLookup": False,
                "displayMetadataUsedForLookup": False,
                "automaticRedirectAllowed": False,
            }
        )

    protein_relations, relation_stats = load_protein_relations(
        mapping_paths["sgdExternalRelations"],
        output_specs["external-accessions.jsonl"]["records"],
        frozenset(proteins),
        current_curies,
        bounds,
    )
    relation_records = [
        {
            "schema": PROTEIN_RECORD_SCHEMA,
            "proteinId": f"UniProtKB:{accession}",
            "sourceAccession": accession,
            "sourceAccessionType": {
                "source": "UniProtKB",
                "type": "UniProtKB ID",
                "namespaceInferred": False,
                "caseNormalization": "none",
            },
            "ncbiTaxon": NCBI_TAXON,
            "currentOrfRelations": list(protein_relations[accession]),
            "currentOrfRelationCount": len(protein_relations[accession]),
            "chooseFirstAllowed": False,
        }
        for accession in sorted(proteins)
    ]
    one_to_many = {
        record["proteinId"]: tuple(record["currentOrfRelations"])
        for record in relation_records
        if record["currentOrfRelationCount"] > 1
    }
    if expected_one_to_many is not None and one_to_many != dict(expected_one_to_many):
        raise ProteomeInventoryError("pinned one-to-many UniProt relation inventory drift")

    reason_counts = Counter(item["reason"] for item in quarantine)
    retired_rows = reason_counts["retired-or-merged-exact-systematic-name"]
    unmatched_rows = sum(
        count for reason, count in reason_counts.items() if reason != "retired-or-merged-exact-systematic-name"
    )
    counts = {
        "metadataRows": len(metadata),
        "knockoutRows": sample_types["ko"],
        "controlRows": sample_types["HIS3"],
        "analyticalQcRows": sample_types["qc"],
        "eligibleKnockoutRows": len(eligible_records),
        "eligibleInterventions": len(eligible),
        "quarantineRows": len(quarantine),
        "quarantineUniqueRawIds": len({item["rawInterventionId"] for item in quarantine}),
        "retiredOrMergedRows": retired_rows,
        "unmatchedRows": unmatched_rows,
        "proteinRecords": len(relation_records),
        "oneToOneProteinRelations": sum(record["currentOrfRelationCount"] == 1 for record in relation_records),
        "oneToManyProteinRelations": sum(record["currentOrfRelationCount"] > 1 for record in relation_records),
    }
    _strict_counts(counts, expected_counts)

    destination_path = Path(destination).resolve()
    if destination_path.exists() or destination_path.is_symlink():
        raise ProteomeInventoryError("destination must not already exist")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination_path.name}-", dir=destination_path.parent) as temporary:
        staging = Path(temporary) / destination_path.name
        intervention_root = staging / "intervention-inventory"
        protein_root = staging / "protein-relations"
        intervention_root.mkdir(parents=True)
        protein_root.mkdir()
        intervention_records = [
            {
                "schema": INTERVENTION_RECORD_SCHEMA,
                "interventionId": curie,
                "ncbiTaxon": NCBI_TAXON,
                "qcPassing": True,
            }
            for curie in sorted(eligible_records)
        ]
        intervention_count, intervention_sha, intervention_bytes = _write_jsonl(
            intervention_root / "interventions.jsonl", intervention_records
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
                    "sha256": intervention_sha,
                    "records": intervention_count,
                }
            ],
        }
        _write_json(intervention_root / "inventory.json", inventory_manifest)
        inventory_manifest_sha = _sha256(intervention_root / "inventory.json")

        protein_count, protein_sha, protein_bytes = _write_jsonl(
            protein_root / "relations.jsonl", relation_records
        )
        protein_manifest = {
            "schema": PROTEIN_INVENTORY_SCHEMA,
            "sourceId": SOURCE_ID,
            "sourceRelease": SOURCE_RELEASE,
            "ncbiTaxon": NCBI_TAXON,
            "identityMappingId": expected_mapping_id,
            "identityMappingSha256": expected_mapping_sha256,
            "relationFormat": PROTEIN_RECORD_SCHEMA,
            "files": [
                {"path": "relations.jsonl", "sha256": protein_sha, "records": protein_count}
            ],
        }
        _write_json(protein_root / "manifest.json", protein_manifest)
        protein_manifest_sha = _sha256(protein_root / "manifest.json")

        audit = {
            "schema": AUDIT_SCHEMA,
            "source": {
                "id": SOURCE_ID,
                "release": SOURCE_RELEASE,
                "ncbiTaxon": NCBI_TAXON,
                "rawFiles": [
                    {"name": spec.name, "bytes": spec.bytes, "sha256": spec.sha256}
                    for spec in sorted(raw_specs, key=lambda item: item.name)
                ],
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
            "accessBoundary": {
                "metadataColumnsParsed": list(METADATA_COLUMNS),
                "matrixHeaderParsed": True,
                "matrixFirstIdentityColumnParsed": "Protein.Group",
                "matrixQuantitativeFieldsDecoded": False,
                "matrixQuantitativeValuesParsed": False,
                "quantitativeCellValuesInterpreted": False,
                "detectionFileAccess": "content-hash-only",
                "documentationPdfAccess": "content-hash-only",
                "outcomeFieldsEmitted": False,
            },
            "mappingPolicy": {
                "systematicNameLookup": "exact-case-sensitive-current-ORF-only",
                "caseNormalization": "none",
                "displaySymbolLookup": False,
                "retiredRedirects": False,
                "ambiguousCurrentLookup": "quarantine",
                "qcPassingMeaning": "passes-source-identity-admissibility-only",
                "proteinTypedKey": ["UniProtKB", "UniProtKB ID", "exact-source-accession"],
                "proteinRelationsChooseFirst": False,
            },
            "counts": counts,
            "mappingReadCounts": relation_stats,
            "quarantineReasonCounts": dict(sorted(reason_counts.items())),
            "quarantineRows": sorted(
                quarantine,
                key=lambda item: (item["rawInterventionId"], item["sampleId"], item["sourceRow"]),
            ),
            "outputs": {
                "interventionInventory": {
                    "manifestSha256": inventory_manifest_sha,
                    "recordsSha256": intervention_sha,
                    "records": intervention_count,
                    "bytes": intervention_bytes,
                },
                "proteinRelations": {
                    "manifestSha256": protein_manifest_sha,
                    "recordsSha256": protein_sha,
                    "records": protein_count,
                    "bytes": protein_bytes,
                },
            },
            "limitations": [
                "identity-only output; no quantitative proteome value was parsed or emitted",
                "retired and merged identities are quarantined without redirects",
                "one-to-many protein relations remain relations and never select one gene",
            ],
        }
        _write_json(staging / "audit.json", audit)
        audit_sha = _sha256(staging / "audit.json")
        staging.replace(destination_path)

    return {
        "audit": audit,
        "auditSha256": audit_sha,
        "inventoryManifestSha256": inventory_manifest_sha,
        "proteinManifestSha256": protein_manifest_sha,
        **counts,
    }
