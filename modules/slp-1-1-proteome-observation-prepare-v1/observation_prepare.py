"""Leakage-safe normalization of the pinned yeast knockout proteome.

This module is intentionally source-specific and architecture-neutral.  It
converts only the pretraining-eligible knockout columns and documented HIS3
controls.  Validation, final, quarantined, and analytical-QC values are never
converted to numbers.  Static features and model-facing query tensors belong
to a later, separately versioned corpus composer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
import csv
import hashlib
import json
import math
import os
import platform
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence
import zipfile

import numpy as np


SOURCE_SCHEMA = "slp.source-observation-archive/v1"
BASAL_SCHEMA = "slp.basal-control-profile/v1"
AUDIT_SCHEMA = "slp.proteome-observation-preparation-audit/v1"
INTERVENTION_INVENTORY_SCHEMA = "slp.intervention-identity-inventory/v1"
INTERVENTION_RECORD_SCHEMA = "slp.intervention-identity-record/v1"
PROTEIN_INVENTORY_SCHEMA = "slp.proteome-protein-relation-inventory/v1"
PROTEIN_RECORD_SCHEMA = "slp.proteome-protein-relation/v1"
ROSTER_SCHEMA = "slp.held-intervention-roster-report/v1"
CURRENT_ORF_SCHEMA = "slp.sgd-current-orf/v1"
MAPPING_MANIFEST_SCHEMA = "slp.sgd-stable-id-mapping/v1"
SOURCE_ID = "mendeley:w8jtmnszd9.2"
SOURCE_RELEASE = "10.17632/w8jtmnszd9.2"
NCBI_TAXON = 4932
MAPPING_ID = "slp-sgd-map:2026-08-28-object-set-v1"
MAPPING_SHA256 = "6fd789df6099b78a8842baa8f1d20ab0a3fe77f27ce512ee783444eb2627ef2a"
VALUE_PROTOCOL = "slp-value:mendeley-w8jtmnszd9-v2-log2-relative-intensity-v1"
VALUE_SPACE = "slp-value:log2-batch-corrected-maxlfq-relative-intensity"
VALUE_UNIT = "slp-unit:log2-relative-intensity"
CONTEXT_ID = "slp-context:mendeley-w8jtmnszd9-v2-prototrophic-sm"
CENTERING_GROUP = "slp-center:mendeley-w8jtmnszd9-v2-prototrophic-sm"
ASSIGNMENT_DOMAIN = b"slp-1.1-yeast-global-held-v1\x00"
ASSIGNMENT_DOMAIN_HEX = ASSIGNMENT_DOMAIN.hex()
ROLE_PRETRAIN = "pretrain"
ROLE_VALIDATION = "molecular-validation"
ROLE_FINAL = "molecular-final"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SGD_CURIE = re.compile(r"^SGD:S[0-9]{9}$")
RESOURCE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
INPUT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,127}$")
SYSTEMATIC_NAME = re.compile(r"^Y[A-P][LR][0-9]{3}[CW](?:-[A-Za-z])?$")
METADATA_COLUMNS = (
    "Filename",
    "Injection nr",
    "Well nr (counted row-wise)",
    "Plate (batch) nr",
    "sampletype",
    "ORF",
)
SHARD_RECORDS = 512


class ProteomeObservationError(ValueError):
    """Raised when source, identity, partition, or numerical contracts drift."""


@dataclass(frozen=True)
class FileSpec:
    name: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class DatasetContract:
    name: str
    revision: str
    manifest_digest: str


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
class SourceContract:
    raw_files: tuple[FileSpec, ...]
    intervention_manifest_sha256: str
    protein_manifest_sha256: str
    protein_records_sha256: str
    roster_sha256: str
    coverage_sha256: str
    current_orfs_sha256: str
    mapping_manifest_sha256: str
    mapping_id: str = MAPPING_ID
    mapping_sha256: str = MAPPING_SHA256
    source_id: str = SOURCE_ID
    source_release: str = SOURCE_RELEASE
    minimum_control_fraction: float = 0.8

    def __post_init__(self) -> None:
        if not self.raw_files or len({item.name for item in self.raw_files}) != len(self.raw_files):
            raise ProteomeObservationError("raw file contract must be non-empty and unique")
        for item in self.raw_files:
            if not item.name or item.bytes < 0 or SHA256.fullmatch(item.sha256) is None:
                raise ProteomeObservationError("raw file contract is invalid")
        for value in (
            self.intervention_manifest_sha256,
            self.protein_manifest_sha256,
            self.protein_records_sha256,
            self.roster_sha256,
            self.coverage_sha256,
            self.current_orfs_sha256,
            self.mapping_manifest_sha256,
            self.mapping_sha256,
        ):
            if SHA256.fullmatch(value) is None:
                raise ProteomeObservationError("source contract contains an invalid SHA-256")
        if not 0.0 < self.minimum_control_fraction <= 1.0:
            raise ProteomeObservationError("minimum control fraction must be in (0, 1]")


@dataclass(frozen=True)
class ExpectedCounts:
    metadata_rows: int
    eligible_rows: int
    eligible_genes: int
    pretrain_records: int
    pretrain_genes: int
    target_values: int
    missing_values: int
    trajectory_genes_sha256: str
    trajectory_gene_set_sha256: str
    protein_readouts: int
    basal_controls: int
    basal_observed_values: int
    basal_supported_readouts: int
    validation_genes: int
    validation_rows: int
    final_genes: int
    final_rows: int
    quarantine_rows: int
    qc_rows: int

    def __post_init__(self) -> None:
        integer_fields = (
            self.metadata_rows,
            self.eligible_rows,
            self.eligible_genes,
            self.pretrain_records,
            self.pretrain_genes,
            self.target_values,
            self.missing_values,
            self.protein_readouts,
            self.basal_controls,
            self.basal_observed_values,
            self.basal_supported_readouts,
            self.validation_genes,
            self.validation_rows,
            self.final_genes,
            self.final_rows,
            self.quarantine_rows,
            self.qc_rows,
        )
        if any(type(value) is not int or value < 0 for value in integer_fields):
            raise ProteomeObservationError("expected counts must be non-negative integers")
        if self.pretrain_records <= 0 or self.pretrain_genes <= 0 or self.protein_readouts <= 0:
            raise ProteomeObservationError("pretrain records, genes, and readouts must be non-empty")
        for value in (self.trajectory_genes_sha256, self.trajectory_gene_set_sha256):
            if SHA256.fullmatch(value) is None:
                raise ProteomeObservationError("expected trajectory digest is invalid")


@dataclass(frozen=True)
class Bounds:
    max_metadata_rows: int = 6_000
    max_readouts: int = 2_000
    max_mapping_records: int = 10_000
    max_jsonl_line_bytes: int = 2_097_152
    max_archive_bytes: int = 2 * 1024**3

    def __post_init__(self) -> None:
        values = (
            ("max_metadata_rows", self.max_metadata_rows, 1, 100_000),
            ("max_readouts", self.max_readouts, 1, 100_000),
            ("max_mapping_records", self.max_mapping_records, 1, 1_000_000),
            ("max_jsonl_line_bytes", self.max_jsonl_line_bytes, 128, 16_777_216),
            ("max_archive_bytes", self.max_archive_bytes, 1_048_576, 64 * 1024**3),
        )
        for name, value, minimum, maximum in values:
            if type(value) is not int or not minimum <= value <= maximum:
                raise ProteomeObservationError(f"{name} is outside its bound")


@dataclass(frozen=True)
class Sample:
    filename: str
    injection: int
    well: int
    plate: str
    sample_type: str
    raw_orf: str
    metadata_row: int
    matrix_column: int
    action_id: str | None


@dataclass(frozen=True)
class PreparationProvenance:
    datasets: Mapping[str, PinnedDataset]
    artifacts: Mapping[str, LiteralArtifact]


PRODUCTION_SOURCE_CONTRACT = SourceContract(
    raw_files=(
        FileSpec("Detection_of_KO_proteins.csv", 30_260, "ca7c8f2ac33272df3763807add7b8982b8a8b52d4276bd929a61ecf19e0ae405"),
        FileSpec("summary_fileupload.pdf", 65_558, "4078289dc86dd6b526d9b0c963e6df61d53acdfdf6260abdeae307588623f828"),
        FileSpec("yeast5k_metadata.csv", 377_047, "48864282c82d516ae929dc87aff7fae9e05e9b922e316c001f3d29dce0ff878b"),
        FileSpec("yeast5k_noimpute_wide.csv", 167_754_298, "69a9df05b6db011f595a4e0b3ce25c1cc247f22cbdd066c79e6da9a706aa1df9"),
    ),
    intervention_manifest_sha256="dd683a2585a15377282e669f61dce38c44ea9d3d9d55be71b24842048c05f3e5",
    protein_manifest_sha256="8d559638f48ee4516f7e6fce9e0248e9a1762d58803fe2ed761eff8734f45f86",
    protein_records_sha256="c72996b4ddc6870a3ab722060eef2fa2747fa9dd121d3e70514dd196c5283b8d",
    roster_sha256="c27eb11a20f593235131f28fc29d8fbd69735f8a0aea88736104850bb875117a",
    coverage_sha256="c746218cbe5a8312e4d00f771d2155ab902d33795381b8c14ada1f9a876e1cbf",
    current_orfs_sha256="df7b717cad88dc3672f72f8148f6a9132d12abe6ba020b220b091a8da8f7004d",
    mapping_manifest_sha256="570557ab1201913a18de9790f8adc5ee2e3cb56c6bb0e8d588fe43660c0214e1",
)

PRODUCTION_EXPECTED_COUNTS = ExpectedCounts(
    metadata_rows=5_476,
    eligible_rows=4_623,
    eligible_genes=4_476,
    pretrain_records=3_811,
    pretrain_genes=3_679,
    target_values=6_865_493,
    missing_values=184_857,
    trajectory_genes_sha256="a37fbd5ba56ba4f38cf4ec0655d7dd9734e4727e77f68064739c24d025d3b7e1",
    trajectory_gene_set_sha256="f6083da5b795d5653e630d41758e52855ab1e931d9a6311a1b7ae7350b59b838",
    protein_readouts=1_850,
    basal_controls=388,
    basal_observed_values=701_619,
    basal_supported_readouts=1_843,
    validation_genes=529,
    validation_rows=537,
    final_genes=268,
    final_rows=275,
    quarantine_rows=76,
    qc_rows=389,
)

PRODUCTION_DATASETS = {
    "rawProteome": DatasetContract(
        "slp-1-1-proteome-raw-v2",
        "sha256:5392d4df7e962c9f59798b83fbdf8e71cd568b30c78a498702d50fabc059397e",
        "sha256:7f25f6e11d3deb73624d1c59f7aead59aef77641be6a54b8d0ee838e305f2213",
    ),
    "interventionInventory": DatasetContract(
        "slp-1-1-proteome-intervention-inventory-v1",
        "sha256:bd688dffdf4d96c01d4147580b1a8705c2149acadbc843a719537817a74505d9",
        "sha256:a1f5222f3dca31d2ca68ca46a271d39cdca3425a903b5dceb7373481450ada36",
    ),
    "proteinRelations": DatasetContract(
        "slp-1-1-proteome-protein-relations-v1",
        "sha256:acad3427907644f8ab8af38ed36066a6e1148ef92557b727351b0a4fba2b446c",
        "sha256:c159573f4f7a2e41b18930d724dea9fb297452a659bdf6050e4718efc1a6c58a",
    ),
    "heldRoster": DatasetContract(
        "slp-1-1-held-roster-v1",
        "sha256:1b9a4800370a5398bf83e0a636007f466bf6ca5a6232e2ebb8fc64c5beb63450",
        "sha256:f8aac504a2d56fdc9e13cc9b1c9fa87a08ebc7ff2d7036c0b6b135c26d187425",
    ),
}

PRODUCTION_ARTIFACTS = {
    "sgdCurrentOrfs": "sha256:e67f0e8773feae108ecdb687139885e01ca972ff4aec95cd1358b33db1ea1192",
    "sgdMappingManifest": "sha256:c74ea81ce604357b998e5f09130dff85bf8a7a26504b9b2426f8038608c52d9c",
}


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def canonical_json_bytes(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ProteomeObservationError(f"could not read {path.name}") from error
    return digest.hexdigest()


def _prefixed_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or SHA256.fullmatch(value[7:]) is None:
        raise ProteomeObservationError(f"{label} must be a lowercase sha256: digest")
    return value


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ProteomeObservationError(f"{label} must be a non-empty trimmed string")
    return value


def _resource_parts(value: object, label: str) -> tuple[str, str]:
    resource = _nonempty(value, label)
    if not resource.startswith("omf://"):
        raise ProteomeObservationError(f"{label} must be an OMF DatasetSnapshot URI")
    identity, separator, revision = resource.removeprefix("omf://").rpartition("@")
    if not separator:
        raise ProteomeObservationError(f"{label} must contain an exact revision")
    _prefixed_digest(revision, f"{label} revision")
    parts = identity.split("/")
    if (
        len(parts) < 3
        or parts[-2] != "datasetsnapshot"
        or RESOURCE_NAME.fullmatch(parts[-1]) is None
        or any(not part or part in {".", ".."} or any(char.isspace() for char in part) for part in parts)
    ):
        raise ProteomeObservationError(f"{label} must identify a DatasetSnapshot")
    return parts[-1], revision


def _resolved_path(value: object, label: str, *, directory: bool) -> Path:
    requested = value if isinstance(value, Path) else Path(_nonempty(value, label))
    cursor = requested.absolute()
    for _ in range(4):
        if cursor.is_symlink():
            raise ProteomeObservationError(f"{label} materialization must not contain a symlink")
        cursor = cursor.parent
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise ProteomeObservationError(f"{label} does not exist") from error
    if directory != resolved.is_dir() or (not directory and not resolved.is_file()):
        raise ProteomeObservationError(f"{label} has the wrong file type")
    return resolved


def resolve_pinned_dataset_input(
    value: object,
    input_name: str,
    contract: DatasetContract,
) -> PinnedDataset:
    if INPUT_NAME.fullmatch(input_name) is None or not isinstance(value, dict):
        raise ProteomeObservationError(f"{input_name} must be a materialized DatasetSnapshot")
    if set(value) != {"resource", "mode", "path", "manifestDigest"}:
        raise ProteomeObservationError(f"{input_name} has a spoofed DatasetSnapshot shape")
    resource_name, revision = _resource_parts(value["resource"], f"{input_name}.resource")
    expected_resource = f"omf://abiome/slp/datasetsnapshot/{contract.name}@{contract.revision}"
    if value["resource"] != expected_resource or resource_name != contract.name or revision != contract.revision:
        raise ProteomeObservationError(f"{input_name} is not the frozen DatasetSnapshot")
    if value["mode"] != "copy":
        raise ProteomeObservationError(f"{input_name} must be an immutable copy")
    manifest_digest = _prefixed_digest(value["manifestDigest"], f"{input_name}.manifestDigest")
    if manifest_digest != contract.manifest_digest:
        raise ProteomeObservationError(f"{input_name} manifest digest drift")
    root = _resolved_path(value["path"], f"{input_name}.path", directory=True)
    if root.name != resource_name or root.parent.name != input_name or root.parent.parent.name != "inputs":
        raise ProteomeObservationError(f"{input_name}.path is inconsistent with OMF materialization")
    return PinnedDataset(root, str(value["resource"]), revision, manifest_digest)


def resolve_literal_artifact(value: object, input_name: str, expected_digest: str) -> LiteralArtifact:
    if INPUT_NAME.fullmatch(input_name) is None or not isinstance(value, dict):
        raise ProteomeObservationError(f"{input_name} must be a literal OMF artifact")
    if set(value) != {"resource", "kind", "artifacts", "paths", "path"} or value["kind"] != "artifact":
        raise ProteomeObservationError(f"{input_name} has a spoofed artifact shape")
    artifacts, paths = value["artifacts"], value["paths"]
    if not isinstance(artifacts, dict) or set(artifacts) != {"payload"}:
        raise ProteomeObservationError(f"{input_name}.artifacts must contain only payload")
    if not isinstance(paths, dict) or set(paths) != {"payload"}:
        raise ProteomeObservationError(f"{input_name}.paths must contain only payload")
    digest = _prefixed_digest(artifacts["payload"], f"{input_name} artifact")
    if digest != expected_digest or value["resource"] != f"artifact:{digest}":
        raise ProteomeObservationError(f"{input_name} does not match its frozen artifact")
    if paths["payload"] != value["path"]:
        raise ProteomeObservationError(f"{input_name}.path is inconsistent with paths.payload")
    path = _resolved_path(value["path"], f"{input_name}.path", directory=False)
    if (
        path.name != "payload"
        or path.parent.name != "payload"
        or path.parent.parent.name != input_name
        or path.parent.parent.parent.name != "inputs"
    ):
        raise ProteomeObservationError(f"{input_name}.path is inconsistent with OMF materialization")
    return LiteralArtifact(path, digest)


def _relative_file(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if (
        relative != posix.as_posix()
        or posix.is_absolute()
        or "\\" in relative
        or ":" in relative
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ProteomeObservationError("relative file path is not canonical")
    cursor = root
    for part in posix.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ProteomeObservationError(f"file must not be a symlink: {relative}")
    try:
        resolved = cursor.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ProteomeObservationError(f"file is missing or escapes its root: {relative}") from error
    if not resolved.is_file():
        raise ProteomeObservationError(f"file is not regular: {relative}")
    return resolved


def _exact_files(root_value: str | Path, expected: set[str], label: str) -> Path:
    root = _resolved_path(root_value, label, directory=True)
    actual = {item.name for item in root.iterdir()}
    if actual != expected:
        raise ProteomeObservationError(f"{label} file set drift; expected={sorted(expected)}, actual={sorted(actual)}")
    for name in expected:
        _relative_file(root, name)
    return root


def verify_raw_snapshot(root_value: str | Path, specs: Sequence[FileSpec]) -> dict[str, Path]:
    root = _exact_files(root_value, {item.name for item in specs}, "raw proteome snapshot")
    paths: dict[str, Path] = {}
    for spec in sorted(specs, key=lambda item: item.name):
        path = _relative_file(root, spec.name)
        if path.stat().st_size != spec.bytes or _sha256(path) != spec.sha256:
            raise ProteomeObservationError(f"raw file byte or digest drift: {spec.name}")
        paths[spec.name] = path
    return paths


def _read_json(path: Path, label: str, maximum_bytes: int = 4 * 1024 * 1024) -> dict[str, Any]:
    if path.stat().st_size > maximum_bytes:
        raise ProteomeObservationError(f"{label} exceeds its byte bound")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProteomeObservationError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ProteomeObservationError(f"{label} must be a JSON object")
    return value


def _expect_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ProteomeObservationError(f"{label} fields drift; expected={sorted(expected)}, actual={actual}")
    return value


def _jsonl(path: Path, max_line_bytes: int) -> Iterator[tuple[int, dict[str, Any], bytes]]:
    try:
        with path.open("rb") as stream:
            line_number = 0
            while True:
                raw = stream.readline(max_line_bytes + 1)
                if not raw:
                    break
                line_number += 1
                if len(raw) > max_line_bytes:
                    raise ProteomeObservationError(f"{path.name}:{line_number} exceeds max line bytes")
                if not raw.endswith(b"\n") or raw in {b"\n", b"\r\n"} or b"\r" in raw:
                    raise ProteomeObservationError(f"{path.name}:{line_number} is not canonical JSONL")
                try:
                    value = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ProteomeObservationError(f"{path.name}:{line_number} is invalid JSON") from error
                if not isinstance(value, dict):
                    raise ProteomeObservationError(f"{path.name}:{line_number} must be an object")
                if canonical_json_bytes(value) != raw:
                    raise ProteomeObservationError(f"{path.name}:{line_number} is not canonical JSONL")
                yield line_number, value, raw
    except OSError as error:
        raise ProteomeObservationError(f"could not read {path.name}") from error


def _validate_file_reference(root: Path, item: object, label: str) -> tuple[Path, int]:
    value = _expect_keys(item, {"path", "sha256", "records"}, label)
    relative = _nonempty(value["path"], f"{label}.path")
    digest = _nonempty(value["sha256"], f"{label}.sha256")
    if SHA256.fullmatch(digest) is None:
        raise ProteomeObservationError(f"{label}.sha256 is invalid")
    records = value["records"]
    if type(records) is not int or records <= 0:
        raise ProteomeObservationError(f"{label}.records must be positive")
    path = _relative_file(root, relative)
    if _sha256(path) != digest:
        raise ProteomeObservationError(f"{label} content digest drift")
    return path, records


def load_mapping(
    current_path: Path,
    manifest_path: Path,
    contract: SourceContract,
    bounds: Bounds,
) -> dict[str, tuple[str, ...]]:
    if _sha256(current_path) != contract.current_orfs_sha256:
        raise ProteomeObservationError("current ORF content digest drift")
    if _sha256(manifest_path) != contract.mapping_manifest_sha256:
        raise ProteomeObservationError("mapping manifest content digest drift")
    manifest = _read_json(manifest_path, "mapping manifest", 1_048_576)
    if (
        manifest.get("schema") != MAPPING_MANIFEST_SCHEMA
        or manifest.get("identityMappingId") != contract.mapping_id
        or manifest.get("identityMappingSha256") != contract.mapping_sha256
        or manifest.get("ncbiTaxon") != NCBI_TAXON
    ):
        raise ProteomeObservationError("mapping manifest identity contract drift")
    basis = manifest.get("digestBasis")
    if not isinstance(basis, dict) or hashlib.sha256(canonical_json_bytes(basis)).hexdigest() != contract.mapping_sha256:
        raise ProteomeObservationError("mapping digest basis does not reproduce the frozen identity digest")
    outputs = basis.get("outputFiles")
    matched = [item for item in outputs if isinstance(item, dict) and item.get("name") == "current-orfs.jsonl"] if isinstance(outputs, list) else []
    if len(matched) != 1 or set(matched[0]) != {"name", "records", "bytes", "sha256"}:
        raise ProteomeObservationError("mapping manifest current-ORF output contract drift")
    output = matched[0]
    if output["sha256"] != contract.current_orfs_sha256 or output["bytes"] != current_path.stat().st_size:
        raise ProteomeObservationError("current ORF file is not bound by the mapping manifest")
    by_systematic: dict[str, set[str]] = defaultdict(set)
    seen_curies: set[str] = set()
    records = 0
    expected_keys = {
        "schema", "canonicalSgdCurie", "systematicName", "featureQualifier",
        "ncbiTaxon", "secondaryIdentifiers", "secondaryIdentifiersResolve", "displayMetadata",
    }
    for line_number, record, _ in _jsonl(current_path, bounds.max_jsonl_line_bytes):
        records += 1
        if records > bounds.max_mapping_records:
            raise ProteomeObservationError("current ORF mapping exceeds its record bound")
        _expect_keys(record, expected_keys, f"current ORF line {line_number}")
        curie, systematic = record["canonicalSgdCurie"], record["systematicName"]
        if (
            record["schema"] != CURRENT_ORF_SCHEMA
            or record["ncbiTaxon"] != NCBI_TAXON
            or not isinstance(curie, str)
            or SGD_CURIE.fullmatch(curie) is None
            or not isinstance(systematic, str)
            or SYSTEMATIC_NAME.fullmatch(systematic) is None
            or curie in seen_curies
        ):
            raise ProteomeObservationError(f"current ORF identity drift at line {line_number}")
        seen_curies.add(curie)
        by_systematic[systematic].add(curie)
    if records != output["records"]:
        raise ProteomeObservationError("current ORF record count drift")
    return {key: tuple(sorted(value)) for key, value in by_systematic.items()}


def load_intervention_inventory(
    root_value: str | Path,
    contract: SourceContract,
    bounds: Bounds,
) -> Counter[str]:
    root = _exact_files(root_value, {"inventory.json", "interventions.jsonl"}, "intervention inventory")
    manifest_path = _relative_file(root, "inventory.json")
    if _sha256(manifest_path) != contract.intervention_manifest_sha256:
        raise ProteomeObservationError("intervention inventory manifest content drift")
    manifest = _expect_keys(
        _read_json(manifest_path, "intervention inventory manifest"),
        {
            "schema", "sourceId", "sourceRelease", "ncbiTaxon", "stableIdNamespace",
            "identityMappingId", "identityMappingSha256", "inventoryFormat", "files",
        },
        "intervention inventory manifest",
    )
    if (
        manifest["schema"] != INTERVENTION_INVENTORY_SCHEMA
        or manifest["sourceId"] != contract.source_id
        or manifest["sourceRelease"] != contract.source_release
        or manifest["ncbiTaxon"] != NCBI_TAXON
        or manifest["stableIdNamespace"] != "SGD"
        or manifest["identityMappingId"] != contract.mapping_id
        or manifest["identityMappingSha256"] != contract.mapping_sha256
        or manifest["inventoryFormat"] != INTERVENTION_RECORD_SCHEMA
        or not isinstance(manifest["files"], list)
        or len(manifest["files"]) != 1
    ):
        raise ProteomeObservationError("intervention inventory identity contract drift")
    records_path, expected_records = _validate_file_reference(root, manifest["files"][0], "intervention inventory file")
    counts: Counter[str] = Counter()
    previous = ""
    for line_number, record, _ in _jsonl(records_path, bounds.max_jsonl_line_bytes):
        _expect_keys(record, {"schema", "interventionId", "ncbiTaxon", "qcPassing"}, f"intervention line {line_number}")
        identifier = record["interventionId"]
        if (
            record["schema"] != INTERVENTION_RECORD_SCHEMA
            or record["ncbiTaxon"] != NCBI_TAXON
            or record["qcPassing"] is not True
            or not isinstance(identifier, str)
            or SGD_CURIE.fullmatch(identifier) is None
            or identifier < previous
        ):
            raise ProteomeObservationError(f"intervention inventory drift at line {line_number}")
        previous = identifier
        counts[identifier] += 1
    if sum(counts.values()) != expected_records:
        raise ProteomeObservationError("intervention inventory record count drift")
    return counts


def load_protein_relations(
    root_value: str | Path,
    contract: SourceContract,
    bounds: Bounds,
    current_curies: frozenset[str],
) -> tuple[bytes, tuple[dict[str, Any], ...], dict[str, int]]:
    root = _exact_files(root_value, {"manifest.json", "relations.jsonl"}, "protein relation inventory")
    manifest_path = _relative_file(root, "manifest.json")
    if _sha256(manifest_path) != contract.protein_manifest_sha256:
        raise ProteomeObservationError("protein relation manifest content drift")
    manifest = _expect_keys(
        _read_json(manifest_path, "protein relation manifest"),
        {
            "schema", "sourceId", "sourceRelease", "ncbiTaxon", "identityMappingId",
            "identityMappingSha256", "relationFormat", "files",
        },
        "protein relation manifest",
    )
    if (
        manifest["schema"] != PROTEIN_INVENTORY_SCHEMA
        or manifest["sourceId"] != contract.source_id
        or manifest["sourceRelease"] != contract.source_release
        or manifest["ncbiTaxon"] != NCBI_TAXON
        or manifest["identityMappingId"] != contract.mapping_id
        or manifest["identityMappingSha256"] != contract.mapping_sha256
        or manifest["relationFormat"] != PROTEIN_RECORD_SCHEMA
        or not isinstance(manifest["files"], list)
        or len(manifest["files"]) != 1
    ):
        raise ProteomeObservationError("protein relation identity contract drift")
    records_path, expected_records = _validate_file_reference(root, manifest["files"][0], "protein relation file")
    if _sha256(records_path) != contract.protein_records_sha256:
        raise ProteomeObservationError("protein relation record content drift")
    rows: list[dict[str, Any]] = []
    raw_records = bytearray()
    previous = ""
    accessions: dict[str, int] = {}
    keys = {
        "schema", "proteinId", "sourceAccession", "sourceAccessionType", "ncbiTaxon",
        "currentOrfRelations", "currentOrfRelationCount", "chooseFirstAllowed",
    }
    for line_number, record, raw in _jsonl(records_path, bounds.max_jsonl_line_bytes):
        if len(rows) >= bounds.max_readouts:
            raise ProteomeObservationError("protein relations exceed maxReadouts")
        _expect_keys(record, keys, f"protein relation line {line_number}")
        protein_id, accession, relations = record["proteinId"], record["sourceAccession"], record["currentOrfRelations"]
        if (
            record["schema"] != PROTEIN_RECORD_SCHEMA
            or record["ncbiTaxon"] != NCBI_TAXON
            or record["chooseFirstAllowed"] is not False
            or not isinstance(protein_id, str)
            or not protein_id.startswith("UniProtKB:")
            or protein_id < previous
            or not isinstance(accession, str)
            or protein_id != f"UniProtKB:{accession}"
            or accession in accessions
            or record["sourceAccessionType"] != {
                "source": "UniProtKB",
                "type": "UniProtKB ID",
                "namespaceInferred": False,
                "caseNormalization": "none",
            }
            or not isinstance(relations, list)
            or not relations
            or relations != sorted(set(relations))
            or any(not isinstance(item, str) or SGD_CURIE.fullmatch(item) is None for item in relations)
            or any(item not in current_curies for item in relations)
            or record["currentOrfRelationCount"] != len(relations)
        ):
            raise ProteomeObservationError(f"protein relation drift at line {line_number}")
        previous = protein_id
        accessions[accession] = len(rows)
        rows.append(record)
        raw_records.extend(raw)
    if len(rows) != expected_records:
        raise ProteomeObservationError("protein relation record count drift")
    return bytes(raw_records), tuple(rows), accessions


def _held_role(identifier: str) -> tuple[str, str]:
    digest = hashlib.sha256(ASSIGNMENT_DOMAIN + identifier.encode("ascii")).hexdigest()
    bucket = int(digest[:16], 16) % 100
    role = ROLE_FINAL if bucket <= 9 else ROLE_VALIDATION if bucket <= 29 else ROLE_PRETRAIN
    return role, digest


def load_held_roster(root_value: str | Path, contract: SourceContract) -> dict[str, str]:
    root = _exact_files(root_value, {"coverage.json", "held-intervention-roster.tsv"}, "held roster")
    roster_path = _relative_file(root, "held-intervention-roster.tsv")
    coverage_path = _relative_file(root, "coverage.json")
    if _sha256(roster_path) != contract.roster_sha256 or _sha256(coverage_path) != contract.coverage_sha256:
        raise ProteomeObservationError("held roster content digest drift")
    roles: dict[str, str] = {}
    try:
        for line_number, raw in enumerate(roster_path.read_bytes().splitlines(keepends=True), start=1):
            if not raw.endswith(b"\n") or b"\r" in raw:
                raise ProteomeObservationError("held roster is not canonical LF text")
            fields = raw[:-1].decode("ascii").split("\t")
            if len(fields) != 3:
                raise ProteomeObservationError("held roster row shape drift")
            identifier, role, digest = fields
            if identifier in roles or SGD_CURIE.fullmatch(identifier) is None or _held_role(identifier) != (role, digest):
                raise ProteomeObservationError(f"held roster assignment drift at line {line_number}")
            roles[identifier] = role
    except (OSError, UnicodeDecodeError) as error:
        raise ProteomeObservationError("could not parse held roster") from error
    if list(roles) != sorted(roles):
        raise ProteomeObservationError("held roster must be sorted")
    coverage = _read_json(coverage_path, "held roster coverage")
    if (
        coverage.get("schema") != ROSTER_SCHEMA
        or coverage.get("rosterPath") != "held-intervention-roster.tsv"
        or coverage.get("rosterSha256") != contract.roster_sha256
        or coverage.get("intersectionSize") != len(roles)
        or coverage.get("identityMapping") != {"id": contract.mapping_id, "sha256": contract.mapping_sha256}
        or coverage.get("assignment", {}).get("domainHex") != ASSIGNMENT_DOMAIN_HEX
    ):
        raise ProteomeObservationError("held roster coverage contract drift")
    observed_counts = Counter(roles.values())
    if coverage.get("roleCounts") != dict(observed_counts):
        raise ProteomeObservationError("held roster role-count drift")
    return roles


def read_metadata(
    path: Path,
    mapping: Mapping[str, tuple[str, ...]],
    inventory_counts: Counter[str],
    bounds: Bounds,
) -> tuple[list[Sample], Counter[str]]:
    samples: list[Sample] = []
    mapped_counts: Counter[str] = Counter()
    filenames: set[str] = set()
    sample_types: Counter[str] = Counter()
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != METADATA_COLUMNS:
                raise ProteomeObservationError("proteome metadata column contract drift")
            for index, row in enumerate(reader, start=0):
                if index >= bounds.max_metadata_rows:
                    raise ProteomeObservationError("proteome metadata exceeds its row bound")
                if set(row) != set(METADATA_COLUMNS) or any(value is None or value != value.strip() for value in row.values()):
                    raise ProteomeObservationError("proteome metadata row shape or whitespace drift")
                filename = row["Filename"]
                if not filename or filename in filenames:
                    raise ProteomeObservationError("proteome sample filenames must be non-empty and unique")
                filenames.add(filename)
                sample_type, raw_orf = row["sampletype"], row["ORF"]
                if sample_type not in {"ko", "HIS3", "qc"}:
                    raise ProteomeObservationError("unknown proteome sample type")
                if sample_type == "HIS3" and raw_orf != "YOR202W":
                    raise ProteomeObservationError("HIS3 biological control ORF drift")
                if sample_type == "qc" and raw_orf != "qc":
                    raise ProteomeObservationError("analytical-QC ORF marker drift")
                try:
                    injection = int(row["Injection nr"])
                    well = int(row["Well nr (counted row-wise)"])
                except ValueError as error:
                    raise ProteomeObservationError("injection and well must be exact integers") from error
                if injection < 0 or well < 0:
                    raise ProteomeObservationError("injection and well must be non-negative")
                action_id: str | None = None
                if sample_type == "ko":
                    candidates = mapping.get(raw_orf, ())
                    if len(candidates) == 1:
                        action_id = candidates[0]
                        mapped_counts[action_id] += 1
                samples.append(
                    Sample(
                        filename=filename,
                        injection=injection,
                        well=well,
                        plate=row["Plate (batch) nr"],
                        sample_type=sample_type,
                        raw_orf=raw_orf,
                        metadata_row=index + 2,
                        matrix_column=index + 1,
                        action_id=action_id,
                    )
                )
                sample_types[sample_type] += 1
    except OSError as error:
        raise ProteomeObservationError("could not parse proteome metadata") from error
    if mapped_counts != inventory_counts:
        raise ProteomeObservationError("raw sample-to-SGD mapping does not exactly reproduce the admitted intervention inventory")
    return samples, sample_types


def _decode_observed(token: str, label: str) -> float | None:
    if token == "NA":
        return None
    if not token or token != token.strip():
        raise ProteomeObservationError(f"{label} is neither literal NA nor a trimmed number")
    try:
        value = float(token)
    except ValueError as error:
        raise ProteomeObservationError(f"{label} is not numeric") from error
    if not math.isfinite(value) or value <= 0.0:
        raise ProteomeObservationError(f"{label} must be finite and strictly positive")
    transformed = math.log2(value)
    if not math.isfinite(transformed):
        raise ProteomeObservationError(f"{label} log2 transform is not finite")
    return transformed


def decode_matrix(
    path: Path,
    samples: Sequence[Sample],
    selected: Sequence[Sample],
    controls: Sequence[Sample],
    accession_to_index: Mapping[str, int],
    matrix: np.memmap,
    bounds: Bounds,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    selected_columns = [sample.matrix_column for sample in selected]
    control_columns = [sample.matrix_column for sample in controls]
    target_values = 0
    basal_values = np.zeros(len(accession_to_index), dtype="<f4")
    basal_counts = np.zeros(len(accession_to_index), dtype="<i4")
    seen: set[str] = set()
    try:
        old_limit = csv.field_size_limit()
        csv.field_size_limit(max(old_limit, bounds.max_jsonl_line_bytes))
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            expected_header = ["Protein.Group", *(sample.filename for sample in samples)]
            if header != expected_header:
                raise ProteomeObservationError("wide matrix header does not exactly match metadata filename order")
            for source_row, row in enumerate(reader, start=2):
                if source_row - 2 >= bounds.max_readouts:
                    raise ProteomeObservationError("wide matrix exceeds maxReadouts")
                if len(row) != len(expected_header):
                    raise ProteomeObservationError(f"wide matrix row shape drift at source row {source_row}")
                accession = row[0]
                if accession not in accession_to_index or accession in seen:
                    raise ProteomeObservationError(f"wide matrix readout identity drift at source row {source_row}")
                seen.add(accession)
                readout_index = accession_to_index[accession]
                target_row = np.empty(len(selected_columns), dtype="<f4")
                target_row.fill(np.nan)
                for selected_index, column in enumerate(selected_columns):
                    decoded = _decode_observed(row[column], f"pretrain cell row={source_row}, column={column + 1}")
                    if decoded is not None:
                        target_row[selected_index] = np.float32(decoded)
                        target_values += 1
                matrix[:, readout_index] = target_row
                control_transformed: list[float] = []
                for column in control_columns:
                    decoded = _decode_observed(row[column], f"HIS3 cell row={source_row}, column={column + 1}")
                    if decoded is not None:
                        control_transformed.append(decoded)
                basal_counts[readout_index] = len(control_transformed)
                if control_transformed:
                    basal_values[readout_index] = np.float32(math.fsum(control_transformed) / len(control_transformed))
    except (OSError, csv.Error) as error:
        raise ProteomeObservationError("could not parse the wide proteome matrix") from error
    finally:
        try:
            csv.field_size_limit(old_limit)
        except UnboundLocalError:
            pass
    if seen != set(accession_to_index):
        raise ProteomeObservationError("wide matrix does not exactly cover the admitted protein relations")
    return target_values, len(selected) * len(accession_to_index) - target_values, basal_values, basal_counts


def _fixed_strings(values: Sequence[str]) -> np.ndarray:
    width = max(1, *(len(value) for value in values))
    return np.asarray(values, dtype=f"<U{width}")


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name in sorted(arrays):
            array = np.ascontiguousarray(arrays[name])
            buffer = BytesIO()
            np.save(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, buffer.getvalue())


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _write_tar(source: Path, destination: Path, prefix: str) -> None:
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists() or destination.exists():
        raise ProteomeObservationError(f"refusing to overwrite output archive: {destination.name}")
    with tarfile.open(partial, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            relative = path.relative_to(source).as_posix()
            info = tarfile.TarInfo(f"{prefix}/{relative}")
            info.size = path.stat().st_size
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    partial.replace(destination)


def _shard_arrays(samples: Sequence[Sample], matrix: np.memmap, plate_index: Mapping[str, int]) -> dict[str, np.ndarray]:
    indptr = [0]
    indices: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for row_index in range(len(samples)):
        row = np.asarray(matrix[row_index])
        observed = np.flatnonzero(np.isfinite(row)).astype("<i4", copy=False)
        indices.append(observed)
        values.append(row[observed].astype("<f4", copy=False))
        indptr.append(indptr[-1] + len(observed))
    return {
        "action_id": _fixed_strings([str(sample.action_id) for sample in samples]),
        "centering_group": _fixed_strings([CENTERING_GROUP] * len(samples)),
        "injection_index": np.asarray([sample.injection for sample in samples], dtype="<i2"),
        "matrix_column": np.asarray([sample.matrix_column for sample in samples], dtype="<i4"),
        "metadata_row": np.asarray([sample.metadata_row for sample in samples], dtype="<i4"),
        "observation_unit_id": _fixed_strings([f"slp-unit:mendeley-w8jtmnszd9.2:metadata-row-{sample.metadata_row:05d}" for sample in samples]),
        "perturbation_id": _fixed_strings([f"slp-perturbation:{sample.action_id}" for sample in samples]),
        "plate_index": np.asarray([plate_index[sample.plate] for sample in samples], dtype="<i2"),
        "record_id": _fixed_strings([f"slp-record:mendeley-w8jtmnszd9.2:metadata-row-{sample.metadata_row:05d}" for sample in samples]),
        "replicate_id": _fixed_strings([f"slp-replicate:mendeley-w8jtmnszd9.2:metadata-row-{sample.metadata_row:05d}" for sample in samples]),
        "species_taxon": np.full(len(samples), NCBI_TAXON, dtype="<i4"),
        "target_indptr": np.asarray(indptr, dtype="<i8"),
        "target_readout_index": np.concatenate(indices) if indices else np.empty(0, dtype="<i4"),
        "target_value": np.concatenate(values) if values else np.empty(0, dtype="<f4"),
        "well_index": np.asarray([sample.well for sample in samples], dtype="<i2"),
    }


def _file_ref(
    path: Path,
    *,
    declared_path: str | None = None,
    records: int | None = None,
    values: int | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "path": declared_path if declared_path is not None else path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }
    if records is not None:
        result["records"] = records
    if values is not None:
        result["values"] = values
    return result


def _dataset_identity(item: PinnedDataset) -> dict[str, str]:
    return {"resource": item.resource, "revision": item.revision, "manifestDigest": item.manifest_digest}


def _artifact_identity(item: LiteralArtifact) -> dict[str, str]:
    return {"resource": f"artifact:{item.artifact_manifest_digest}", "manifestDigest": item.artifact_manifest_digest}


def _runtime_identity() -> dict[str, str]:
    return {
        "pythonImplementation": platform.python_implementation(),
        "pythonVersion": platform.python_version(),
        "numpyVersion": np.__version__,
    }


def _source_identity(contract: SourceContract) -> dict[str, object]:
    return {
        "id": contract.source_id,
        "immutableRelease": contract.source_release,
        "rawFiles": [
            {"name": spec.name, "bytes": spec.bytes, "sha256": spec.sha256}
            for spec in sorted(contract.raw_files, key=lambda item: item.name)
        ],
    }


def _identity_provenance(
    contract: SourceContract,
    provenance: PreparationProvenance | None,
) -> dict[str, object]:
    datasets = (
        {}
        if provenance is None
        else {name: _dataset_identity(value) for name, value in sorted(provenance.datasets.items())}
    )
    artifacts = (
        {}
        if provenance is None
        else {name: _artifact_identity(value) for name, value in sorted(provenance.artifacts.items())}
    )
    return {
        "mappingId": contract.mapping_id,
        "mappingSha256": contract.mapping_sha256,
        "interventionManifestSha256": contract.intervention_manifest_sha256,
        "proteinManifestSha256": contract.protein_manifest_sha256,
        "proteinRecordsSha256": contract.protein_records_sha256,
        "datasets": datasets,
        "artifacts": artifacts,
    }


def _partition_identity(
    contract: SourceContract,
    validation_genes: int,
    final_genes: int,
) -> dict[str, object]:
    return {
        "rosterSha256": contract.roster_sha256,
        "coverageSha256": contract.coverage_sha256,
        "assignmentDomainHex": ASSIGNMENT_DOMAIN_HEX,
        "bucketRule": "int(first-16-lowercase-hex,16) mod 100",
        "protectedValidationGenes": validation_genes,
        "protectedFinalGenes": final_genes,
    }


def _measurement_identity() -> dict[str, object]:
    return {
        "protocolId": VALUE_PROTOCOL,
        "valueSpace": VALUE_SPACE,
        "unit": VALUE_UNIT,
        "inputQuantity": "positive-batch-corrected-MaxLFQ-relative-intensity",
        "transform": "log2-no-pseudocount",
        "missingness": "literal-NA-is-unobserved-and-omitted-from-CSR",
        "implicitZero": False,
        "additionalCentering": "none",
    }


def _parse_canonical_jsonl_payload(
    payload: bytes,
    label: str,
    max_line_bytes: int,
) -> list[dict[str, Any]]:
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise ProteomeObservationError(f"{label} is not canonical LF JSONL")
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(payload.splitlines(keepends=True), start=1):
        if len(raw) > max_line_bytes:
            raise ProteomeObservationError(f"{label}:{line_number} exceeds max line bytes")
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProteomeObservationError(f"{label}:{line_number} is invalid JSON") from error
        if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
            raise ProteomeObservationError(f"{label}:{line_number} is not canonical JSONL")
        records.append(value)
    return records


def _validate_readout_dictionary(
    payload: bytes,
    expected_records: int,
    bounds: Bounds,
) -> tuple[str, ...]:
    records = _parse_canonical_jsonl_payload(
        payload, "readout dictionary", bounds.max_jsonl_line_bytes
    )
    if len(records) != expected_records:
        raise ProteomeObservationError("readout dictionary record count drift")
    expected_fields = {
        "schema", "proteinId", "sourceAccession", "sourceAccessionType", "ncbiTaxon",
        "currentOrfRelations", "currentOrfRelationCount", "chooseFirstAllowed",
    }
    identifiers: list[str] = []
    for index, record in enumerate(records, start=1):
        _expect_keys(record, expected_fields, f"readout dictionary line {index}")
        protein_id = record["proteinId"]
        accession = record["sourceAccession"]
        relations = record["currentOrfRelations"]
        if (
            record["schema"] != PROTEIN_RECORD_SCHEMA
            or record["ncbiTaxon"] != NCBI_TAXON
            or record["chooseFirstAllowed"] is not False
            or not isinstance(protein_id, str)
            or not isinstance(accession, str)
            or protein_id != f"UniProtKB:{accession}"
            or record["sourceAccessionType"] != {
                "source": "UniProtKB",
                "type": "UniProtKB ID",
                "namespaceInferred": False,
                "caseNormalization": "none",
            }
            or not isinstance(relations, list)
            or not relations
            or relations != sorted(set(relations))
            or any(not isinstance(value, str) or SGD_CURIE.fullmatch(value) is None for value in relations)
            or record["currentOrfRelationCount"] != len(relations)
        ):
            raise ProteomeObservationError(f"readout dictionary identity drift at line {index}")
        identifiers.append(protein_id)
    if identifiers != sorted(set(identifiers)):
        raise ProteomeObservationError("readout dictionary must be strictly sorted and unique")
    return tuple(identifiers)


def _validate_trajectory_genes(payload: bytes, expected_records: int) -> tuple[str, ...]:
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise ProteomeObservationError("trajectory genes are not canonical LF text")
    try:
        identifiers = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ProteomeObservationError("trajectory genes are not ASCII") from error
    if (
        len(identifiers) != expected_records
        or identifiers != sorted(set(identifiers))
        or any(SGD_CURIE.fullmatch(value) is None for value in identifiers)
    ):
        raise ProteomeObservationError("trajectory gene identity/count/order drift")
    return tuple(identifiers)


def _validate_shard_bytes(
    payload: bytes,
    records: int,
    readouts: int,
    expected_values: int,
    plate_count: int,
) -> dict[str, set[object]]:
    required = {
        "action_id", "centering_group", "injection_index", "matrix_column", "metadata_row",
        "observation_unit_id", "perturbation_id", "plate_index", "record_id", "replicate_id",
        "species_taxon", "target_indptr", "target_readout_index", "target_value", "well_index",
    }
    try:
        with np.load(BytesIO(payload), allow_pickle=False) as loaded:
            if set(loaded.files) != required:
                raise ProteomeObservationError("observation shard array set drift")
            arrays = {name: loaded[name] for name in loaded.files}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ProteomeObservationError("observation shard is not a safe NPZ") from error
    row_arrays = required - {"target_indptr", "target_readout_index", "target_value"}
    if any(arrays[name].ndim != 1 or len(arrays[name]) != records for name in row_arrays):
        raise ProteomeObservationError("observation shard row-array shape drift")
    if any(arrays[name].dtype.kind not in {"U"} for name in {"action_id", "centering_group", "observation_unit_id", "perturbation_id", "record_id", "replicate_id"}):
        raise ProteomeObservationError("observation shard identifiers must be fixed-width strings")
    expected_dtypes = {
        "injection_index": np.dtype("<i2"), "well_index": np.dtype("<i2"), "plate_index": np.dtype("<i2"),
        "matrix_column": np.dtype("<i4"), "metadata_row": np.dtype("<i4"), "species_taxon": np.dtype("<i4"),
        "target_indptr": np.dtype("<i8"), "target_readout_index": np.dtype("<i4"), "target_value": np.dtype("<f4"),
    }
    if any(arrays[name].dtype != dtype for name, dtype in expected_dtypes.items()):
        raise ProteomeObservationError("observation shard dtype drift")
    indptr, indices, values = arrays["target_indptr"], arrays["target_readout_index"], arrays["target_value"]
    if (
        indptr.ndim != 1
        or len(indptr) != records + 1
        or indptr[0] != 0
        or np.any(indptr[1:] < indptr[:-1])
        or indptr[-1] != len(indices)
        or len(indices) != len(values)
        or len(values) != expected_values
        or np.any(indices < 0)
        or np.any(indices >= readouts)
        or not np.isfinite(values).all()
    ):
        raise ProteomeObservationError("observation shard CSR contract drift")
    for start, stop in zip(indptr[:-1], indptr[1:]):
        if np.any(indices[start + 1 : stop] <= indices[start : stop - 1]):
            raise ProteomeObservationError("observation shard CSR indices are not strictly sorted")
    actions = arrays["action_id"].tolist()
    record_ids = arrays["record_id"].tolist()
    observation_ids = arrays["observation_unit_id"].tolist()
    replicate_ids = arrays["replicate_id"].tolist()
    perturbation_ids = arrays["perturbation_id"].tolist()
    metadata_rows = arrays["metadata_row"].tolist()
    matrix_columns = arrays["matrix_column"].tolist()
    if (
        np.any(arrays["species_taxon"] != NCBI_TAXON)
        or np.any(arrays["injection_index"] < 0)
        or np.any(arrays["well_index"] < 0)
        or np.any(arrays["plate_index"] < 0)
        or np.any(arrays["plate_index"] >= plate_count)
        or any(SGD_CURIE.fullmatch(value) is None for value in actions)
        or any(value != CENTERING_GROUP for value in arrays["centering_group"].tolist())
    ):
        raise ProteomeObservationError("observation shard row identity or covariate drift")
    for index, metadata_row in enumerate(metadata_rows):
        expected_suffix = f"metadata-row-{metadata_row:05d}"
        if (
            record_ids[index] != f"slp-record:mendeley-w8jtmnszd9.2:{expected_suffix}"
            or observation_ids[index] != f"slp-unit:mendeley-w8jtmnszd9.2:{expected_suffix}"
            or replicate_ids[index] != f"slp-replicate:mendeley-w8jtmnszd9.2:{expected_suffix}"
            or perturbation_ids[index] != f"slp-perturbation:{actions[index]}"
        ):
            raise ProteomeObservationError("observation shard derived identity drift")
    return {
        "actions": set(actions),
        "recordIds": set(record_ids),
        "observationIds": set(observation_ids),
        "replicateIds": set(replicate_ids),
        "metadataRows": set(metadata_rows),
        "matrixColumns": set(matrix_columns),
    }


def validate_observation_archive(
    path: Path,
    bounds: Bounds = Bounds(),
    *,
    source_contract: SourceContract = PRODUCTION_SOURCE_CONTRACT,
    expected: ExpectedCounts = PRODUCTION_EXPECTED_COUNTS,
    provenance: PreparationProvenance | None = None,
    expected_runtime: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if path.stat().st_size > bounds.max_archive_bytes:
        raise ProteomeObservationError("observation archive exceeds its byte bound")
    try:
        with tarfile.open(path, mode="r:") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if not members or len(members) > 64 or names != sorted(names) or len(names) != len(set(names)):
                raise ProteomeObservationError("observation archive members must be path-sorted and unique")
            if any(
                not member.isfile()
                or member.mode != 0o644
                or member.uid != 0
                or member.gid != 0
                or member.mtime != 0
                or member.uname != ""
                or member.gname != ""
                or member.size < 0
                or member.size > bounds.max_archive_bytes
                or not member.name.startswith("proteome-observations/")
                or ".." in PurePosixPath(member.name).parts
                for member in members
            ):
                raise ProteomeObservationError("observation archive member contract drift")
            if sum(member.size for member in members) > bounds.max_archive_bytes:
                raise ProteomeObservationError("observation archive payload exceeds its byte bound")
            blobs = {member.name.removeprefix("proteome-observations/"): archive.extractfile(member).read() for member in members}
    except (OSError, tarfile.TarError, AttributeError) as error:
        raise ProteomeObservationError("observation archive is invalid") from error
    if "manifest.json" not in blobs or len(blobs["manifest.json"]) > 4 * 1024 * 1024:
        raise ProteomeObservationError("observation archive lacks a bounded manifest")
    try:
        manifest = json.loads(blobs["manifest.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProteomeObservationError("observation manifest is invalid JSON") from error
    required_manifest = {
        "schema", "archiveId", "version", "role", "labelClass", "benchmarkLabelsPresent",
        "source", "identity", "partition", "speciesTaxa", "modalities", "context",
        "measurement", "covariateDefinitions", "assayedPanel", "readoutDictionary",
        "trajectoryGenes", "plateVocabulary", "bounds", "counts", "shards", "runtime",
    }
    _expect_keys(manifest, required_manifest, "observation manifest")
    if (
        canonical_json_bytes(manifest) != blobs["manifest.json"]
        or manifest["schema"] != SOURCE_SCHEMA
        or manifest["archiveId"] != "slp-observations:mendeley-w8jtmnszd9-v2-pretrain-v1"
        or manifest["version"] != "v1"
        or manifest["role"] != ROLE_PRETRAIN
        or manifest["labelClass"] != "molecular"
        or manifest["benchmarkLabelsPresent"] is not False
        or manifest["speciesTaxa"] != [NCBI_TAXON]
        or manifest["modalities"] != ["slp-modality:quantitative-proteome"]
        or manifest["source"] != _source_identity(source_contract)
        or manifest["identity"] != _identity_provenance(source_contract, provenance)
        or manifest["partition"] != _partition_identity(
            source_contract, expected.validation_genes, expected.final_genes
        )
        or manifest["context"] != {"id": CONTEXT_ID, "centeringGroup": CENTERING_GROUP}
        or manifest["measurement"] != _measurement_identity()
        or manifest["covariateDefinitions"] != [
            {"id": "slp-covariate:injection-index", "access": "audit"},
            {"id": "slp-covariate:well-index", "access": "audit"},
            {"id": "slp-covariate:plate-index", "access": "audit"},
        ]
        or manifest["assayedPanel"] != {
            "dictionary": "readouts.jsonl", "readouts": expected.protein_readouts
        }
        or manifest["bounds"] != {
            "maxRecordsPerSourceShard": SHARD_RECORDS, "maxReadouts": bounds.max_readouts
        }
    ):
        raise ProteomeObservationError("observation manifest top-level contract drift")
    runtime = manifest["runtime"]
    if (
        not isinstance(runtime, dict)
        or set(runtime) != {"pythonImplementation", "pythonVersion", "numpyVersion"}
        or any(not isinstance(value, str) or not value or value != value.strip() for value in runtime.values())
        or (expected_runtime is not None and runtime != dict(expected_runtime))
    ):
        raise ProteomeObservationError("observation runtime provenance drift")
    expected_counts = {
        "records": expected.pretrain_records,
        "interventionGenes": expected.pretrain_genes,
        "readouts": expected.protein_readouts,
        "assayedValues": expected.pretrain_records * expected.protein_readouts,
        "observedValues": expected.target_values,
        "missingValues": expected.missing_values,
    }
    if manifest["counts"] != expected_counts or expected.target_values + expected.missing_values != expected_counts["assayedValues"]:
        raise ProteomeObservationError("observation manifest count arithmetic drift")
    plates = manifest["plateVocabulary"]
    if (
        not isinstance(plates, list)
        or not plates
        or plates != sorted(set(plates))
        or any(not isinstance(value, str) or not value or value != value.strip() for value in plates)
    ):
        raise ProteomeObservationError("observation plate vocabulary drift")
    declared = {"manifest.json"}
    referenced_payloads: dict[str, bytes] = {}
    for key in ("readoutDictionary", "trajectoryGenes"):
        ref = manifest[key]
        if (
            not isinstance(ref, dict)
            or set(ref) != {"path", "sha256", "bytes", "records"}
            or type(ref["bytes"]) is not int
            or ref["bytes"] < 0
            or type(ref["records"]) is not int
            or ref["records"] <= 0
            or not isinstance(ref["sha256"], str)
            or SHA256.fullmatch(ref["sha256"]) is None
        ):
            raise ProteomeObservationError(f"observation manifest {key} reference drift")
        data = blobs.get(ref["path"])
        if data is None or len(data) != ref["bytes"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ProteomeObservationError(f"observation manifest {key} content drift")
        declared.add(ref["path"])
        referenced_payloads[key] = data
    if (
        manifest["readoutDictionary"]["path"] != "readouts.jsonl"
        or manifest["readoutDictionary"]["records"] != expected.protein_readouts
        or manifest["readoutDictionary"]["sha256"] != source_contract.protein_records_sha256
        or manifest["trajectoryGenes"]["path"] != "trajectory-genes.txt"
        or manifest["trajectoryGenes"]["records"] != expected.pretrain_genes
    ):
        raise ProteomeObservationError("observation identity-file declaration drift")
    _validate_readout_dictionary(
        referenced_payloads["readoutDictionary"], expected.protein_readouts, bounds
    )
    trajectory_genes = set(
        _validate_trajectory_genes(
            referenced_payloads["trajectoryGenes"], expected.pretrain_genes
        )
    )
    if (
        hashlib.sha256(referenced_payloads["trajectoryGenes"]).hexdigest()
        != expected.trajectory_genes_sha256
        or canonical_sha256(sorted(trajectory_genes)) != expected.trajectory_gene_set_sha256
    ):
        raise ProteomeObservationError("trajectory gene digest drift")
    total_records = total_values = 0
    shards = manifest["shards"]
    if not isinstance(shards, list) or not shards:
        raise ProteomeObservationError("observation manifest requires shards")
    paths: list[str] = []
    all_actions: set[object] = set()
    unique_fields = {
        "recordIds": set(), "observationIds": set(), "replicateIds": set(),
        "metadataRows": set(), "matrixColumns": set(),
    }
    expected_shards = math.ceil(expected.pretrain_records / SHARD_RECORDS)
    if len(shards) != expected_shards:
        raise ProteomeObservationError("observation shard count drift")
    for shard_index, ref in enumerate(shards):
        if (
            not isinstance(ref, dict)
            or set(ref) != {"path", "sha256", "bytes", "records", "values"}
            or ref["path"] != f"shards/shard-{shard_index:05d}.npz"
            or type(ref["bytes"]) is not int
            or ref["bytes"] <= 0
            or type(ref["records"]) is not int
            or ref["records"] <= 0
            or ref["records"] > SHARD_RECORDS
            or type(ref["values"]) is not int
            or ref["values"] < 0
            or not isinstance(ref["sha256"], str)
            or SHA256.fullmatch(ref["sha256"]) is None
        ):
            raise ProteomeObservationError("observation shard reference drift")
        expected_records = min(
            SHARD_RECORDS, expected.pretrain_records - shard_index * SHARD_RECORDS
        )
        if ref["records"] != expected_records:
            raise ProteomeObservationError("observation shard record partition drift")
        data = blobs.get(ref["path"])
        if data is None or len(data) != ref["bytes"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ProteomeObservationError("observation shard content drift")
        shard_identity = _validate_shard_bytes(
            data,
            ref["records"],
            manifest["counts"]["readouts"],
            ref["values"],
            len(plates),
        )
        all_actions.update(shard_identity["actions"])
        for field, accumulated in unique_fields.items():
            observed = shard_identity[field]
            if len(observed) != ref["records"] or accumulated & observed:
                raise ProteomeObservationError(f"observation {field} must be globally unique")
            accumulated.update(observed)
        declared.add(ref["path"])
        paths.append(ref["path"])
        total_records += ref["records"]
        total_values += ref["values"]
    if paths != sorted(paths) or set(blobs) != declared:
        raise ProteomeObservationError("observation archive declared file set drift")
    if total_records != manifest["counts"]["records"] or total_values != manifest["counts"]["observedValues"]:
        raise ProteomeObservationError("observation archive aggregate count drift")
    if all_actions != trajectory_genes:
        raise ProteomeObservationError("observation action identities do not equal trajectory genes")
    return manifest


def validate_basal_archive(
    path: Path,
    bounds: Bounds = Bounds(),
    *,
    source_contract: SourceContract = PRODUCTION_SOURCE_CONTRACT,
    expected: ExpectedCounts = PRODUCTION_EXPECTED_COUNTS,
    provenance: PreparationProvenance | None = None,
    expected_runtime: Mapping[str, str] | None = None,
    expected_control_locator_sha256: str | None = None,
    expected_readout_dictionary_sha256: str | None = None,
) -> dict[str, Any]:
    if path.stat().st_size > bounds.max_archive_bytes:
        raise ProteomeObservationError("basal archive exceeds its byte bound")
    try:
        with tarfile.open(path, mode="r:") as archive:
            members = archive.getmembers()
            if [member.name for member in members] != ["basal-control/basal.json", "basal-control/basal.npz"]:
                raise ProteomeObservationError("basal archive file set/order drift")
            if any(
                not member.isfile()
                or member.mode != 0o644
                or member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
                or member.mtime != 0
                or member.size < 0
                or member.size > bounds.max_archive_bytes
                for member in members
            ):
                raise ProteomeObservationError("basal archive member contract drift")
            manifest_bytes = archive.extractfile(members[0]).read()
            npz_bytes = archive.extractfile(members[1]).read()
    except (OSError, tarfile.TarError, AttributeError) as error:
        raise ProteomeObservationError("basal archive is invalid") from error
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProteomeObservationError("basal manifest is invalid JSON") from error
    required_fields = {
        "schema", "profileId", "source", "identity", "partition", "runtime",
        "speciesTaxon", "contextId", "measurement", "controlPopulation",
        "readoutDictionarySha256", "profileFile", "counts",
    }
    if not isinstance(manifest, dict) or set(manifest) != required_fields:
        raise ProteomeObservationError("basal manifest provenance or identity fields drift")
    runtime = manifest["runtime"]
    control = manifest["controlPopulation"]
    profile = manifest["profileFile"]
    counts = manifest["counts"]
    minimum_controls = math.ceil(source_contract.minimum_control_fraction * expected.basal_controls)
    expected_control = {
        "sampleType": "HIS3",
        "semantics": "his3Delta::kanMX-complemented-biological-WT-control",
        "records": expected.basal_controls,
        "minimumObservedFraction": source_contract.minimum_control_fraction,
        "minimumObservedControls": minimum_controls,
        "locatorSchema": "metadata-row-and-matrix-column/v1",
        "locatorSetSha256": control.get("locatorSetSha256") if isinstance(control, dict) else None,
        "usesKnockoutOutcomes": False,
        "usesAnalyticalQc": False,
    }
    if (
        canonical_json_bytes(manifest) != manifest_bytes
        or manifest["schema"] != BASAL_SCHEMA
        or manifest["profileId"] != "slp-basal:mendeley-w8jtmnszd9-v2-his3-controls-v1"
        or manifest["source"] != _source_identity(source_contract)
        or manifest["identity"] != _identity_provenance(source_contract, provenance)
        or manifest["partition"] != _partition_identity(
            source_contract, expected.validation_genes, expected.final_genes
        )
        or manifest["speciesTaxon"] != NCBI_TAXON
        or manifest["contextId"] != CONTEXT_ID
        or manifest["measurement"] != _measurement_identity()
        or not isinstance(runtime, dict)
        or set(runtime) != {"pythonImplementation", "pythonVersion", "numpyVersion"}
        or any(not isinstance(value, str) or not value or value != value.strip() for value in runtime.values())
        or (expected_runtime is not None and runtime != dict(expected_runtime))
        or not isinstance(control, dict)
        or control != expected_control
        or not isinstance(control.get("locatorSetSha256"), str)
        or SHA256.fullmatch(control["locatorSetSha256"]) is None
        or (
            expected_control_locator_sha256 is not None
            and control["locatorSetSha256"] != expected_control_locator_sha256
        )
        or not isinstance(manifest["readoutDictionarySha256"], str)
        or SHA256.fullmatch(manifest["readoutDictionarySha256"]) is None
        or (
            expected_readout_dictionary_sha256 is not None
            and manifest["readoutDictionarySha256"] != expected_readout_dictionary_sha256
        )
        or not isinstance(profile, dict)
        or set(profile) != {"path", "sha256", "bytes", "records"}
        or profile["path"] != "basal.npz"
        or profile["sha256"] != hashlib.sha256(npz_bytes).hexdigest()
        or type(profile["bytes"]) is not int
        or profile["bytes"] != len(npz_bytes)
        or type(profile["records"]) is not int
        or profile["records"] != expected.protein_readouts
        or counts != {
            "readouts": expected.protein_readouts,
            "supportedReadouts": expected.basal_supported_readouts,
            "observedControlValues": expected.basal_observed_values,
        }
    ):
        raise ProteomeObservationError("basal source provenance, identity, or binding drift")
    try:
        with np.load(BytesIO(npz_bytes), allow_pickle=False) as loaded:
            if set(loaded.files) != {"control_observed", "readout_index", "value", "value_present"}:
                raise ProteomeObservationError("basal array set drift")
            arrays = {name: loaded[name] for name in loaded.files}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ProteomeObservationError("basal profile is not a safe NPZ") from error
    readouts = counts["readouts"]
    if any(array.ndim != 1 or len(array) != readouts for array in arrays.values()):
        raise ProteomeObservationError("basal array shape drift")
    if (
        arrays["readout_index"].dtype != np.dtype("<i4")
        or arrays["control_observed"].dtype != np.dtype("<i4")
        or arrays["value"].dtype != np.dtype("<f4")
        or arrays["value_present"].dtype != np.dtype("bool")
        or not np.array_equal(arrays["readout_index"], np.arange(readouts, dtype="<i4"))
        or not np.isfinite(arrays["value"]).all()
        or np.any(arrays["control_observed"] < 0)
        or np.any(arrays["control_observed"] > expected.basal_controls)
    ):
        raise ProteomeObservationError("basal array value contract drift")
    expected_support = arrays["control_observed"] >= minimum_controls
    if (
        not np.array_equal(arrays["value_present"], expected_support)
        or np.any(arrays["value"][~expected_support] != np.float32(0.0))
        or int(arrays["value_present"].sum()) != counts["supportedReadouts"]
        or int(arrays["control_observed"].sum()) != counts["observedControlValues"]
    ):
        raise ProteomeObservationError("basal support mask or observed-count contract drift")
    return manifest


def build_pretrain_observations(
    raw_root: str | Path,
    intervention_root: str | Path,
    protein_root: str | Path,
    roster_root: str | Path,
    current_orfs_path: str | Path,
    mapping_manifest_path: str | Path,
    destination: str | Path,
    *,
    source_contract: SourceContract = PRODUCTION_SOURCE_CONTRACT,
    expected: ExpectedCounts = PRODUCTION_EXPECTED_COUNTS,
    bounds: Bounds = Bounds(),
    provenance: PreparationProvenance | None = None,
) -> dict[str, Any]:
    destination_path = Path(destination)
    if destination_path.exists():
        raise ProteomeObservationError("refusing to overwrite an observation preparation")
    current_path = _resolved_path(current_orfs_path, "current ORFs", directory=False)
    mapping_path = _resolved_path(mapping_manifest_path, "mapping manifest", directory=False)
    raw_paths = verify_raw_snapshot(raw_root, source_contract.raw_files)
    mapping = load_mapping(current_path, mapping_path, source_contract, bounds)
    current_curies = frozenset(curie for values in mapping.values() for curie in values)
    inventory_counts = load_intervention_inventory(intervention_root, source_contract, bounds)
    readout_bytes, relation_rows, accession_to_index = load_protein_relations(
        protein_root, source_contract, bounds, current_curies
    )
    roster = load_held_roster(roster_root, source_contract)
    samples, sample_types = read_metadata(raw_paths["yeast5k_metadata.csv"], mapping, inventory_counts, bounds)
    if len(samples) != expected.metadata_rows or sum(inventory_counts.values()) != expected.eligible_rows or len(inventory_counts) != expected.eligible_genes:
        raise ProteomeObservationError("metadata or eligible intervention count drift")
    if len(relation_rows) != expected.protein_readouts:
        raise ProteomeObservationError("protein readout count drift")
    eligible = [sample for sample in samples if sample.sample_type == "ko" and sample.action_id is not None]
    selected = [sample for sample in eligible if roster.get(str(sample.action_id)) not in {ROLE_VALIDATION, ROLE_FINAL}]
    controls = [sample for sample in samples if sample.sample_type == "HIS3"]
    qc = [sample for sample in samples if sample.sample_type == "qc"]
    quarantine = [sample for sample in samples if sample.sample_type == "ko" and sample.action_id is None]
    validation = [sample for sample in eligible if roster.get(str(sample.action_id)) == ROLE_VALIDATION]
    final = [sample for sample in eligible if roster.get(str(sample.action_id)) == ROLE_FINAL]
    selected_genes = sorted({str(sample.action_id) for sample in selected})
    validation_genes = {str(sample.action_id) for sample in validation}
    final_genes = {str(sample.action_id) for sample in final}
    control_locators = [
        {"metadataRow": sample.metadata_row, "matrixColumn": sample.matrix_column}
        for sample in controls
    ]
    control_locator_sha256 = canonical_sha256(control_locators)
    if set(selected_genes) & (validation_genes | final_genes):
        raise ProteomeObservationError("held validation/final intervention leaked into pretraining")
    count_tuple = (
        len(selected), len(selected_genes), len(controls), len(validation_genes), len(validation),
        len(final_genes), len(final), len(quarantine), len(qc),
    )
    expected_tuple = (
        expected.pretrain_records, expected.pretrain_genes, expected.basal_controls,
        expected.validation_genes, expected.validation_rows, expected.final_genes,
        expected.final_rows, expected.quarantine_rows, expected.qc_rows,
    )
    if count_tuple != expected_tuple:
        raise ProteomeObservationError(f"frozen role partition drift; observed={count_tuple}, expected={expected_tuple}")
    trajectory_bytes = "".join(identifier + "\n" for identifier in selected_genes).encode("ascii")
    if (
        hashlib.sha256(trajectory_bytes).hexdigest() != expected.trajectory_genes_sha256
        or canonical_sha256(selected_genes) != expected.trajectory_gene_set_sha256
    ):
        raise ProteomeObservationError("pretrain trajectory-gene population digest drift")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination_path.name}.", dir=destination_path.parent))
    try:
        observation_root = staging / "observation-content"
        shard_root = observation_root / "shards"
        basal_root = staging / "basal-content"
        observation_root.mkdir()
        shard_root.mkdir()
        basal_root.mkdir()
        (observation_root / "readouts.jsonl").write_bytes(readout_bytes)
        (observation_root / "trajectory-genes.txt").write_bytes(trajectory_bytes)
        plate_vocabulary = sorted({sample.plate for sample in samples})
        plate_index = {value: index for index, value in enumerate(plate_vocabulary)}
        with tempfile.TemporaryDirectory(prefix="slp-proteome-memmap-") as temporary:
            memmap_path = Path(temporary) / "targets.f32"
            matrix = np.memmap(memmap_path, mode="w+", dtype="<f4", shape=(len(selected), len(relation_rows)))
            try:
                matrix[:] = np.nan
                target_values, missing_values, basal_values, basal_counts = decode_matrix(
                    raw_paths["yeast5k_noimpute_wide.csv"], samples, selected, controls,
                    accession_to_index, matrix, bounds,
                )
                matrix.flush()
                if target_values != expected.target_values or missing_values != expected.missing_values:
                    raise ProteomeObservationError("frozen pretrain observed/missing value count drift")
                minimum_controls = math.ceil(source_contract.minimum_control_fraction * len(controls))
                basal_present = basal_counts >= minimum_controls
                if int(basal_counts.sum()) != expected.basal_observed_values or int(basal_present.sum()) != expected.basal_supported_readouts:
                    raise ProteomeObservationError("frozen HIS3 basal count drift")
                shard_refs: list[dict[str, object]] = []
                for start in range(0, len(selected), SHARD_RECORDS):
                    stop = min(start + SHARD_RECORDS, len(selected))
                    relative = Path("shards") / f"shard-{len(shard_refs):05d}.npz"
                    path = observation_root / relative
                    arrays = _shard_arrays(selected[start:stop], matrix[start:stop], plate_index)
                    _write_deterministic_npz(path, arrays)
                    shard_refs.append(
                        _file_ref(
                            path,
                            declared_path=relative.as_posix(),
                            records=stop - start,
                            values=len(arrays["target_value"]),
                        )
                    )
            finally:
                matrix.flush()
                matrix._mmap.close()
                del matrix
        basal_npz = basal_root / "basal.npz"
        padded_values = np.where(basal_present, basal_values, np.float32(0.0)).astype("<f4")
        _write_deterministic_npz(
            basal_npz,
            {
                "control_observed": basal_counts.astype("<i4", copy=False),
                "readout_index": np.arange(len(relation_rows), dtype="<i4"),
                "value": padded_values,
                "value_present": basal_present.astype(np.bool_, copy=False),
            },
        )
        runtime = _runtime_identity()
        source = _source_identity(source_contract)
        identity = _identity_provenance(source_contract, provenance)
        partition = _partition_identity(
            source_contract, len(validation_genes), len(final_genes)
        )
        measurement = _measurement_identity()
        manifest = {
            "schema": SOURCE_SCHEMA,
            "archiveId": "slp-observations:mendeley-w8jtmnszd9-v2-pretrain-v1",
            "version": "v1",
            "role": ROLE_PRETRAIN,
            "labelClass": "molecular",
            "benchmarkLabelsPresent": False,
            "source": source,
            "identity": identity,
            "partition": partition,
            "runtime": runtime,
            "speciesTaxa": [NCBI_TAXON],
            "modalities": ["slp-modality:quantitative-proteome"],
            "context": {"id": CONTEXT_ID, "centeringGroup": CENTERING_GROUP},
            "measurement": measurement,
            "covariateDefinitions": [
                {"id": "slp-covariate:injection-index", "access": "audit"},
                {"id": "slp-covariate:well-index", "access": "audit"},
                {"id": "slp-covariate:plate-index", "access": "audit"},
            ],
            "assayedPanel": {"dictionary": "readouts.jsonl", "readouts": len(relation_rows)},
            "readoutDictionary": _file_ref(
                observation_root / "readouts.jsonl",
                declared_path="readouts.jsonl",
                records=len(relation_rows),
            ),
            "trajectoryGenes": _file_ref(
                observation_root / "trajectory-genes.txt",
                declared_path="trajectory-genes.txt",
                records=len(selected_genes),
            ),
            "plateVocabulary": plate_vocabulary,
            "bounds": {"maxRecordsPerSourceShard": SHARD_RECORDS, "maxReadouts": bounds.max_readouts},
            "counts": {
                "records": len(selected),
                "interventionGenes": len(selected_genes),
                "readouts": len(relation_rows),
                "assayedValues": len(selected) * len(relation_rows),
                "observedValues": target_values,
                "missingValues": missing_values,
            },
            "shards": shard_refs,
        }
        _write_json(observation_root / "manifest.json", manifest)
        basal_manifest = {
            "schema": BASAL_SCHEMA,
            "profileId": "slp-basal:mendeley-w8jtmnszd9-v2-his3-controls-v1",
            "source": source,
            "identity": identity,
            "partition": partition,
            "runtime": runtime,
            "speciesTaxon": NCBI_TAXON,
            "contextId": CONTEXT_ID,
            "measurement": measurement,
            "controlPopulation": {
                "sampleType": "HIS3",
                "semantics": "his3Delta::kanMX-complemented-biological-WT-control",
                "records": len(controls),
                "minimumObservedFraction": source_contract.minimum_control_fraction,
                "minimumObservedControls": minimum_controls,
                "locatorSchema": "metadata-row-and-matrix-column/v1",
                "locatorSetSha256": control_locator_sha256,
                "usesKnockoutOutcomes": False,
                "usesAnalyticalQc": False,
            },
            "readoutDictionarySha256": hashlib.sha256(readout_bytes).hexdigest(),
            "profileFile": _file_ref(
                basal_npz,
                declared_path="basal.npz",
                records=len(relation_rows),
            ),
            "counts": {
                "readouts": len(relation_rows),
                "supportedReadouts": int(basal_present.sum()),
                "observedControlValues": int(basal_counts.sum()),
            },
        }
        _write_json(basal_root / "basal.json", basal_manifest)
        publication_root = staging / "publication"
        publication_root.mkdir()
        observation_archive = publication_root / "observation-corpus.tar"
        basal_archive = publication_root / "basal-control.tar"
        _write_tar(observation_root, observation_archive, "proteome-observations")
        _write_tar(basal_root, basal_archive, "basal-control")
        validated_manifest = validate_observation_archive(
            observation_archive,
            bounds,
            source_contract=source_contract,
            expected=expected,
            provenance=provenance,
            expected_runtime=runtime,
        )
        validated_basal = validate_basal_archive(
            basal_archive,
            bounds,
            source_contract=source_contract,
            expected=expected,
            provenance=provenance,
            expected_runtime=runtime,
            expected_control_locator_sha256=control_locator_sha256,
            expected_readout_dictionary_sha256=hashlib.sha256(readout_bytes).hexdigest(),
        )
        if validated_manifest["counts"] != manifest["counts"] or validated_basal["counts"] != basal_manifest["counts"]:
            raise ProteomeObservationError("post-write archive validation changed the content contract")
        audit = {
            "schema": AUDIT_SCHEMA,
            "role": ROLE_PRETRAIN,
            "source": source,
            "identity": identity,
            "partition": partition,
            "runtime": runtime,
            "accessBoundary": {
                "metadataColumnsParsed": list(METADATA_COLUMNS),
                "matrixHeaderParsed": True,
                "readoutIdentityColumnParsed": "Protein.Group",
                "numericColumnsConverted": {"pretrain": len(selected), "HIS3-controls": len(controls)},
                "numericColumnsNotConverted": {
                    "molecular-validation": len(validation),
                    "molecular-final": len(final),
                    "quarantine": len(quarantine),
                    "analytical-qc": len(qc),
                },
                "excludedNumericValuesInspectedOrValidated": False,
                "filenameSuffixesUsed": False,
                "knockoutOutcomesUsedForBasal": False,
                "qcOutcomesUsedForBasal": False,
            },
            "counts": {
                "metadataRows": len(samples),
                "sampleTypes": dict(sorted(sample_types.items())),
                "eligibleKnockoutRows": len(eligible),
                "eligibleInterventionGenes": len(inventory_counts),
                "pretrainRecords": len(selected),
                "pretrainInterventionGenes": len(selected_genes),
                "pretrainObservedValues": target_values,
                "pretrainMissingValues": missing_values,
                "protectedValidationGenes": len(validation_genes),
                "protectedValidationRowsNotDecoded": len(validation),
                "protectedFinalGenes": len(final_genes),
                "protectedFinalRowsNotDecoded": len(final),
                "quarantineRowsNotDecoded": len(quarantine),
                "analyticalQcRowsNotDecoded": len(qc),
                "basalControlRows": len(controls),
                "basalControlLocatorSetSha256": control_locator_sha256,
                "basalObservedValues": int(basal_counts.sum()),
                "basalSupportedReadouts": int(basal_present.sum()),
            },
            "outputs": {
                "observationArchiveSha256": _sha256(observation_archive),
                "basalArchiveSha256": _sha256(basal_archive),
                "trajectoryGenesSha256": expected.trajectory_genes_sha256,
                "trajectoryGeneSetSha256": expected.trajectory_gene_set_sha256,
            },
            "limitations": [
                "source-normalized observations contain no static feature vectors or model-facing query tensors",
                "plate, injection, and well identifiers are audit-only",
                "the basal profile is emitted separately and is not subtracted from target values",
                "this preparation is not an admitted slp.corpus/v1.1 training snapshot",
            ],
        }
        _write_json(publication_root / "preparation-audit.json", audit)
        audit_sha256 = _sha256(publication_root / "preparation-audit.json")
        result = {
            "audit": audit,
            "auditSha256": audit_sha256,
            "observationArchiveSha256": audit["outputs"]["observationArchiveSha256"],
            "basalArchiveSha256": audit["outputs"]["basalArchiveSha256"],
            "records": len(selected),
            "interventionGenes": len(selected_genes),
            "targetValues": target_values,
            "missingValues": missing_values,
            "basalControls": len(controls),
            "basalObservedValues": int(basal_counts.sum()),
            "basalSupportedReadouts": int(basal_present.sum()),
            "excludedValidationRows": len(validation),
            "excludedFinalRows": len(final),
            "excludedQuarantineRows": len(quarantine),
            "excludedQcRows": len(qc),
        }
        publication_root.replace(destination_path)
        return result
    except Exception:
        if destination_path.exists():
            for item in destination_path.iterdir():
                item.unlink()
            destination_path.rmdir()
        raise
    finally:
        for item in sorted(staging.rglob("*"), key=lambda value: len(value.parts), reverse=True):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                item.rmdir()
        if staging.exists():
            staging.rmdir()
