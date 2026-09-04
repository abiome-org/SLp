"""Deterministic, species-native SGD stable-identity normalization.

This module intentionally exposes relations rather than performing lookup for a
downstream dataset.  Systematic names are case-sensitive.  Display symbols,
free-text aliases, and lexical accession shapes are never identity evidence.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Iterator, Mapping


MAPPING_SCHEMA = "slp.sgd-stable-id-mapping/v1"
DIGEST_SCHEMA = "slp.sgd-stable-id-mapping-digest/v1"
ORF_SCHEMA = "slp.sgd-current-orf/v1"
EXTERNAL_SCHEMA = "slp.sgd-external-accession-relation/v1"
QUARANTINE_SCHEMA = "slp.sgd-retired-quarantine/v1"
IDENTITY_MAPPING_ID = "slp-sgd-map:2026-08-28-object-set-v1"
NCBI_TAXON = 4932
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SGD_ID = re.compile(r"^S[0-9]{9}$")
RESOURCE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
INPUT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")

OUTPUT_FILES = (
    "current-orfs.jsonl",
    "external-accessions.jsonl",
    "retired-merged-quarantine.jsonl",
)


class SgdMapError(ValueError):
    """Raised when the pinned raw identity snapshot violates its contract."""


@dataclass(frozen=True)
class FileSpec:
    name: str
    bytes: int
    sha256: str
    physical_lines: int | None = None
    data_records: int | None = None
    irregular_records: int | None = None

    def __post_init__(self) -> None:
        if not self.name or Path(self.name).name != self.name:
            raise SgdMapError("pinned file names must be simple basenames")
        if not isinstance(self.bytes, int) or isinstance(self.bytes, bool) or self.bytes < 0:
            raise SgdMapError(f"invalid pinned byte count for {self.name}")
        if not isinstance(self.sha256, str) or SHA256.fullmatch(self.sha256) is None:
            raise SgdMapError(f"invalid pinned SHA-256 for {self.name}")
        for label, value in (
            ("physical_lines", self.physical_lines),
            ("data_records", self.data_records),
            ("irregular_records", self.irregular_records),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise SgdMapError(f"invalid {label} for {self.name}")


PINNED_FILES = (
    FileSpec(
        "SGD_features.tab",
        3_382_715,
        "636b4fc0407dd9f4fe74dceb5f5cd056194623d36a25c620a3c1ec2394af3dcc",
        physical_lines=16_461,
        data_records=16_459,
        irregular_records=2,
    ),
    FileSpec(
        "SGD_features.README",
        1_586,
        "befbd275783dda3772a19a20fadc4b1d62916dfe04e436540d4d935693a49328",
    ),
    FileSpec(
        "dbxref.tab",
        15_365_158,
        "2ff198c4127c226fa965cf514e10d10e2e2e581be228a22f4da69773fd7df5b7",
        physical_lines=267_097,
        data_records=267_097,
        irregular_records=0,
    ),
    FileSpec(
        "dbxref.README",
        3_338,
        "96deaf79c0ae0bc05fdcc7cd42bf31ee34261970ad422476038e7b737250cb2c",
    ),
    FileSpec(
        "deleted_merged_features.tab",
        50_549,
        "df979ff33732eb90220ef93c7ec38f817578dffbd31ebe3eb7978c98dd3f35b2",
        physical_lines=105,
        data_records=100,
        irregular_records=5,
    ),
    FileSpec(
        "deleted_merged_features.README",
        1_024,
        "b68857febe13cb07f6127e50069d80e784f01d4f3e7a48c5119848869eb4ade3",
    ),
)

README_MARKERS = {
    "SGD_features.README": (
        "based on Genome Version R64-5-1",
        "1.   Primary SGDID (mandatory)",
        "2.   Feature type (mandatory)",
        "4.   Feature name (optional)",
        "5.   Standard gene name (optional)",
        "6.   Alias (optional, multiples separated by |)",
        "8.   Secondary SGDID (optional, multiples separated by |)",
        "updated weekly on Friday night",
    ),
    "dbxref.README": (
        "The SGDID is the recommended identifier for features in SGD",
        "1) DBXREF ID",
        "2) DBXREF ID source",
        "3) DBXREF ID type",
        "4) S. cerevisiae feature name",
        "5) SGDID",
        "updated weekly on Friday night",
    ),
    "deleted_merged_features.README": (
        "based on Genome Version R64-5-1",
        "1.  Systematic name (mandatory)",
        "2.  Feature_type|qualifier (mandatory)",
        "7.  Primary SGDID (mandatory)",
        "9.  New systematic name (mandatory for Merged features only)",
        "10. New Primary SGDID (mandatory for Merged features only)",
        "13. Date of action (mandatory)",
        "updated weekly on Friday night",
    ),
}


@dataclass(frozen=True)
class MapBounds:
    max_feature_records: int = 20_000
    max_external_records: int = 300_000
    max_retired_physical_lines: int = 256
    max_line_bytes: int = 4_096
    max_targets_per_external_key: int = 1_024
    max_assertions_per_external_target: int = 64
    max_display_aliases_per_orf: int = 128

    def __post_init__(self) -> None:
        for name, value, minimum, maximum in (
            ("maxFeatureRecords", self.max_feature_records, 1, 50_000),
            ("maxExternalRecords", self.max_external_records, 1, 500_000),
            ("maxRetiredPhysicalLines", self.max_retired_physical_lines, 1, 10_000),
            ("maxLineBytes", self.max_line_bytes, 128, 65_536),
            ("maxTargetsPerExternalKey", self.max_targets_per_external_key, 1, 4_096),
            (
                "maxAssertionsPerExternalTarget",
                self.max_assertions_per_external_target,
                1,
                1_024,
            ),
            ("maxDisplayAliasesPerOrf", self.max_display_aliases_per_orf, 1, 1_024),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not minimum <= value <= maximum
            ):
                raise SgdMapError(f"{name} must be an integer in [{minimum}, {maximum}]")


@dataclass(frozen=True)
class PinnedDatasetInput:
    input_name: str
    path: str
    resource: str
    revision: str
    manifest_digest: str


@dataclass(frozen=True)
class SourceProvenance:
    resource: str
    revision: str
    manifest_digest: str


@dataclass(frozen=True)
class FeatureState:
    orf_records: tuple[dict[str, object], ...]
    current_primary_ids: frozenset[str]
    current_orf_ids: frozenset[str]
    feature_types: Mapping[str, tuple[str, ...]]
    physical_lines: int
    blank_lines: int


@dataclass(frozen=True)
class RetiredState:
    records: tuple[dict[str, object], ...]
    retired_primary_ids: frozenset[str]
    physical_lines: int
    valid_records: int
    irregular_records: int


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_mapping_digest(digest_basis: object) -> str:
    """Return the frozen digest over canonical UTF-8 JSON plus one LF."""
    return hashlib.sha256((canonical_json(digest_basis) + "\n").encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pinned_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or SHA256.fullmatch(value.removeprefix("sha256:")) is None
    ):
        raise SgdMapError(f"{label} must be an admission-pinned SHA-256")
    return value


def _trimmed(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SgdMapError(f"{label} must be a non-empty, exact trimmed string")
    if "\x00" in value:
        raise SgdMapError(f"{label} contains a NUL byte")
    return value


def _dataset_resource(value: object, label: str) -> tuple[str, str]:
    resource = _trimmed(value, label)
    if not resource.startswith("omf://"):
        raise SgdMapError(f"{label} must be an OMF DatasetSnapshot resource URI")
    identity, separator, revision = resource.removeprefix("omf://").rpartition("@")
    if not separator:
        raise SgdMapError(f"{label} must contain an exact resource revision")
    revision = _pinned_digest(revision, f"{label} revision")
    parts = identity.split("/")
    if (
        len(parts) < 3
        or any(
            not part
            or part in {".", ".."}
            or any(character.isspace() for character in part)
            for part in parts
        )
        or parts[-2] != "datasetsnapshot"
        or RESOURCE_NAME.fullmatch(parts[-1]) is None
    ):
        raise SgdMapError(f"{label} kind must be DatasetSnapshot with a valid name")
    return parts[-1], revision


def resolve_pinned_dataset_input(value: object, input_name: str) -> PinnedDatasetInput:
    """Validate OMF's exact copied DatasetSnapshot request object."""
    if INPUT_NAME.fullmatch(input_name) is None:
        raise SgdMapError("input name is not canonical")
    if not isinstance(value, dict):
        raise SgdMapError(f"{input_name} must be a materialized OMF DatasetSnapshot object")
    if set(value) != {"resource", "mode", "path", "manifestDigest"}:
        raise SgdMapError(f"{input_name} has a spoofed materialized DatasetSnapshot shape")
    resource_name, revision = _dataset_resource(value["resource"], f"{input_name}.resource")
    if value["mode"] != "copy":
        raise SgdMapError(f"{input_name} must be an immutable copied DatasetSnapshot")
    manifest_digest = _pinned_digest(value["manifestDigest"], f"{input_name}.manifestDigest")
    path_value = _trimmed(value["path"], f"{input_name}.path")
    requested = Path(path_value)
    if requested.is_symlink():
        raise SgdMapError(f"{input_name}.path must not be a symlink")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise SgdMapError(f"{input_name}.path does not exist") from error
    if not resolved.is_dir():
        raise SgdMapError(f"{input_name}.path must materialize a directory")
    if (
        resolved.name != resource_name
        or resolved.parent.name != input_name
        or resolved.parent.parent.name != "inputs"
    ):
        raise SgdMapError(
            f"{input_name}.path is inconsistent with its input name and DatasetSnapshot resource"
        )
    return PinnedDatasetInput(
        input_name=input_name,
        path=str(resolved),
        resource=value["resource"],
        revision=revision,
        manifest_digest=manifest_digest,
    )


def _snapshot_root(path_value: str | Path) -> Path:
    requested = Path(path_value)
    if requested.is_symlink():
        raise SgdMapError("raw snapshot root must not be a symlink")
    try:
        root = requested.resolve(strict=True)
    except OSError as error:
        raise SgdMapError("raw snapshot root does not exist") from error
    if not root.is_dir():
        raise SgdMapError("raw snapshot root must be a directory")
    return root


def verify_raw_snapshot(
    path_value: str | Path, file_specs: Iterable[FileSpec] = PINNED_FILES
) -> tuple[Path, tuple[dict[str, object], ...]]:
    """Require the exact six-file set and every expected byte and SHA-256."""
    root = _snapshot_root(path_value)
    specs = tuple(file_specs)
    expected = {item.name: item for item in specs}
    if len(expected) != len(specs) or len(expected) != 6:
        raise SgdMapError("raw mapping contract must declare six unique files")
    actual: dict[str, Path] = {}
    for child in root.iterdir():
        if child.is_symlink():
            raise SgdMapError(f"raw snapshot member must not be a symlink: {child.name}")
        if not child.is_file():
            raise SgdMapError(f"raw snapshot member must be a regular file: {child.name}")
        actual[child.name] = child
    if set(actual) != set(expected):
        raise SgdMapError(
            "raw snapshot file set drift; "
            f"missing={sorted(set(expected) - set(actual))}, "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    verified: list[dict[str, object]] = []
    for name in sorted(expected):
        spec = expected[name]
        path = actual[name]
        size = path.stat().st_size
        if size != spec.bytes:
            raise SgdMapError(
                f"raw byte-count drift for {name}: expected {spec.bytes}, got {size}"
            )
        digest = _sha256(path)
        if digest != spec.sha256:
            raise SgdMapError(f"raw SHA-256 drift for {name}")
        verified.append({"name": name, "bytes": size, "sha256": digest})
    return root, tuple(verified)


def _read_physical_lines(path: Path, max_line_bytes: int) -> Iterator[tuple[int, str]]:
    try:
        with path.open("rb") as stream:
            line_number = 0
            while True:
                raw = stream.readline(max_line_bytes + 1)
                if not raw:
                    break
                line_number += 1
                if len(raw) > max_line_bytes:
                    raise SgdMapError(
                        f"physical line exceeds maxLineBytes in {path.name}:{line_number}"
                    )
                if raw.endswith(b"\n"):
                    raw = raw[:-1]
                if raw.endswith(b"\r"):
                    raw = raw[:-1]
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise SgdMapError(
                        f"invalid UTF-8 in {path.name}:{line_number}"
                    ) from error
                if "\x00" in text:
                    raise SgdMapError(f"NUL byte in {path.name}:{line_number}")
                yield line_number, text
    except OSError as error:
        raise SgdMapError(f"could not read {path.name}") from error


def _validate_readmes(root: Path) -> None:
    for name, markers in README_MARKERS.items():
        try:
            text = (root / name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise SgdMapError(f"could not read pinned UTF-8 documentation: {name}") from error
        missing = [marker for marker in markers if marker not in text]
        if missing:
            raise SgdMapError(f"README column contract drift in {name}: missing {missing}")


def _assert_counts(
    spec: FileSpec,
    *,
    physical_lines: int,
    data_records: int,
    irregular_records: int,
) -> None:
    for label, actual, expected in (
        ("physical line", physical_lines, spec.physical_lines),
        ("data record", data_records, spec.data_records),
        ("irregular record", irregular_records, spec.irregular_records),
    ):
        if expected is not None and actual != expected:
            raise SgdMapError(
                f"{label} count drift for {spec.name}: expected {expected}, got {actual}"
            )


def _optional_exact(value: str, label: str) -> str | None:
    if not value:
        return None
    return _trimmed(value, label)


def _pipe_values(value: str, label: str, maximum: int) -> list[str]:
    if not value:
        return []
    values = value.split("|")
    if len(values) > maximum:
        raise SgdMapError(f"{label} exceeds its bounded item count")
    for index, item in enumerate(values):
        _trimmed(item, f"{label}[{index}]")
    return values


def parse_features(path: Path, spec: FileSpec, bounds: MapBounds) -> FeatureState:
    records: list[dict[str, object]] = []
    current_ids: set[str] = set()
    current_orf_ids: set[str] = set()
    systematic_to_primary: dict[str, str] = {}
    types: dict[str, set[str]] = defaultdict(set)
    physical_lines = 0
    blank_lines = 0
    data_records = 0
    for line_number, line in _read_physical_lines(path, bounds.max_line_bytes):
        physical_lines += 1
        if not line:
            blank_lines += 1
            continue
        data_records += 1
        if data_records > bounds.max_feature_records:
            raise SgdMapError("SGD_features.tab exceeds maxFeatureRecords")
        fields = line.split("\t")
        if len(fields) != 16:
            raise SgdMapError(
                f"SGD_features.tab:{line_number} must contain exactly 16 columns"
            )
        primary = _trimmed(fields[0], f"SGD_features.tab:{line_number} primary SGDID")
        if SGD_ID.fullmatch(primary) is None:
            raise SgdMapError(f"malformed primary SGDID in SGD_features.tab:{line_number}")
        feature_type = _trimmed(
            fields[1], f"SGD_features.tab:{line_number} feature type"
        )
        current_ids.add(primary)
        types[primary].add(feature_type)
        if feature_type != "ORF":
            continue
        systematic = _trimmed(
            fields[3], f"SGD_features.tab:{line_number} systematic feature name"
        )
        previous = systematic_to_primary.get(systematic)
        if previous is not None:
            raise SgdMapError(
                "ambiguous exact systematic ORF key in SGD_features.tab: "
                f"{systematic} maps to {previous} and {primary}"
            )
        if primary in current_orf_ids:
            raise SgdMapError(f"duplicate current ORF primary SGDID: {primary}")
        systematic_to_primary[systematic] = primary
        current_orf_ids.add(primary)
        aliases = _pipe_values(
            fields[5],
            f"SGD_features.tab:{line_number} display aliases",
            bounds.max_display_aliases_per_orf,
        )
        secondary = _pipe_values(
            fields[7],
            f"SGD_features.tab:{line_number} secondary identifiers",
            bounds.max_display_aliases_per_orf,
        )
        records.append(
            {
                "schema": ORF_SCHEMA,
                "ncbiTaxon": NCBI_TAXON,
                "canonicalSgdCurie": f"SGD:{primary}",
                "systematicName": systematic,
                "featureQualifier": _optional_exact(
                    fields[2], f"SGD_features.tab:{line_number} feature qualifier"
                ),
                "secondaryIdentifiers": secondary,
                "secondaryIdentifiersResolve": False,
                "displayMetadata": {
                    "standardGeneName": _optional_exact(
                        fields[4], f"SGD_features.tab:{line_number} standard gene name"
                    ),
                    "aliases": aliases,
                    "resolvesIdentity": False,
                },
            }
        )
    _assert_counts(
        spec,
        physical_lines=physical_lines,
        data_records=data_records,
        irregular_records=blank_lines,
    )
    records.sort(key=lambda item: str(item["canonicalSgdCurie"]))
    frozen_types = {
        identifier: tuple(sorted(values)) for identifier, values in sorted(types.items())
    }
    return FeatureState(
        orf_records=tuple(records),
        current_primary_ids=frozenset(current_ids),
        current_orf_ids=frozenset(current_orf_ids),
        feature_types=frozen_types,
        physical_lines=physical_lines,
        blank_lines=blank_lines,
    )


def _retired_record(fields: list[str], line_number: int) -> dict[str, object]:
    systematic = _trimmed(
        fields[0], f"deleted_merged_features.tab:{line_number} systematic name"
    )
    type_status = _trimmed(
        fields[1], f"deleted_merged_features.tab:{line_number} feature type/status"
    )
    feature_type, separator, status = type_status.rpartition("|")
    if not separator or not feature_type or status not in {"Deleted", "Merged"}:
        raise SgdMapError(
            f"invalid retired feature type/status in deleted_merged_features.tab:{line_number}"
        )
    _trimmed(fields[2], f"deleted_merged_features.tab:{line_number} chromosome")
    primary = _trimmed(
        fields[6], f"deleted_merged_features.tab:{line_number} primary SGDID"
    )
    issues: list[str] = []
    retired_curie: str | None = None
    if SGD_ID.fullmatch(primary) is None:
        issues.append("noncanonical-legacy-primary-identifier")
    else:
        retired_curie = f"SGD:{primary}"
    new_systematic = _optional_exact(
        fields[8], f"deleted_merged_features.tab:{line_number} new systematic name"
    )
    new_primary = _optional_exact(
        fields[9], f"deleted_merged_features.tab:{line_number} new primary SGDID"
    )
    if status == "Merged" and (new_systematic is None or new_primary is None):
        raise SgdMapError(
            f"merged row lacks its README-mandatory replacement in line {line_number}"
        )
    if new_primary is not None and SGD_ID.fullmatch(new_primary) is None:
        raise SgdMapError(
            f"malformed replacement SGDID in deleted_merged_features.tab:{line_number}"
        )
    if not fields[11]:
        issues.append("missing-readme-mandatory-annotation-note")
    if not fields[12]:
        issues.append("missing-readme-mandatory-action-date")
    return {
        "schema": QUARANTINE_SCHEMA,
        "recordKind": "retired-or-merged",
        "sourceLine": line_number,
        "ncbiTaxon": NCBI_TAXON,
        "systematicName": systematic,
        "featureType": feature_type,
        "status": status,
        "sourcePrimaryIdentifier": primary,
        "retiredPrimarySgdCurie": retired_curie,
        "secondaryIdentifier": _optional_exact(
            fields[7], f"deleted_merged_features.tab:{line_number} secondary identifier"
        ),
        "reportedReplacement": {
            "systematicName": new_systematic,
            "primarySgdCurie": f"SGD:{new_primary}" if new_primary else None,
            "evidenceOnly": True,
        },
        "description": fields[10],
        "annotationNote": fields[11] or None,
        "actionDate": fields[12] or None,
        "sourceContractIssues": issues,
        "automaticRedirectAllowed": False,
    }


def parse_retired(path: Path, spec: FileSpec, bounds: MapBounds) -> RetiredState:
    records: list[dict[str, object]] = []
    retired_ids: set[str] = set()
    physical_lines = 0
    valid_records = 0
    irregular_records = 0
    seen_primary: set[str] = set()
    for line_number, line in _read_physical_lines(path, bounds.max_line_bytes):
        physical_lines += 1
        if physical_lines > bounds.max_retired_physical_lines:
            raise SgdMapError(
                "deleted_merged_features.tab exceeds maxRetiredPhysicalLines"
            )
        fields = line.split("\t")
        if len(fields) != 13:
            irregular_records += 1
            records.append(
                {
                    "schema": QUARANTINE_SCHEMA,
                    "recordKind": "malformed-source-row",
                    "sourceLine": line_number,
                    "fieldCount": len(fields),
                    "rawLine": line,
                    "reason": "README declares exactly 13 tab-separated columns",
                    "automaticRedirectAllowed": False,
                }
            )
            continue
        record = _retired_record(fields, line_number)
        valid_records += 1
        primary = str(record["sourcePrimaryIdentifier"])
        if primary in seen_primary:
            raise SgdMapError(f"duplicate retired source primary identifier: {primary}")
        seen_primary.add(primary)
        if SGD_ID.fullmatch(primary) is not None:
            retired_ids.add(primary)
        records.append(record)
    _assert_counts(
        spec,
        physical_lines=physical_lines,
        data_records=valid_records,
        irregular_records=irregular_records,
    )
    return RetiredState(
        records=tuple(records),
        retired_primary_ids=frozenset(retired_ids),
        physical_lines=physical_lines,
        valid_records=valid_records,
        irregular_records=irregular_records,
    )


def _target_status(
    primary: str, features: FeatureState, retired: RetiredState
) -> str:
    if primary in features.current_orf_ids:
        return "current-orf"
    if primary in features.current_primary_ids:
        return "current-non-orf"
    if primary in retired.retired_primary_ids:
        return "retired-or-merged"
    return "not-in-current-feature-table"


def parse_external_relations(
    path: Path,
    spec: FileSpec,
    features: FeatureState,
    retired: RetiredState,
    bounds: MapBounds,
) -> tuple[tuple[dict[str, object], ...], dict[str, int]]:
    # typed key -> primary ID -> exact (feature name, undocumented display column) assertions
    groups: dict[
        tuple[str, str, str], dict[str, set[tuple[str, str]]]
    ] = defaultdict(lambda: defaultdict(set))
    physical_lines = 0
    records = 0
    for line_number, line in _read_physical_lines(path, bounds.max_line_bytes):
        physical_lines += 1
        if not line:
            raise SgdMapError(f"blank dbxref row in line {line_number}")
        records += 1
        if records > bounds.max_external_records:
            raise SgdMapError("dbxref.tab exceeds maxExternalRecords")
        fields = line.split("\t")
        if len(fields) != 6:
            raise SgdMapError(
                "dbxref.tab payload must contain exactly six columns; its pinned README "
                f"documents columns 1-5 and column 6 is non-resolving display metadata (line {line_number})"
            )
        accession = _trimmed(fields[0], f"dbxref.tab:{line_number} accession")
        source = _trimmed(fields[1], f"dbxref.tab:{line_number} accession source")
        accession_type = _trimmed(fields[2], f"dbxref.tab:{line_number} accession type")
        feature_name = _trimmed(fields[3], f"dbxref.tab:{line_number} feature name")
        primary = _trimmed(fields[4], f"dbxref.tab:{line_number} primary SGDID")
        if SGD_ID.fullmatch(primary) is None:
            raise SgdMapError(f"malformed primary SGDID in dbxref.tab:{line_number}")
        display = fields[5]
        if display:
            _trimmed(display, f"dbxref.tab:{line_number} undocumented display column")
        target_assertions = groups[(accession, source, accession_type)][primary]
        target_assertions.add((feature_name, display))
        if len(target_assertions) > bounds.max_assertions_per_external_target:
            raise SgdMapError("external target exceeds maxAssertionsPerExternalTarget")
        if len(groups[(accession, source, accession_type)]) > bounds.max_targets_per_external_key:
            raise SgdMapError("typed external key exceeds maxTargetsPerExternalKey")
    _assert_counts(
        spec,
        physical_lines=physical_lines,
        data_records=records,
        irregular_records=0,
    )
    output: list[dict[str, object]] = []
    one_to_many = 0
    status_counts: dict[str, int] = defaultdict(int)
    for (accession, source, accession_type), targets in sorted(groups.items()):
        normalized_targets: list[dict[str, object]] = []
        for primary, assertions in sorted(targets.items()):
            status = _target_status(primary, features, retired)
            status_counts[status] += 1
            normalized_targets.append(
                {
                    "canonicalSgdCurie": f"SGD:{primary}",
                    "targetStatus": status,
                    "assertions": [
                        {
                            "sourceFeatureName": feature_name,
                            "undocumentedDisplayColumn6": display or None,
                            "displayResolvesIdentity": False,
                        }
                        for feature_name, display in sorted(assertions)
                    ],
                }
            )
        if len(normalized_targets) > 1:
            one_to_many += 1
        output.append(
            {
                "schema": EXTERNAL_SCHEMA,
                "ncbiTaxon": NCBI_TAXON,
                "typedAccession": {
                    "value": accession,
                    "source": source,
                    "type": accession_type,
                    "caseNormalization": "none",
                    "namespaceInferred": False,
                },
                "relationOnly": True,
                "targetCount": len(normalized_targets),
                "targets": normalized_targets,
            }
        )
    summary = {
        "physicalRows": physical_lines,
        "typedRelations": len(output),
        "oneToManyTypedRelations": one_to_many,
        "targetAssertionsByStatus": dict(sorted(status_counts.items())),
    }
    return tuple(output), summary


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> tuple[int, str, int]:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(canonical_json(record) + "\n")
            count += 1
    return count, _sha256(path), path.stat().st_size


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def normalize_sgd_snapshot(
    raw_snapshot: str | Path,
    destination: str | Path,
    bounds: MapBounds,
    *,
    file_specs: Iterable[FileSpec] = PINNED_FILES,
    source_provenance: SourceProvenance | None = None,
) -> dict[str, object]:
    """Verify and deterministically normalize one exact raw SGD snapshot."""
    specs = tuple(file_specs)
    spec_by_name = {item.name: item for item in specs}
    root, verified_inputs = verify_raw_snapshot(raw_snapshot, specs)
    _validate_readmes(root)

    features = parse_features(root / "SGD_features.tab", spec_by_name["SGD_features.tab"], bounds)
    retired = parse_retired(
        root / "deleted_merged_features.tab",
        spec_by_name["deleted_merged_features.tab"],
        bounds,
    )
    external, external_summary = parse_external_relations(
        root / "dbxref.tab",
        spec_by_name["dbxref.tab"],
        features,
        retired,
        bounds,
    )

    output_root = Path(destination)
    if output_root.exists():
        raise SgdMapError("destination must not already exist")
    output_root.mkdir(parents=True)
    output_specs: list[dict[str, object]] = []
    for name, records in (
        ("current-orfs.jsonl", features.orf_records),
        ("external-accessions.jsonl", external),
        ("retired-merged-quarantine.jsonl", retired.records),
    ):
        count, digest, size = _write_jsonl(output_root / name, records)
        output_specs.append(
            {"name": name, "records": count, "bytes": size, "sha256": digest}
        )

    digest_basis = {
        "schema": DIGEST_SCHEMA,
        "identityMappingId": IDENTITY_MAPPING_ID,
        "ncbiTaxon": NCBI_TAXON,
        "inputFiles": list(verified_inputs),
        "outputFiles": output_specs,
        "normalizationPolicy": {
            "systematicNameMatch": "exact-case-sensitive",
            "displayMetadataResolvesIdentity": False,
            "externalKey": ["value", "source", "type"],
            "externalRelationsChooseFirst": False,
            "retiredIdentifiersAutoRedirect": False,
            "benchmarkFieldsAllowed": False,
        },
    }
    identity_digest = canonical_mapping_digest(digest_basis)
    manifest: dict[str, object] = {
        "schema": MAPPING_SCHEMA,
        "identityMappingId": IDENTITY_MAPPING_ID,
        "identityMappingSha256": identity_digest,
        "identityDigest": {
            "algorithm": "sha256",
            "canonicalization": "UTF-8 RFC-8259 JSON with sorted keys, compact separators, and one LF",
            "basis": "digestBasis",
        },
        "ncbiTaxon": NCBI_TAXON,
        "upstreamGenomeAnnotationRelease": "R64.5.1",
        "digestBasis": digest_basis,
        "counts": {
            "currentOrfs": len(features.orf_records),
            "featurePhysicalLines": features.physical_lines,
            "featureBlankLines": features.blank_lines,
            "externalPhysicalRows": external_summary["physicalRows"],
            "typedExternalRelations": external_summary["typedRelations"],
            "oneToManyTypedExternalRelations": external_summary[
                "oneToManyTypedRelations"
            ],
            "retiredValidRows": retired.valid_records,
            "retiredIrregularRows": retired.irregular_records,
        },
        "externalTargetAssertionsByStatus": external_summary[
            "targetAssertionsByStatus"
        ],
        "sourceDataset": (
            {
                "resource": source_provenance.resource,
                "revision": source_provenance.revision,
                "manifestDigest": source_provenance.manifest_digest,
            }
            if source_provenance is not None
            else None
        ),
        "limitations": [
            "dbxref README documents five columns while the pinned payload has six; column 6 is preserved only as non-resolving display metadata",
            "malformed physical rows from deleted_merged_features.tab remain quarantined and are never reconstructed into redirects",
            "typed external relations are not a one-gene lookup and may contain multiple SGD CURIE targets",
        ],
    }
    manifest_path = output_root / "mapping-manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha256 = _sha256(manifest_path)
    return {
        "mappingManifest": manifest,
        "identityMappingId": IDENTITY_MAPPING_ID,
        "identityMappingSha256": identity_digest,
        "mappingManifestSha256": manifest_sha256,
        "currentOrfCount": len(features.orf_records),
        "typedExternalRelationCount": int(external_summary["typedRelations"]),
        "oneToManyExternalRelationCount": int(
            external_summary["oneToManyTypedRelations"]
        ),
        "retiredQuarantineCount": len(retired.records),
        "retiredIrregularCount": retired.irregular_records,
    }
