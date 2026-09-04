"""Outcome-blind, deterministic held-intervention roster construction.

Only immutable identity inventories enter this module.  Quantitative outcomes,
display symbols, inferred cross-species identities, and mutable releases are
outside its contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Iterable, Iterator


INVENTORY_SCHEMA = "slp.intervention-identity-inventory/v1"
RECORD_SCHEMA = "slp.intervention-identity-record/v1"
REPORT_SCHEMA = "slp.held-intervention-roster-report/v1"
ASSIGNMENT_DOMAIN = b"slp-1.1-yeast-global-held-v1\x00"
YEAST_TAXON = 4932
SGD_NAMESPACE = "SGD"
SGD_IDENTIFIER = re.compile(r"^SGD:S[0-9]{9}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SOURCE_IDENTIFIER = re.compile(r"^[^\s:]+:[^\s:]+$")
RESOURCE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
INPUT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
MUTABLE_RELEASE = re.compile(
    r"(?:^|[-_.:/])(latest|current|head|main|master|nightly)(?:$|[-_.:/])",
    re.IGNORECASE,
)
MANIFEST_FIELDS = {
    "schema",
    "sourceId",
    "sourceRelease",
    "ncbiTaxon",
    "stableIdNamespace",
    "identityMappingId",
    "identityMappingSha256",
    "inventoryFormat",
    "files",
}
RECORD_FIELDS = {"schema", "interventionId", "ncbiTaxon", "qcPassing"}
OUTCOME_FIELD_PARTS = {
    "abundance",
    "effect",
    "expression",
    "fitness",
    "foldchange",
    "label",
    "logfc",
    "measurement",
    "outcome",
    "phenotype",
    "pvalue",
    "readout",
    "score",
    "target",
    "value",
}
MAX_MANIFEST_BYTES = 1_048_576


class HeldRosterError(ValueError):
    """Raised when an identity inventory cannot safely define a roster."""


@dataclass(frozen=True)
class RosterBounds:
    minimum_intersection_size: int
    max_sources: int = 16
    max_files_per_source: int = 32
    max_records_per_source: int = 200_000
    max_line_bytes: int = 4_096

    def __post_init__(self) -> None:
        for name, value, minimum, maximum in (
            ("minimumIntersectionSize", self.minimum_intersection_size, 1, 100_000),
            ("maxSources", self.max_sources, 2, 64),
            ("maxFilesPerSource", self.max_files_per_source, 1, 256),
            ("maxRecordsPerSource", self.max_records_per_source, 1, 2_000_000),
            ("maxLineBytes", self.max_line_bytes, 128, 65_536),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
                raise HeldRosterError(f"{name} must be an integer in [{minimum}, {maximum}]")


@dataclass(frozen=True)
class Assignment:
    intervention_id: str
    role: str
    digest: str
    bucket: int


@dataclass(frozen=True)
class LoadedInventory:
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
class PinnedDatasetInput:
    input_name: str
    path: str
    resource: str
    revision: str
    manifest_digest: str


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise HeldRosterError(f"{label} must be a non-empty, trimmed string")
    return value


def _outcome_like(fields: Iterable[object]) -> list[str]:
    matches: list[str] = []
    for field in fields:
        if not isinstance(field, str):
            continue
        normalized = re.sub(r"[^a-z0-9]", "", field.casefold())
        if any(part in normalized for part in OUTCOME_FIELD_PARTS):
            matches.append(field)
    return sorted(matches)


def _pinned_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or SHA256.fullmatch(value.removeprefix("sha256:")) is None
    ):
        raise HeldRosterError(f"{label} must be an admission-pinned SHA-256")
    return value


def _dataset_resource(value: object, label: str) -> tuple[str, str]:
    resource = _nonempty_string(value, label)
    if not resource.startswith("omf://"):
        raise HeldRosterError(f"{label} must be an OMF DatasetSnapshot resource URI")
    body = resource.removeprefix("omf://")
    identity, separator, revision = body.rpartition("@")
    if not separator:
        raise HeldRosterError(f"{label} must contain an exact resource revision")
    revision = _pinned_digest(revision, f"{label} revision")
    parts = identity.split("/")
    if (
        len(parts) < 3
        or any(not part or part in {".", ".."} or any(char.isspace() for char in part) for part in parts)
        or parts[-2] != "datasetsnapshot"
        or RESOURCE_NAME.fullmatch(parts[-1]) is None
    ):
        raise HeldRosterError(f"{label} kind must be DatasetSnapshot with a valid resource name")
    return parts[-1], revision


def resolve_pinned_dataset_input(value: object, input_name: str) -> PinnedDatasetInput:
    """Validate OMF's literal materialized DatasetSnapshot request object."""
    if INPUT_NAME.fullmatch(input_name) is None:
        raise HeldRosterError("inventory input name is not canonical")
    if not isinstance(value, dict):
        raise HeldRosterError(f"{input_name} must be a materialized OMF DatasetSnapshot object")
    expected = {"resource", "mode", "path", "manifestDigest"}
    if set(value) != expected:
        raise HeldRosterError(
            f"{input_name} must have the exact materialized OMF DatasetSnapshot shape"
        )
    resource_name, revision = _dataset_resource(value["resource"], f"{input_name}.resource")
    if value["mode"] != "copy":
        raise HeldRosterError(f"{input_name} must be an immutable copied DatasetSnapshot")
    manifest_digest = _pinned_digest(
        value["manifestDigest"], f"{input_name}.manifestDigest"
    )
    path_value = _nonempty_string(value["path"], f"{input_name}.path")
    requested = Path(path_value)
    if requested.is_symlink():
        raise HeldRosterError(f"{input_name}.path must not be a symlink")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise HeldRosterError(f"{input_name}.path does not exist") from error
    if not resolved.is_dir():
        raise HeldRosterError(f"{input_name}.path must materialize an inventory directory")
    if (
        resolved.name != resource_name
        or resolved.parent.name != input_name
        or resolved.parent.parent.name != "inputs"
    ):
        raise HeldRosterError(
            f"{input_name}.path is inconsistent with its input name and DatasetSnapshot resource"
        )
    return PinnedDatasetInput(
        input_name=input_name,
        path=str(resolved),
        resource=value["resource"],
        revision=revision,
        manifest_digest=manifest_digest,
    )


def _strict_fields(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HeldRosterError(f"{label} must be a JSON object")
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    outcome_fields = _outcome_like(extra)
    if outcome_fields:
        raise HeldRosterError(
            f"{label} contains forbidden outcome-like fields: {', '.join(outcome_fields)}"
        )
    if missing or extra:
        raise HeldRosterError(
            f"{label} fields do not match the contract; missing={missing}, extra={extra}"
        )
    return value


def _inventory_root(path_value: str | Path) -> Path:
    requested = Path(path_value)
    if requested.is_symlink():
        raise HeldRosterError("inventory artifact root must not be a symlink")
    try:
        root = requested.resolve(strict=True)
    except OSError as error:
        raise HeldRosterError("inventory artifact root does not exist") from error
    if not root.is_dir():
        raise HeldRosterError("inventory artifact root must be a directory")
    return root


def _relative_regular_file(root: Path, value: object, label: str) -> Path:
    relative = _nonempty_string(value, label)
    posix = PurePosixPath(relative)
    if (
        posix.is_absolute()
        or relative != posix.as_posix()
        or "\\" in relative
        or ":" in relative
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise HeldRosterError(f"{label} must be a canonical relative POSIX path")
    candidate = root.joinpath(*posix.parts)
    cursor = root
    for part in posix.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise HeldRosterError(f"{label} must not contain a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise HeldRosterError(f"{label} resolves outside or is missing from the artifact") from error
    if not resolved.is_file():
        raise HeldRosterError(f"{label} must be a regular file")
    return resolved


def role_from_digest(digest: str) -> tuple[str, int]:
    """Return the frozen role and bucket for a lowercase SHA-256 digest."""
    if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
        raise HeldRosterError("assignment digest must be a lowercase SHA-256")
    bucket = int(digest[:16], 16) % 100
    if bucket <= 9:
        return "molecular-final", bucket
    if bucket <= 29:
        return "molecular-validation", bucket
    return "pretrain", bucket


def assign_intervention(intervention_id: str) -> Assignment:
    """Pure frozen assignment for one canonical species-native SGD CURIE."""
    if not isinstance(intervention_id, str) or SGD_IDENTIFIER.fullmatch(intervention_id) is None:
        raise HeldRosterError(
            "interventionId must be a canonical SGD CURIE, not a display symbol"
        )
    digest = hashlib.sha256(ASSIGNMENT_DOMAIN + intervention_id.encode("ascii")).hexdigest()
    role, bucket = role_from_digest(digest)
    return Assignment(intervention_id, role, digest, bucket)


def _load_manifest(root: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = _relative_regular_file(root, "inventory.json", "inventory manifest")
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise HeldRosterError("inventory manifest exceeds the one MiB bound")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HeldRosterError("inventory manifest is not valid UTF-8 JSON") from error
    return _strict_fields(manifest, MANIFEST_FIELDS, "inventory manifest"), manifest_path


def _read_records(path: Path, max_line_bytes: int) -> Iterator[dict[str, Any]]:
    try:
        with path.open("rb") as stream:
            line_number = 0
            while True:
                raw = stream.readline(max_line_bytes + 1)
                if not raw:
                    break
                line_number += 1
                if len(raw) > max_line_bytes:
                    raise HeldRosterError(
                        f"inventory line exceeds maxLineBytes in {path.name}:{line_number}"
                    )
                if not raw.strip():
                    raise HeldRosterError(f"blank inventory record in {path.name}:{line_number}")
                try:
                    record = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise HeldRosterError(
                        f"invalid UTF-8 JSON record in {path.name}:{line_number}"
                    ) from error
                yield _strict_fields(record, RECORD_FIELDS, f"record {path.name}:{line_number}")
    except OSError as error:
        raise HeldRosterError(f"could not read inventory file {path.name}") from error


def load_inventory(path_value: str | Path, bounds: RosterBounds) -> LoadedInventory:
    root = _inventory_root(path_value)
    manifest, manifest_path = _load_manifest(root)
    if manifest["schema"] != INVENTORY_SCHEMA:
        raise HeldRosterError(f"unsupported inventory schema: {manifest['schema']!r}")
    if manifest["inventoryFormat"] != RECORD_SCHEMA:
        raise HeldRosterError(f"unsupported inventoryFormat: {manifest['inventoryFormat']!r}")
    source_id = _nonempty_string(manifest["sourceId"], "sourceId")
    if SOURCE_IDENTIFIER.fullmatch(source_id) is None:
        raise HeldRosterError("sourceId must be an unambiguous CURIE-like identifier")
    source_release = _nonempty_string(manifest["sourceRelease"], "sourceRelease")
    if MUTABLE_RELEASE.search(source_release):
        raise HeldRosterError("sourceRelease must be immutable, not a mutable alias")
    if manifest["ncbiTaxon"] != YEAST_TAXON:
        raise HeldRosterError("inventory must explicitly retain NCBI taxon 4932")
    if manifest["stableIdNamespace"] != SGD_NAMESPACE:
        raise HeldRosterError("inventory stableIdNamespace must be exactly SGD")
    identity_mapping_id = _nonempty_string(
        manifest["identityMappingId"], "identityMappingId"
    )
    if SOURCE_IDENTIFIER.fullmatch(identity_mapping_id) is None:
        raise HeldRosterError("identityMappingId must identify one pinned SGD mapping snapshot")
    identity_mapping_sha256 = manifest["identityMappingSha256"]
    if not isinstance(identity_mapping_sha256, str) or SHA256.fullmatch(identity_mapping_sha256) is None:
        raise HeldRosterError("identityMappingSha256 must be a lowercase SHA-256")

    files = manifest["files"]
    if not isinstance(files, list) or not files or len(files) > bounds.max_files_per_source:
        raise HeldRosterError("files must be a non-empty list within maxFilesPerSource")
    paths: list[str] = []
    for index, item in enumerate(files):
        item = _strict_fields(item, {"path", "sha256", "records"}, f"files[{index}]")
        relative_path = _nonempty_string(item["path"], f"files[{index}].path")
        if not relative_path.endswith(".jsonl"):
            raise HeldRosterError("inventory data files must use the .jsonl extension")
        paths.append(relative_path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise HeldRosterError("inventory file paths must be unique and path-sorted")

    seen: dict[str, bool] = {}
    duplicate_records = 0
    records_total = 0
    for index, item in enumerate(files):
        inventory_path = _relative_regular_file(root, item["path"], f"files[{index}].path")
        digest = item["sha256"]
        if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
            raise HeldRosterError("inventory file sha256 must be lowercase hexadecimal")
        if _sha256(inventory_path) != digest:
            raise HeldRosterError(f"inventory file digest drift: {item['path']}")
        expected_records = item["records"]
        if (
            not isinstance(expected_records, int)
            or isinstance(expected_records, bool)
            or expected_records < 1
        ):
            raise HeldRosterError("inventory file records must be a positive integer")
        actual_records = 0
        for record in _read_records(inventory_path, bounds.max_line_bytes):
            actual_records += 1
            records_total += 1
            if records_total > bounds.max_records_per_source:
                raise HeldRosterError("inventory exceeds maxRecordsPerSource")
            if record["schema"] != RECORD_SCHEMA:
                raise HeldRosterError(f"unsupported record schema: {record['schema']!r}")
            if record["ncbiTaxon"] != YEAST_TAXON:
                raise HeldRosterError("record must explicitly retain NCBI taxon 4932")
            intervention_id = record["interventionId"]
            if not isinstance(intervention_id, str) or SGD_IDENTIFIER.fullmatch(intervention_id) is None:
                raise HeldRosterError(
                    "interventionId must be a canonical SGD CURIE, not a display symbol"
                )
            qc_passing = record["qcPassing"]
            if not isinstance(qc_passing, bool):
                raise HeldRosterError("qcPassing must be a JSON boolean")
            previous = seen.get(intervention_id)
            if previous is not None:
                if previous != qc_passing:
                    raise HeldRosterError(
                        f"conflicting duplicate interventionId in {source_id}: {intervention_id}"
                    )
                duplicate_records += 1
            else:
                seen[intervention_id] = qc_passing
        if actual_records != expected_records:
            raise HeldRosterError(
                f"inventory record count drift for {item['path']}: "
                f"expected {expected_records}, got {actual_records}"
            )

    qc_passing_ids = frozenset(identifier for identifier, passes in seen.items() if passes)
    qc_failed_ids = frozenset(identifier for identifier, passes in seen.items() if not passes)
    return LoadedInventory(
        source_id=source_id,
        source_release=source_release,
        identity_mapping_id=identity_mapping_id,
        identity_mapping_sha256=identity_mapping_sha256,
        manifest_sha256=_sha256(manifest_path),
        records=records_total,
        duplicate_records=duplicate_records,
        qc_passing=qc_passing_ids,
        qc_failed=qc_failed_ids,
    )


def _source_coverage(source: LoadedInventory, intersection: frozenset[str]) -> dict[str, object]:
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


def build_held_roster(
    inventories: Iterable[str | Path],
    destination: str | Path,
    bounds: RosterBounds,
) -> dict[str, object]:
    """Validate inventories and write one deterministic roster plus coverage report."""
    inventory_paths = list(inventories)
    if len(inventory_paths) < 2:
        raise HeldRosterError("at least two protected source inventories are required")
    if len(inventory_paths) > bounds.max_sources:
        raise HeldRosterError("protected source count exceeds maxSources")
    loaded = [load_inventory(path, bounds) for path in inventory_paths]
    source_ids = [item.source_id for item in loaded]
    if len(source_ids) != len(set(source_ids)):
        raise HeldRosterError("protected sourceId values must be unique")
    loaded.sort(key=lambda item: item.source_id)

    identity_mappings = {
        (item.identity_mapping_id, item.identity_mapping_sha256) for item in loaded
    }
    if len(identity_mappings) != 1:
        raise HeldRosterError(
            "protected inventories must declare the exact same identityMappingId "
            "and identityMappingSha256"
        )
    identity_mapping_id, identity_mapping_sha256 = next(iter(identity_mappings))

    intersection = frozenset.intersection(*(item.qc_passing for item in loaded))
    if len(intersection) < bounds.minimum_intersection_size:
        raise HeldRosterError(
            "QC-passing protected-source intersection is undersized: "
            f"required {bounds.minimum_intersection_size}, found {len(intersection)}; "
            "the frozen roster must not be rerolled or weakened"
        )

    assignments = [assign_intervention(identifier) for identifier in sorted(intersection)]
    roster_bytes = "".join(
        f"{item.intervention_id}\t{item.role}\t{item.digest}\n" for item in assignments
    ).encode("ascii")
    roster_sha256 = hashlib.sha256(roster_bytes).hexdigest()
    role_counts = {
        role: sum(item.role == role for item in assignments)
        for role in ("pretrain", "molecular-validation", "molecular-final")
    }
    rejection_counts = {
        "qcFailed": sum(len(item.qc_failed) for item in loaded),
        "notPassingAllProtectedSources": sum(
            len(item.qc_passing - intersection) for item in loaded
        ),
        "identicalDuplicatesCollapsed": sum(item.duplicate_records for item in loaded),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "assignment": {
            "domainHex": ASSIGNMENT_DOMAIN.hex(),
            "digest": "sha256",
            "bucketRule": "int(first-16-lowercase-hex,16) mod 100",
            "roles": {
                "0-9": "molecular-final",
                "10-29": "molecular-validation",
                "30-99": "pretrain",
            },
        },
        "sourceCount": len(loaded),
        "identityMapping": {
            "id": identity_mapping_id,
            "sha256": identity_mapping_sha256,
        },
        "minimumIntersectionSize": bounds.minimum_intersection_size,
        "intersectionSize": len(intersection),
        "roleCounts": role_counts,
        "rejectionCounts": rejection_counts,
        "rosterPath": "held-intervention-roster.tsv",
        "rosterSha256": roster_sha256,
        "sources": [_source_coverage(item, intersection) for item in loaded],
    }

    destination_path = Path(destination).resolve()
    if destination_path.exists():
        raise HeldRosterError("destination must not already exist")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination_path.name}-", dir=destination_path.parent
    ) as temporary:
        staging = Path(temporary) / destination_path.name
        staging.mkdir()
        (staging / "held-intervention-roster.tsv").write_bytes(roster_bytes)
        _write_json(staging / "coverage.json", report)
        staging.replace(destination_path)
    report["coverageSha256"] = _sha256(destination_path / "coverage.json")
    return report
