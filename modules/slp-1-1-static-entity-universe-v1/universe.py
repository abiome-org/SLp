"""Deterministic, outcome-blind static entity-universe construction.

Only two copied DatasetSnapshots are accepted: the proteome intervention
identity inventory and the typed protein-relation inventory.  This module does
not consume held-out assignments, measurements, labels, or numeric features.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

UNIVERSE_SCHEMA = "slp.static-entity-universe/v1"
ENTITY_SCHEMA = "slp.static-entity/v1"
AUDIT_SCHEMA = "slp.static-entity-universe-audit/v1"
INTERVENTION_MANIFEST_SCHEMA = "slp.intervention-identity-inventory/v1"
INTERVENTION_RECORD_SCHEMA = "slp.intervention-identity-record/v1"
PROTEIN_MANIFEST_SCHEMA = "slp.proteome-protein-relation-inventory/v1"
PROTEIN_RECORD_SCHEMA = "slp.proteome-protein-relation/v1"

SGD_CURIE = re.compile(r"^SGD:S[0-9]{9}$")
UNIPROT_CURIE = re.compile(r"^UniProtKB:[A-Z0-9][A-Z0-9-]{0,31}$")
UNIPROT_ACCESSION = re.compile(r"^[A-Z0-9][A-Z0-9-]{0,31}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RESOURCE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
INPUT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")

INTERVENTION_MANIFEST_FIELDS = {
    "schema", "sourceId", "sourceRelease", "ncbiTaxon", "stableIdNamespace",
    "identityMappingId", "identityMappingSha256", "inventoryFormat", "files",
}
INTERVENTION_RECORD_FIELDS = {"schema", "interventionId", "ncbiTaxon", "qcPassing"}
PROTEIN_MANIFEST_FIELDS = {
    "schema", "sourceId", "sourceRelease", "ncbiTaxon", "identityMappingId",
    "identityMappingSha256", "relationFormat", "files",
}
PROTEIN_RECORD_FIELDS = {
    "schema", "proteinId", "sourceAccession", "sourceAccessionType", "ncbiTaxon",
    "currentOrfRelations", "currentOrfRelationCount", "chooseFirstAllowed",
}
ACCESSION_TYPE_FIELDS = {
    "source", "type", "namespaceInferred", "caseNormalization",
}
FILE_FIELDS = {"path", "sha256", "records"}
ENTITY_FIELDS = {"schema", "ncbiTaxon", "entityId", "entityClass", "usages"}

# Any appearance of these as input object keys is incompatible with an
# identity-only universe.  The source contracts use none of these keys.
FORBIDDEN_INPUT_KEY_PARTS = {
    "abundance", "benchmark", "embedding", "expression", "feature", "fitness",
    "fold", "label", "measurement", "outcome", "phenotype", "reward", "role",
    "score", "split", "value", "vector",
}

PRODUCTION_MAPPING_ID = "slp-sgd-map:2026-08-28-object-set-v1"
PRODUCTION_MAPPING_SHA256 = (
    "6fd789df6099b78a8842baa8f1d20ab0a3fe77f27ce512ee783444eb2627ef2a"
)


class StaticEntityUniverseError(ValueError):
    """Raised when an identity source or output violates the frozen contract."""


@dataclass(frozen=True)
class Bounds:
    max_manifest_bytes: int = 1_048_576
    max_line_bytes: int = 16_384
    max_intervention_records: int = 10_000
    max_relation_records: int = 10_000
    max_archive_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        for name, value, minimum, maximum in (
            ("maxManifestBytes", self.max_manifest_bytes, 256, 16 * 1024 * 1024),
            ("maxLineBytes", self.max_line_bytes, 128, 1024 * 1024),
            ("maxInterventionRecords", self.max_intervention_records, 1, 10_000_000),
            ("maxRelationRecords", self.max_relation_records, 1, 10_000_000),
            ("maxArchiveBytes", self.max_archive_bytes, 1024, 1024**3),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
                raise StaticEntityUniverseError(
                    f"{name} must be an integer in [{minimum}, {maximum}]"
                )


@dataclass(frozen=True)
class PinnedDataset:
    input_name: str
    path: Path
    resource: str
    revision: str
    manifest_digest: str


@dataclass(frozen=True)
class ExpectedSnapshot:
    resource: str
    manifest_digest: str
    inner_manifest_sha256: str
    records_sha256: str

    def __post_init__(self) -> None:
        _dataset_resource(self.resource, "expected resource")
        _prefixed_digest(self.manifest_digest, "expected outer manifest")
        _bare_digest(self.inner_manifest_sha256, "expected inner manifest")
        _bare_digest(self.records_sha256, "expected record set")


@dataclass(frozen=True)
class ExpectedContract:
    intervention: ExpectedSnapshot
    relations: ExpectedSnapshot
    mapping_id: str
    mapping_sha256: str
    intervention_records: int
    action_entities: int
    relation_records: int
    relation_edges: int
    relation_target_genes: int
    relation_targets_in_action_universe: int
    relation_support_only: int
    one_to_many_relations: int
    readout_entities: int
    total_entities: int
    action_id_set_sha256: str
    protein_id_set_sha256: str
    relation_edge_set_sha256: str
    full_entity_id_set_sha256: str
    full_entity_key_set_sha256: str

    def __post_init__(self) -> None:
        _nonempty(self.mapping_id, "expected mapping ID")
        _bare_digest(self.mapping_sha256, "expected mapping SHA-256")
        _bare_digest(self.action_id_set_sha256, "expected action ID-set SHA-256")
        _bare_digest(self.protein_id_set_sha256, "expected protein ID-set SHA-256")
        _bare_digest(self.relation_edge_set_sha256, "expected relation edge-set SHA-256")
        _bare_digest(self.full_entity_id_set_sha256, "expected full entity ID-set SHA-256")
        _bare_digest(self.full_entity_key_set_sha256, "expected full entity key-set SHA-256")
        for name, value in (
            ("interventionRecords", self.intervention_records),
            ("actionEntities", self.action_entities),
            ("relationRecords", self.relation_records),
            ("relationEdges", self.relation_edges),
            ("relationTargetGenes", self.relation_target_genes),
            ("relationTargetsInActionUniverse", self.relation_targets_in_action_universe),
            ("relationSupportOnly", self.relation_support_only),
            ("oneToManyRelations", self.one_to_many_relations),
            ("readoutEntities", self.readout_entities),
            ("totalEntities", self.total_entities),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise StaticEntityUniverseError(f"{name} must be a non-negative integer")


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _canonical_json_bytes(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _pretty_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise StaticEntityUniverseError(f"could not hash {path.name}") from error
    return digest.hexdigest()


def _bounded_file_sha256(path: Path, max_bytes: int, label: str) -> str:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise StaticEntityUniverseError(f"could not stat {label}") from error
    if size > max_bytes:
        raise StaticEntityUniverseError(f"{label} exceeds its configured byte bound")
    return _sha256_file(path)


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise StaticEntityUniverseError(f"{label} must be a non-empty trimmed string")
    return value


def _bare_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise StaticEntityUniverseError(f"{label} must be a lowercase SHA-256")
    return value


def _prefixed_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise StaticEntityUniverseError(f"{label} must use the sha256: prefix")
    _bare_digest(value.removeprefix("sha256:"), label)
    return value


def _dataset_resource(value: object, label: str) -> tuple[str, str]:
    resource = _nonempty(value, label)
    if not resource.startswith("omf://"):
        raise StaticEntityUniverseError(f"{label} must be an OMF DatasetSnapshot URI")
    identity, separator, revision = resource.removeprefix("omf://").rpartition("@")
    if not separator:
        raise StaticEntityUniverseError(f"{label} must carry an exact revision")
    _prefixed_digest(revision, f"{label} revision")
    parts = identity.split("/")
    if (
        len(parts) < 3
        or parts[-2] != "datasetsnapshot"
        or RESOURCE_NAME.fullmatch(parts[-1]) is None
        or any(not part or part in {".", ".."} or any(char.isspace() for char in part) for part in parts)
    ):
        raise StaticEntityUniverseError(f"{label} must identify a DatasetSnapshot")
    return parts[-1], revision


PRODUCTION_CONTRACT = ExpectedContract(
    intervention=ExpectedSnapshot(
        resource=(
            "omf://abiome/slp/datasetsnapshot/"
            "slp-1-1-proteome-intervention-inventory-v1@"
            "sha256:bd688dffdf4d96c01d4147580b1a8705c2149acadbc843a719537817a74505d9"
        ),
        manifest_digest=(
            "sha256:a1f5222f3dca31d2ca68ca46a271d39cdca3425a903b5dceb7373481450ada36"
        ),
        inner_manifest_sha256=(
            "dd683a2585a15377282e669f61dce38c44ea9d3d9d55be71b24842048c05f3e5"
        ),
        records_sha256=(
            "15e011d9f3bbea2e034f47dd06b260f834475ffee8adb452046dbb2701ead497"
        ),
    ),
    relations=ExpectedSnapshot(
        resource=(
            "omf://abiome/slp/datasetsnapshot/"
            "slp-1-1-proteome-protein-relations-v1@"
            "sha256:acad3427907644f8ab8af38ed36066a6e1148ef92557b727351b0a4fba2b446c"
        ),
        manifest_digest=(
            "sha256:c159573f4f7a2e41b18930d724dea9fb297452a659bdf6050e4718efc1a6c58a"
        ),
        inner_manifest_sha256=(
            "8d559638f48ee4516f7e6fce9e0248e9a1762d58803fe2ed761eff8734f45f86"
        ),
        records_sha256=(
            "c72996b4ddc6870a3ab722060eef2fa2747fa9dd121d3e70514dd196c5283b8d"
        ),
    ),
    mapping_id=PRODUCTION_MAPPING_ID,
    mapping_sha256=PRODUCTION_MAPPING_SHA256,
    intervention_records=4_623,
    action_entities=4_476,
    relation_records=1_850,
    relation_edges=1_855,
    relation_target_genes=1_855,
    relation_targets_in_action_universe=1_144,
    relation_support_only=711,
    one_to_many_relations=5,
    readout_entities=1_850,
    total_entities=7_037,
    action_id_set_sha256="7424b17b63f504b419fb1c52930ede3d1cbb2a0fabf3a0621624df1d098c4d88",
    protein_id_set_sha256="25ec666023cc97610797b4e915e537bc3f0212b0f3a02972cb3b60eac160d12d",
    relation_edge_set_sha256="8a75c42d5a0f24a86be16ecea2616d6d13d25d90de18a80d3dd22cd188afc6d1",
    full_entity_id_set_sha256="e7231d3bb859ca4818364c76d9aa9fee54d6b1d9a64050c2d3ab8af81a9b3eb9",
    full_entity_key_set_sha256="82b8e2885939577fe6946e3b974a10cb947834118f2070e1bcbe4c2f2e6a5fd9",
)


def _resolved_directory(value: object, label: str) -> Path:
    path = Path(_nonempty(value, label))
    cursor = path.absolute()
    while True:
        if cursor.is_symlink():
            raise StaticEntityUniverseError(f"{label} must not contain a symlink")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise StaticEntityUniverseError(f"{label} does not exist") from error
    if not resolved.is_dir():
        raise StaticEntityUniverseError(f"{label} must be a directory")
    return resolved


def resolve_pinned_dataset(value: object, input_name: str) -> PinnedDataset:
    """Accept only OMF's exact copy-materialized DatasetSnapshot shape."""
    if INPUT_NAME.fullmatch(input_name) is None or not isinstance(value, dict):
        raise StaticEntityUniverseError(f"{input_name} must be a materialized DatasetSnapshot")
    if set(value) != {"resource", "mode", "path", "manifestDigest"}:
        raise StaticEntityUniverseError(f"{input_name} has a spoofed DatasetSnapshot shape")
    resource_name, revision = _dataset_resource(value["resource"], f"{input_name}.resource")
    if value["mode"] != "copy":
        raise StaticEntityUniverseError(f"{input_name} must be copied, not mutable")
    manifest_digest = _prefixed_digest(
        value["manifestDigest"], f"{input_name}.manifestDigest"
    )
    root = _resolved_directory(value["path"], f"{input_name}.path")
    if root.name != resource_name or root.parent.name != input_name or root.parent.parent.name != "inputs":
        raise StaticEntityUniverseError(
            f"{input_name}.path is inconsistent with OMF materialization"
        )
    return PinnedDataset(
        input_name=input_name,
        path=root,
        resource=str(value["resource"]),
        revision=revision,
        manifest_digest=manifest_digest,
    )


def _strict_fields(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StaticEntityUniverseError(f"{label} must be a JSON object")
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise StaticEntityUniverseError(
            f"{label} fields do not match the contract; missing={missing}, extra={extra}"
        )
    return value


def _forbidden_input_keys(value: object, label: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(key, str):
                normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
                matched = sorted(part for part in FORBIDDEN_INPUT_KEY_PARTS if part in normalized)
                if matched:
                    raise StaticEntityUniverseError(
                        f"{label} contains forbidden non-identity field {key!r}"
                    )
            _forbidden_input_keys(nested, label)
    elif isinstance(value, list):
        for nested in value:
            _forbidden_input_keys(nested, label)


def _relative_file(root: Path, relative: str, label: str) -> Path:
    posix = PurePosixPath(relative)
    if (
        relative != posix.as_posix()
        or posix.is_absolute()
        or "\\" in relative
        or ":" in relative
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise StaticEntityUniverseError(f"{label} path must be canonical relative POSIX")
    cursor = root
    for part in posix.parts:
        cursor /= part
        if cursor.is_symlink():
            raise StaticEntityUniverseError(f"{label} must not be a symlink")
    try:
        resolved = cursor.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise StaticEntityUniverseError(f"{label} is missing or escapes its snapshot") from error
    if not resolved.is_file():
        raise StaticEntityUniverseError(f"{label} must be a regular file")
    return resolved


def _snapshot_files(root: Path, expected: set[str]) -> Mapping[str, Path]:
    try:
        actual = {item.name for item in root.iterdir()}
    except OSError as error:
        raise StaticEntityUniverseError("could not enumerate DatasetSnapshot") from error
    if actual != expected:
        raise StaticEntityUniverseError(
            f"DatasetSnapshot file set drift; expected={sorted(expected)}, actual={sorted(actual)}"
        )
    return {name: _relative_file(root, name, name) for name in sorted(expected)}


def _read_manifest(path: Path, bounds: Bounds, label: str) -> tuple[dict[str, Any], str]:
    try:
        size = path.stat().st_size
        if size > bounds.max_manifest_bytes:
            raise StaticEntityUniverseError(f"{label} exceeds maxManifestBytes")
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StaticEntityUniverseError(f"{label} is not bounded UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise StaticEntityUniverseError(f"{label} must be an object")
    _forbidden_input_keys(value, label)
    return value, _sha256_bytes(raw)


def _jsonl(path: Path, bounds: Bounds, limit: int, label: str) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        with path.open("rb") as stream:
            line_number = 0
            while True:
                raw = stream.readline(bounds.max_line_bytes + 1)
                if not raw:
                    break
                line_number += 1
                if line_number > limit:
                    raise StaticEntityUniverseError(f"{label} exceeds its record bound")
                if len(raw) > bounds.max_line_bytes:
                    raise StaticEntityUniverseError(f"{label}:{line_number} exceeds maxLineBytes")
                if not raw.endswith(b"\n") or raw in {b"\n", b"\r\n"}:
                    raise StaticEntityUniverseError(f"{label}:{line_number} is not canonical JSONL")
                try:
                    record = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise StaticEntityUniverseError(f"{label}:{line_number} is invalid JSON") from error
                if not isinstance(record, dict):
                    raise StaticEntityUniverseError(f"{label}:{line_number} must be an object")
                if raw != _canonical_json_bytes(record):
                    raise StaticEntityUniverseError(f"{label}:{line_number} is not canonical JSONL")
                _forbidden_input_keys(record, f"{label}:{line_number}")
                yield line_number, record
    except OSError as error:
        raise StaticEntityUniverseError(f"could not read {label}") from error


def _one_file(manifest: Mapping[str, Any], expected_path: str, label: str) -> dict[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 1:
        raise StaticEntityUniverseError(f"{label}.files must declare exactly one file")
    item = _strict_fields(files[0], FILE_FIELDS, f"{label}.files[0]")
    if item["path"] != expected_path:
        raise StaticEntityUniverseError(f"{label} record path drift")
    _bare_digest(item["sha256"], f"{label} record SHA-256")
    if not isinstance(item["records"], int) or isinstance(item["records"], bool) or item["records"] < 0:
        raise StaticEntityUniverseError(f"{label} record count is invalid")
    return item


def _input_identity(dataset: PinnedDataset, manifest_sha: str, file_spec: Mapping[str, Any]) -> dict[str, object]:
    return {
        "resource": dataset.resource,
        "revision": dataset.revision,
        "manifestDigest": dataset.manifest_digest,
        "innerManifestSha256": manifest_sha,
        "recordsSha256": file_spec["sha256"],
        "records": file_spec["records"],
    }


def framed_ascii_set_sha256(values: Iterable[str]) -> str:
    """Hash unique ASCII strings in ordinal order, each terminated by one LF."""
    unique = set(values)
    try:
        payload = b"".join(item.encode("ascii") + b"\n" for item in sorted(unique))
    except (AttributeError, UnicodeEncodeError) as error:
        raise StaticEntityUniverseError("semantic identity sets must contain ASCII strings") from error
    return _sha256_bytes(payload)


def framed_composite_key_set_sha256(values: Iterable[tuple[int, str]]) -> str:
    """Hash unique taxon-qualified keys as ``taxon<TAB>identifier<LF>``."""
    framed: list[str] = []
    for taxon, identifier in values:
        if not isinstance(taxon, int) or isinstance(taxon, bool) or taxon <= 0:
            raise StaticEntityUniverseError("semantic entity keys require positive integer taxa")
        if not isinstance(identifier, str) or not identifier or identifier != identifier.strip():
            raise StaticEntityUniverseError("semantic entity keys require trimmed identifiers")
        if "\t" in identifier or "\n" in identifier or "\r" in identifier:
            raise StaticEntityUniverseError("semantic entity keys cannot contain framing characters")
        framed.append(f"{taxon}\t{identifier}")
    return framed_ascii_set_sha256(framed)


def _check_snapshot_pin(dataset: PinnedDataset, expected: ExpectedSnapshot, label: str) -> None:
    if dataset.resource != expected.resource:
        raise StaticEntityUniverseError(f"{label} resource revision is not the frozen source")
    if dataset.manifest_digest != expected.manifest_digest:
        raise StaticEntityUniverseError(f"{label} outer manifest digest drift")


def _load_interventions(
    dataset: PinnedDataset,
    bounds: Bounds,
    expected: ExpectedContract,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, Any]]:
    _check_snapshot_pin(dataset, expected.intervention, "interventionInventory")
    files = _snapshot_files(dataset.path, {"inventory.json", "interventions.jsonl"})
    manifest, manifest_sha = _read_manifest(files["inventory.json"], bounds, "intervention manifest")
    if manifest_sha != expected.intervention.inner_manifest_sha256:
        raise StaticEntityUniverseError("intervention inner manifest digest drift")
    manifest = _strict_fields(manifest, INTERVENTION_MANIFEST_FIELDS, "intervention manifest")
    if (
        manifest["schema"] != INTERVENTION_MANIFEST_SCHEMA
        or manifest["inventoryFormat"] != INTERVENTION_RECORD_SCHEMA
        or manifest["stableIdNamespace"] != "SGD"
    ):
        raise StaticEntityUniverseError("intervention manifest schema or namespace drift")
    taxon = manifest["ncbiTaxon"]
    if not isinstance(taxon, int) or isinstance(taxon, bool) or taxon <= 0:
        raise StaticEntityUniverseError("intervention ncbiTaxon must be a positive integer")
    mapping_id = _nonempty(manifest["identityMappingId"], "identity mapping ID")
    mapping_sha = _bare_digest(manifest["identityMappingSha256"], "identity mapping SHA-256")
    file_spec = _one_file(manifest, "interventions.jsonl", "intervention manifest")
    actual_records_sha = _bounded_file_sha256(
        files["interventions.jsonl"],
        bounds.max_line_bytes * bounds.max_intervention_records,
        "interventions.jsonl",
    )
    if file_spec["sha256"] != actual_records_sha or actual_records_sha != expected.intervention.records_sha256:
        raise StaticEntityUniverseError("intervention record-set digest drift")

    by_key: dict[tuple[int, str], bool] = {}
    record_count = 0
    for line_number, raw in _jsonl(
        files["interventions.jsonl"], bounds, bounds.max_intervention_records, "interventions.jsonl"
    ):
        record_count += 1
        record = _strict_fields(raw, INTERVENTION_RECORD_FIELDS, f"interventions.jsonl:{line_number}")
        if record["schema"] != INTERVENTION_RECORD_SCHEMA:
            raise StaticEntityUniverseError("intervention record schema drift")
        identifier = record["interventionId"]
        if not isinstance(identifier, str) or SGD_CURIE.fullmatch(identifier) is None:
            raise StaticEntityUniverseError("interventionId must be a canonical SGD CURIE")
        if record["ncbiTaxon"] != taxon:
            raise StaticEntityUniverseError("intervention record taxon differs from its manifest")
        if not isinstance(record["qcPassing"], bool):
            raise StaticEntityUniverseError("qcPassing must be boolean identity-admission metadata")
        key = (taxon, identifier)
        previous = by_key.get(key)
        if previous is not None and previous != record["qcPassing"]:
            raise StaticEntityUniverseError("conflicting duplicate intervention identity")
        by_key[key] = record["qcPassing"]
    if record_count != file_spec["records"] or record_count != expected.intervention_records:
        raise StaticEntityUniverseError("intervention record count drift")
    actions = [
        {
            "schema": ENTITY_SCHEMA,
            "ncbiTaxon": key[0],
            "entityId": key[1],
            "entityClass": "gene",
            "usages": ["action"],
        }
        for key, passing in sorted(by_key.items())
        if passing
    ]
    if len(actions) != expected.action_entities:
        raise StaticEntityUniverseError("action entity count drift")
    action_id_set_sha256 = framed_ascii_set_sha256(item["entityId"] for item in actions)
    if action_id_set_sha256 != expected.action_id_set_sha256:
        raise StaticEntityUniverseError("action semantic ID-set digest drift")
    identity = _input_identity(dataset, manifest_sha, file_spec)
    details = {
        "sourceId": _nonempty(manifest["sourceId"], "intervention sourceId"),
        "sourceRelease": _nonempty(manifest["sourceRelease"], "intervention sourceRelease"),
        "ncbiTaxon": taxon,
        "mappingId": mapping_id,
        "mappingSha256": mapping_sha,
        "records": record_count,
        "duplicateRecords": record_count - len(by_key),
        "actionIdSetSha256": action_id_set_sha256,
    }
    return actions, identity, details


def _load_relations(
    dataset: PinnedDataset,
    bounds: Bounds,
    expected: ExpectedContract,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, Any]],
    dict[str, object],
    dict[str, Any],
]:
    _check_snapshot_pin(dataset, expected.relations, "proteinRelations")
    files = _snapshot_files(dataset.path, {"manifest.json", "relations.jsonl"})
    manifest, manifest_sha = _read_manifest(files["manifest.json"], bounds, "protein relation manifest")
    if manifest_sha != expected.relations.inner_manifest_sha256:
        raise StaticEntityUniverseError("protein relation inner manifest digest drift")
    manifest = _strict_fields(manifest, PROTEIN_MANIFEST_FIELDS, "protein relation manifest")
    if manifest["schema"] != PROTEIN_MANIFEST_SCHEMA or manifest["relationFormat"] != PROTEIN_RECORD_SCHEMA:
        raise StaticEntityUniverseError("protein relation manifest schema drift")
    taxon = manifest["ncbiTaxon"]
    if not isinstance(taxon, int) or isinstance(taxon, bool) or taxon <= 0:
        raise StaticEntityUniverseError("protein relation ncbiTaxon must be a positive integer")
    mapping_id = _nonempty(manifest["identityMappingId"], "identity mapping ID")
    mapping_sha = _bare_digest(manifest["identityMappingSha256"], "identity mapping SHA-256")
    file_spec = _one_file(manifest, "relations.jsonl", "protein relation manifest")
    actual_records_sha = _bounded_file_sha256(
        files["relations.jsonl"],
        bounds.max_line_bytes * bounds.max_relation_records,
        "relations.jsonl",
    )
    if file_spec["sha256"] != actual_records_sha or actual_records_sha != expected.relations.records_sha256:
        raise StaticEntityUniverseError("protein relation record-set digest drift")

    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    edge_count = 0
    one_to_many: list[dict[str, object]] = []
    for line_number, raw in _jsonl(
        files["relations.jsonl"], bounds, bounds.max_relation_records, "relations.jsonl"
    ):
        record = _strict_fields(raw, PROTEIN_RECORD_FIELDS, f"relations.jsonl:{line_number}")
        if record["schema"] != PROTEIN_RECORD_SCHEMA:
            raise StaticEntityUniverseError("protein relation record schema drift")
        protein_id = record["proteinId"]
        accession = record["sourceAccession"]
        if not isinstance(protein_id, str) or UNIPROT_CURIE.fullmatch(protein_id) is None:
            raise StaticEntityUniverseError("proteinId must be a canonical UniProtKB CURIE")
        if not isinstance(accession, str) or UNIPROT_ACCESSION.fullmatch(accession) is None:
            raise StaticEntityUniverseError("sourceAccession is malformed")
        if protein_id != f"UniProtKB:{accession}":
            raise StaticEntityUniverseError("proteinId and sourceAccession differ")
        if record["ncbiTaxon"] != taxon:
            raise StaticEntityUniverseError("protein relation taxon differs from its manifest")
        accession_type = _strict_fields(
            record["sourceAccessionType"], ACCESSION_TYPE_FIELDS, "sourceAccessionType"
        )
        if accession_type != {
            "source": "UniProtKB", "type": "UniProtKB ID",
            "namespaceInferred": False, "caseNormalization": "none",
        }:
            raise StaticEntityUniverseError("typed UniProt accession contract drift")
        targets = record["currentOrfRelations"]
        if (
            not isinstance(targets, list)
            or not targets
            or any(not isinstance(item, str) or SGD_CURIE.fullmatch(item) is None for item in targets)
            or targets != sorted(targets)
            or len(targets) != len(set(targets))
        ):
            raise StaticEntityUniverseError("currentOrfRelations must be sorted unique SGD CURIEs")
        if (
            type(record["currentOrfRelationCount"]) is not int
            or record["currentOrfRelationCount"] != len(targets)
        ):
            raise StaticEntityUniverseError("protein relation cardinality drift")
        if record["chooseFirstAllowed"] is not False:
            raise StaticEntityUniverseError("protein relations may never select a first gene")
        key = (taxon, protein_id)
        if key in by_key:
            raise StaticEntityUniverseError("duplicate protein relation identity")
        normalized = json.loads(canonical_json(record))
        by_key[key] = normalized
        edge_count += len(targets)
        if len(targets) > 1:
            one_to_many.append(
                {"ncbiTaxon": taxon, "proteinId": protein_id, "currentOrfRelations": list(targets)}
            )
    if len(by_key) != file_spec["records"] or len(by_key) != expected.relation_records:
        raise StaticEntityUniverseError("protein relation record count drift")
    if edge_count != expected.relation_edges:
        raise StaticEntityUniverseError("protein relation edge count drift")
    if len(one_to_many) != expected.one_to_many_relations:
        raise StaticEntityUniverseError("one-to-many protein relation count drift")
    readouts = [
        {
            "schema": ENTITY_SCHEMA,
            "ncbiTaxon": key[0],
            "entityId": key[1],
            "entityClass": "protein",
            "usages": ["readout-query"],
        }
        for key in sorted(by_key)
    ]
    if len(readouts) != expected.readout_entities:
        raise StaticEntityUniverseError("readout-query entity count drift")
    protein_id_set_sha256 = framed_ascii_set_sha256(item["entityId"] for item in readouts)
    if protein_id_set_sha256 != expected.protein_id_set_sha256:
        raise StaticEntityUniverseError("protein semantic ID-set digest drift")
    relations = [by_key[key] for key in sorted(by_key)]
    relation_edge_set_sha256 = framed_ascii_set_sha256(
        f"{record['proteinId']}\t{target}"
        for record in relations
        for target in record["currentOrfRelations"]
    )
    if relation_edge_set_sha256 != expected.relation_edge_set_sha256:
        raise StaticEntityUniverseError("protein relation semantic edge-set digest drift")
    relation_target_keys = sorted(
        {
            (record["ncbiTaxon"], target)
            for record in relations
            for target in record["currentOrfRelations"]
        }
    )
    if len(relation_target_keys) != expected.relation_target_genes:
        raise StaticEntityUniverseError("unique protein-relation target-gene count drift")
    relation_support = [
        {
            "schema": ENTITY_SCHEMA,
            "ncbiTaxon": key[0],
            "entityId": key[1],
            "entityClass": "gene",
            "usages": ["relation-support"],
        }
        for key in relation_target_keys
    ]
    identity = _input_identity(dataset, manifest_sha, file_spec)
    details = {
        "sourceId": _nonempty(manifest["sourceId"], "protein sourceId"),
        "sourceRelease": _nonempty(manifest["sourceRelease"], "protein sourceRelease"),
        "ncbiTaxon": taxon,
        "mappingId": mapping_id,
        "mappingSha256": mapping_sha,
        "records": len(relations),
        "edges": edge_count,
        "relationTargetGenes": len(relation_target_keys),
        "proteinIdSetSha256": protein_id_set_sha256,
        "relationEdgeSetSha256": relation_edge_set_sha256,
        "oneToMany": sorted(one_to_many, key=lambda item: (item["ncbiTaxon"], item["proteinId"])),
    }
    return readouts, relation_support, relations, identity, details


def canonicalize_entities(records: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Validate and sort entities by the composite species-native identity key."""
    by_key: dict[tuple[int, str], dict[str, object]] = {}
    for index, raw in enumerate(records, start=1):
        record = _strict_fields(dict(raw), ENTITY_FIELDS, f"entity {index}")
        if record["schema"] != ENTITY_SCHEMA:
            raise StaticEntityUniverseError("entity schema drift")
        taxon = record["ncbiTaxon"]
        identifier = record["entityId"]
        entity_class = record["entityClass"]
        usages = record["usages"]
        if not isinstance(taxon, int) or isinstance(taxon, bool) or taxon <= 0:
            raise StaticEntityUniverseError("entity ncbiTaxon must be positive")
        if not isinstance(identifier, str) or not identifier or identifier != identifier.strip():
            raise StaticEntityUniverseError("entityId must be non-empty and trimmed")
        if (
            not isinstance(usages, list)
            or not usages
            or any(not isinstance(item, str) for item in usages)
            or usages != sorted(set(usages))
        ):
            raise StaticEntityUniverseError("entity usages must be sorted, unique, and non-empty")
        if entity_class == "gene":
            if SGD_CURIE.fullmatch(identifier) is None:
                raise StaticEntityUniverseError("gene entityId must be a canonical SGD CURIE")
            if not set(usages) <= {"action", "relation-support"}:
                raise StaticEntityUniverseError("gene usage must be action or relation-support")
        elif entity_class == "protein":
            if UNIPROT_CURIE.fullmatch(identifier) is None:
                raise StaticEntityUniverseError("protein entityId must be a canonical UniProtKB CURIE")
            if usages != ["readout-query"]:
                raise StaticEntityUniverseError("protein usage must be readout-query")
        else:
            raise StaticEntityUniverseError("entityClass must be gene or protein")
        key = (taxon, identifier)
        normalized = {
            "schema": ENTITY_SCHEMA, "ncbiTaxon": taxon,
            "entityId": identifier, "entityClass": entity_class,
            "usages": list(usages),
        }
        if key in by_key:
            previous = by_key[key]
            if previous["entityClass"] != entity_class:
                raise StaticEntityUniverseError("one composite entity identity has conflicting classes")
            previous["usages"] = sorted(set(previous["usages"]) | set(usages))
        else:
            by_key[key] = normalized
    return [by_key[key] for key in sorted(by_key)]


def _jsonl_bytes(records: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json_bytes(record) for record in records)


def _file_ref(path: str, content: bytes, records: int) -> dict[str, object]:
    return {"path": path, "sha256": _sha256_bytes(content), "bytes": len(content), "records": records}


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def _write_tar(path: Path, members: Mapping[str, bytes]) -> None:
    if path.exists() or path.is_symlink():
        raise StaticEntityUniverseError("refusing to overwrite entity-universe archive")
    path.write_bytes(_tar_bytes(members))


def _tar_bytes(members: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(members):
            payload = members[name]
            archive.addfile(_tar_info(name, len(payload)), io.BytesIO(payload))
    return output.getvalue()


def _jsonl_blob(
    payload: bytes,
    bounds: Bounds,
    limit: int,
    label: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    stream = io.BytesIO(payload)
    while True:
        raw = stream.readline(bounds.max_line_bytes + 1)
        if not raw:
            break
        line_number = len(records) + 1
        if line_number > limit:
            raise StaticEntityUniverseError(f"{label} exceeds its record bound")
        if len(raw) > bounds.max_line_bytes:
            raise StaticEntityUniverseError(f"{label}:{line_number} exceeds maxLineBytes")
        if not raw.endswith(b"\n") or raw in {b"\n", b"\r\n"}:
            raise StaticEntityUniverseError(f"{label}:{line_number} is not canonical JSONL")
        try:
            record = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise StaticEntityUniverseError(f"{label}:{line_number} is invalid JSON") from error
        if not isinstance(record, dict) or raw != _canonical_json_bytes(record):
            raise StaticEntityUniverseError(f"{label}:{line_number} is not canonical JSONL")
        _forbidden_input_keys(record, f"{label}:{line_number}")
        records.append(record)
    return records


def _validate_output_relation(
    raw: object,
    label: str,
) -> dict[str, Any]:
    record = _strict_fields(raw, PROTEIN_RECORD_FIELDS, label)
    if record["schema"] != PROTEIN_RECORD_SCHEMA:
        raise StaticEntityUniverseError("output protein relation schema drift")
    taxon = record["ncbiTaxon"]
    protein_id = record["proteinId"]
    accession = record["sourceAccession"]
    if not isinstance(taxon, int) or isinstance(taxon, bool) or taxon <= 0:
        raise StaticEntityUniverseError("output protein relation taxon is invalid")
    if not isinstance(protein_id, str) or UNIPROT_CURIE.fullmatch(protein_id) is None:
        raise StaticEntityUniverseError("output proteinId must be a canonical UniProtKB CURIE")
    if not isinstance(accession, str) or UNIPROT_ACCESSION.fullmatch(accession) is None:
        raise StaticEntityUniverseError("output sourceAccession is malformed")
    if protein_id != f"UniProtKB:{accession}":
        raise StaticEntityUniverseError("output proteinId and sourceAccession differ")
    if _strict_fields(record["sourceAccessionType"], ACCESSION_TYPE_FIELDS, label) != {
        "source": "UniProtKB",
        "type": "UniProtKB ID",
        "namespaceInferred": False,
        "caseNormalization": "none",
    }:
        raise StaticEntityUniverseError("output typed UniProt accession contract drift")
    targets = record["currentOrfRelations"]
    if (
        not isinstance(targets, list)
        or not targets
        or any(not isinstance(item, str) or SGD_CURIE.fullmatch(item) is None for item in targets)
        or targets != sorted(targets)
        or len(targets) != len(set(targets))
    ):
        raise StaticEntityUniverseError("output currentOrfRelations must be sorted unique SGD CURIEs")
    if (
        type(record["currentOrfRelationCount"]) is not int
        or record["currentOrfRelationCount"] != len(targets)
    ):
        raise StaticEntityUniverseError("output protein relation cardinality drift")
    if record["chooseFirstAllowed"] is not False:
        raise StaticEntityUniverseError("output protein relations may never select a first gene")
    return record


def _validate_snapshot_identity(value: object, label: str) -> dict[str, Any]:
    identity = _strict_fields(
        value,
        {"resource", "revision", "manifestDigest", "innerManifestSha256", "recordsSha256", "records"},
        label,
    )
    _, revision = _dataset_resource(identity["resource"], f"{label}.resource")
    if identity["revision"] != revision:
        raise StaticEntityUniverseError(f"{label} resource/revision mismatch")
    _prefixed_digest(identity["manifestDigest"], f"{label}.manifestDigest")
    _bare_digest(identity["innerManifestSha256"], f"{label}.innerManifestSha256")
    _bare_digest(identity["recordsSha256"], f"{label}.recordsSha256")
    if not isinstance(identity["records"], int) or isinstance(identity["records"], bool) or identity["records"] < 0:
        raise StaticEntityUniverseError(f"{label}.records must be a non-negative integer")
    return identity


def _validate_file_ref(
    value: object,
    expected_path: str,
    payload: bytes,
    records: int,
    label: str,
) -> dict[str, Any]:
    reference = _strict_fields(value, {"path", "sha256", "bytes", "records"}, label)
    expected = _file_ref(expected_path, payload, records)
    if (
        reference["path"] != expected_path
        or not isinstance(reference["sha256"], str)
        or reference["sha256"] != expected["sha256"]
        or type(reference["bytes"]) is not int
        or reference["bytes"] != expected["bytes"]
        or type(reference["records"]) is not int
        or reference["records"] != records
    ):
        raise StaticEntityUniverseError(f"{label} content declaration drift")
    return reference


def validate_archive(path: str | Path, bounds: Bounds) -> dict[str, Any]:
    """Re-verify deterministic bytes, schemas, identities, closure, and hashes."""
    archive_path = Path(path)
    if archive_path.is_symlink() or not archive_path.is_file():
        raise StaticEntityUniverseError("entity-universe archive must be a regular file")
    if archive_path.stat().st_size > bounds.max_archive_bytes:
        raise StaticEntityUniverseError("entity-universe archive exceeds maxArchiveBytes")
    expected_names = [
        "static-entity-universe/entities.jsonl",
        "static-entity-universe/manifest.json",
        "static-entity-universe/relations.jsonl",
    ]
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            members = archive.getmembers()
            if [member.name for member in members] != expected_names:
                raise StaticEntityUniverseError("archive member set or order drift")
            for member in members:
                if (
                    not member.isfile() or member.mode != 0o644 or member.mtime != 0
                    or member.uid != 0 or member.gid != 0 or member.uname != "" or member.gname != ""
                    or member.pax_headers
                ):
                    raise StaticEntityUniverseError("archive member metadata drift")
            blobs = {}
            for member in members:
                stream = archive.extractfile(member)
                if stream is None:
                    raise StaticEntityUniverseError("archive member is not readable")
                blobs[member.name] = stream.read()
    except (OSError, tarfile.TarError) as error:
        raise StaticEntityUniverseError("entity-universe archive is invalid") from error
    if _tar_bytes(blobs) != archive_path.read_bytes():
        raise StaticEntityUniverseError("archive bytes are not canonical deterministic USTAR")
    manifest_payload = blobs["static-entity-universe/manifest.json"]
    try:
        manifest = json.loads(manifest_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StaticEntityUniverseError("entity-universe manifest is invalid") from error
    if not isinstance(manifest, dict) or manifest_payload != _pretty_json_bytes(manifest):
        raise StaticEntityUniverseError("entity-universe manifest is not canonical JSON")
    manifest = _strict_fields(
        manifest,
        {
            "schema", "version", "identityKey", "ordering", "source",
            "identityMapping", "semanticSetHashes", "inputs", "entities",
            "relations", "contentPolicy",
        },
        "entity-universe manifest",
    )
    if (
        manifest["schema"] != UNIVERSE_SCHEMA
        or type(manifest["version"]) is not int
        or manifest["version"] != 1
    ):
        raise StaticEntityUniverseError("entity-universe manifest schema drift")
    if manifest["identityKey"] != ["ncbiTaxon", "entityId"] or manifest["ordering"] != "ascending-ncbiTaxon-then-codepoint-entityId":
        raise StaticEntityUniverseError("entity-universe identity or ordering contract drift")

    source = _strict_fields(manifest["source"], {"id", "release", "ncbiTaxon"}, "source")
    _nonempty(source["id"], "source.id")
    _nonempty(source["release"], "source.release")
    source_taxon = source["ncbiTaxon"]
    if not isinstance(source_taxon, int) or isinstance(source_taxon, bool) or source_taxon <= 0:
        raise StaticEntityUniverseError("source.ncbiTaxon must be positive")
    mapping = _strict_fields(manifest["identityMapping"], {"id", "sha256"}, "identityMapping")
    _nonempty(mapping["id"], "identityMapping.id")
    _bare_digest(mapping["sha256"], "identityMapping.sha256")
    inputs = _strict_fields(
        manifest["inputs"], {"interventionInventory", "proteinRelations"}, "inputs"
    )
    intervention_identity = _validate_snapshot_identity(
        inputs["interventionInventory"], "inputs.interventionInventory"
    )
    relation_identity = _validate_snapshot_identity(
        inputs["proteinRelations"], "inputs.proteinRelations"
    )

    entity_payload = blobs["static-entity-universe/entities.jsonl"]
    relation_payload = blobs["static-entity-universe/relations.jsonl"]
    entity_records = _jsonl_blob(
        entity_payload,
        bounds,
        min(20_000_000, bounds.max_archive_bytes // 64 + 1),
        "entities.jsonl",
    )
    entities = canonicalize_entities(entity_records)
    if entities != entity_records:
        raise StaticEntityUniverseError("entity records are duplicated or not in composite-key order")
    relation_records = _jsonl_blob(
        relation_payload, bounds, bounds.max_relation_records, "relations.jsonl"
    )
    relations = [
        _validate_output_relation(record, f"relations.jsonl:{index}")
        for index, record in enumerate(relation_records, start=1)
    ]
    relation_keys = [(record["ncbiTaxon"], record["proteinId"]) for record in relations]
    if relation_keys != sorted(relation_keys) or len(relation_keys) != len(set(relation_keys)):
        raise StaticEntityUniverseError("relation records are duplicated or not in composite-key order")
    if any(item["ncbiTaxon"] != source_taxon for item in entities) or any(
        item["ncbiTaxon"] != source_taxon for item in relations
    ):
        raise StaticEntityUniverseError("output entity or relation taxon differs from source")

    entity_by_key = {(item["ncbiTaxon"], item["entityId"]): item for item in entities}
    action_keys = {
        key for key, item in entity_by_key.items() if "action" in item["usages"]
    }
    support_keys = {
        key for key, item in entity_by_key.items() if "relation-support" in item["usages"]
    }
    protein_keys = {
        key for key, item in entity_by_key.items() if item["entityClass"] == "protein"
    }
    related_keys = {
        (record["ncbiTaxon"], target)
        for record in relations
        for target in record["currentOrfRelations"]
    }
    relation_protein_keys = set(relation_keys)
    gene_keys = {
        key for key, item in entity_by_key.items() if item["entityClass"] == "gene"
    }
    if support_keys != related_keys or protein_keys != relation_protein_keys:
        raise StaticEntityUniverseError("entity usages do not preserve exact relation closure")
    if gene_keys != action_keys | related_keys:
        raise StaticEntityUniverseError("gene universe contains an unbound or missing identity")

    entity_section = _strict_fields(
        manifest["entities"], {"format", "file", "counts"}, "entities"
    )
    if entity_section["format"] != ENTITY_SCHEMA:
        raise StaticEntityUniverseError("entity output format drift")
    _validate_file_ref(
        entity_section["file"], "entities.jsonl", entity_payload, len(entities), "entities.file"
    )
    entity_counts = _strict_fields(
        entity_section["counts"],
        {
            "genes", "proteins", "actionEligible", "readoutQueryEligible",
            "relationSupport", "relationSupportOnly", "currentModelEligibleKeys",
            "totalSourceUniverse",
        },
        "entities.counts",
    )
    expected_entity_counts = {
        "genes": len(gene_keys),
        "proteins": len(protein_keys),
        "actionEligible": len(action_keys),
        "readoutQueryEligible": len(protein_keys),
        "relationSupport": len(support_keys),
        "relationSupportOnly": len(support_keys - action_keys),
        "currentModelEligibleKeys": len(action_keys) + len(protein_keys),
        "totalSourceUniverse": len(entities),
    }
    for key, expected_value in expected_entity_counts.items():
        if type(entity_counts[key]) is not int or entity_counts[key] != expected_value:
            raise StaticEntityUniverseError("entity counts do not match canonical records")

    relation_section = _strict_fields(
        manifest["relations"],
        {
            "format", "file", "relationSetSha256", "edges", "oneToManyRecords",
            "chooseFirstAllowed", "targetGenes", "targetsInUniverse",
        },
        "relations",
    )
    if relation_section["format"] != PROTEIN_RECORD_SCHEMA or relation_section["chooseFirstAllowed"] is not False:
        raise StaticEntityUniverseError("relation output format or ambiguity policy drift")
    relation_ref = _validate_file_ref(
        relation_section["file"], "relations.jsonl", relation_payload, len(relations), "relations.file"
    )
    edge_count = sum(len(record["currentOrfRelations"]) for record in relations)
    expected_relation_values = {
        "relationSetSha256": relation_ref["sha256"],
        "edges": edge_count,
        "oneToManyRecords": sum(len(record["currentOrfRelations"]) > 1 for record in relations),
        "targetGenes": len(related_keys),
        "targetsInUniverse": len(related_keys),
    }
    for key, value in expected_relation_values.items():
        if type(value) is int and type(relation_section[key]) is not int:
            raise StaticEntityUniverseError(f"relations.{key} must be an integer")
        if relation_section[key] != value:
            raise StaticEntityUniverseError(f"relations.{key} does not match canonical records")

    hashes = _strict_fields(
        manifest["semanticSetHashes"],
        {
            "framing", "actionIdSet", "proteinIdSet", "relationEdgeSet",
            "fullEntityIdSet", "fullEntityKeySet",
        },
        "semanticSetHashes",
    )
    if hashes["framing"] != "unique-ASCII-items-ordinal-sort-LF-terminated":
        raise StaticEntityUniverseError("semantic-set framing drift")
    expected_hashes = {
        "actionIdSet": (
            "interventionId",
            framed_ascii_set_sha256(key[1] for key in action_keys),
        ),
        "proteinIdSet": (
            "proteinId",
            framed_ascii_set_sha256(key[1] for key in protein_keys),
        ),
        "relationEdgeSet": (
            "proteinId-TAB-currentOrfRelation",
            framed_ascii_set_sha256(
                f"{record['proteinId']}\t{target}"
                for record in relations
                for target in record["currentOrfRelations"]
            ),
        ),
        "fullEntityKeySet": (
            "ncbiTaxon-TAB-entityId",
            framed_composite_key_set_sha256(entity_by_key),
        ),
    }
    for name, (basis, digest) in expected_hashes.items():
        value = _strict_fields(hashes[name], {"basis", "sha256"}, f"semanticSetHashes.{name}")
        if value["basis"] != basis or value["sha256"] != digest:
            raise StaticEntityUniverseError(f"semanticSetHashes.{name} digest drift")
    full_id_hash = _strict_fields(
        hashes["fullEntityIdSet"], {"basis", "ncbiTaxon", "sha256"}, "semanticSetHashes.fullEntityIdSet"
    )
    if (
        full_id_hash["basis"] != "union-of-action-protein-and-relation-target-IDs-under-one-bound-ncbiTaxon"
        or full_id_hash["ncbiTaxon"] != source_taxon
        or full_id_hash["sha256"] != framed_ascii_set_sha256(key[1] for key in entity_by_key)
    ):
        raise StaticEntityUniverseError("semanticSetHashes.fullEntityIdSet drift")

    policy = _strict_fields(
        manifest["contentPolicy"],
        {
            "containsDisplaySymbols", "containsNumericFeatures", "containsOutcomesOrLabels",
            "containsTrainingPartitionAssignments", "crossTaxonIdentityMerge",
        },
        "contentPolicy",
    )
    if any(value is not False for value in policy.values()):
        raise StaticEntityUniverseError("content policy must remain entirely false")
    if intervention_identity["records"] < len(action_keys) or relation_identity["records"] != len(relations):
        raise StaticEntityUniverseError("input provenance record counts conflict with output")
    return manifest


def build_entity_universe(
    intervention_dataset: PinnedDataset,
    relation_dataset: PinnedDataset,
    destination: str | Path,
    bounds: Bounds,
    *,
    expected: ExpectedContract = PRODUCTION_CONTRACT,
) -> dict[str, object]:
    """Construct one deterministic archive and an external audit file."""
    actions, intervention_identity, intervention_details = _load_interventions(
        intervention_dataset, bounds, expected
    )
    readouts, relation_support, relations, relation_identity, relation_details = _load_relations(
        relation_dataset, bounds, expected
    )
    for label, details in (
        ("intervention", intervention_details), ("protein relation", relation_details)
    ):
        if details["mappingId"] != expected.mapping_id or details["mappingSha256"] != expected.mapping_sha256:
            raise StaticEntityUniverseError(f"{label} identity mapping drift")
    for key in ("sourceId", "sourceRelease", "ncbiTaxon"):
        if intervention_details[key] != relation_details[key]:
            raise StaticEntityUniverseError(f"source {key} differs between identity snapshots")

    entities = canonicalize_entities([*actions, *relation_support, *readouts])
    entity_bytes = _jsonl_bytes(entities)
    relation_bytes = _jsonl_bytes(relations)
    action_count = sum("action" in item["usages"] for item in entities)
    readout_count = sum("readout-query" in item["usages"] for item in entities)
    relation_support_count = sum("relation-support" in item["usages"] for item in entities)
    support_only_count = sum(item["usages"] == ["relation-support"] for item in entities)
    action_keys = {
        (item["ncbiTaxon"], item["entityId"])
        for item in entities
        if "action" in item["usages"]
    }
    related_keys = {
        (record["ncbiTaxon"], target)
        for record in relations
        for target in record["currentOrfRelations"]
    }
    relation_targets_present = len(action_keys & related_keys)
    relation_targets_external = len(related_keys - action_keys)
    if relation_targets_present != expected.relation_targets_in_action_universe:
        raise StaticEntityUniverseError("relation target/action overlap count drift")
    if relation_targets_external != expected.relation_support_only or support_only_count != expected.relation_support_only:
        raise StaticEntityUniverseError("relation-support-only entity count drift")
    if len(entities) != expected.total_entities:
        raise StaticEntityUniverseError("relation-closed entity-universe count drift")
    full_entity_id_set_sha256 = framed_ascii_set_sha256(item["entityId"] for item in entities)
    if full_entity_id_set_sha256 != expected.full_entity_id_set_sha256:
        raise StaticEntityUniverseError("full semantic entity ID-set digest drift")
    full_entity_key_set_sha256 = framed_composite_key_set_sha256(
        (item["ncbiTaxon"], item["entityId"]) for item in entities
    )
    if full_entity_key_set_sha256 != expected.full_entity_key_set_sha256:
        raise StaticEntityUniverseError("full semantic composite entity-key digest drift")
    entity_ref = _file_ref("entities.jsonl", entity_bytes, len(entities))
    relation_ref = _file_ref("relations.jsonl", relation_bytes, len(relations))
    one_to_many = relation_details["oneToMany"]
    manifest = {
        "schema": UNIVERSE_SCHEMA,
        "version": 1,
        "identityKey": ["ncbiTaxon", "entityId"],
        "ordering": "ascending-ncbiTaxon-then-codepoint-entityId",
        "source": {
            "id": intervention_details["sourceId"],
            "release": intervention_details["sourceRelease"],
            "ncbiTaxon": intervention_details["ncbiTaxon"],
        },
        "identityMapping": {"id": expected.mapping_id, "sha256": expected.mapping_sha256},
        "semanticSetHashes": {
            "framing": "unique-ASCII-items-ordinal-sort-LF-terminated",
            "actionIdSet": {
                "basis": "interventionId",
                "sha256": intervention_details["actionIdSetSha256"],
            },
            "proteinIdSet": {
                "basis": "proteinId",
                "sha256": relation_details["proteinIdSetSha256"],
            },
            "relationEdgeSet": {
                "basis": "proteinId-TAB-currentOrfRelation",
                "sha256": relation_details["relationEdgeSetSha256"],
            },
            "fullEntityIdSet": {
                "basis": "union-of-action-protein-and-relation-target-IDs-under-one-bound-ncbiTaxon",
                "ncbiTaxon": intervention_details["ncbiTaxon"],
                "sha256": full_entity_id_set_sha256,
            },
            "fullEntityKeySet": {
                "basis": "ncbiTaxon-TAB-entityId",
                "sha256": full_entity_key_set_sha256,
            },
        },
        "inputs": {
            "interventionInventory": intervention_identity,
            "proteinRelations": relation_identity,
        },
        "entities": {
            "format": ENTITY_SCHEMA,
            "file": entity_ref,
            "counts": {
                "genes": sum(item["entityClass"] == "gene" for item in entities),
                "proteins": sum(item["entityClass"] == "protein" for item in entities),
                "actionEligible": action_count,
                "readoutQueryEligible": readout_count,
                "relationSupport": relation_support_count,
                "relationSupportOnly": support_only_count,
                "currentModelEligibleKeys": action_count + readout_count,
                "totalSourceUniverse": len(entities),
            },
        },
        "relations": {
            "format": PROTEIN_RECORD_SCHEMA,
            "file": relation_ref,
            "relationSetSha256": relation_ref["sha256"],
            "edges": relation_details["edges"],
            "oneToManyRecords": len(one_to_many),
            "chooseFirstAllowed": False,
            "targetGenes": len(related_keys),
            "targetsInUniverse": len(related_keys & {
                (item["ncbiTaxon"], item["entityId"])
                for item in entities
                if item["entityClass"] == "gene"
            }),
        },
        "contentPolicy": {
            "containsDisplaySymbols": False,
            "containsNumericFeatures": False,
            "containsOutcomesOrLabels": False,
            "containsTrainingPartitionAssignments": False,
            "crossTaxonIdentityMerge": False,
        },
    }
    manifest_bytes = _pretty_json_bytes(manifest)
    destination_path = Path(destination).resolve()
    if destination_path.exists() or destination_path.is_symlink():
        raise StaticEntityUniverseError("destination must not already exist")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination_path.name}-", dir=destination_path.parent) as temporary:
        staging = Path(temporary) / destination_path.name
        staging.mkdir()
        archive_path = staging / "entity-universe.tar"
        members = {
            "static-entity-universe/entities.jsonl": entity_bytes,
            "static-entity-universe/manifest.json": manifest_bytes,
            "static-entity-universe/relations.jsonl": relation_bytes,
        }
        _write_tar(archive_path, members)
        validated = validate_archive(archive_path, bounds)
        if validated != manifest:
            raise StaticEntityUniverseError("post-write archive validation changed the manifest")
        archive_sha = _sha256_file(archive_path)
        audit = {
            "schema": AUDIT_SCHEMA,
            "inputs": manifest["inputs"],
            "source": manifest["source"],
            "identityMapping": manifest["identityMapping"],
            "semanticSetHashes": manifest["semanticSetHashes"],
            "outputs": {
                "archiveSha256": archive_sha,
                "manifestSha256": _sha256_bytes(manifest_bytes),
                "entitySetSha256": entity_ref["sha256"],
                "relationSetSha256": relation_ref["sha256"],
            },
            "counts": {
                "interventionRecords": intervention_details["records"],
                "duplicateInterventionRecords": intervention_details["duplicateRecords"],
                "actionEntities": action_count,
                "readoutQueryEntities": readout_count,
                "totalEntities": len(entities),
                "relationRecords": len(relations),
                "relationEdges": relation_details["edges"],
                "oneToManyRelations": len(one_to_many),
                "relationTargetGenes": len(related_keys),
                "relationTargetsInUniverse": len(related_keys),
                "relationTargetsInActionUniverse": relation_targets_present,
                "relationSupportGenes": relation_support_count,
                "relationSupportOnly": relation_targets_external,
                "currentModelEligibleKeys": action_count + readout_count,
            },
            "oneToManyRelations": one_to_many,
            "accessBoundary": {
                "inputSnapshots": ["interventionInventory", "proteinRelations"],
                "heldRosterConsumed": False,
                "outcomesOrLabelsConsumed": False,
                "trainingPartitionAssignmentsConsumed": False,
                "numericFeaturesConsumed": False,
                "displaySymbolsConsumed": False,
                "identifiersMergedAcrossTaxa": False,
            },
            "limitations": [
                "identity and typed relations only; this artifact is not a numeric static feature pack",
                "711 relation-support-only genes close the source relation graph but are not action eligible",
                "this artifact assigns no pretraining, validation, final, benchmark, or reward roles",
            ],
        }
        audit_path = staging / "entity-universe-audit.json"
        audit_path.write_bytes(_pretty_json_bytes(audit))
        audit_sha = _sha256_file(audit_path)
        staging.replace(destination_path)
    return {
        "audit": audit,
        "archiveSha256": archive_sha,
        "auditSha256": audit_sha,
        "manifestSha256": audit["outputs"]["manifestSha256"],
        "entitySetSha256": entity_ref["sha256"],
        "relationSetSha256": relation_ref["sha256"],
        "actionIdSetSha256": intervention_details["actionIdSetSha256"],
        "proteinIdSetSha256": relation_details["proteinIdSetSha256"],
        "relationEdgeSetSha256": relation_details["relationEdgeSetSha256"],
        "fullEntityIdSetSha256": full_entity_id_set_sha256,
        "fullEntityKeySetSha256": full_entity_key_set_sha256,
        "actionEntities": action_count,
        "readoutQueryEntities": readout_count,
        "totalEntities": len(entities),
        "relationRecords": len(relations),
        "relationEdges": relation_details["edges"],
        "relationTargetGenes": len(related_keys),
        "relationTargetsInUniverse": len(related_keys),
        "relationSupportOnly": relation_targets_external,
        "oneToManyRelations": len(one_to_many),
    }
