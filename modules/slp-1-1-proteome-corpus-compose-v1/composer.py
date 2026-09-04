"""Compose the exact yeast proteome observations and static features as corpus v1.2.

This module is intentionally production-pinned.  It accepts only the three
copy-materialized DatasetSnapshots named in ``PRODUCTION_CONTRACT`` and emits
one deterministic, composite-keyed corpus.  Identifier strings remain
provenance; numerical model inputs contain only declared static and species
features plus the molecular targets already present in the fitting snapshot.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
import tempfile
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

CORPUS_SCHEMA = "slp.corpus/v1.2"
FEATURE_PACK_SCHEMA = "slp.static-feature-pack/v1"
TRAJECTORY_SCHEMA = "slp.trajectory-intervention/v1"
AUDIT_SCHEMA = "slp.proteome-corpus-compose-audit/v1"
SOURCE_SCHEMA = "slp.source-observation-archive/v1"
FEATURE_SCHEMA = "slp.sequence-statistics-feature-block/v1"
FEATURE_ENTITY_SCHEMA = "slp.static-feature-entity/v1"
ROSTER_SCHEMA = "slp.held-intervention-roster-report/v1"
SPECIES_TAXON = 4932
FEATURE_DIM = 21
SPECIES_FEATURE_DIM = 1
CONTEXT_ID = "slp-context:mendeley-w8jtmnszd9-v2-prototrophic-sm"
CONTEXT_TYPE = "slp-context-type:prototrophic-synthetic-medium"
ACTION_TYPE = "slp-action:gene-deletion"
READOUT_TYPE = "slp-readout:proteome-relative-intensity"
ENTITY_TYPES = [
    "slp-entity-type:gene",
    "slp-entity-type:protein",
    "slp-entity-type:context",
]
SAMPLING_SCHEME = "slp.source-intervention-replicate-record/v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RESOURCE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
INPUT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
SGD_RE = re.compile(r"^SGD:S[0-9]{9}$")
UNIPROT_RE = re.compile(r"^UniProtKB:[A-Z0-9][A-Z0-9-]{0,31}$")

FEATURE_MEMBERS = (
    "static-feature-block/entities.jsonl",
    "static-feature-block/excluded-non-current.jsonl",
    "static-feature-block/manifest.json",
    "static-feature-block/present.npy",
    "static-feature-block/sequence-provenance.jsonl",
    "static-feature-block/values.npy",
)
OUTPUT_STATIC_MEMBERS = (
    "composite-corpus/corpus.json",
    "composite-corpus/entities.npz",
    "composite-corpus/queries.npz",
    "composite-corpus/query-panels.npz",
    "composite-corpus/trajectory-interventions.jsonl",
)
SOURCE_SHARD_ARRAYS = {
    "action_id",
    "centering_group",
    "injection_index",
    "matrix_column",
    "metadata_row",
    "observation_unit_id",
    "perturbation_id",
    "plate_index",
    "record_id",
    "replicate_id",
    "species_taxon",
    "target_indptr",
    "target_readout_index",
    "target_value",
    "well_index",
}
OUTPUT_SHARD_ARRAYS = {
    "record_id",
    "observation_unit_id",
    "source_index",
    "replicate_id",
    "perturbation_id",
    "species_taxon",
    "species_feature_value",
    "species_feature_present",
    "context_entity_index",
    "context_type",
    "context_mask",
    "context_covariate_value",
    "context_covariate_present",
    "record_covariate_value",
    "record_covariate_present",
    "action_entity_index",
    "action_type",
    "action_mask",
    "action_covariate_value",
    "action_covariate_present",
    "observation_covariate_value",
    "observation_covariate_present",
    "query_panel_index",
    "target_indptr",
    "target_query_index",
    "target_value",
}


class CorpusComposeError(ValueError):
    """An input or output violates the frozen composite-corpus contract."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def canonical_json_bytes(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise CorpusComposeError(f"could not hash {path.name}") from error
    return digest.hexdigest()


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CorpusComposeError(f"{label} must be a non-empty trimmed string")
    return value


def _bare_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise CorpusComposeError(f"{label} must be a lowercase SHA-256")
    return value


def _prefixed_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise CorpusComposeError(f"{label} must use the sha256: prefix")
    _bare_digest(value.removeprefix("sha256:"), label)
    return value


def _canonical_relative(value: object, label: str) -> str:
    relative = _nonempty(value, label)
    posix = PurePosixPath(relative)
    if (
        relative != posix.as_posix()
        or posix.is_absolute()
        or "\\" in relative
        or ":" in relative
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise CorpusComposeError(f"{label} is not a canonical relative path")
    return relative


def _strict(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise CorpusComposeError(f"{label} fields drift")
    return value


def _fixed_strings(values: Sequence[str]) -> np.ndarray:
    width = max(1, *(len(value) for value in values))
    return np.asarray(values, dtype=f"<U{width}")


@dataclass(frozen=True)
class FileSpec:
    name: str
    bytes: int
    sha256: str
    mode: int = 0o644

    def __post_init__(self) -> None:
        _canonical_relative(self.name, "file name")
        if type(self.bytes) is not int or self.bytes < 0:
            raise CorpusComposeError("file bytes must be non-negative")
        _bare_digest(self.sha256, "file SHA-256")


@dataclass(frozen=True)
class ExpectedDataset:
    resource: str
    manifest_digest: str
    tree_digest: str
    files: tuple[FileSpec, ...]


@dataclass(frozen=True)
class ExpectedContract:
    observations: ExpectedDataset
    features: ExpectedDataset
    roster: ExpectedDataset
    records: int
    trajectory_interventions: int
    readouts: int
    target_values: int
    feature_rows: int
    roster_rows: int


@dataclass(frozen=True)
class PinnedDataset:
    input_name: str
    path: Path
    resource: str
    revision: str
    manifest_digest: str


@dataclass(frozen=True)
class Bounds:
    max_manifest_bytes: int = 2 * 1024 * 1024
    max_line_bytes: int = 64 * 1024
    max_archive_bytes: int = 256 * 1024 * 1024
    max_records: int = 100_000
    max_entities: int = 100_000
    max_readouts: int = 100_000
    max_target_values: int = 20_000_000

    def __post_init__(self) -> None:
        for name, value, minimum, maximum in (
            ("maxManifestBytes", self.max_manifest_bytes, 256, 32 * 1024 * 1024),
            ("maxLineBytes", self.max_line_bytes, 128, 4 * 1024 * 1024),
            ("maxArchiveBytes", self.max_archive_bytes, 1024, 4 * 1024**3),
            ("maxRecords", self.max_records, 1, 20_000_000),
            ("maxEntities", self.max_entities, 1, 20_000_000),
            ("maxReadouts", self.max_readouts, 1, 20_000_000),
            ("maxTargetValues", self.max_target_values, 1, 2_000_000_000),
        ):
            if type(value) is not int or not minimum <= value <= maximum:
                raise CorpusComposeError(
                    f"{name} must be an integer in [{minimum}, {maximum}]"
                )


PRODUCTION_CONTRACT = ExpectedContract(
    observations=ExpectedDataset(
        resource=(
            "omf://abiome/slp/datasetsnapshot/slp-1-1-proteome-observation-pretrain-v1@"
            "sha256:631f66e32a218e167af9edb60115a04514d0bcf675a13bcb244c465ffab2f751"
        ),
        manifest_digest="sha256:0bc00463f8641fc91d6fcb82266b6f41d4c55cc78275b737eaad257dd2053130",
        tree_digest="sha256:fc1f812308af999c601bee9b53ce21035bdd6fd9952cead11451c72b612a833f",
        files=(
            FileSpec(
                "observation-corpus.tar",
                59_535_360,
                "1f533d7dfb5bd76489b5b4576268e5d5b58fc6200416362876b5a2301c611f0b",
            ),
        ),
    ),
    features=ExpectedDataset(
        resource=(
            "omf://abiome/slp/datasetsnapshot/slp-1-1-sequence-statistics-feature-block-v1@"
            "sha256:e9733974c551bca3af93c4cb488972f5167da5e7e3cf48ef5803348cd20d91e5"
        ),
        manifest_digest="sha256:6b4b32c794d7787b9b9076d78726ea0ad7706d64fd82b5f918f0c6da20da0d2a",
        tree_digest="sha256:3f4549114a181c162596d60ef1b94d222ec494282d23ece8da7e19142135cb8d",
        files=(
            FileSpec(
                "sequence-feature-block-audit.json",
                7_851,
                "5d3a9fba29e9c31979fbda5a07951f244b66b35cf6c45de53c27fd231586a5e7",
            ),
            FileSpec(
                "sequence-feature-block.tar",
                4_392_960,
                "1b0aaec738b10ad3baa082d907d0c962c35c9b159b89fffca893fa1ecf5a7bed",
            ),
        ),
    ),
    roster=ExpectedDataset(
        resource=(
            "omf://abiome/slp/datasetsnapshot/slp-1-1-held-roster-v1@"
            "sha256:1b9a4800370a5398bf83e0a636007f466bf6ca5a6232e2ebb8fc64c5beb63450"
        ),
        manifest_digest="sha256:f8aac504a2d56fdc9e13cc9b1c9fa87a08ebc7ff2d7036c0b6b135c26d187425",
        tree_digest="sha256:ba62f5855f46e693f2a27f4ed06efeec046ccd99c9993c145f9983807dfed0b1",
        files=(
            FileSpec(
                "coverage.json",
                262_302,
                "c746218cbe5a8312e4d00f771d2155ab902d33795381b8c14ada1f9a876e1cbf",
            ),
            FileSpec(
                "held-intervention-roster.tsv",
                248_524,
                "c27eb11a20f593235131f28fc29d8fbd69735f8a0aea88736104850bb875117a",
            ),
        ),
    ),
    records=3_811,
    trajectory_interventions=3_679,
    readouts=1_850,
    target_values=6_865_493,
    feature_rows=7_037,
    roster_rows=2_700,
)


def _dataset_resource(value: object, label: str) -> tuple[str, str]:
    resource = _nonempty(value, label)
    if not resource.startswith("omf://"):
        raise CorpusComposeError(f"{label} must be an OMF DatasetSnapshot URI")
    identity, separator, revision = resource.removeprefix("omf://").rpartition("@")
    if not separator:
        raise CorpusComposeError(f"{label} must carry an exact revision")
    _prefixed_digest(revision, f"{label} revision")
    parts = identity.split("/")
    if (
        len(parts) < 3
        or parts[-2] != "datasetsnapshot"
        or RESOURCE_NAME_RE.fullmatch(parts[-1]) is None
    ):
        raise CorpusComposeError(f"{label} must identify a DatasetSnapshot")
    return parts[-1], revision


def _resolved_directory(value: object, label: str) -> Path:
    path = Path(_nonempty(str(value), label)).absolute()
    cursor = path
    while True:
        if cursor.is_symlink():
            raise CorpusComposeError(f"{label} must not contain a symlink")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise CorpusComposeError(f"{label} does not exist") from error
    if not resolved.is_dir():
        raise CorpusComposeError(f"{label} must be a directory")
    return resolved


def _reject_symlink_components(path: Path, label: str) -> None:
    cursor = path.absolute()
    while True:
        if cursor.is_symlink():
            raise CorpusComposeError(f"{label} must not contain a symlink")
        if cursor.parent == cursor:
            return
        cursor = cursor.parent


def resolve_pinned_dataset(value: object, input_name: str) -> PinnedDataset:
    if INPUT_NAME_RE.fullmatch(input_name) is None or not isinstance(value, dict):
        raise CorpusComposeError(f"{input_name} must be a materialized DatasetSnapshot")
    if set(value) != {"resource", "mode", "path", "manifestDigest"}:
        raise CorpusComposeError(f"{input_name} has a spoofed DatasetSnapshot shape")
    resource_name, revision = _dataset_resource(
        value["resource"], f"{input_name}.resource"
    )
    if value["mode"] != "copy":
        raise CorpusComposeError(f"{input_name} must be copied, not mutable")
    manifest = _prefixed_digest(value["manifestDigest"], f"{input_name}.manifestDigest")
    root = _resolved_directory(value["path"], f"{input_name}.path")
    if (
        root.name != resource_name
        or root.parent.name != input_name
        or root.parent.parent.name != "inputs"
    ):
        raise CorpusComposeError(
            f"{input_name}.path is inconsistent with OMF materialization"
        )
    return PinnedDataset(input_name, root, str(value["resource"]), revision, manifest)


def _omf_tree_digest(files: Iterable[FileSpec]) -> str:
    entries = [
        {
            "path": item.name,
            "mode": item.mode,
            "size": item.bytes,
            "digest": f"sha256:{item.sha256}",
        }
        for item in sorted(files, key=lambda item: item.name)
    ]
    return f"sha256:{sha256_bytes(canonical_json(entries).encode('utf-8'))}"


def _regular_child(root: Path, name: str) -> Path:
    relative = _canonical_relative(name, "snapshot file")
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor /= part
        if cursor.is_symlink():
            raise CorpusComposeError(f"snapshot file must not be a symlink: {name}")
    try:
        path = cursor.resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as error:
        raise CorpusComposeError(
            f"snapshot file is missing or escapes its root: {name}"
        ) from error
    if not path.is_file():
        raise CorpusComposeError(f"snapshot member is not a regular file: {name}")
    return path


def verify_dataset(
    dataset: PinnedDataset, expected: ExpectedDataset
) -> dict[str, Path]:
    if (
        dataset.resource != expected.resource
        or dataset.manifest_digest != expected.manifest_digest
    ):
        raise CorpusComposeError(f"{dataset.input_name} immutable identity drift")
    if _omf_tree_digest(expected.files) != expected.tree_digest:
        raise CorpusComposeError(
            f"{dataset.input_name} expected tree is internally inconsistent"
        )
    expected_names = {item.name for item in expected.files}
    actual_paths = [item for item in dataset.path.rglob("*")]
    if any(item.is_symlink() for item in actual_paths):
        raise CorpusComposeError(f"{dataset.input_name} contains a symlink")
    if any(item.is_dir() for item in actual_paths):
        raise CorpusComposeError(
            f"{dataset.input_name} contains an undeclared directory"
        )
    actual_names = {
        item.relative_to(dataset.path).as_posix()
        for item in actual_paths
        if item.is_file()
    }
    if actual_names != expected_names:
        raise CorpusComposeError(f"{dataset.input_name} file set drift")
    result: dict[str, Path] = {}
    for spec in expected.files:
        path = _regular_child(dataset.path, spec.name)
        if path.stat().st_size != spec.bytes or sha256_file(path) != spec.sha256:
            raise CorpusComposeError(f"{dataset.input_name}/{spec.name} content drift")
        result[spec.name] = path
    return result


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    _canonical_relative(name, "tar member")
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.type = tarfile.REGTYPE
    return info


def deterministic_tar_bytes(members: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(members):
            payload = members[name]
            archive.addfile(_tar_info(name, len(payload)), io.BytesIO(payload))
    return output.getvalue()


def read_canonical_tar(path: Path, bounds: Bounds, label: str) -> dict[str, bytes]:
    _reject_symlink_components(path, label)
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > bounds.max_archive_bytes
    ):
        raise CorpusComposeError(f"{label} is not a bounded regular archive")
    blobs: dict[str, bytes] = {}
    total = 0
    try:
        with tarfile.open(path, mode="r:") as archive:
            members = archive.getmembers()
            if [item.name for item in members] != sorted(item.name for item in members):
                raise CorpusComposeError(f"{label} member order is not canonical")
            for member in members:
                _canonical_relative(member.name, f"{label} member")
                if member.name in blobs:
                    raise CorpusComposeError(f"{label} has a duplicate member")
                if (
                    not member.isfile()
                    or member.mode != 0o644
                    or member.mtime != 0
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.pax_headers
                ):
                    raise CorpusComposeError(f"{label} member metadata drift")
                total += member.size
                if total > bounds.max_archive_bytes:
                    raise CorpusComposeError(f"{label} expanded bytes exceed the bound")
                stream = archive.extractfile(member)
                if stream is None:
                    raise CorpusComposeError(f"{label} member is unreadable")
                payload = stream.read(member.size + 1)
                if len(payload) != member.size:
                    raise CorpusComposeError(f"{label} member size drift")
                blobs[member.name] = payload
    except (OSError, tarfile.TarError) as error:
        raise CorpusComposeError(f"{label} is invalid") from error
    if deterministic_tar_bytes(blobs) != path.read_bytes():
        raise CorpusComposeError(f"{label} is not canonical deterministic USTAR")
    return blobs


def _json_payload(payload: bytes, label: str, bounds: Bounds) -> dict[str, Any]:
    if not payload or len(payload) > bounds.max_manifest_bytes:
        raise CorpusComposeError(f"{label} is empty or exceeds its byte bound")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusComposeError(f"{label} is invalid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise CorpusComposeError(f"{label} must be an object")
    return value


def _canonical_jsonl(
    payload: bytes, label: str, bounds: Bounds, maximum: int
) -> list[dict[str, Any]]:
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise CorpusComposeError(f"{label} is not canonical LF JSONL")
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(payload.splitlines(keepends=True), start=1):
        if number > maximum or len(raw) > bounds.max_line_bytes:
            raise CorpusComposeError(f"{label} exceeds its row or line bound")
        try:
            row = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CorpusComposeError(f"{label}:{number} is invalid JSON") from error
        if not isinstance(row, dict) or canonical_json_bytes(row) != raw:
            raise CorpusComposeError(f"{label}:{number} is not canonical JSONL")
        rows.append(row)
    return rows


def _read_npy(
    payload: bytes, dtype: np.dtype[Any], shape: tuple[int, ...], label: str
) -> np.ndarray:
    stream = io.BytesIO(payload)
    try:
        array = np.lib.format.read_array(stream, allow_pickle=False)
    except (ValueError, EOFError) as error:
        raise CorpusComposeError(f"{label} is not a valid non-object NPY") from error
    if stream.tell() != len(payload):
        raise CorpusComposeError(f"{label} has trailing bytes")
    if array.dtype != dtype or array.shape != shape or not array.flags.c_contiguous:
        raise CorpusComposeError(f"{label} dtype, shape, or layout drift")
    return array


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for name in sorted(arrays):
            if not name or "/" in name or "\\" in name:
                raise CorpusComposeError("NPZ array name is invalid")
            array = np.ascontiguousarray(arrays[name])
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, buffer.getvalue())
    return output.getvalue()


def read_deterministic_npz(
    payload: bytes, expected_names: set[str], label: str
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            names = archive.namelist()
            expected_members = [f"{name}.npy" for name in sorted(expected_names)]
            if names != expected_members:
                raise CorpusComposeError(f"{label} member set or order drift")
            for info in archive.infolist():
                if (
                    info.compress_type != zipfile.ZIP_STORED
                    or info.date_time != (1980, 1, 1, 0, 0, 0)
                    or info.create_system != 3
                    or info.external_attr != 0o100644 << 16
                ):
                    raise CorpusComposeError(f"{label} ZIP metadata drift")
                name = info.filename.removesuffix(".npy")
                stream = io.BytesIO(archive.read(info))
                arrays[name] = np.lib.format.read_array(stream, allow_pickle=False)
                if stream.tell() != len(stream.getbuffer()):
                    raise CorpusComposeError(f"{label}/{name} has trailing NPY bytes")
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, CorpusComposeError):
            raise
        raise CorpusComposeError(f"{label} is not a valid deterministic NPZ") from error
    if deterministic_npz_bytes(arrays) != payload:
        raise CorpusComposeError(f"{label} is not byte-deterministic")
    return arrays


def _file_ref(path: str, payload: bytes) -> dict[str, object]:
    return {"path": path, "bytes": len(payload), "sha256": sha256_bytes(payload)}


def _input_identity(expected: ExpectedDataset) -> dict[str, object]:
    _, revision = _dataset_resource(expected.resource, "expected resource")
    return {
        "resource": expected.resource,
        "revision": revision,
        "outerManifestDigest": expected.manifest_digest,
        "treeDigest": expected.tree_digest,
        "files": [
            {"path": item.name, "bytes": item.bytes, "sha256": item.sha256}
            for item in sorted(expected.files, key=lambda item: item.name)
        ],
    }


def composite_key_sha256(keys: Iterable[tuple[int, str]]) -> str:
    documents = [
        {"entityId": entity, "ncbiTaxon": taxon} for taxon, entity in sorted(set(keys))
    ]
    return sha256_bytes(canonical_json(documents).encode("ascii"))


def composite_perturbation_id(keys: Iterable[tuple[int, str]]) -> str:
    documents = [
        {"entityId": entity, "ncbiTaxon": taxon} for taxon, entity in sorted(set(keys))
    ]
    if not documents:
        raise CorpusComposeError(
            "a perturbation requires at least one composite action"
        )
    digest = sha256_bytes(canonical_json(documents).encode("ascii"))
    return f"slp-perturbation:sha256-{digest}"


def validate_held_intervention_boundary(
    trajectory_keys: Iterable[tuple[int, str]],
    roles: Mapping[tuple[int, str], str],
) -> None:
    protected = sorted(
        key
        for key in set(trajectory_keys)
        if roles.get(key) in {"molecular-validation", "molecular-final"}
    )
    if protected:
        raise CorpusComposeError(
            f"fitting source contains protected roster interventions: {protected[:10]}"
        )


def _array_bytes_sha256(arrays: Iterable[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class FeatureData:
    manifest: dict[str, Any]
    keys: tuple[tuple[int, str], ...]
    values: np.ndarray
    present: np.ndarray
    manifest_sha256: str
    entity_rows_sha256: str
    values_sha256: str
    present_sha256: str


@dataclass(frozen=True)
class RosterData:
    roles: dict[tuple[int, str], str]
    roster_sha256: str
    coverage_sha256: str


@dataclass(frozen=True)
class ObservationData:
    manifest: dict[str, Any]
    readout_keys: tuple[tuple[int, str], ...]
    trajectory_keys: tuple[tuple[int, str], ...]
    shard_blobs: tuple[bytes, ...]
    manifest_sha256: str
    readout_sha256: str


def _validate_ref(
    value: object, path: str, payload: bytes, label: str, **counts: int
) -> None:
    expected: dict[str, object] = _file_ref(path, payload)
    expected.update(counts)
    if value != expected:
        raise CorpusComposeError(f"{label} does not match its payload")


def parse_feature_snapshot(
    paths: Mapping[str, Path], bounds: Bounds, expected: ExpectedContract
) -> FeatureData:
    blobs = read_canonical_tar(
        paths["sequence-feature-block.tar"], bounds, "feature archive"
    )
    if tuple(blobs) != FEATURE_MEMBERS:
        raise CorpusComposeError("feature archive member set drift")
    manifest_bytes = blobs["static-feature-block/manifest.json"]
    manifest = _json_payload(manifest_bytes, "feature manifest", bounds)
    required = {
        "schema",
        "version",
        "source",
        "identityMapping",
        "identityKey",
        "ordering",
        "featureDefinition",
        "contentPolicy",
        "counts",
        "inputs",
        "files",
        "semanticHashes",
    }
    _strict(manifest, required, "feature manifest")
    counts = manifest["counts"]
    if (
        manifest["schema"] != FEATURE_SCHEMA
        or manifest["version"] != 1
        or manifest["identityKey"] != ["ncbiTaxon", "entityId"]
        or manifest["ordering"] != "ascending-ncbiTaxon-then-codepoint-entityId"
        or not isinstance(counts, dict)
        or counts.get("rows") != expected.feature_rows
        or counts.get("featuresPerRow") != FEATURE_DIM
        or counts.get("missingEntities") != 0
        or counts.get("presentValues") != expected.feature_rows * FEATURE_DIM
    ):
        raise CorpusComposeError("feature manifest identity or count drift")
    policy = manifest["contentPolicy"]
    if not isinstance(policy, dict) or any(
        policy.get(key) is not False
        for key in (
            "containsBenchmarkData",
            "containsIdentifiersAsValues",
            "containsOutcomesOrLabels",
            "containsTrainingPartitionAssignments",
            "crossTaxonIdentityMerge",
        )
    ):
        raise CorpusComposeError("feature block content policy is unsafe")
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != {
        "entities",
        "excludedNonCurrent",
        "present",
        "sequenceProvenance",
        "values",
    }:
        raise CorpusComposeError("feature file references drift")
    _validate_ref(
        files["entities"],
        "entities.jsonl",
        blobs["static-feature-block/entities.jsonl"],
        "feature entities",
        records=expected.feature_rows,
    )
    _validate_ref(
        files["values"],
        "values.npy",
        blobs["static-feature-block/values.npy"],
        "feature values",
    )
    _validate_ref(
        files["present"],
        "present.npy",
        blobs["static-feature-block/present.npy"],
        "feature presence",
    )
    entity_rows = _canonical_jsonl(
        blobs["static-feature-block/entities.jsonl"],
        "feature entities",
        bounds,
        bounds.max_entities,
    )
    if len(entity_rows) != expected.feature_rows:
        raise CorpusComposeError("feature row count drift")
    keys: list[tuple[int, str]] = []
    for index, row in enumerate(entity_rows):
        _strict(
            row,
            {"schema", "ncbiTaxon", "entityId", "rowIndex"},
            f"feature entity {index}",
        )
        key = (row["ncbiTaxon"], row["entityId"])
        if (
            row["schema"] != FEATURE_ENTITY_SCHEMA
            or row["rowIndex"] != index
            or type(key[0]) is not int
            or key[0] != SPECIES_TAXON
            or not isinstance(key[1], str)
            or not (SGD_RE.fullmatch(key[1]) or UNIPROT_RE.fullmatch(key[1]))
        ):
            raise CorpusComposeError(f"feature entity {index} identity drift")
        keys.append(key)
    if keys != sorted(set(keys)):
        raise CorpusComposeError(
            "feature composite keys must be strictly ordered and unique"
        )
    semantic = manifest["semanticHashes"]
    if (
        not isinstance(semantic, dict)
        or semantic.get("entityKeySetSha256")
        != "82b8e2885939577fe6946e3b974a10cb947834118f2070e1bcbe4c2f2e6a5fd9"
    ):
        raise CorpusComposeError("feature entity-key semantic digest drift")
    values = _read_npy(
        blobs["static-feature-block/values.npy"],
        np.dtype("<f4"),
        (expected.feature_rows, FEATURE_DIM),
        "feature values",
    )
    present = _read_npy(
        blobs["static-feature-block/present.npy"],
        np.dtype("|b1"),
        (expected.feature_rows, FEATURE_DIM),
        "feature presence",
    )
    if not bool(present.all()) or not np.isfinite(values).all():
        raise CorpusComposeError("production feature rows must be complete and finite")
    return FeatureData(
        manifest,
        tuple(keys),
        values,
        present,
        sha256_bytes(manifest_bytes),
        sha256_bytes(blobs["static-feature-block/entities.jsonl"]),
        sha256_bytes(blobs["static-feature-block/values.npy"]),
        sha256_bytes(blobs["static-feature-block/present.npy"]),
    )


def parse_roster_snapshot(
    paths: Mapping[str, Path], bounds: Bounds, expected: ExpectedContract
) -> RosterData:
    roster_bytes = paths["held-intervention-roster.tsv"].read_bytes()
    coverage_bytes = paths["coverage.json"].read_bytes()
    if not roster_bytes.endswith(b"\n") or b"\r" in roster_bytes:
        raise CorpusComposeError("held roster is not canonical LF TSV")
    coverage = _json_payload(coverage_bytes, "held roster coverage", bounds)
    _strict(
        coverage,
        {
            "schema",
            "assignment",
            "identityMapping",
            "intersectionSize",
            "minimumIntersectionSize",
            "rejectionCounts",
            "roleCounts",
            "rosterPath",
            "rosterSha256",
            "sourceCount",
            "sources",
        },
        "held roster coverage",
    )
    if (
        coverage["schema"] != ROSTER_SCHEMA
        or coverage["intersectionSize"] != expected.roster_rows
        or coverage["rosterPath"] != "held-intervention-roster.tsv"
        or coverage["rosterSha256"] != sha256_bytes(roster_bytes)
    ):
        raise CorpusComposeError("held roster coverage drift")
    assignment = coverage["assignment"]
    _strict(
        assignment,
        {"bucketRule", "digest", "domainHex", "roles"},
        "held roster assignment",
    )
    if (
        assignment["digest"] != "sha256"
        or assignment["bucketRule"] != "int(first-16-lowercase-hex,16) mod 100"
    ):
        raise CorpusComposeError("held roster assignment algorithm drift")
    try:
        domain = bytes.fromhex(assignment["domainHex"])
    except (TypeError, ValueError) as error:
        raise CorpusComposeError("held roster assignment domain is invalid") from error
    roles: dict[tuple[int, str], str] = {}
    role_counts: dict[str, int] = {
        "pretrain": 0,
        "molecular-validation": 0,
        "molecular-final": 0,
    }
    previous = ""
    for number, raw in enumerate(roster_bytes.splitlines(), start=1):
        if number > bounds.max_entities or len(raw) > bounds.max_line_bytes:
            raise CorpusComposeError("held roster exceeds its row or line bound")
        try:
            identifier, role, digest = raw.decode("ascii").split("\t")
        except (UnicodeDecodeError, ValueError) as error:
            raise CorpusComposeError(
                f"held roster row {number} is malformed"
            ) from error
        if (
            not SGD_RE.fullmatch(identifier)
            or identifier <= previous
            or role not in role_counts
        ):
            raise CorpusComposeError(
                f"held roster row {number} identity or order drift"
            )
        expected_digest = hashlib.sha256(
            domain + identifier.encode("ascii")
        ).hexdigest()
        bucket = int(expected_digest[:16], 16) % 100
        expected_role = (
            "molecular-final"
            if bucket < 10
            else "molecular-validation"
            if bucket < 30
            else "pretrain"
        )
        if digest != expected_digest or role != expected_role:
            raise CorpusComposeError(f"held roster row {number} assignment drift")
        previous = identifier
        roles[(SPECIES_TAXON, identifier)] = role
        role_counts[role] += 1
    if len(roles) != expected.roster_rows or role_counts != coverage["roleCounts"]:
        raise CorpusComposeError("held roster role counts drift")
    return RosterData(roles, sha256_bytes(roster_bytes), sha256_bytes(coverage_bytes))


def _source_members(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    names = [
        "proteome-observations/manifest.json",
        "proteome-observations/readouts.jsonl",
    ]
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise CorpusComposeError("source shard list is invalid")
    for item in shards:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise CorpusComposeError("source shard reference is invalid")
        names.append(
            "proteome-observations/"
            + _canonical_relative(item["path"], "source shard path")
        )
    names.append("proteome-observations/trajectory-genes.txt")
    return tuple(sorted(names))


def parse_observation_snapshot(
    paths: Mapping[str, Path], bounds: Bounds, expected: ExpectedContract
) -> ObservationData:
    blobs = read_canonical_tar(
        paths["observation-corpus.tar"], bounds, "observation archive"
    )
    manifest_bytes = blobs.get("proteome-observations/manifest.json")
    if manifest_bytes is None:
        raise CorpusComposeError("observation manifest is missing")
    manifest = _json_payload(manifest_bytes, "observation manifest", bounds)
    required = {
        "archiveId",
        "assayedPanel",
        "benchmarkLabelsPresent",
        "bounds",
        "context",
        "counts",
        "covariateDefinitions",
        "identity",
        "labelClass",
        "measurement",
        "modalities",
        "partition",
        "plateVocabulary",
        "readoutDictionary",
        "role",
        "runtime",
        "schema",
        "shards",
        "source",
        "speciesTaxa",
        "trajectoryGenes",
        "version",
    }
    _strict(manifest, required, "observation manifest")
    counts = manifest["counts"]
    if (
        tuple(blobs) != _source_members(manifest)
        or manifest["schema"] != SOURCE_SCHEMA
        or manifest["role"] != "pretrain"
        or manifest["labelClass"] != "molecular"
        or manifest["benchmarkLabelsPresent"] is not False
        or manifest["speciesTaxa"] != [SPECIES_TAXON]
        or manifest["context"]
        != {
            "id": CONTEXT_ID,
            "centeringGroup": "slp-center:mendeley-w8jtmnszd9-v2-prototrophic-sm",
        }
        or not isinstance(counts, dict)
        or counts.get("records") != expected.records
        or counts.get("interventionGenes") != expected.trajectory_interventions
        or counts.get("readouts") != expected.readouts
        or counts.get("observedValues") != expected.target_values
    ):
        raise CorpusComposeError("observation boundary, count, or identity drift")
    readout_payload = blobs["proteome-observations/readouts.jsonl"]
    _validate_ref(
        manifest["readoutDictionary"],
        "readouts.jsonl",
        readout_payload,
        "source readout dictionary",
        records=expected.readouts,
    )
    readout_rows = _canonical_jsonl(
        readout_payload, "source readouts", bounds, bounds.max_readouts
    )
    readout_keys: list[tuple[int, str]] = []
    for number, row in enumerate(readout_rows, start=1):
        expected_fields = {
            "schema",
            "proteinId",
            "sourceAccession",
            "sourceAccessionType",
            "ncbiTaxon",
            "currentOrfRelations",
            "currentOrfRelationCount",
            "chooseFirstAllowed",
        }
        _strict(row, expected_fields, f"source readout {number}")
        key = (row["ncbiTaxon"], row["proteinId"])
        if (
            type(key[0]) is not int
            or key[0] != SPECIES_TAXON
            or not UNIPROT_RE.fullmatch(str(key[1]))
        ):
            raise CorpusComposeError(
                f"source readout {number} has an invalid composite identity"
            )
        readout_keys.append((key[0], str(key[1])))
    if (
        readout_keys != sorted(set(readout_keys))
        or len(readout_keys) != expected.readouts
    ):
        raise CorpusComposeError(
            "source readout composite keys are duplicated or unordered"
        )
    trajectory_payload = blobs["proteome-observations/trajectory-genes.txt"]
    _validate_ref(
        manifest["trajectoryGenes"],
        "trajectory-genes.txt",
        trajectory_payload,
        "source trajectory genes",
        records=expected.trajectory_interventions,
    )
    if not trajectory_payload.endswith(b"\n") or b"\r" in trajectory_payload:
        raise CorpusComposeError("source trajectory genes are not canonical LF text")
    genes = trajectory_payload.decode("ascii").splitlines()
    if (
        len(genes) != expected.trajectory_interventions
        or genes != sorted(set(genes))
        or any(SGD_RE.fullmatch(item) is None for item in genes)
    ):
        raise CorpusComposeError("source trajectory gene identity or order drift")
    shard_blobs: list[bytes] = []
    total_records = total_values = 0
    seen_records: set[str] = set()
    active_actions: set[str] = set()
    for index, ref in enumerate(manifest["shards"]):
        _strict(
            ref,
            {"path", "sha256", "bytes", "records", "values"},
            f"source shard {index}",
        )
        path = _canonical_relative(ref["path"], f"source shard {index} path")
        if path != f"shards/shard-{index:05d}.npz":
            raise CorpusComposeError(
                "source shard paths are not contiguous and canonical"
            )
        payload = blobs["proteome-observations/" + path]
        _validate_ref(
            ref,
            path,
            payload,
            f"source shard {index}",
            records=ref["records"],
            values=ref["values"],
        )
        arrays = read_deterministic_npz(
            payload, SOURCE_SHARD_ARRAYS, f"source shard {index}"
        )
        records = ref["records"]
        values = ref["values"]
        _validate_source_shard(arrays, records, values, expected.readouts, index)
        record_ids = {str(item) for item in arrays["record_id"]}
        if len(record_ids) != records or seen_records & record_ids:
            raise CorpusComposeError("source record identities are not globally unique")
        seen_records.update(record_ids)
        active_actions.update(str(item) for item in arrays["action_id"])
        total_records += records
        total_values += values
        shard_blobs.append(payload)
    if (
        total_records != expected.records
        or total_values != expected.target_values
        or active_actions != set(genes)
    ):
        raise CorpusComposeError("source shard totals or active actions drift")
    return ObservationData(
        manifest,
        tuple(readout_keys),
        tuple((SPECIES_TAXON, item) for item in genes),
        tuple(shard_blobs),
        sha256_bytes(manifest_bytes),
        sha256_bytes(readout_payload),
    )


def _require_shape_dtype(
    array: np.ndarray, shape: tuple[int, ...], dtype: str, label: str
) -> None:
    if (
        array.shape != shape
        or array.dtype != np.dtype(dtype)
        or not array.flags.c_contiguous
    ):
        raise CorpusComposeError(f"{label} dtype, shape, or layout drift")


def _validate_source_shard(
    arrays: Mapping[str, np.ndarray],
    records: int,
    values: int,
    readouts: int,
    index: int,
) -> None:
    if (
        type(records) is not int
        or type(values) is not int
        or records <= 0
        or values <= 0
    ):
        raise CorpusComposeError(f"source shard {index} has invalid counts")
    for name in (
        "action_id",
        "centering_group",
        "observation_unit_id",
        "perturbation_id",
        "record_id",
        "replicate_id",
    ):
        array = arrays[name]
        if (
            array.shape != (records,)
            or array.dtype.kind != "U"
            or any(not str(item) for item in array)
        ):
            raise CorpusComposeError(f"source shard {index}/{name} drift")
    for name, dtype in (
        ("injection_index", "<i2"),
        ("well_index", "<i2"),
        ("plate_index", "<i2"),
        ("matrix_column", "<i4"),
        ("metadata_row", "<i4"),
        ("species_taxon", "<i4"),
    ):
        _require_shape_dtype(
            arrays[name], (records,), dtype, f"source shard {index}/{name}"
        )
    _require_shape_dtype(
        arrays["target_indptr"],
        (records + 1,),
        "<i8",
        f"source shard {index}/target_indptr",
    )
    _require_shape_dtype(
        arrays["target_readout_index"],
        (values,),
        "<i4",
        f"source shard {index}/target_readout_index",
    )
    _require_shape_dtype(
        arrays["target_value"], (values,), "<f4", f"source shard {index}/target_value"
    )
    indptr = arrays["target_indptr"]
    if indptr[0] != 0 or indptr[-1] != values or np.any(indptr[1:] < indptr[:-1]):
        raise CorpusComposeError(f"source shard {index} target CSR drift")
    indices = arrays["target_readout_index"]
    if (
        np.any(indices < 0)
        or np.any(indices >= readouts)
        or not np.isfinite(arrays["target_value"]).all()
    ):
        raise CorpusComposeError(f"source shard {index} target values or indices drift")
    if not bool(np.all(arrays["species_taxon"] == SPECIES_TAXON)):
        raise CorpusComposeError(f"source shard {index} contains a swapped taxon")
    for row in range(records):
        selected = indices[int(indptr[row]) : int(indptr[row + 1])]
        if len(selected) != len({int(item) for item in selected}):
            raise CorpusComposeError(f"source shard {index} repeats a target readout")


def _dataset_snapshot_identity(expected: ExpectedDataset) -> dict[str, str]:
    _, revision = _dataset_resource(expected.resource, "expected DatasetSnapshot")
    return {
        "resource": expected.resource,
        "revision": revision,
        "outerManifestDigest": expected.manifest_digest,
        "treeDigest": expected.tree_digest,
    }


def _lineage_input(
    expected: ExpectedDataset, semantic_sha256: str
) -> dict[str, object]:
    _bare_digest(semantic_sha256, "input semanticSha256")
    return {
        "datasetSnapshot": _dataset_snapshot_identity(expected),
        "semanticSha256": semantic_sha256,
        "files": [
            {"path": item.name, "sha256": item.sha256, "bytes": item.bytes}
            for item in sorted(expected.files, key=lambda item: item.name)
        ],
    }


def _feature_pack(
    feature: FeatureData, expected: ExpectedContract
) -> dict[str, object]:
    semantic = _bare_digest(
        feature.manifest["semanticHashes"]["featureDefinitionSha256"],
        "feature definition digest",
    )
    block = {
        "id": "slp-feature-block:sequence-statistics-r64-5-1-v1",
        "offset": 0,
        "dimension": FEATURE_DIM,
        "datasetSnapshot": _dataset_snapshot_identity(expected.features),
        "semanticSha256": semantic,
        "entityKeySetSha256": composite_key_sha256(feature.keys),
        "files": [
            {"path": item.name, "sha256": item.sha256, "bytes": item.bytes}
            for item in sorted(expected.features.files, key=lambda item: item.name)
        ],
    }
    without_sha: dict[str, object] = {
        "schema": FEATURE_PACK_SCHEMA,
        "revision": "slp-feature-pack:sequence-statistics-r64-5-1-v1",
        "entityFeatureDim": FEATURE_DIM,
        "speciesFeatureDim": SPECIES_FEATURE_DIM,
        "blocks": [block],
    }
    return {
        **without_sha,
        "sha256": sha256_bytes(canonical_json(without_sha).encode("ascii")),
    }


def _rewrite_shard(
    payload: bytes,
    shard_index: int,
    entity_lookup: Mapping[tuple[int, str], int],
    query_map: np.ndarray,
    context_index: int,
) -> tuple[bytes, int, int, set[tuple[int, str]], set[int]]:
    source = read_deterministic_npz(
        payload, SOURCE_SHARD_ARRAYS, f"source shard {shard_index}"
    )
    records = len(source["record_id"])
    values = len(source["target_value"])
    actions: list[int] = []
    perturbations: list[str] = []
    active_keys: set[tuple[int, str]] = set()
    for taxon, identifier in zip(source["species_taxon"], source["action_id"]):
        key = (int(taxon), str(identifier))
        if key not in entity_lookup:
            raise CorpusComposeError(
                f"source shard {shard_index} action lacks static entity: {key}"
            )
        active_keys.add(key)
        actions.append(entity_lookup[key])
        perturbations.append(composite_perturbation_id([key]))
    source_target_index = source["target_readout_index"]
    if np.any(source_target_index < 0) or np.any(source_target_index >= len(query_map)):
        raise CorpusComposeError(
            f"source shard {shard_index} readout index is outside the dictionary"
        )
    target_query_index = query_map[source_target_index].astype("<i8", copy=False)
    observation_values = np.column_stack(
        [
            source["injection_index"],
            source["well_index"],
            source["plate_index"],
            source["metadata_row"],
            source["matrix_column"],
        ]
    ).astype("<f4", copy=False)
    output = {
        "record_id": source["record_id"],
        "observation_unit_id": source["observation_unit_id"],
        "source_index": np.zeros(records, dtype="<i8"),
        "replicate_id": source["replicate_id"],
        "perturbation_id": _fixed_strings(perturbations),
        "species_taxon": source["species_taxon"].astype("<i8", copy=False),
        "species_feature_value": np.ones((records, 1), dtype="<f4"),
        "species_feature_present": np.ones((records, 1), dtype="|b1"),
        "context_entity_index": np.full((records, 1), context_index, dtype="<i8"),
        "context_type": np.zeros((records, 1), dtype="<i8"),
        "context_mask": np.ones((records, 1), dtype="|b1"),
        "context_covariate_value": np.zeros((records, 1, 0), dtype="<f4"),
        "context_covariate_present": np.zeros((records, 1, 0), dtype="|b1"),
        "record_covariate_value": np.zeros((records, 0), dtype="<f4"),
        "record_covariate_present": np.zeros((records, 0), dtype="|b1"),
        "action_entity_index": np.asarray(actions, dtype="<i8")[:, None],
        "action_type": np.zeros((records, 1), dtype="<i8"),
        "action_mask": np.ones((records, 1), dtype="|b1"),
        "action_covariate_value": np.zeros((records, 1, 0), dtype="<f4"),
        "action_covariate_present": np.zeros((records, 1, 0), dtype="|b1"),
        "observation_covariate_value": observation_values,
        "observation_covariate_present": np.ones((records, 5), dtype="|b1"),
        "query_panel_index": np.zeros(records, dtype="<i8"),
        "target_indptr": source["target_indptr"],
        "target_query_index": target_query_index,
        "target_value": source["target_value"],
    }
    return (
        deterministic_npz_bytes(output),
        records,
        values,
        active_keys,
        {int(item) for item in target_query_index},
    )


def build_composite_corpus(
    observations: PinnedDataset,
    static_features: PinnedDataset,
    held_roster: PinnedDataset,
    destination: Path,
    bounds: Bounds | None = None,
    expected: ExpectedContract = PRODUCTION_CONTRACT,
) -> dict[str, object]:
    """Build, independently validate, and atomically publish corpus v1.2."""

    bounds = Bounds() if bounds is None else bounds
    observation_paths = verify_dataset(observations, expected.observations)
    feature_paths = verify_dataset(static_features, expected.features)
    roster_paths = verify_dataset(held_roster, expected.roster)
    feature = parse_feature_snapshot(feature_paths, bounds, expected)
    roster = parse_roster_snapshot(roster_paths, bounds, expected)
    source = parse_observation_snapshot(observation_paths, bounds, expected)

    feature_key_set = set(feature.keys)
    if CONTEXT_ID in {entity for _, entity in feature.keys}:
        raise CorpusComposeError(
            "the dedicated context identity collides with a feature entity"
        )
    corpus_keys = (*feature.keys, (SPECIES_TAXON, CONTEXT_ID))
    if list(corpus_keys) != sorted(set(corpus_keys)):
        raise CorpusComposeError(
            "corpus composite entity keys are not strictly ordered and unique"
        )
    if not set(source.readout_keys).issubset(feature_key_set) or not set(
        source.trajectory_keys
    ).issubset(feature_key_set):
        raise CorpusComposeError(
            "source actions or readouts do not resolve through composite feature keys"
        )
    validate_held_intervention_boundary(source.trajectory_keys, roster.roles)

    entity_ids = [entity for _, entity in corpus_keys]
    entity_taxa = np.asarray([taxon for taxon, _ in corpus_keys], dtype="<i8")
    entity_types = np.asarray(
        [
            0 if SGD_RE.fullmatch(entity) else 1 if UNIPROT_RE.fullmatch(entity) else 2
            for entity in entity_ids
        ],
        dtype="<i8",
    )
    entity_values = np.zeros((len(corpus_keys), FEATURE_DIM), dtype="<f4")
    entity_present = np.zeros((len(corpus_keys), FEATURE_DIM), dtype="|b1")
    entity_values[: expected.feature_rows] = feature.values
    entity_present[: expected.feature_rows] = feature.present
    entity_arrays = {
        "entity_taxon": entity_taxa,
        "entity_id": _fixed_strings(entity_ids),
        "entity_type": entity_types,
        "entity_feature_value": entity_values,
        "entity_feature_present": entity_present,
    }
    entity_payload = deterministic_npz_bytes(entity_arrays)
    source_feature_value_sha = _array_bytes_sha256([feature.values])
    source_feature_present_sha = _array_bytes_sha256([feature.present])
    composed_feature_value_sha = _array_bytes_sha256(
        [entity_values[: expected.feature_rows]]
    )
    composed_feature_present_sha = _array_bytes_sha256(
        [entity_present[: expected.feature_rows]]
    )
    if (
        source_feature_value_sha != composed_feature_value_sha
        or source_feature_present_sha != composed_feature_present_sha
    ):
        raise CorpusComposeError("static feature row bytes changed during composition")
    entity_lookup = {key: index for index, key in enumerate(corpus_keys)}
    context_index = entity_lookup[(SPECIES_TAXON, CONTEXT_ID)]

    query_rows = sorted((key, 0) for key in source.readout_keys)
    query_entity_index = np.asarray(
        [entity_lookup[key] for key, _ in query_rows], dtype="<i8"
    )
    query_readout_index = np.asarray(
        [readout for _, readout in query_rows], dtype="<i8"
    )
    query_payload = deterministic_npz_bytes(
        {
            "query_entity_index": query_entity_index,
            "query_readout_index": query_readout_index,
        }
    )
    query_lookup = {
        (corpus_keys[int(entity)], int(readout)): index
        for index, (entity, readout) in enumerate(
            zip(query_entity_index, query_readout_index)
        )
    }
    query_map = np.asarray(
        [query_lookup[(key, 0)] for key in source.readout_keys], dtype="<i8"
    )
    panel_payload = deterministic_npz_bytes(
        {
            "panel_id": _fixed_strings(
                ["slp-panel:mendeley-w8jtmnszd9-v2-proteome-v1"]
            ),
            "panel_indptr": np.asarray([0, expected.readouts], dtype="<i8"),
            "panel_query_index": np.arange(expected.readouts, dtype="<i8"),
        }
    )
    trajectory_payload = b"".join(
        canonical_json_bytes(
            {"schema": TRAJECTORY_SCHEMA, "ncbiTaxon": taxon, "entityId": entity}
        )
        for taxon, entity in source.trajectory_keys
    )

    output_shards: list[bytes] = []
    shard_refs: list[dict[str, object]] = []
    output_target_arrays: list[np.ndarray] = []
    active_actions: set[tuple[int, str]] = set()
    active_queries: set[int] = set()
    for index, source_payload in enumerate(source.shard_blobs):
        payload, records, values, actions, queries = _rewrite_shard(
            source_payload, index, entity_lookup, query_map, context_index
        )
        path = f"shards/shard-{index:05d}.npz"
        output_shards.append(payload)
        shard_refs.append(
            {**_file_ref(path, payload), "records": records, "targetValues": values}
        )
        arrays = read_deterministic_npz(
            payload, OUTPUT_SHARD_ARRAYS, f"composed shard {index}"
        )
        output_target_arrays.append(arrays["target_value"])
        active_actions.update(actions)
        active_queries.update(queries)
    if active_actions != set(source.trajectory_keys) or active_queries != set(
        range(expected.readouts)
    ):
        raise CorpusComposeError("composed action or query population drift")
    source_target_sha = _array_bytes_sha256(
        read_deterministic_npz(blob, SOURCE_SHARD_ARRAYS, f"source shard {index}")[
            "target_value"
        ]
        for index, blob in enumerate(source.shard_blobs)
    )
    output_target_sha = _array_bytes_sha256(output_target_arrays)
    if source_target_sha != output_target_sha:
        raise CorpusComposeError("quantitative target bytes changed during composition")

    feature_pack = _feature_pack(feature, expected)
    inputs = {
        "observations": _lineage_input(expected.observations, source.manifest_sha256),
        "staticFeatures": _lineage_input(
            expected.features, str(feature_pack["blocks"][0]["semanticSha256"])
        ),
        "heldInterventionRoster": _lineage_input(expected.roster, roster.roster_sha256),
    }
    covariates = {
        "record": [],
        "context": [],
        "action": [],
        "observation": [
            {
                "id": "slp-covariate:injection-index",
                "unit": "slp-unit:index",
                "access": "audit",
            },
            {
                "id": "slp-covariate:well-index",
                "unit": "slp-unit:index",
                "access": "audit",
            },
            {
                "id": "slp-covariate:plate-index",
                "unit": "slp-unit:index",
                "access": "audit",
            },
            {
                "id": "slp-covariate:metadata-row",
                "unit": "slp-unit:index",
                "access": "audit",
            },
            {
                "id": "slp-covariate:matrix-column",
                "unit": "slp-unit:index",
                "access": "audit",
            },
        ],
    }
    counts = {
        "entities": len(corpus_keys),
        "featureRows": expected.feature_rows,
        "contexts": 1,
        "queries": expected.readouts,
        "panels": 1,
        "trajectoryInterventions": expected.trajectory_interventions,
        "records": expected.records,
        "targetValues": expected.target_values,
        "shards": len(output_shards),
    }
    manifest: dict[str, object] = {
        "schema": CORPUS_SCHEMA,
        "datasetId": "slp-corpus:mendeley-w8jtmnszd9-v2-pretrain-composite-v1-2",
        "version": "v1.2",
        "role": "pretrain",
        "labelClass": "molecular",
        "benchmarkLabelsPresent": False,
        "rewardEnabled": False,
        "identityKey": ["ncbiTaxon", "entityId"],
        "rights": {
            "revision": "slp-rights:proteome-composite-corpus-v1-cc-by-4.0",
            "trainingAllowed": True,
            "redistributionAllowed": True,
        },
        "modalities": sorted(
            set(source.manifest["modalities"])
            | {"slp-modality:protein-sequence-statistics"}
        ),
        "sources": [{"id": source.manifest["source"]["id"]}],
        "sampling": {"scheme": SAMPLING_SCHEME, "sourceWeights": [1.0]},
        "species": [
            {"taxon": SPECIES_TAXON, "featureValue": [1.0], "featurePresent": [True]}
        ],
        "featurePack": feature_pack,
        "entityTypes": ENTITY_TYPES,
        "contextTypes": [CONTEXT_TYPE],
        "actionTypes": [ACTION_TYPE],
        "covariates": covariates,
        "readoutTypes": [
            {
                "id": READOUT_TYPE,
                "likelihood": "gaussian",
                "unit": source.manifest["measurement"]["unit"],
                "implicitZero": False,
            }
        ],
        "inputs": inputs,
        "counts": counts,
        "entityDictionary": {
            **_file_ref("entities.npz", entity_payload),
            "count": len(corpus_keys),
        },
        "queryDictionary": {
            **_file_ref("queries.npz", query_payload),
            "count": expected.readouts,
        },
        "queryPanels": {**_file_ref("query-panels.npz", panel_payload), "count": 1},
        "trajectoryInterventions": {
            **_file_ref("trajectory-interventions.jsonl", trajectory_payload),
            "count": expected.trajectory_interventions,
        },
        "normalization": {
            "id": source.manifest["measurement"]["protocolId"],
            "valueSpace": source.manifest["measurement"]["valueSpace"],
        },
        "bounds": {
            "maxRecordsPerShard": source.manifest["bounds"]["maxRecordsPerSourceShard"],
            "maxContextTokens": 1,
            "maxActionTokens": 1,
            "maxPanelQueries": expected.readouts,
            "maxTargetsPerRecord": expected.readouts,
        },
        "shards": shard_refs,
    }
    manifest_payload = canonical_json_bytes(manifest)
    members: dict[str, bytes] = {
        "composite-corpus/corpus.json": manifest_payload,
        "composite-corpus/entities.npz": entity_payload,
        "composite-corpus/queries.npz": query_payload,
        "composite-corpus/query-panels.npz": panel_payload,
        "composite-corpus/trajectory-interventions.jsonl": trajectory_payload,
    }
    for ref, payload in zip(shard_refs, output_shards):
        members["composite-corpus/" + str(ref["path"])] = payload
    archive_payload = deterministic_tar_bytes(members)
    if len(archive_payload) > bounds.max_archive_bytes:
        raise CorpusComposeError("composed corpus exceeds maxArchiveBytes")
    audit: dict[str, object] = {
        "schema": AUDIT_SCHEMA,
        "archive": _file_ref("corpus-v1-2.tar", archive_payload),
        "corpusManifestSha256": sha256_bytes(manifest_payload),
        "inputs": inputs,
        "counts": counts,
        "identity": {
            "key": ["ncbiTaxon", "entityId"],
            "corpusEntityKeySetSha256": composite_key_sha256(corpus_keys),
            "featureEntityKeySetSha256": composite_key_sha256(feature.keys),
            "contextEntity": {"ncbiTaxon": SPECIES_TAXON, "entityId": CONTEXT_ID},
        },
        "featurePackSha256": feature_pack["sha256"],
        "featurePreservation": {
            "rows": expected.feature_rows,
            "dimension": FEATURE_DIM,
            "sourceValueBytesSha256": source_feature_value_sha,
            "composedValueBytesSha256": composed_feature_value_sha,
            "sourcePresentBytesSha256": source_feature_present_sha,
            "composedPresentBytesSha256": composed_feature_present_sha,
            "byteExact": True,
        },
        "targetPreservation": {
            "dtype": "little-endian-float32",
            "values": expected.target_values,
            "sourceBytesSha256": source_target_sha,
            "composedBytesSha256": output_target_sha,
            "byteExact": True,
        },
        "leakage": {
            "heldRosterChecked": True,
            "protectedInterventionOverlap": 0,
            "benchmarkLabelsPresent": False,
            "rewardDataPresent": False,
        },
        "formats": {
            "archive": "canonical-USTAR",
            "arrays": "deterministic-uncompressed-NPZ",
        },
        "limitations": [
            "this is a fitting-only yeast proteome corpus, not molecular validation or final-holdout evidence",
            "the 21-dimensional hand-designed sequence block is a weak static baseline, not a frontier representation",
            "composition preserves quantitative targets but does not train or evaluate a world model",
        ],
    }
    audit_payload = pretty_json_bytes(audit)

    destination = destination.absolute()
    _reject_symlink_components(destination.parent, "corpus destination parent")
    if destination.exists() or destination.is_symlink():
        raise CorpusComposeError("refusing to overwrite corpus destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temp_name:
        publication = Path(temp_name) / "publication"
        publication.mkdir()
        archive_path = publication / "corpus-v1-2.tar"
        audit_path = publication / "corpus-compose-audit.json"
        archive_path.write_bytes(archive_payload)
        audit_path.write_bytes(audit_payload)
        summary = validate_composite_corpus_archive(archive_path, bounds)
        validate_composition_audit(audit_path, archive_path, summary, inputs)
        publication.replace(destination)
    return {
        "archiveSha256": sha256_bytes(archive_payload),
        "auditSha256": sha256_bytes(audit_payload),
        "manifestSha256": sha256_bytes(manifest_payload),
        "featurePackSha256": feature_pack["sha256"],
        "entityKeySetSha256": composite_key_sha256(corpus_keys),
        "featureEntityKeySetSha256": composite_key_sha256(feature.keys),
        "targetValueBytesSha256": output_target_sha,
        "featureValueBytesSha256": composed_feature_value_sha,
        "featurePresentBytesSha256": composed_feature_present_sha,
        **counts,
    }


def _validate_dataset_snapshot_identity(value: object, label: str) -> None:
    item = _strict(
        value, {"resource", "revision", "outerManifestDigest", "treeDigest"}, label
    )
    _, revision = _dataset_resource(item["resource"], f"{label}.resource")
    if item["revision"] != revision:
        raise CorpusComposeError(f"{label} revision does not match resource")
    _prefixed_digest(item["outerManifestDigest"], f"{label}.outerManifestDigest")
    _prefixed_digest(item["treeDigest"], f"{label}.treeDigest")


def _validate_lineage(value: object, label: str) -> None:
    item = _strict(value, {"datasetSnapshot", "semanticSha256", "files"}, label)
    _validate_dataset_snapshot_identity(
        item["datasetSnapshot"], f"{label}.datasetSnapshot"
    )
    _bare_digest(item["semanticSha256"], f"{label}.semanticSha256")
    files = item["files"]
    if not isinstance(files, list) or not files:
        raise CorpusComposeError(f"{label}.files must be non-empty")
    paths: list[str] = []
    for index, file_ref in enumerate(files):
        ref = _strict(file_ref, {"path", "sha256", "bytes"}, f"{label}.files[{index}]")
        paths.append(_canonical_relative(ref["path"], f"{label}.files[{index}].path"))
        _bare_digest(ref["sha256"], f"{label}.files[{index}].sha256")
        if type(ref["bytes"]) is not int or ref["bytes"] <= 0:
            raise CorpusComposeError(f"{label}.files[{index}].bytes is invalid")
    if paths != sorted(set(paths)):
        raise CorpusComposeError(f"{label}.files are duplicated or unordered")


def _validate_feature_pack(value: object) -> None:
    pack = _strict(
        value,
        {
            "schema",
            "revision",
            "sha256",
            "entityFeatureDim",
            "speciesFeatureDim",
            "blocks",
        },
        "featurePack",
    )
    if (
        pack["schema"] != FEATURE_PACK_SCHEMA
        or pack["revision"] != "slp-feature-pack:sequence-statistics-r64-5-1-v1"
        or pack["entityFeatureDim"] != FEATURE_DIM
        or pack["speciesFeatureDim"] != SPECIES_FEATURE_DIM
    ):
        raise CorpusComposeError("featurePack declaration drift")
    declared_sha = _bare_digest(pack["sha256"], "featurePack.sha256")
    without_sha = {key: child for key, child in pack.items() if key != "sha256"}
    if declared_sha != sha256_bytes(canonical_json(without_sha).encode("ascii")):
        raise CorpusComposeError("featurePack canonical digest mismatch")
    blocks = pack["blocks"]
    if not isinstance(blocks, list) or len(blocks) != 1:
        raise CorpusComposeError("this featurePack requires exactly one block")
    end = 0
    ids: set[str] = set()
    for index, block_value in enumerate(blocks):
        block = _strict(
            block_value,
            {
                "id",
                "offset",
                "dimension",
                "datasetSnapshot",
                "semanticSha256",
                "entityKeySetSha256",
                "files",
            },
            f"featurePack.blocks[{index}]",
        )
        identifier = _nonempty(block["id"], f"featurePack.blocks[{index}].id")
        if ":" not in identifier or identifier in ids or block["offset"] != end:
            raise CorpusComposeError(
                "featurePack blocks must have unique IDs and contiguous offsets"
            )
        ids.add(identifier)
        if type(block["dimension"]) is not int or block["dimension"] <= 0:
            raise CorpusComposeError("featurePack block dimension is invalid")
        end += block["dimension"]
        _validate_dataset_snapshot_identity(
            block["datasetSnapshot"], f"featurePack.blocks[{index}].datasetSnapshot"
        )
        _bare_digest(
            block["semanticSha256"], f"featurePack.blocks[{index}].semanticSha256"
        )
        _bare_digest(
            block["entityKeySetSha256"],
            f"featurePack.blocks[{index}].entityKeySetSha256",
        )
        _validate_lineage(
            {
                "datasetSnapshot": block["datasetSnapshot"],
                "semanticSha256": block["semanticSha256"],
                "files": block["files"],
            },
            f"featurePack.blocks[{index}]",
        )
    if end != FEATURE_DIM:
        raise CorpusComposeError(
            "featurePack block dimensions do not cover entityFeatureDim"
        )


def _validate_manifest_ref(
    value: object, path: str, payload: bytes, count: int, label: str
) -> None:
    expected = {**_file_ref(path, payload), "count": count}
    if value != expected:
        raise CorpusComposeError(f"{label} reference drift")


def _validate_output_shard(
    arrays: Mapping[str, np.ndarray],
    records: int,
    values: int,
    entities: int,
    queries: int,
    context_index: int,
    entity_taxon: np.ndarray,
    entity_id: np.ndarray,
    query_entity_index: np.ndarray,
    seen_records: set[str],
    active_actions: set[int],
    active_contexts: set[int],
) -> None:
    for name in ("record_id", "observation_unit_id", "replicate_id", "perturbation_id"):
        array = arrays[name]
        if (
            array.shape != (records,)
            or array.dtype.kind != "U"
            or any(not str(item) for item in array)
        ):
            raise CorpusComposeError(f"composed shard {name} drift")
    records_now = {str(item) for item in arrays["record_id"]}
    if len(records_now) != records or seen_records & records_now:
        raise CorpusComposeError("composed record identities are not globally unique")
    seen_records.update(records_now)
    for name in ("source_index", "species_taxon", "query_panel_index"):
        _require_shape_dtype(arrays[name], (records,), "<i8", f"composed shard/{name}")
    if (
        np.any(arrays["source_index"] != 0)
        or np.any(arrays["species_taxon"] != SPECIES_TAXON)
        or np.any(arrays["query_panel_index"] != 0)
    ):
        raise CorpusComposeError("composed shard source, species, or panel drift")
    _require_shape_dtype(
        arrays["species_feature_value"], (records, 1), "<f4", "species_feature_value"
    )
    _require_shape_dtype(
        arrays["species_feature_present"],
        (records, 1),
        "|b1",
        "species_feature_present",
    )
    if not np.array_equal(
        arrays["species_feature_value"], np.ones((records, 1), np.float32)
    ) or not bool(arrays["species_feature_present"].all()):
        raise CorpusComposeError("composed species features drift")
    for axis, width, covariates in (("context", 1, 0), ("action", 1, 0)):
        _require_shape_dtype(
            arrays[f"{axis}_entity_index"],
            (records, width),
            "<i8",
            f"{axis}_entity_index",
        )
        _require_shape_dtype(
            arrays[f"{axis}_type"], (records, width), "<i8", f"{axis}_type"
        )
        _require_shape_dtype(
            arrays[f"{axis}_mask"], (records, width), "|b1", f"{axis}_mask"
        )
        _require_shape_dtype(
            arrays[f"{axis}_covariate_value"],
            (records, width, covariates),
            "<f4",
            f"{axis}_covariate_value",
        )
        _require_shape_dtype(
            arrays[f"{axis}_covariate_present"],
            (records, width, covariates),
            "|b1",
            f"{axis}_covariate_present",
        )
        if not bool(arrays[f"{axis}_mask"].all()):
            raise CorpusComposeError(f"every composed {axis} token must be active")
        indices = arrays[f"{axis}_entity_index"].reshape(-1)
        if (
            np.any(indices < 0)
            or np.any(indices >= entities)
            or np.any(entity_taxon[indices] != arrays["species_taxon"])
        ):
            raise CorpusComposeError(f"composed {axis} composite taxon join failed")
    contexts = {int(item) for item in arrays["context_entity_index"].reshape(-1)}
    if contexts != {context_index}:
        raise CorpusComposeError("composed context identity drift")
    active_contexts.update(contexts)
    actions = {int(item) for item in arrays["action_entity_index"].reshape(-1)}
    active_actions.update(actions)
    for row, entity_index in enumerate(arrays["action_entity_index"].reshape(-1)):
        key = (int(entity_taxon[int(entity_index)]), str(entity_id[int(entity_index)]))
        if str(arrays["perturbation_id"][row]) != composite_perturbation_id([key]):
            raise CorpusComposeError(
                "perturbation_id is not derived from the composite action key"
            )
    if np.any(arrays["context_type"] != 0) or np.any(arrays["action_type"] != 0):
        raise CorpusComposeError("composed context or action type drift")
    _require_shape_dtype(
        arrays["record_covariate_value"], (records, 0), "<f4", "record_covariate_value"
    )
    _require_shape_dtype(
        arrays["record_covariate_present"],
        (records, 0),
        "|b1",
        "record_covariate_present",
    )
    _require_shape_dtype(
        arrays["observation_covariate_value"],
        (records, 5),
        "<f4",
        "observation_covariate_value",
    )
    _require_shape_dtype(
        arrays["observation_covariate_present"],
        (records, 5),
        "|b1",
        "observation_covariate_present",
    )
    if not bool(arrays["observation_covariate_present"].all()):
        raise CorpusComposeError("audit-only observation covariates must be present")
    _require_shape_dtype(
        arrays["target_indptr"], (records + 1,), "<i8", "target_indptr"
    )
    _require_shape_dtype(
        arrays["target_query_index"], (values,), "<i8", "target_query_index"
    )
    _require_shape_dtype(arrays["target_value"], (values,), "<f4", "target_value")
    indptr = arrays["target_indptr"]
    target_queries = arrays["target_query_index"]
    if indptr[0] != 0 or indptr[-1] != values or np.any(indptr[1:] < indptr[:-1]):
        raise CorpusComposeError("composed target CSR drift")
    if (
        np.any(target_queries < 0)
        or np.any(target_queries >= queries)
        or not np.isfinite(arrays["target_value"]).all()
    ):
        raise CorpusComposeError("composed target query or value drift")
    if np.any(
        entity_taxon[query_entity_index[target_queries]]
        != np.repeat(arrays["species_taxon"], np.diff(indptr))
    ):
        raise CorpusComposeError("composed target query taxon join failed")
    for row in range(records):
        selected = target_queries[int(indptr[row]) : int(indptr[row + 1])]
        if len(selected) != len({int(item) for item in selected}):
            raise CorpusComposeError("composed record repeats a target query")


def validate_composite_corpus_archive(
    path: Path, bounds: Bounds | None = None
) -> dict[str, object]:
    bounds = Bounds() if bounds is None else bounds
    blobs = read_canonical_tar(path, bounds, "composite corpus")
    manifest_payload = blobs.get("composite-corpus/corpus.json")
    if manifest_payload is None:
        raise CorpusComposeError("composite corpus manifest is missing")
    manifest = _json_payload(manifest_payload, "composite corpus manifest", bounds)
    required = {
        "schema",
        "datasetId",
        "version",
        "role",
        "labelClass",
        "benchmarkLabelsPresent",
        "rewardEnabled",
        "identityKey",
        "rights",
        "modalities",
        "sources",
        "sampling",
        "species",
        "featurePack",
        "entityTypes",
        "contextTypes",
        "actionTypes",
        "covariates",
        "readoutTypes",
        "inputs",
        "counts",
        "entityDictionary",
        "queryDictionary",
        "queryPanels",
        "trajectoryInterventions",
        "normalization",
        "bounds",
        "shards",
    }
    _strict(manifest, required, "composite corpus manifest")
    if (
        manifest["schema"] != CORPUS_SCHEMA
        or manifest["version"] != "v1.2"
        or manifest["role"] != "pretrain"
        or manifest["labelClass"] != "molecular"
        or manifest["benchmarkLabelsPresent"] is not False
        or manifest["rewardEnabled"] is not False
        or manifest["identityKey"] != ["ncbiTaxon", "entityId"]
        or manifest["entityTypes"] != ENTITY_TYPES
        or manifest["contextTypes"] != [CONTEXT_TYPE]
        or manifest["actionTypes"] != [ACTION_TYPE]
    ):
        raise CorpusComposeError("composite corpus boundary or type declaration drift")
    expected_observation_covariates = [
        {
            "id": "slp-covariate:injection-index",
            "unit": "slp-unit:index",
            "access": "audit",
        },
        {"id": "slp-covariate:well-index", "unit": "slp-unit:index", "access": "audit"},
        {
            "id": "slp-covariate:plate-index",
            "unit": "slp-unit:index",
            "access": "audit",
        },
        {
            "id": "slp-covariate:metadata-row",
            "unit": "slp-unit:index",
            "access": "audit",
        },
        {
            "id": "slp-covariate:matrix-column",
            "unit": "slp-unit:index",
            "access": "audit",
        },
    ]
    if (
        manifest["rights"]
        != {
            "revision": "slp-rights:proteome-composite-corpus-v1-cc-by-4.0",
            "trainingAllowed": True,
            "redistributionAllowed": True,
        }
        or manifest["modalities"]
        != [
            "slp-modality:protein-sequence-statistics",
            "slp-modality:quantitative-proteome",
        ]
        or manifest["sources"] != [{"id": "mendeley:w8jtmnszd9.2"}]
        or manifest["sampling"] != {"scheme": SAMPLING_SCHEME, "sourceWeights": [1.0]}
        or manifest["species"]
        != [{"taxon": SPECIES_TAXON, "featureValue": [1.0], "featurePresent": [True]}]
        or manifest["covariates"]
        != {
            "record": [],
            "context": [],
            "action": [],
            "observation": expected_observation_covariates,
        }
    ):
        raise CorpusComposeError(
            "corpus rights, modalities, source, species, or covariate drift"
        )
    readout_types = manifest["readoutTypes"]
    if (
        not isinstance(readout_types, list)
        or len(readout_types) != 1
        or not isinstance(readout_types[0], dict)
        or set(readout_types[0]) != {"id", "likelihood", "unit", "implicitZero"}
        or readout_types[0]["id"] != READOUT_TYPE
        or readout_types[0]["likelihood"] != "gaussian"
        or readout_types[0]["implicitZero"] is not False
        or not isinstance(readout_types[0]["unit"], str)
        or not readout_types[0]["unit"].startswith("slp-unit:")
    ):
        raise CorpusComposeError("corpus readout type drift")
    inputs = _strict(
        manifest["inputs"],
        {"observations", "staticFeatures", "heldInterventionRoster"},
        "inputs",
    )
    for name, item in inputs.items():
        _validate_lineage(item, f"inputs.{name}")
    _validate_feature_pack(manifest["featurePack"])
    pack_block = manifest["featurePack"]["blocks"][0]
    if (
        inputs["staticFeatures"]["semanticSha256"] != pack_block["semanticSha256"]
        or inputs["staticFeatures"]["datasetSnapshot"] != pack_block["datasetSnapshot"]
        or inputs["staticFeatures"]["files"] != pack_block["files"]
    ):
        raise CorpusComposeError("static feature input and featurePack lineage differ")
    counts = _strict(
        manifest["counts"],
        {
            "entities",
            "featureRows",
            "contexts",
            "queries",
            "panels",
            "trajectoryInterventions",
            "records",
            "targetValues",
            "shards",
        },
        "counts",
    )
    if any(type(value) is not int or value <= 0 for value in counts.values()):
        raise CorpusComposeError("corpus counts must be positive integers")
    if manifest["normalization"] != {
        "id": "slp-value:mendeley-w8jtmnszd9-v2-log2-relative-intensity-v1",
        "valueSpace": "slp-value:log2-batch-corrected-maxlfq-relative-intensity",
    }:
        raise CorpusComposeError("corpus normalization drift")
    if manifest["bounds"] != {
        "maxRecordsPerShard": 512,
        "maxContextTokens": 1,
        "maxActionTokens": 1,
        "maxPanelQueries": counts["queries"],
        "maxTargetsPerRecord": counts["queries"],
    }:
        raise CorpusComposeError("corpus operational bounds drift")
    shards = manifest["shards"]
    if not isinstance(shards, list) or len(shards) != counts["shards"]:
        raise CorpusComposeError("corpus shard count drift")
    expected_members = set(OUTPUT_STATIC_MEMBERS) | {
        "composite-corpus/" + str(item.get("path"))
        for item in shards
        if isinstance(item, dict)
    }
    if set(blobs) != expected_members:
        raise CorpusComposeError("composite corpus member set drift")

    entity_payload = blobs["composite-corpus/entities.npz"]
    _validate_manifest_ref(
        manifest["entityDictionary"],
        "entities.npz",
        entity_payload,
        counts["entities"],
        "entityDictionary",
    )
    entities = read_deterministic_npz(
        entity_payload,
        {
            "entity_taxon",
            "entity_id",
            "entity_type",
            "entity_feature_value",
            "entity_feature_present",
        },
        "entityDictionary",
    )
    _require_shape_dtype(
        entities["entity_taxon"], (counts["entities"],), "<i8", "entity_taxon"
    )
    if (
        entities["entity_id"].shape != (counts["entities"],)
        or entities["entity_id"].dtype.kind != "U"
    ):
        raise CorpusComposeError("entity_id dtype or shape drift")
    _require_shape_dtype(
        entities["entity_type"], (counts["entities"],), "<i8", "entity_type"
    )
    _require_shape_dtype(
        entities["entity_feature_value"],
        (counts["entities"], FEATURE_DIM),
        "<f4",
        "entity_feature_value",
    )
    _require_shape_dtype(
        entities["entity_feature_present"],
        (counts["entities"], FEATURE_DIM),
        "|b1",
        "entity_feature_present",
    )
    keys = [
        (int(taxon), str(entity))
        for taxon, entity in zip(entities["entity_taxon"], entities["entity_id"])
    ]
    if keys != sorted(set(keys)):
        raise CorpusComposeError(
            "entity dictionary composite keys are duplicated or unordered"
        )
    for key, entity_type in zip(keys, entities["entity_type"]):
        expected_type = (
            0
            if SGD_RE.fullmatch(key[1])
            else 1
            if UNIPROT_RE.fullmatch(key[1])
            else 2
            if key == (SPECIES_TAXON, CONTEXT_ID)
            else -1
        )
        if int(entity_type) != expected_type:
            raise CorpusComposeError("entity identity and entity_type disagree")
    present = entities["entity_feature_present"]
    full_rows = present.all(axis=1)
    empty_rows = ~present.any(axis=1)
    if (
        int(full_rows.sum()) != counts["featureRows"]
        or int(empty_rows.sum()) != counts["contexts"]
        or not bool((full_rows | empty_rows).all())
    ):
        raise CorpusComposeError(
            "entity feature rows must be either complete or explicitly missing"
        )
    if np.any(entities["entity_feature_value"][empty_rows] != 0):
        raise CorpusComposeError("missing entity feature storage must be zero")
    context_rows = np.flatnonzero(entities["entity_type"] == 2)
    if len(context_rows) != counts["contexts"] or not bool(
        empty_rows[context_rows].all()
    ):
        raise CorpusComposeError(
            "context rows must be exactly the missing-feature rows"
        )
    context_index = int(context_rows[0])
    feature_keys = [key for key, keep in zip(keys, full_rows) if bool(keep)]
    if pack_block["entityKeySetSha256"] != composite_key_sha256(feature_keys):
        raise CorpusComposeError("featurePack composite entity-key digest mismatch")

    query_payload = blobs["composite-corpus/queries.npz"]
    _validate_manifest_ref(
        manifest["queryDictionary"],
        "queries.npz",
        query_payload,
        counts["queries"],
        "queryDictionary",
    )
    queries = read_deterministic_npz(
        query_payload, {"query_entity_index", "query_readout_index"}, "queryDictionary"
    )
    _require_shape_dtype(
        queries["query_entity_index"], (counts["queries"],), "<i8", "query_entity_index"
    )
    _require_shape_dtype(
        queries["query_readout_index"],
        (counts["queries"],),
        "<i8",
        "query_readout_index",
    )
    if np.any(queries["query_entity_index"] < 0) or np.any(
        queries["query_entity_index"] >= counts["entities"]
    ):
        raise CorpusComposeError("query entity index is outside the dictionary")
    query_keys = [
        (keys[int(entity)], int(readout))
        for entity, readout in zip(
            queries["query_entity_index"], queries["query_readout_index"]
        )
    ]
    if query_keys != sorted(set(query_keys)) or any(
        readout != 0 for _, readout in query_keys
    ):
        raise CorpusComposeError(
            "query composite identities are duplicated, unordered, or mistyped"
        )

    panel_payload = blobs["composite-corpus/query-panels.npz"]
    _validate_manifest_ref(
        manifest["queryPanels"],
        "query-panels.npz",
        panel_payload,
        counts["panels"],
        "queryPanels",
    )
    panels = read_deterministic_npz(
        panel_payload, {"panel_id", "panel_indptr", "panel_query_index"}, "queryPanels"
    )
    if (
        panels["panel_id"].shape != (counts["panels"],)
        or panels["panel_id"].dtype.kind != "U"
    ):
        raise CorpusComposeError("panel_id dtype or shape drift")
    _require_shape_dtype(
        panels["panel_indptr"], (counts["panels"] + 1,), "<i8", "panel_indptr"
    )
    _require_shape_dtype(
        panels["panel_query_index"], (counts["queries"],), "<i8", "panel_query_index"
    )
    if not np.array_equal(
        panels["panel_indptr"], np.asarray([0, counts["queries"]], dtype="<i8")
    ) or not np.array_equal(
        panels["panel_query_index"], np.arange(counts["queries"], dtype="<i8")
    ):
        raise CorpusComposeError(
            "query panel does not contain the exact ordered query dictionary"
        )

    trajectory_payload = blobs["composite-corpus/trajectory-interventions.jsonl"]
    _validate_manifest_ref(
        manifest["trajectoryInterventions"],
        "trajectory-interventions.jsonl",
        trajectory_payload,
        counts["trajectoryInterventions"],
        "trajectoryInterventions",
    )
    trajectory_rows = _canonical_jsonl(
        trajectory_payload, "trajectoryInterventions", bounds, bounds.max_entities
    )
    trajectory_keys: list[tuple[int, str]] = []
    for index, row in enumerate(trajectory_rows):
        _strict(
            row,
            {"schema", "ncbiTaxon", "entityId"},
            f"trajectoryInterventions[{index}]",
        )
        key = (row["ncbiTaxon"], row["entityId"])
        if (
            row["schema"] != TRAJECTORY_SCHEMA
            or key not in set(keys)
            or SGD_RE.fullmatch(str(key[1])) is None
        ):
            raise CorpusComposeError(
                "trajectory intervention has invalid composite identity"
            )
        trajectory_keys.append((int(key[0]), str(key[1])))
    if trajectory_keys != sorted(set(trajectory_keys)):
        raise CorpusComposeError("trajectory interventions are duplicated or unordered")

    seen_records: set[str] = set()
    active_actions: set[int] = set()
    active_contexts: set[int] = set()
    target_arrays: list[np.ndarray] = []
    record_total = target_total = 0
    for index, ref_value in enumerate(shards):
        ref = _strict(
            ref_value,
            {"path", "sha256", "bytes", "records", "targetValues"},
            f"shards[{index}]",
        )
        expected_path = f"shards/shard-{index:05d}.npz"
        payload = blobs["composite-corpus/" + expected_path]
        if ref != {
            **_file_ref(expected_path, payload),
            "records": ref["records"],
            "targetValues": ref["targetValues"],
        }:
            raise CorpusComposeError(f"shards[{index}] reference drift")
        arrays = read_deterministic_npz(
            payload, OUTPUT_SHARD_ARRAYS, f"shards[{index}]"
        )
        _validate_output_shard(
            arrays,
            ref["records"],
            ref["targetValues"],
            counts["entities"],
            counts["queries"],
            context_index,
            entities["entity_taxon"],
            entities["entity_id"],
            queries["query_entity_index"],
            seen_records,
            active_actions,
            active_contexts,
        )
        record_total += ref["records"]
        target_total += ref["targetValues"]
        target_arrays.append(arrays["target_value"])
    active_action_keys = {keys[index] for index in active_actions}
    if active_action_keys != set(trajectory_keys) or active_contexts != {context_index}:
        raise CorpusComposeError(
            "active action/context populations do not match declarations"
        )
    if record_total != counts["records"] or target_total != counts["targetValues"]:
        raise CorpusComposeError("record or target totals do not match counts")
    return {
        "manifest": manifest,
        "counts": counts,
        "corpusManifestSha256": sha256_bytes(manifest_payload),
        "archiveSha256": sha256_file(path),
        "entityKeySetSha256": composite_key_sha256(keys),
        "featureEntityKeySetSha256": composite_key_sha256(feature_keys),
        "targetValueBytesSha256": _array_bytes_sha256(target_arrays),
        "featurePackSha256": manifest["featurePack"]["sha256"],
        "featureValueBytesSha256": _array_bytes_sha256(
            [entities["entity_feature_value"][full_rows]]
        ),
        "featurePresentBytesSha256": _array_bytes_sha256(
            [entities["entity_feature_present"][full_rows]]
        ),
    }


def validate_composition_audit(
    audit_path: Path,
    archive_path: Path,
    summary: Mapping[str, object],
    inputs: Mapping[str, object],
) -> None:
    if audit_path.is_symlink() or not audit_path.is_file():
        raise CorpusComposeError("composition audit must be a regular file")
    audit = _json_payload(audit_path.read_bytes(), "composition audit", Bounds())
    required = {
        "schema",
        "archive",
        "corpusManifestSha256",
        "inputs",
        "counts",
        "identity",
        "featurePackSha256",
        "featurePreservation",
        "targetPreservation",
        "leakage",
        "formats",
        "limitations",
    }
    _strict(audit, required, "composition audit")
    expected_archive = {
        "path": "corpus-v1-2.tar",
        "bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
    }
    if (
        audit["schema"] != AUDIT_SCHEMA
        or audit["archive"] != expected_archive
        or audit["corpusManifestSha256"] != summary["corpusManifestSha256"]
        or audit["inputs"] != inputs
        or audit["counts"] != summary["counts"]
        or audit["featurePackSha256"] != summary["featurePackSha256"]
    ):
        raise CorpusComposeError("composition audit lineage or digest drift")
    identity = _strict(
        audit["identity"],
        {
            "key",
            "corpusEntityKeySetSha256",
            "featureEntityKeySetSha256",
            "contextEntity",
        },
        "composition audit identity",
    )
    if (
        identity["key"] != ["ncbiTaxon", "entityId"]
        or identity["corpusEntityKeySetSha256"] != summary["entityKeySetSha256"]
        or identity["featureEntityKeySetSha256"] != summary["featureEntityKeySetSha256"]
        or identity["contextEntity"]
        != {"ncbiTaxon": SPECIES_TAXON, "entityId": CONTEXT_ID}
    ):
        raise CorpusComposeError("composition audit identity drift")
    target = _strict(
        audit["targetPreservation"],
        {"dtype", "values", "sourceBytesSha256", "composedBytesSha256", "byteExact"},
        "composition audit targetPreservation",
    )
    if (
        target["dtype"] != "little-endian-float32"
        or target["values"] != summary["counts"]["targetValues"]
        or target["sourceBytesSha256"] != target["composedBytesSha256"]
        or target["composedBytesSha256"] != summary["targetValueBytesSha256"]
        or target["byteExact"] is not True
    ):
        raise CorpusComposeError("composition audit target preservation drift")
    feature = _strict(
        audit["featurePreservation"],
        {
            "rows",
            "dimension",
            "sourceValueBytesSha256",
            "composedValueBytesSha256",
            "sourcePresentBytesSha256",
            "composedPresentBytesSha256",
            "byteExact",
        },
        "composition audit featurePreservation",
    )
    if (
        feature["rows"] != summary["counts"]["featureRows"]
        or feature["dimension"] != FEATURE_DIM
        or feature["sourceValueBytesSha256"] != feature["composedValueBytesSha256"]
        or feature["composedValueBytesSha256"] != summary["featureValueBytesSha256"]
        or feature["sourcePresentBytesSha256"] != feature["composedPresentBytesSha256"]
        or feature["composedPresentBytesSha256"] != summary["featurePresentBytesSha256"]
        or feature["byteExact"] is not True
    ):
        raise CorpusComposeError("composition audit feature preservation drift")
    if audit["leakage"] != {
        "heldRosterChecked": True,
        "protectedInterventionOverlap": 0,
        "benchmarkLabelsPresent": False,
        "rewardDataPresent": False,
    }:
        raise CorpusComposeError("composition audit leakage declaration drift")
