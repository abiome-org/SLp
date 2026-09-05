"""Bounded fitting-only loader for the SLp composite molecular corpus v1.2.

Identifiers are returned for provenance and splitting.  Numerical model inputs
are restricted to the static entity feature values and their presence mask.
Technical observation locators declared with ``access: audit`` are validated
but deliberately never returned.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_RECORDS = 10_000
MAX_QUERIES = 10_000
MAX_DENSE_CELLS = MAX_RECORDS * MAX_QUERIES
PRODUCTION_ARCHIVE_SHA256 = (
    "0a5322c46e15e8a15d17000e8993c0ad642fcc70bc8fff00cbba8fb2905708bf"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHARD_ARRAYS = {
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


class CorpusLoadError(ValueError):
    """The fitting corpus violates the bounded v1.2 consumer contract."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CorpusLoadError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if (
        value != path.as_posix()
        or path.is_absolute()
        or "\\" in value
        or ":" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise CorpusLoadError(f"{label} is not a canonical relative path")
    return value


def _load_tar(path: Path) -> tuple[dict[str, bytes], str]:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise CorpusLoadError("corpus archive does not exist") from error
    if path.is_symlink() or not resolved.is_file():
        raise CorpusLoadError("corpus archive must be a regular file")
    size = resolved.stat().st_size
    if size <= 0 or size > MAX_ARCHIVE_BYTES:
        raise CorpusLoadError("corpus archive exceeds the 256 MiB bound")

    blobs: dict[str, bytes] = {}
    expanded = 0
    archive_hash = hashlib.sha256()
    try:
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                archive_hash.update(chunk)
        with tarfile.open(resolved, mode="r:") as archive:
            for member in archive.getmembers():
                name = _canonical_relative(member.name, "tar member")
                if not name.startswith("composite-corpus/") or name in blobs:
                    raise CorpusLoadError("tar member set is duplicated or out of scope")
                if not member.isfile() or member.pax_headers:
                    raise CorpusLoadError("tar archive may contain only regular files")
                expanded += member.size
                if expanded > MAX_ARCHIVE_BYTES:
                    raise CorpusLoadError("expanded corpus exceeds the 256 MiB bound")
                reader = archive.extractfile(member)
                if reader is None:
                    raise CorpusLoadError(f"tar member is unreadable: {name}")
                payload = reader.read(member.size + 1)
                if len(payload) != member.size:
                    raise CorpusLoadError(f"tar member size drift: {name}")
                blobs[name] = payload
    except (OSError, tarfile.TarError) as error:
        if isinstance(error, CorpusLoadError):
            raise
        raise CorpusLoadError("corpus is not a valid uncompressed tar archive") from error
    return blobs, archive_hash.hexdigest()


def _json(payload: bytes, label: str) -> dict[str, Any]:
    if not payload or len(payload) > MAX_MANIFEST_BYTES:
        raise CorpusLoadError(f"{label} is empty or too large")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusLoadError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise CorpusLoadError(f"{label} must be a JSON object")
    return value


def _ref_payload(
    blobs: Mapping[str, bytes], reference: object, label: str
) -> tuple[str, bytes]:
    if not isinstance(reference, dict) or not {"path", "bytes", "sha256"} <= set(
        reference
    ):
        raise CorpusLoadError(f"{label} file reference is malformed")
    relative = _canonical_relative(reference["path"], f"{label}.path")
    name = "composite-corpus/" + relative
    payload = blobs.get(name)
    digest = reference["sha256"]
    if (
        payload is None
        or type(reference["bytes"]) is not int
        or reference["bytes"] != len(payload)
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or _sha256(payload) != digest
    ):
        raise CorpusLoadError(f"{label} bytes or SHA-256 do not match the manifest")
    return name, payload


def _read_npz(payload: bytes, expected: set[str], label: str) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    expanded = 0
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or set(names) != {
                f"{name}.npy" for name in expected
            }:
                raise CorpusLoadError(f"{label} NPZ array set drift")
            for info in infos:
                name = info.filename
                if (
                    info.is_dir()
                    or info.compress_type != zipfile.ZIP_STORED
                    or PurePosixPath(name).name != name
                    or not name.endswith(".npy")
                ):
                    raise CorpusLoadError(f"{label} contains an unsafe NPZ member")
                expanded += info.file_size
                if expanded > MAX_ARCHIVE_BYTES:
                    raise CorpusLoadError(f"{label} expanded NPZ bytes exceed the bound")
                raw = archive.read(info)
                stream = io.BytesIO(raw)
                array = np.lib.format.read_array(stream, allow_pickle=False)
                if stream.tell() != len(raw) or array.dtype.hasobject:
                    raise CorpusLoadError(f"{label}/{name} is not a safe NPY array")
                arrays[name.removesuffix(".npy")] = array
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, CorpusLoadError):
            raise
        raise CorpusLoadError(f"{label} is not a valid pickle-free NPZ") from error
    return arrays


def _require_array(
    array: np.ndarray, shape: tuple[int, ...], dtype: np.dtype[Any], label: str
) -> None:
    if array.shape != shape or array.dtype != dtype or not array.flags.c_contiguous:
        raise CorpusLoadError(f"{label} dtype, shape, or layout drift")


def _positive_count(counts: Mapping[str, object], name: str, maximum: int | None = None) -> int:
    value = counts.get(name)
    if type(value) is not int or value <= 0 or (maximum is not None and value > maximum):
        raise CorpusLoadError(f"manifest count {name} is invalid or exceeds its bound")
    return value


def _require_ref_count(reference: object, expected: int, label: str) -> None:
    if not isinstance(reference, dict) or reference.get("count") != expected:
        raise CorpusLoadError(f"{label} count does not match the manifest")


def _validate_boundary(manifest: Mapping[str, object]) -> tuple[str, tuple[str, ...]]:
    if (
        manifest.get("schema") != "slp.corpus/v1.2"
        or manifest.get("version") != "v1.2"
        or manifest.get("role") != "pretrain"
        or manifest.get("labelClass") != "molecular"
        or manifest.get("benchmarkLabelsPresent") is not False
        or manifest.get("rewardEnabled") is not False
        or manifest.get("identityKey") != ["ncbiTaxon", "entityId"]
        or manifest.get("actionTypes") != ["slp-action:gene-deletion"]
    ):
        raise CorpusLoadError("corpus is not a fitting-only composite molecular corpus")
    readout_types = manifest.get("readoutTypes")
    if (
        not isinstance(readout_types, list)
        or len(readout_types) != 1
        or not isinstance(readout_types[0], dict)
        or readout_types[0].get("likelihood") != "gaussian"
        or readout_types[0].get("implicitZero") is not False
    ):
        raise CorpusLoadError("pilot loader requires one explicit Gaussian readout type")
    rights = manifest.get("rights")
    if not isinstance(rights, dict) or rights.get("trainingAllowed") is not True:
        raise CorpusLoadError("corpus rights do not allow fitting")
    sources = manifest.get("sources")
    if (
        not isinstance(sources, list)
        or len(sources) != 1
        or not isinstance(sources[0], dict)
        or not isinstance(sources[0].get("id"), str)
        or not sources[0]["id"].strip()
    ):
        raise CorpusLoadError("pilot loader requires exactly one declared source")
    covariates = manifest.get("covariates")
    if not isinstance(covariates, dict):
        raise CorpusLoadError("covariate declarations are missing")
    audit_ids: list[str] = []
    for axis in ("record", "context", "action", "observation"):
        declarations = covariates.get(axis)
        if not isinstance(declarations, list):
            raise CorpusLoadError(f"covariates.{axis} must be a list")
        for item in declarations:
            if not isinstance(item, dict) or item.get("access") != "audit":
                raise CorpusLoadError(
                    "pilot loader does not expose undeclared or non-audit covariates"
                )
            identifier = item.get("id")
            if not isinstance(identifier, str) or not identifier:
                raise CorpusLoadError("covariate identifier is invalid")
            audit_ids.append(identifier)
    return sources[0]["id"], tuple(audit_ids)


def _lineage_hashes(manifest: Mapping[str, object]) -> dict[str, str]:
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise CorpusLoadError("input lineage is missing")
    hashes: dict[str, str] = {}
    for name, value in sorted(inputs.items()):
        if not isinstance(name, str) or not isinstance(value, dict):
            raise CorpusLoadError("input lineage is malformed")
        digest = value.get("semanticSha256")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise CorpusLoadError(f"inputs.{name}.semanticSha256 is invalid")
        hashes[name] = digest
    return hashes


def load_corpus(path: str | Path) -> dict[str, object]:
    """Load one bounded, fitting-only composite corpus into pilot arrays.

    Missing target cells have zero storage and are distinguished from observed
    numerical zeros by the returned ``observed`` mask.
    """

    blobs, archive_sha256 = _load_tar(Path(path))
    manifest_payload = blobs.get("composite-corpus/corpus.json")
    if manifest_payload is None:
        raise CorpusLoadError("corpus manifest is missing")
    manifest = _json(manifest_payload, "corpus manifest")
    source_id, audit_covariates = _validate_boundary(manifest)
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise CorpusLoadError("manifest counts are missing")
    entity_count = _positive_count(counts, "entities")
    query_count = _positive_count(counts, "queries", MAX_QUERIES)
    record_count = _positive_count(counts, "records", MAX_RECORDS)
    shard_count = _positive_count(counts, "shards", MAX_RECORDS)
    panel_count = _positive_count(counts, "panels", MAX_QUERIES)
    if panel_count != 1:
        raise CorpusLoadError("pilot loader requires exactly one query panel")
    if record_count * query_count > MAX_DENSE_CELLS:
        raise CorpusLoadError("dense pilot target allocation exceeds its bound")

    entity_name, entity_payload = _ref_payload(
        blobs, manifest.get("entityDictionary"), "entityDictionary"
    )
    query_name, query_payload = _ref_payload(
        blobs, manifest.get("queryDictionary"), "queryDictionary"
    )
    panel_name, panel_payload = _ref_payload(
        blobs, manifest.get("queryPanels"), "queryPanels"
    )
    trajectory_name, trajectory_payload = _ref_payload(
        blobs, manifest.get("trajectoryInterventions"), "trajectoryInterventions"
    )
    _require_ref_count(manifest.get("entityDictionary"), entity_count, "entityDictionary")
    _require_ref_count(manifest.get("queryDictionary"), query_count, "queryDictionary")
    _require_ref_count(manifest.get("queryPanels"), panel_count, "queryPanels")
    trajectory_count = counts.get("trajectoryInterventions")
    if type(trajectory_count) is not int or trajectory_count < 0:
        raise CorpusLoadError("manifest count trajectoryInterventions is invalid")
    _require_ref_count(
        manifest.get("trajectoryInterventions"),
        trajectory_count,
        "trajectoryInterventions",
    )

    entities = _read_npz(
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
    feature_dim = manifest.get("featurePack", {}).get("entityFeatureDim") if isinstance(manifest.get("featurePack"), dict) else None
    if type(feature_dim) is not int or feature_dim <= 0:
        raise CorpusLoadError("entity feature dimension is invalid")
    _require_array(entities["entity_taxon"], (entity_count,), np.dtype("<i8"), "entity_taxon")
    _require_array(entities["entity_type"], (entity_count,), np.dtype("<i8"), "entity_type")
    _require_array(
        entities["entity_feature_value"],
        (entity_count, feature_dim),
        np.dtype("<f4"),
        "entity_feature_value",
    )
    _require_array(
        entities["entity_feature_present"],
        (entity_count, feature_dim),
        np.dtype("|b1"),
        "entity_feature_present",
    )
    ids = entities["entity_id"]
    if ids.shape != (entity_count,) or ids.dtype.kind != "U":
        raise CorpusLoadError("entity_id dtype or shape drift")
    entity_keys = tuple(
        (int(taxon), str(identifier))
        for taxon, identifier in zip(entities["entity_taxon"], ids)
    )
    if (
        any(taxon <= 0 or not identifier or identifier != identifier.strip() for taxon, identifier in entity_keys)
        or len(set(entity_keys)) != entity_count
    ):
        raise CorpusLoadError("entity composite identities are invalid or duplicated")
    feature_present = entities["entity_feature_present"]
    feature_values = entities["entity_feature_value"]
    if not np.isfinite(feature_values[feature_present]).all() or np.any(
        feature_values[~feature_present] != 0
    ):
        raise CorpusLoadError("entity features are non-finite or missing storage is nonzero")

    queries = _read_npz(
        query_payload, {"query_entity_index", "query_readout_index"}, "queryDictionary"
    )
    _require_array(
        queries["query_entity_index"], (query_count,), np.dtype("<i8"), "query_entity_index"
    )
    _require_array(
        queries["query_readout_index"], (query_count,), np.dtype("<i8"), "query_readout_index"
    )
    query_entity_index = queries["query_entity_index"]
    if np.any(query_entity_index < 0) or np.any(query_entity_index >= entity_count):
        raise CorpusLoadError("query entity index is outside the entity dictionary")
    query_identities = tuple(
        (entity_keys[int(entity)], int(readout))
        for entity, readout in zip(query_entity_index, queries["query_readout_index"])
    )
    if len(set(query_identities)) != query_count:
        raise CorpusLoadError("query composite identities are duplicated")
    if np.any(queries["query_readout_index"] != 0):
        raise CorpusLoadError("query dictionary uses an undeclared readout type")
    panels = _read_npz(
        panel_payload,
        {"panel_id", "panel_indptr", "panel_query_index"},
        "queryPanels",
    )
    if panels["panel_id"].shape != (1,) or panels["panel_id"].dtype.kind != "U":
        raise CorpusLoadError("query panel ID dtype or shape drift")
    _require_array(panels["panel_indptr"], (2,), np.dtype("<i8"), "panel_indptr")
    _require_array(
        panels["panel_query_index"],
        (query_count,),
        np.dtype("<i8"),
        "panel_query_index",
    )
    if not np.array_equal(panels["panel_indptr"], np.asarray([0, query_count])) or not np.array_equal(
        panels["panel_query_index"], np.arange(query_count, dtype=np.int64)
    ):
        raise CorpusLoadError("query panel does not contain the ordered query dictionary")

    trajectory_keys: list[tuple[int, str]] = []
    try:
        text = trajectory_payload.decode("utf-8")
        if trajectory_payload and not trajectory_payload.endswith(b"\n"):
            raise CorpusLoadError("trajectory intervention JSONL is not LF terminated")
        for number, line in enumerate(text.splitlines(), start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise CorpusLoadError(f"trajectory intervention {number} is malformed")
            key = (row.get("ncbiTaxon"), row.get("entityId"))
            if type(key[0]) is not int or not isinstance(key[1], str):
                raise CorpusLoadError(f"trajectory intervention {number} is malformed")
            trajectory_keys.append((key[0], key[1]))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusLoadError("trajectory intervention JSONL is invalid") from error
    if (
        len(trajectory_keys) != trajectory_count
        or len(trajectory_keys) != len(set(trajectory_keys))
        or not set(trajectory_keys) <= set(entity_keys)
    ):
        raise CorpusLoadError("trajectory intervention identities are duplicated or unresolved")

    shard_refs = manifest.get("shards")
    if not isinstance(shard_refs, list) or len(shard_refs) != shard_count:
        raise CorpusLoadError("manifest shard count drift")
    allowed_members = {
        "composite-corpus/corpus.json",
        entity_name,
        query_name,
        panel_name,
        trajectory_name,
    }
    loaded_shards: list[tuple[Mapping[str, object], dict[str, np.ndarray], str]] = []
    shard_hashes: dict[str, str] = {}
    for index, reference in enumerate(shard_refs):
        name, payload = _ref_payload(blobs, reference, f"shards[{index}]")
        expected_name = f"composite-corpus/shards/shard-{index:05d}.npz"
        if name != expected_name:
            raise CorpusLoadError("shard paths are not canonical and contiguous")
        arrays = _read_npz(payload, _SHARD_ARRAYS, f"shards[{index}]")
        loaded_shards.append((reference, arrays, name))
        shard_hashes[name.removeprefix("composite-corpus/")] = _sha256(payload)
        allowed_members.add(name)
    if set(blobs) != allowed_members:
        raise CorpusLoadError("archive contains undeclared members")

    targets = np.zeros((record_count, query_count), dtype=np.float32)
    observed = np.zeros((record_count, query_count), dtype=np.bool_)
    action_index = np.empty(record_count, dtype=np.int64)
    record_ids: list[str] = []
    action_keys: list[tuple[int, str]] = []
    cursor = 0
    observed_total = 0
    for shard_number, (reference, arrays, _) in enumerate(loaded_shards):
        records = reference.get("records")
        values = reference.get("targetValues")
        if type(records) is not int or records <= 0 or type(values) is not int or values < 0:
            raise CorpusLoadError(f"shards[{shard_number}] counts are invalid")
        if cursor + records > record_count:
            raise CorpusLoadError("shard records exceed the manifest count")
        shard_ids = arrays["record_id"]
        if shard_ids.shape != (records,) or shard_ids.dtype.kind != "U":
            raise CorpusLoadError("record_id dtype or shape drift")
        local_ids = [str(item) for item in shard_ids]
        if any(not item or item != item.strip() for item in local_ids):
            raise CorpusLoadError("record IDs must be non-empty trimmed strings")

        actions = arrays["action_entity_index"]
        masks = arrays["action_mask"]
        if actions.ndim != 2 or actions.shape[0] != records or masks.shape != actions.shape or masks.dtype != np.dtype("|b1"):
            raise CorpusLoadError("action index or mask shape drift")
        active_counts = masks.sum(axis=1)
        if not np.array_equal(active_counts, np.ones(records, dtype=active_counts.dtype)):
            raise CorpusLoadError("current pilot accepts exactly one action per record")
        chosen = actions[np.arange(records), np.argmax(masks, axis=1)]
        if actions.dtype != np.dtype("<i8") or np.any(chosen < 0) or np.any(chosen >= entity_count):
            raise CorpusLoadError("action entity index is invalid")
        action_types = arrays["action_type"]
        _require_array(action_types, actions.shape, np.dtype("<i8"), "action_type")
        if np.any(action_types[masks] != 0):
            raise CorpusLoadError("active action uses an undeclared action type")
        species = arrays["species_taxon"]
        _require_array(species, (records,), np.dtype("<i8"), "species_taxon")
        if np.any(entities["entity_taxon"][chosen] != species):
            raise CorpusLoadError("action composite taxon join failed")
        source_indices = arrays["source_index"]
        _require_array(source_indices, (records,), np.dtype("<i8"), "source_index")
        if np.any(source_indices != 0):
            raise CorpusLoadError("record source index disagrees with the pilot source")

        indptr = arrays["target_indptr"]
        query_indices = arrays["target_query_index"]
        values_array = arrays["target_value"]
        _require_array(indptr, (records + 1,), np.dtype("<i8"), "target_indptr")
        _require_array(query_indices, (values,), np.dtype("<i8"), "target_query_index")
        _require_array(values_array, (values,), np.dtype("<f4"), "target_value")
        if indptr[0] != 0 or indptr[-1] != values or np.any(indptr[1:] < indptr[:-1]):
            raise CorpusLoadError("target CSR offsets are invalid")
        if (
            np.any(query_indices < 0)
            or np.any(query_indices >= query_count)
            or not np.isfinite(values_array).all()
        ):
            raise CorpusLoadError("target indices or observed target values are invalid")
        target_taxa = entities["entity_taxon"][query_entity_index[query_indices]]
        if np.any(target_taxa != np.repeat(species, np.diff(indptr))):
            raise CorpusLoadError("target query composite taxon join failed")
        for local_row in range(records):
            start, stop = int(indptr[local_row]), int(indptr[local_row + 1])
            selected = query_indices[start:stop]
            if len(selected) != len({int(item) for item in selected}):
                raise CorpusLoadError("a record repeats a target query")
            global_row = cursor + local_row
            targets[global_row, selected] = values_array[start:stop]
            observed[global_row, selected] = True
        action_index[cursor : cursor + records] = chosen
        record_ids.extend(local_ids)
        action_keys.extend(entity_keys[int(item)] for item in chosen)
        cursor += records
        observed_total += values

    if cursor != record_count or len(set(record_ids)) != record_count:
        raise CorpusLoadError("record count drift or duplicate record IDs")
    declared_targets = counts.get("targetValues")
    if type(declared_targets) is not int or declared_targets != observed_total or int(observed.sum()) != observed_total:
        raise CorpusLoadError("observed target count does not match the manifest")
    if np.any(targets[~observed] != 0):
        raise CorpusLoadError("missing target storage must be zero")
    if set(action_keys) != set(trajectory_keys):
        raise CorpusLoadError("active actions do not match trajectory intervention identities")

    source_hashes = {
        "archive": archive_sha256,
        "corpus_manifest": _sha256(manifest_payload),
        "entity_dictionary": _sha256(entity_payload),
        "query_dictionary": _sha256(query_payload),
        "query_panels": _sha256(panel_payload),
        "trajectory_interventions": _sha256(trajectory_payload),
        **{f"input:{name}": digest for name, digest in _lineage_hashes(manifest).items()},
        **{f"shard:{name}": digest for name, digest in shard_hashes.items()},
    }
    return {
        "entity_features": feature_values.astype(np.float32, copy=False),
        "entity_present": feature_present.astype(np.bool_, copy=False),
        "entity_keys": entity_keys,
        "action_index": action_index,
        "query_entity_index": query_entity_index.astype(np.int64, copy=False),
        "targets": targets,
        "observed": observed,
        "record_ids": tuple(record_ids),
        "action_keys": tuple(action_keys),
        "source_id": source_id,
        "metadata": {
            "schema": manifest["schema"],
            "dataset_id": manifest.get("datasetId"),
            "normalization": manifest.get("normalization"),
            "counts": dict(counts),
            "source_hashes": source_hashes,
            "production_archive_match": archive_sha256 == PRODUCTION_ARCHIVE_SHA256,
            "masked_audit_covariates": audit_covariates,
        },
    }


def split_by_gene(
    action_keys: tuple[tuple[int, str], ...] | list[tuple[int, str]], seed: int = 731
) -> dict[str, np.ndarray]:
    """Return deterministic record indices grouped by composite action identity."""

    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    groups: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    prefix = f"slp11-development-v1|{seed}|"
    for index, key in enumerate(action_keys):
        if (
            not isinstance(key, (tuple, list))
            or len(key) != 2
            or type(key[0]) is not int
            or key[0] <= 0
            or not isinstance(key[1], str)
            or not key[1]
            or key[1] != key[1].strip()
        ):
            raise ValueError(f"action_keys[{index}] is not a composite identity")
        digest = hashlib.sha256(f"{prefix}{key[0]}|{key[1]}".encode()).digest()
        bucket = int.from_bytes(digest[:8], byteorder="big", signed=False) % 100
        group = "train" if bucket < 70 else "validation" if bucket < 85 else "test"
        groups[group].append(index)
    return {name: np.asarray(indices, dtype=np.int64) for name, indices in groups.items()}
