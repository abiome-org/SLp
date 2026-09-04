"""Deterministic preparation of species-native yeast molecular corpora.

The input is an immutable, rights-bearing source snapshot.  This module does
not download upstream data and does not interpret display symbols as identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile
from typing import Any, Iterator
import zipfile

import numpy as np


SOURCE_SCHEMA = "slp.yeast-source/v1"
RECORD_SCHEMA = "slp.yeast-molecular-record/v1"
CORPUS_SCHEMA = "slp.corpus/v1"
YEAST_TAXON = 4932
SGD_IDENTIFIER = re.compile(r"^SGD:S[0-9]{9}$")
ALLOWED_ROLES = {
    "pretrain",
    "molecular-validation",
    "molecular-reward",
    "molecular-final",
}
SOURCE_FIELDS = {
    "schema",
    "sourceId",
    "sourceRelease",
    "ncbiTaxon",
    "stableIdNamespace",
    "labelClass",
    "benchmarkLabelsPresent",
    "modalities",
    "rightsFile",
    "rightsSha256",
    "rawFormat",
    "rawFiles",
}
RECORD_FIELDS = {
    "schema",
    "recordId",
    "perturbationId",
    "ncbiTaxon",
    "modality",
    "assay",
    "protocol",
    "endpoint",
    "normalization",
    "experimentalMetadata",
    "speciesFeatures",
    "context",
    "actions",
    "queries",
}


class YeastPreparationError(ValueError):
    """Raised when source admission or normalization must fail closed."""


@dataclass(frozen=True)
class PreparationConfig:
    dataset_id: str
    version: str
    role: str
    entity_feature_dim: int
    species_feature_dim: int
    species_feature_vector: tuple[float, ...]
    action_covariate_dim: int
    readout_types: tuple[str, ...]
    max_context_tokens: int
    max_action_tokens: int
    max_query_tokens: int
    shard_records: int

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> "PreparationConfig":
        required = {
            "datasetId",
            "version",
            "role",
            "entityFeatureDim",
            "speciesFeatureDim",
            "speciesFeatureVector",
            "actionCovariateDim",
            "readoutTypes",
            "maxContextTokens",
            "maxActionTokens",
            "maxQueryTokens",
            "shardRecords",
        }
        missing = sorted(required - value.keys())
        extra = sorted(value.keys() - required)
        if missing:
            raise YeastPreparationError(
                f"preparation config is missing fields: {', '.join(missing)}"
            )
        if extra:
            raise YeastPreparationError(
                f"unsupported preparation config fields: {', '.join(extra)}"
            )
        dataset_id = _nonempty_string(value["datasetId"], "datasetId")
        version = _immutable_version(value["version"], "version")
        role = _nonempty_string(value["role"], "role")
        if role not in ALLOWED_ROLES:
            raise YeastPreparationError(f"unsupported molecular corpus role: {role!r}")
        readouts = value["readoutTypes"]
        if (
            not isinstance(readouts, list)
            or not readouts
            or any(not isinstance(item, str) or not item.strip() for item in readouts)
            or len(readouts) != len(set(readouts))
        ):
            raise YeastPreparationError("readoutTypes must be unique non-empty strings")
        entity_feature_dim = _bounded_integer(
            value["entityFeatureDim"], "entityFeatureDim", 1, 16384
        )
        species_feature_dim = _bounded_integer(
            value["speciesFeatureDim"], "speciesFeatureDim", 1, 1024
        )
        species_feature_vector = tuple(
            _finite_vector(
                value["speciesFeatureVector"],
                species_feature_dim,
                "speciesFeatureVector",
            )
        )
        return cls(
            dataset_id=dataset_id,
            version=version,
            role=role,
            entity_feature_dim=entity_feature_dim,
            species_feature_dim=species_feature_dim,
            species_feature_vector=species_feature_vector,
            action_covariate_dim=_bounded_integer(
                value["actionCovariateDim"], "actionCovariateDim", 1, 1024
            ),
            readout_types=tuple(readouts),
            max_context_tokens=_bounded_integer(
                value["maxContextTokens"], "maxContextTokens", 1, 4096
            ),
            max_action_tokens=_bounded_integer(
                value["maxActionTokens"], "maxActionTokens", 1, 128
            ),
            max_query_tokens=_bounded_integer(
                value["maxQueryTokens"], "maxQueryTokens", 1, 16384
            ),
            shard_records=_bounded_integer(
                value["shardRecords"], "shardRecords", 1, 65536
            ),
        )

    def provenance(self) -> dict[str, object]:
        return {
            "datasetId": self.dataset_id,
            "version": self.version,
            "role": self.role,
            "entityFeatureDim": self.entity_feature_dim,
            "speciesFeatureDim": self.species_feature_dim,
            "speciesFeatureVector": list(self.species_feature_vector),
            "actionCovariateDim": self.action_covariate_dim,
            "readoutTypes": list(self.readout_types),
            "maxContextTokens": self.max_context_tokens,
            "maxActionTokens": self.max_action_tokens,
            "maxQueryTokens": self.max_query_tokens,
            "shardRecords": self.shard_records,
        }


def _bounded_integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or value > maximum
    ):
        raise YeastPreparationError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise YeastPreparationError(f"{name} must be a non-empty string")
    return value.strip()


def _immutable_version(value: object, name: str) -> str:
    version = _nonempty_string(value, name)
    if version.casefold() in {"latest", "current", "head", "main", "master"}:
        raise YeastPreparationError(f"{name} must identify an immutable release")
    return version


def _json_loads(value: str, label: str) -> Any:
    def reject_constant(constant: str) -> None:
        raise YeastPreparationError(f"{label} contains non-finite JSON value {constant}")

    try:
        return json.loads(value, parse_constant=reject_constant)
    except YeastPreparationError:
        raise
    except (json.JSONDecodeError, TypeError) as error:
        raise YeastPreparationError(f"invalid JSON in {label}: {error}") from error


def _relative_file(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise YeastPreparationError(f"{label} must be a portable relative path")
    portable = PurePosixPath(value)
    if portable.is_absolute() or ".." in portable.parts:
        raise YeastPreparationError(f"unsafe {label}: {value!r}")
    path = root.joinpath(*portable.parts)
    if path.is_symlink() or not path.is_file():
        raise YeastPreparationError(f"missing or symlinked {label}: {value}")
    try:
        path.resolve().relative_to(root)
    except ValueError as error:
        raise YeastPreparationError(f"{label} resolves outside the source snapshot") from error
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _parse_rights(path: Path) -> dict[str, object]:
    """Parse the deliberately small, flat rights declaration without PyYAML."""
    result: dict[str, object] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line[:1].isspace() or ":" not in raw_line:
            raise YeastPreparationError(
                f"rights declaration must be a flat mapping (line {line_number})"
            )
        key, raw_value = raw_line.split(":", 1)
        key = key.strip()
        scalar = raw_value.strip()
        if not key or not scalar or key in result:
            raise YeastPreparationError(f"invalid rights declaration line {line_number}")
        if scalar == "true":
            parsed: object = True
        elif scalar == "false":
            parsed = False
        elif scalar.startswith(('"', "'")):
            if scalar.startswith("'") and scalar.endswith("'"):
                parsed = scalar[1:-1]
            else:
                parsed = _json_loads(scalar, f"rights line {line_number}")
        else:
            parsed = scalar
        result[key] = parsed
    if result.get("trainingAllowed") is not True:
        raise YeastPreparationError("source rights do not explicitly allow model training")
    license_name = _nonempty_string(result.get("license"), "rights.license")
    if license_name.casefold() in {"unknown", "unverified", "none"}:
        raise YeastPreparationError("source license must be verified before preparation")
    _nonempty_string(result.get("source"), "rights.source")
    redistribution = result.get("redistributionAllowed")
    if redistribution is not None and not isinstance(redistribution, bool):
        raise YeastPreparationError("rights.redistributionAllowed must be boolean when present")
    return result


def _load_source(source_root: Path) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    manifest_path = source_root / "source.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise YeastPreparationError("source snapshot must contain a regular source.json")
    manifest = _json_loads(manifest_path.read_text(encoding="utf-8"), "source.json")
    if not isinstance(manifest, dict) or set(manifest) != SOURCE_FIELDS:
        missing = sorted(SOURCE_FIELDS - set(manifest) if isinstance(manifest, dict) else SOURCE_FIELDS)
        extra = sorted(set(manifest) - SOURCE_FIELDS if isinstance(manifest, dict) else set())
        raise YeastPreparationError(
            f"source.json fields do not match the contract; missing={missing}, extra={extra}"
        )
    if manifest["schema"] != SOURCE_SCHEMA:
        raise YeastPreparationError(f"unsupported source schema: {manifest['schema']!r}")
    manifest["sourceId"] = _nonempty_string(manifest["sourceId"], "sourceId")
    manifest["sourceRelease"] = _immutable_version(
        manifest["sourceRelease"], "sourceRelease"
    )
    if manifest["ncbiTaxon"] != YEAST_TAXON:
        raise YeastPreparationError("source must retain NCBI taxon 4932 yeast identity")
    if manifest["stableIdNamespace"] != "SGD":
        raise YeastPreparationError("source stableIdNamespace must be SGD")
    if manifest["labelClass"] != "molecular" or manifest["benchmarkLabelsPresent"] is not False:
        raise YeastPreparationError("world-model sources may contain only molecular labels")
    if manifest["rawFormat"] != "slp.yeast-molecular-jsonl/v1":
        raise YeastPreparationError(f"unsupported rawFormat: {manifest['rawFormat']!r}")
    modalities = manifest["modalities"]
    if (
        not isinstance(modalities, list)
        or not modalities
        or any(not isinstance(item, str) or not item for item in modalities)
        or len(modalities) != len(set(modalities))
    ):
        raise YeastPreparationError("modalities must be unique non-empty strings")

    rights_path = _relative_file(source_root, manifest["rightsFile"], "rightsFile")
    if _sha256(rights_path) != manifest["rightsSha256"]:
        raise YeastPreparationError("rights declaration digest mismatch")
    rights = _parse_rights(rights_path)

    raw_files = manifest["rawFiles"]
    if not isinstance(raw_files, list) or not raw_files:
        raise YeastPreparationError("rawFiles must contain at least one pinned file")
    if any(
        not isinstance(item, dict) or not isinstance(item.get("path"), str)
        for item in raw_files
    ):
        raise YeastPreparationError("rawFiles entries require string paths")
    expected_paths = [item["path"] for item in raw_files]
    if expected_paths != sorted(expected_paths):
        raise YeastPreparationError("rawFiles must be strictly path-sorted")
    if len(expected_paths) != len(set(expected_paths)):
        raise YeastPreparationError("rawFiles contains duplicate paths")
    verified: list[dict[str, object]] = []
    for item in raw_files:
        if set(item) != {"path", "sha256", "records"}:
            raise YeastPreparationError("each raw file requires only path, sha256, and records")
        path = _relative_file(source_root, item["path"], "raw file")
        digest = item["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise YeastPreparationError("raw file sha256 must be lowercase hexadecimal")
        if _sha256(path) != digest:
            raise YeastPreparationError(f"raw file digest mismatch: {item['path']}")
        count = item["records"]
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise YeastPreparationError("raw file records must be a positive integer")
        verified.append({**item, "resolvedPath": path})
    return manifest, rights, verified


def _finite_vector(value: object, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise YeastPreparationError(f"{label} must contain exactly {length} numeric values")
    vector: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(item):
            raise YeastPreparationError(f"{label} must contain only finite numeric values")
        vector.append(float(item))
    return vector


def _sgd_id(value: object, label: str) -> str:
    identifier = _nonempty_string(value, label)
    if SGD_IDENTIFIER.fullmatch(identifier) is None:
        raise YeastPreparationError(
            f"{label} must be a stable SGD CURIE, not a display symbol: {identifier!r}"
        )
    return identifier


def _ordered_unique(keys: list[tuple[object, ...]], label: str) -> None:
    if keys != sorted(keys):
        raise YeastPreparationError(f"{label} must be canonically sorted")
    if len(keys) != len(set(keys)):
        raise YeastPreparationError(f"{label} contains duplicate identities")


def _validate_record(record: object, config: PreparationConfig, modalities: set[str]) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != RECORD_FIELDS:
        missing = sorted(RECORD_FIELDS - set(record) if isinstance(record, dict) else RECORD_FIELDS)
        extra = sorted(set(record) - RECORD_FIELDS if isinstance(record, dict) else set())
        raise YeastPreparationError(
            f"record fields do not match the contract; missing={missing}, extra={extra}"
        )
    if record["schema"] != RECORD_SCHEMA:
        raise YeastPreparationError(f"unsupported record schema: {record['schema']!r}")
    record_id = _nonempty_string(record["recordId"], "recordId")
    perturbation_id = _nonempty_string(record["perturbationId"], "perturbationId")
    if record["ncbiTaxon"] != YEAST_TAXON:
        raise YeastPreparationError(f"record {record_id!r} is not NCBI taxon 4932")
    modality = _nonempty_string(record["modality"], "modality")
    if modality not in modalities:
        raise YeastPreparationError(f"record {record_id!r} has undeclared modality {modality!r}")
    for field in ("assay", "protocol", "endpoint", "normalization"):
        record[field] = _nonempty_string(record[field], field)
    if not isinstance(record["experimentalMetadata"], dict):
        raise YeastPreparationError("experimentalMetadata must be an object")
    record["speciesFeatures"] = _finite_vector(
        record["speciesFeatures"], config.species_feature_dim, "speciesFeatures"
    )
    if tuple(record["speciesFeatures"]) != config.species_feature_vector:
        raise YeastPreparationError(
            "record speciesFeatures must exactly match the configured yeast speciesFeatureVector"
        )

    context = record["context"]
    actions = record["actions"]
    queries = record["queries"]
    if not isinstance(context, list) or not context or len(context) > config.max_context_tokens:
        raise YeastPreparationError("context token count is outside the configured bound")
    if not isinstance(actions, list) or not actions or len(actions) > config.max_action_tokens:
        raise YeastPreparationError("action token count is outside the configured bound")
    if not isinstance(queries, list) or not queries or len(queries) > config.max_query_tokens:
        raise YeastPreparationError("query token count is outside the configured bound")

    normalized_context: list[dict[str, Any]] = []
    for index, item in enumerate(context):
        if not isinstance(item, dict) or set(item) != {"entityId", "features"}:
            raise YeastPreparationError("context tokens require only entityId and features")
        normalized_context.append(
            {
                "entityId": _sgd_id(item["entityId"], f"context[{index}].entityId"),
                "features": _finite_vector(
                    item["features"], config.entity_feature_dim, f"context[{index}].features"
                ),
            }
        )
    _ordered_unique(
        [(item["entityId"],) for item in normalized_context],
        "context tokens",
    )

    normalized_actions: list[dict[str, Any]] = []
    for index, item in enumerate(actions):
        if not isinstance(item, dict) or set(item) != {"entityId", "features", "covariates"}:
            raise YeastPreparationError(
                "action tokens require only entityId, features, and covariates"
            )
        normalized_actions.append(
            {
                "entityId": _sgd_id(item["entityId"], f"actions[{index}].entityId"),
                "features": _finite_vector(
                    item["features"], config.entity_feature_dim, f"actions[{index}].features"
                ),
                "covariates": _finite_vector(
                    item["covariates"],
                    config.action_covariate_dim,
                    f"actions[{index}].covariates",
                ),
            }
        )
    _ordered_unique(
        [(item["entityId"],) for item in normalized_actions],
        "action tokens",
    )

    normalized_queries: list[dict[str, Any]] = []
    for index, item in enumerate(queries):
        expected = {"entityId", "features", "readoutType", "target", "observed"}
        if not isinstance(item, dict) or set(item) != expected:
            raise YeastPreparationError(
                "query tokens require entityId, features, readoutType, target, and observed"
            )
        readout = _nonempty_string(item["readoutType"], f"queries[{index}].readoutType")
        if readout not in config.readout_types:
            raise YeastPreparationError(f"unknown readout type: {readout!r}")
        target = item["target"]
        if not isinstance(target, (int, float)) or isinstance(target, bool) or not math.isfinite(target):
            raise YeastPreparationError("query targets must be finite numeric values")
        if not isinstance(item["observed"], bool):
            raise YeastPreparationError("query observed flags must be boolean")
        normalized_queries.append(
            {
                "entityId": _sgd_id(item["entityId"], f"queries[{index}].entityId"),
                "features": _finite_vector(
                    item["features"], config.entity_feature_dim, f"queries[{index}].features"
                ),
                "readoutType": readout,
                "target": float(target),
                "observed": item["observed"],
            }
        )
    _ordered_unique(
        [(item["readoutType"], item["entityId"]) for item in normalized_queries],
        "query tokens",
    )
    if not any(item["observed"] for item in normalized_queries):
        raise YeastPreparationError("each record must contain at least one observed query")

    return {
        **record,
        "recordId": record_id,
        "perturbationId": perturbation_id,
        "modality": modality,
        "context": normalized_context,
        "actions": normalized_actions,
        "queries": normalized_queries,
    }


def _iter_records(
    raw_files: list[dict[str, object]],
    config: PreparationConfig,
    modalities: set[str],
) -> Iterator[dict[str, Any]]:
    previous_record_id: str | None = None
    for raw in raw_files:
        path = raw["resolvedPath"]
        assert isinstance(path, Path)
        count = 0
        with path.open("r", encoding="utf-8", newline="") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise YeastPreparationError(
                        f"blank records are forbidden in {raw['path']}:{line_number}"
                    )
                record = _validate_record(
                    _json_loads(line, f"{raw['path']}:{line_number}"), config, modalities
                )
                record_id = record["recordId"]
                if previous_record_id is not None and record_id <= previous_record_id:
                    raise YeastPreparationError(
                        "recordId values must be unique and strictly sorted across raw files"
                    )
                previous_record_id = record_id
                count += 1
                yield record
        if count != raw["records"]:
            raise YeastPreparationError(
                f"raw record count mismatch for {raw['path']}: expected {raw['records']}, got {count}"
            )


def _string_array(shape: tuple[int, ...], values: list[str]) -> np.ndarray:
    maximum = max((len(item) for item in values), default=1)
    return np.full(shape, "", dtype=f"U{maximum}")


def _record_arrays(
    records: list[dict[str, Any]],
    config: PreparationConfig,
    source_id: str,
) -> dict[str, np.ndarray]:
    count = len(records)
    shape_context = (count, config.max_context_tokens)
    shape_action = (count, config.max_action_tokens)
    shape_query = (count, config.max_query_tokens)
    identifiers = [
        item["entityId"]
        for record in records
        for field in ("context", "actions", "queries")
        for item in record[field]
    ]
    text_values = [
        str(record[field])
        for record in records
        for field in (
            "recordId",
            "perturbationId",
            "modality",
            "assay",
            "protocol",
            "endpoint",
            "normalization",
        )
    ] + identifiers
    text_values.append(source_id)
    metadata_values = [_canonical_json(record["experimentalMetadata"]) for record in records]
    arrays: dict[str, np.ndarray] = {
        "context_features": np.zeros(
            (*shape_context, config.entity_feature_dim), dtype="float32"
        ),
        "context_mask": np.zeros(shape_context, dtype=bool),
        "action_features": np.zeros(
            (*shape_action, config.entity_feature_dim), dtype="float32"
        ),
        "action_covariates": np.zeros(
            (*shape_action, config.action_covariate_dim), dtype="float32"
        ),
        "action_mask": np.zeros(shape_action, dtype=bool),
        "query_features": np.zeros(
            (*shape_query, config.entity_feature_dim), dtype="float32"
        ),
        "query_mask": np.zeros(shape_query, dtype=bool),
        "readout_type": np.zeros(shape_query, dtype="int64"),
        "species_features": np.zeros((count, config.species_feature_dim), dtype="float32"),
        "species_taxon": np.full((count,), YEAST_TAXON, dtype="int64"),
        "target": np.zeros(shape_query, dtype="float32"),
        "target_mask": np.zeros(shape_query, dtype=bool),
        "record_id": _string_array((count,), text_values),
        "source_id": _string_array((count,), text_values),
        "perturbation_id": _string_array((count,), text_values),
        "context_entity_id": _string_array(shape_context, identifiers),
        "action_curies": _string_array(shape_action, identifiers),
        "query_entity_id": _string_array(shape_query, identifiers),
        "modality": _string_array((count,), text_values),
        "assay": _string_array((count,), text_values),
        "protocol": _string_array((count,), text_values),
        "endpoint": _string_array((count,), text_values),
        "normalization": _string_array((count,), text_values),
        "experimental_metadata_json": _string_array((count,), metadata_values),
    }
    readout_index = {name: index for index, name in enumerate(config.readout_types)}
    for row, record in enumerate(records):
        arrays["record_id"][row] = record["recordId"]
        arrays["source_id"][row] = source_id
        arrays["perturbation_id"][row] = record["perturbationId"]
        arrays["species_features"][row] = record["speciesFeatures"]
        for field in ("modality", "assay", "protocol", "endpoint", "normalization"):
            arrays[field][row] = record[field]
        arrays["experimental_metadata_json"][row] = _canonical_json(
            record["experimentalMetadata"]
        )
        for column, token in enumerate(record["context"]):
            arrays["context_features"][row, column] = token["features"]
            arrays["context_mask"][row, column] = True
            arrays["context_entity_id"][row, column] = token["entityId"]
        for column, token in enumerate(record["actions"]):
            arrays["action_features"][row, column] = token["features"]
            arrays["action_covariates"][row, column] = token["covariates"]
            arrays["action_mask"][row, column] = True
            arrays["action_curies"][row, column] = token["entityId"]
        for column, token in enumerate(record["queries"]):
            arrays["query_features"][row, column] = token["features"]
            arrays["query_mask"][row, column] = True
            arrays["readout_type"][row, column] = readout_index[token["readoutType"]]
            arrays["target"][row, column] = token["target"]
            arrays["target_mask"][row, column] = token["observed"]
            arrays["query_entity_id"][row, column] = token["entityId"]
    return arrays


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name in sorted(arrays):
            buffer = BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, buffer.getvalue())


def _content_digest(files: list[dict[str, object]]) -> str:
    projection = [{"path": item["path"], "sha256": item["sha256"]} for item in files]
    return hashlib.sha256(_canonical_json(projection).encode("utf-8")).hexdigest()


def prepare_yeast_snapshot(
    source: str | Path,
    destination: str | Path,
    config_value: dict[str, object],
) -> dict[str, object]:
    """Prepare a bounded, deterministic SLp corpus directory."""
    source_root = Path(source).resolve()
    if not source_root.is_dir():
        raise YeastPreparationError("source input must be a snapshot directory")
    destination = Path(destination).resolve()
    if destination.exists():
        raise YeastPreparationError("destination must not already exist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    config = PreparationConfig.from_mapping(config_value)
    source_manifest, rights, raw_files = _load_source(source_root)
    source_id = str(source_manifest["sourceId"])
    modalities = set(source_manifest["modalities"])

    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / destination.name
        staging.mkdir()
        shard_manifests: list[dict[str, object]] = []
        trajectory_genes: set[str] = set()
        records_total = 0
        chunk: list[dict[str, Any]] = []

        def flush() -> None:
            nonlocal chunk
            if not chunk:
                return
            name = f"shard-{len(shard_manifests):05d}.npz"
            path = staging / name
            _write_deterministic_npz(path, _record_arrays(chunk, config, source_id))
            shard_manifests.append(
                {"path": name, "sha256": _sha256(path), "records": len(chunk)}
            )
            chunk = []

        for record in _iter_records(raw_files, config, modalities):
            trajectory_genes.update(item["entityId"] for item in record["actions"])
            chunk.append(record)
            records_total += 1
            if len(chunk) == config.shard_records:
                flush()
        flush()
        if not records_total or not trajectory_genes:
            raise YeastPreparationError("source contains no trainable molecular records")

        genes_path = staging / "trajectory-genes.txt"
        genes_path.write_text(
            "".join(f"{identifier}\n" for identifier in sorted(trajectory_genes)),
            encoding="utf-8",
            newline="\n",
        )
        corpus = {
            "schema": CORPUS_SCHEMA,
            "datasetId": config.dataset_id,
            "version": config.version,
            "role": config.role,
            "labelClass": "molecular",
            "benchmarkLabelsPresent": False,
            "speciesTaxa": [YEAST_TAXON],
            "modalities": sorted(modalities),
            "trajectoryGenes": genes_path.name,
            "entityFeatureDim": config.entity_feature_dim,
            "speciesFeatureDim": config.species_feature_dim,
            "speciesFeatureVectors": {
                str(YEAST_TAXON): list(config.species_feature_vector)
            },
            "actionCovariateDim": config.action_covariate_dim,
            "readoutTypes": list(config.readout_types),
            "shards": shard_manifests,
        }
        corpus_path = staging / "corpus.json"
        _write_json(corpus_path, corpus)
        generated_files = [
            {"path": corpus_path.name, "sha256": _sha256(corpus_path)},
            {"path": genes_path.name, "sha256": _sha256(genes_path)},
            *[
                {"path": item["path"], "sha256": item["sha256"]}
                for item in shard_manifests
            ],
        ]
        content_digest = _content_digest(generated_files)
        provenance = {
            "schema": "slp.yeast-preparation/v1",
            "source": {
                "sourceId": source_manifest["sourceId"],
                "sourceRelease": source_manifest["sourceRelease"],
                "ncbiTaxon": YEAST_TAXON,
                "stableIdNamespace": "SGD",
                "rawFiles": [
                    {key: item[key] for key in ("path", "sha256", "records")}
                    for item in raw_files
                ],
            },
            "rights": {
                "path": source_manifest["rightsFile"],
                "sha256": source_manifest["rightsSha256"],
                "license": rights["license"],
                "trainingAllowed": True,
                "redistributionAllowed": rights.get("redistributionAllowed"),
                "source": rights["source"],
            },
            "preparation": config.provenance(),
            "records": records_total,
            "trajectoryGenes": len(trajectory_genes),
            "generatedFiles": generated_files,
            "contentSha256": content_digest,
        }
        _write_json(staging / "provenance.json", provenance)
        staging.replace(destination)
    return provenance


def write_deterministic_tar(corpus: str | Path, destination: str | Path) -> str:
    corpus = Path(corpus).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise YeastPreparationError("archive destination must not already exist")
    files = sorted(path for path in corpus.rglob("*") if path.is_file())
    with tarfile.open(destination, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in files:
            relative = path.relative_to(corpus).as_posix()
            info = tarfile.TarInfo(name=f"prepared-corpus/{relative}")
            info.size = path.stat().st_size
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = 0o644
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    return _sha256(destination)
