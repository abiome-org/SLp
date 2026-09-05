#!/usr/bin/env python
"""Freeze metadata, then evaluate frozen shared-context count forecasts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "modules/slp-1-1-count-world-evaluation-v1/evaluator.py"
K_MANIFEST = ROOT / "data/derived/slp11-human-k562-essential-raw-cells-v2/manifest.json"
R_MANIFEST = ROOT / "data/derived/slp11-human-rpe1-essential-raw-cells-v1/manifest.json"
K_CONTROL = ROOT / "data/derived/slp11-human-k562-essential-count-control/reconstruction-train-nt-gem-v1/gem-control-reference.npz"
R_CONTROL = ROOT / "data/derived/slp11-human-rpe1-essential-count-control/reconstruction-train-nt-gem-v1/gem-control-reference.npz"
DEFAULT_OUTPUT = ROOT / "results/slp11-transition/human-essential-count-shared-context-development-evaluation-v2"
FREEZE_NAME = "FORECASTS-FROZEN-BEFORE-DEVELOPMENT.json"

EXPECTED = {
    "k562": {
        "sourceId": "k562",
        "contextId": "replogle-2022-k562-essential-day-6",
        "genes": 305,
        "queries": 8563,
        "cells": 47914,
        "gems": 48,
    },
    "rpe1": {
        "sourceId": "rpe1",
        "contextId": "replogle-2022-rpe1-essential-day-7",
        "genes": 360,
        "queries": 8749,
        "cells": 39014,
        "gems": 56,
    },
}


def _load_module():
    spec = importlib.util.spec_from_file_location("slp11_count_world_evaluation_v1", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load evaluation module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EVAL = _load_module()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable output already exists with different bytes: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata_from_k562(manifest: dict) -> dict[str, np.ndarray]:
    base = K_MANIFEST.parent
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    source_rows: list[np.ndarray] = []
    cells: list[np.ndarray] = []
    dev_entries = [item for item in manifest["shards"] if item["role"] == "development-validation"]
    for item in dev_entries:
        # Deliberately load metadata members only. The raw CSR members are not accessed here.
        with np.load(base / item["path"], allow_pickle=False) as data:
            action = data["action_ids"].astype(str)
            gem = data["gem_group"].astype(np.int16)
            source_row = data["source_row_index"].astype(np.int64)
            cell = data["cell_ids"].astype(str)
            if not np.all(data["intervention_role"].astype(str) == "validation"):
                raise ValueError("K562 development shard contains a non-validation intervention")
            if not np.all(data["reconstruction_role"].astype(str) == "none"):
                raise ValueError("K562 development shard contains a reconstruction role")
            if np.any(data["is_control"].astype(bool)) or not np.all(data["entity_taxon"] == 9606):
                raise ValueError("K562 development shard contains control or non-human rows")
            if str(data["source_sha256"].item()) != manifest["source"]["sha256"]:
                raise ValueError("K562 shard/source hash identity mismatch")
            if len(action) != item["rows"]:
                raise ValueError("K562 shard row count mismatch")
            rows.append((action, gem))
            source_rows.append(source_row)
            cells.append(cell)
    action = np.concatenate([item[0] for item in rows])
    gem = np.concatenate([item[1] for item in rows])
    genes = np.array(sorted(set(action.tolist())))
    gene_lookup = {item: i for i, item in enumerate(genes.tolist())}
    with np.load(K_CONTROL, allow_pickle=False) as control:
        query = control["query_ids"].astype(str)
        gems = control["gem_group"].astype(np.int16)
    gem_lookup = {int(item): i for i, item in enumerate(gems)}
    gene_index = np.array([gene_lookup[item] for item in action], dtype=np.int32)
    gem_index = np.array([gem_lookup[int(item)] for item in gem], dtype=np.int16)
    gem_count = np.zeros((len(genes), len(gems)), dtype=np.int64)
    np.add.at(gem_count, (gene_index, gem_index), 1)
    return {
        "gene_ids": genes,
        "query_ids": query,
        "gem_group_ids": gems,
        "cell_count": np.bincount(gene_index, minlength=len(genes)).astype(np.int64),
        "gem_cell_count": gem_count,
        "source_row_index": np.concatenate(source_rows),
        "cell_ids": np.concatenate(cells),
        "gene_index": gene_index,
        "gem_index": gem_index,
    }


def _metadata_from_rpe1(manifest: dict) -> dict[str, np.ndarray]:
    routing_path = ROOT / manifest["routing"]["path"]
    if sha256(routing_path) != manifest["routing"]["sha256"]:
        raise ValueError("RPE1 routing hash mismatch")
    with np.load(routing_path, allow_pickle=False) as route:
        role = route["intervention_role"].astype(str)
        recon = route["reconstruction_role"].astype(str)
        is_control = route["is_control"].astype(bool)
        unresolved = route["unresolved_action"].astype(bool)
        select = (role == "validation") & (recon == "none") & (~is_control) & (~unresolved)
        if np.any(select & ((role == "test-excluded") | unresolved)):
            raise ValueError("RPE1 allowlist overlaps excluded rows")
        action = route["action_ids"][select].astype(str)
        gem = route["gem_group"][select].astype(np.int16)
        source_row = route["source_row_index"][select].astype(np.int64)
        cell = route["cell_ids"][select].astype(str)
        query_from_route = route["query_ids"].astype(str)
    with np.load(R_CONTROL, allow_pickle=False) as control:
        query = control["query_ids"].astype(str)
        gems = control["gem_group"].astype(np.int16)
    if not np.array_equal(query, query_from_route):
        raise ValueError("RPE1 routing/control query order mismatch")
    genes = np.array(sorted(set(action.tolist())))
    gene_lookup = {item: i for i, item in enumerate(genes.tolist())}
    gem_lookup = {int(item): i for i, item in enumerate(gems)}
    gene_index = np.array([gene_lookup[item] for item in action], dtype=np.int32)
    gem_index = np.array([gem_lookup[int(item)] for item in gem], dtype=np.int16)
    gem_count = np.zeros((len(genes), len(gems)), dtype=np.int64)
    np.add.at(gem_count, (gene_index, gem_index), 1)
    return {
        "gene_ids": genes,
        "query_ids": query,
        "gem_group_ids": gems,
        "cell_count": np.bincount(gene_index, minlength=len(genes)).astype(np.int64),
        "gem_cell_count": gem_count,
        "source_row_index": source_row,
        "cell_ids": cell,
        "gene_index": gene_index,
        "gem_index": gem_index,
    }


def _check_expected(source: str, arrays: dict[str, np.ndarray]) -> None:
    expected = EXPECTED[source]
    actual = {
        "genes": len(arrays["gene_ids"]),
        "queries": len(arrays["query_ids"]),
        "cells": int(arrays["cell_count"].sum()),
        "gems": len(arrays["gem_group_ids"]),
    }
    for key, value in actual.items():
        if value != expected[key]:
            raise ValueError(f"{source} expected {key}={expected[key]}, observed {value}")
    if not np.array_equal(arrays["gem_cell_count"].sum(1), arrays["cell_count"]):
        raise ValueError(f"{source} GEM counts do not close")
    if list(arrays["gene_ids"].astype(str)) != sorted(arrays["gene_ids"].astype(str).tolist()):
        raise ValueError(f"{source} gene roster is not ascending")


def prepare(output: Path) -> dict:
    k_manifest = _json(K_MANIFEST)
    r_manifest = _json(R_MANIFEST)
    pins = {
        "k562RawManifest": sha256(K_MANIFEST),
        "rpe1RawManifest": sha256(R_MANIFEST),
        "k562ControlReference": sha256(K_CONTROL),
        "rpe1ControlReference": sha256(R_CONTROL),
        "k562Routing": sha256(ROOT / k_manifest["routing"]["path"]),
        "rpe1Routing": sha256(ROOT / r_manifest["routing"]["path"]),
        "evaluationModule": sha256(MODULE_PATH),
        "evaluationRunner": sha256(Path(__file__).resolve()),
    }
    if pins["k562Routing"] != k_manifest["routing"]["sha256"] or pins["rpe1Routing"] != r_manifest["routing"]["sha256"]:
        raise ValueError("routing pin differs from raw manifest")
    k = _metadata_from_k562(k_manifest)
    r = _metadata_from_rpe1(r_manifest)
    _check_expected("k562", k)
    _check_expected("rpe1", r)
    metadata_arrays: dict[str, np.ndarray] = {
        "schema": np.array("slp.human-essential-count-development-evaluation-metadata/v1"),
    }
    for source, arrays in (("k562", k), ("rpe1", r)):
        for key, value in arrays.items():
            metadata_arrays[f"{source}_{key}"] = value
    metadata_path = output / "metadata-contract.npz"
    _write_new(metadata_path, deterministic_npz(metadata_arrays))
    protocol = {
        "schema": "slp.human-essential-count-shared-context-development-evaluation-protocol/v1",
        "status": "metadata-only-prepared",
        "hypothesis": "Alternating shared-context count training improves intervention-gene transfer in each source while retaining the K562-only source behavior.",
        "fixedAdvancementRule": {
            "k562": "joint gene-macro raw MSE <=0.99*K562-only, <=0.99*anchored-mean, <=0.99*static-ridge; centered residual r>=0.10 and >=static-ridge",
            "rpe1": "joint gene-macro raw MSE <=0.99*anchored-mean and <=0.99*static-ridge; centered residual r>=0.10 and >=static-ridge",
            "allChecksRequired": True,
        },
        "endpoint": "equal-cell mean per-cell CP10k over every native source query, transformed once as ln1p(mean CP10k)",
        "correlation": "subtract the same GEM-composition-matched control anchor from truth and prediction; translate by the first ascending gene; center each query independently across genes; mean defined per-gene query Pearson",
        "forecastContract": {
            "freezeFile": FREEZE_NAME,
            "filePattern": "development-forecasts-{k562,rpe1}.npz",
            "requiredKeys": ["schema", "source_id", "context_id", "gene_ids", "query_ids", "gem_group_ids", "cell_count", "gem_cell_count", *EVAL.PREDICTION_KEYS],
            "predictionShape": "[genes, native-source queries]",
            "values": "finite ln1p(mean CP10k) molecular forecasts; no uncertainty claim",
        },
        "sources": {
            "k562": {
                **EXPECTED["k562"],
                "rawManifest": str(K_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
                "rawManifestSha256": pins["k562RawManifest"],
                "controlReference": str(K_CONTROL.relative_to(ROOT)).replace("\\", "/"),
                "controlReferenceSha256": pins["k562ControlReference"],
                "developmentAccess": "canonical CSR shards whose manifest role is development-validation",
            },
            "rpe1": {
                **EXPECTED["rpe1"],
                "rawManifest": str(R_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
                "rawManifestSha256": pins["rpe1RawManifest"],
                "controlReference": str(R_CONTROL.relative_to(ROOT)).replace("\\", "/"),
                "controlReferenceSha256": pins["rpe1ControlReference"],
                "developmentAccess": "only routing rows with intervention_role=validation, reconstruction_role=none, is_control=false, unresolved_action=false; contiguous float32 H5 matrix spans copied in chunks<=2048 and memmaps closed immediately",
            },
        },
        "pins": pins,
        "metadataContract": {
            "path": metadata_path.name,
            "sha256": sha256(metadata_path),
        },
        "exclusions": {
            "testRowsOpened": 0,
            "unresolvedRpe1RowsOpened": 0,
            "reconstructionHeldRowsOpened": 0,
            "developmentCountValuesAccessedDuringPreparation": False,
        },
        "reconstructionPreservation": "K562 reconstruction evidence is reported separately from this development evaluator.",
    }
    protocol_path = output / "protocol.json"
    _write_new(protocol_path, canonical_json(protocol))
    receipt = {
        "schema": "slp.human-essential-count-development-evaluation-metadata-receipt/v1",
        "status": "complete",
        "protocol": {"path": protocol_path.name, "sha256": sha256(protocol_path)},
        "metadata": {"path": metadata_path.name, "sha256": sha256(metadata_path)},
        "developmentCountValuesAccessed": False,
        "testRowsOpened": 0,
    }
    _write_new(output / "PREPARED-METADATA-ONLY.json", canonical_json(receipt))
    return receipt


def _resolve_pin(artifact_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        local = (artifact_dir / candidate).resolve()
        repository = (ROOT / candidate).resolve()
        resolved = local if local.exists() else repository
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"pinned path escapes repository: {raw_path}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"pinned file is absent: {resolved}")
    return resolved


def validate_freeze(artifact_dir: Path, protocol: dict) -> tuple[dict, dict[str, Path]]:
    freeze_path = artifact_dir / FREEZE_NAME
    if not freeze_path.is_file():
        raise FileNotFoundError(f"authoritative forecast freeze absent: {freeze_path}")
    freeze = _json(freeze_path)
    if freeze.get("forecastsFrozenBeforeDevelopmentCountAccess") is not True:
        raise ValueError("forecast freeze does not attest pre-development freezing")
    if freeze.get("developmentCountMembersOpened") is not False or freeze.get("testOpened") is not False:
        raise ValueError("forecast freeze reports development or test access")
    pinned: dict[str, Path] = {"freeze": freeze_path}
    for section in ("forecasts", "models", "references", "baselines", "routingMetadata"):
        entries = freeze.get(section)
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"freeze is missing nonempty {section}")
        for name, item in entries.items():
            if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
                raise ValueError(f"malformed freeze pin {section}.{name}")
            path = _resolve_pin(artifact_dir, item["path"])
            actual = sha256(path)
            if actual != item["sha256"]:
                raise ValueError(f"hash mismatch before development access: {section}.{name}")
            pinned[f"{section}.{name}"] = path
    for source in ("k562", "rpe1"):
        entry = freeze["forecasts"].get(source)
        if entry is None:
            raise ValueError(f"missing {source} forecast pin")
        expected = EXPECTED[source]
        for field in ("sourceId", "contextId", "genes", "queries"):
            if entry.get(field) != expected[field]:
                raise ValueError(f"{source} forecast freeze {field} mismatch")
        represented = entry.get("cellsRepresentedByMetadata", entry.get("cells"))
        if represented != expected["cells"]:
            raise ValueError(f"{source} forecast freeze cell count mismatch")
    for key, expected_hash in protocol["pins"].items():
        if key in ("evaluationModule", "evaluationRunner"):
            actual = sha256(MODULE_PATH if key == "evaluationModule" else Path(__file__).resolve())
            if actual != expected_hash:
                raise ValueError(f"prepared evaluator source changed: {key}")
    return freeze, pinned


def _load_metadata(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        if str(data["schema"].item()) != "slp.human-essential-count-development-evaluation-metadata/v1":
            raise ValueError("metadata contract schema mismatch")
        result = {}
        for source in ("k562", "rpe1"):
            result[source] = {
                key: data[f"{source}_{key}"]
                for key in ("gene_ids", "query_ids", "gem_group_ids", "cell_count", "gem_cell_count", "source_row_index", "cell_ids", "gene_index", "gem_index")
            }
    return result


def _load_forecast(path: Path, source: str, metadata: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    EVAL.validate_forecast_arrays(arrays)
    if str(arrays["source_id"].item()) != EXPECTED[source]["sourceId"]:
        raise ValueError(f"{source} source identity mismatch")
    if str(arrays["context_id"].item()) != EXPECTED[source]["contextId"]:
        raise ValueError(f"{source} context identity mismatch")
    for key in ("gene_ids", "query_ids", "gem_group_ids", "cell_count", "gem_cell_count"):
        if not np.array_equal(arrays[key], metadata[key]):
            raise ValueError(f"{source} forecast {key} differs from prepared metadata")
    control_path = K_CONTROL if source == "k562" else R_CONTROL
    with np.load(control_path, allow_pickle=False) as control:
        if not np.array_equal(control["query_ids"].astype(str), metadata["query_ids"].astype(str)):
            raise ValueError(f"{source} canonical control query mismatch")
        if not np.array_equal(control["gem_group"], metadata["gem_group_ids"]):
            raise ValueError(f"{source} canonical control GEM mismatch")
        expected_control = EVAL.control_prediction(control["basal_rate"], metadata["gem_cell_count"])
    np.testing.assert_allclose(arrays["control_prediction"], expected_control, atol=2e-7, rtol=2e-7)
    return arrays


def _validate_before_access(artifact_dir: Path, output: Path) -> tuple[dict, dict[str, dict[str, np.ndarray]], dict]:
    protocol_path = output / "protocol.json"
    receipt_path = output / "PREPARED-METADATA-ONLY.json"
    if not protocol_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("run prepare before score")
    protocol = _json(protocol_path)
    receipt = _json(receipt_path)
    metadata_path = output / protocol["metadataContract"]["path"]
    if sha256(metadata_path) != protocol["metadataContract"]["sha256"]:
        raise ValueError("prepared metadata contract hash mismatch")
    if receipt.get("developmentCountValuesAccessed") is not False:
        raise ValueError("metadata receipt does not preserve the no-count-access boundary")
    prepared_paths = {
        "k562RawManifest": K_MANIFEST,
        "rpe1RawManifest": R_MANIFEST,
        "k562ControlReference": K_CONTROL,
        "rpe1ControlReference": R_CONTROL,
        "k562Routing": ROOT / _json(K_MANIFEST)["routing"]["path"],
        "rpe1Routing": ROOT / _json(R_MANIFEST)["routing"]["path"],
    }
    for key, path in prepared_paths.items():
        if sha256(path) != protocol["pins"][key]:
            raise ValueError(f"prepared source dependency changed before development access: {key}")
    freeze, pinned = validate_freeze(artifact_dir, protocol)
    metadata = _load_metadata(metadata_path)
    forecasts = {}
    for source in ("k562", "rpe1"):
        forecasts[source] = _load_forecast(pinned[f"forecasts.{source}"], source, metadata[source])
    validation = {
        "schema": "slp.human-essential-count-development-forecasts-preaccess-validation/v1",
        "status": "complete",
        "forecastFreeze": {"path": str((artifact_dir / FREEZE_NAME).relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(artifact_dir / FREEZE_NAME)},
        "validatedPins": {key: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path)} for key, path in sorted(pinned.items())},
        "forecastContracts": {
            source: {"genes": len(metadata[source]["gene_ids"]), "queries": len(metadata[source]["query_ids"]), "cells": int(metadata[source]["cell_count"].sum()), "gems": len(metadata[source]["gem_group_ids"])}
            for source in ("k562", "rpe1")
        },
        "allForecastsAndDependenciesValidatedBeforeDevelopmentCountAccess": True,
        "developmentCountValuesAccessed": False,
        "testRowsOpened": 0,
    }
    validation_path = output / "FORECASTS-VALIDATED-BEFORE-DEVELOPMENT.json"
    _write_new(validation_path, canonical_json(validation))
    return metadata, forecasts, validation


def _aggregate_k562(metadata: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict]:
    manifest = _json(K_MANIFEST)
    sums = np.zeros((len(metadata["gene_ids"]), len(metadata["query_ids"])), dtype=np.float64)
    cells = np.zeros(len(metadata["gene_ids"]), dtype=np.int64)
    expected_offset = 0
    read_bytes = 0
    shards = [item for item in manifest["shards"] if item["role"] == "development-validation"]
    for item in shards:
        path = K_MANIFEST.parent / item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"K562 development shard hash mismatch: {item['path']}")
        read_bytes += path.stat().st_size
        with np.load(path, allow_pickle=False) as data:
            rows = int(data["raw_shape"][0])
            stop = expected_offset + rows
            if not np.array_equal(data["source_row_index"], metadata["source_row_index"][expected_offset:stop]):
                raise ValueError("K562 source row order drift after freeze")
            if not np.array_equal(data["cell_ids"].astype(str), metadata["cell_ids"][expected_offset:stop].astype(str)):
                raise ValueError("K562 cell identity drift after freeze")
            matrix = sparse.csr_matrix(
                (data["raw_data"], data["raw_indices"], data["raw_indptr"]),
                shape=tuple(data["raw_shape"].astype(int)),
            )
            library = data["library_size"].astype(np.int64)
            measured_library = np.asarray(matrix.sum(axis=1)).reshape(-1).astype(np.int64)
            if not np.array_equal(library, measured_library):
                raise ValueError("K562 raw library does not close over the 8563-query panel")
            raw = matrix.toarray()
            EVAL.accumulate_cp10k(sums, cells, raw, library, metadata["gene_index"][expected_offset:stop])
            expected_offset = stop
    if expected_offset != len(metadata["source_row_index"]) or not np.array_equal(cells, metadata["cell_count"]):
        raise ValueError("K562 aggregated cells differ from prepared metadata")
    return EVAL.aggregate_truth(sums, cells), cells, {"shardsOpened": len(shards), "bytesOpened": read_bytes, "rowsOpened": int(cells.sum())}


def _aggregate_rpe1(metadata: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict]:
    manifest = _json(R_MANIFEST)
    source = ROOT / manifest["source"]["path"]
    stat = source.stat()
    if stat.st_size != manifest["source"]["bytes"] or stat.st_mtime_ns != manifest["source"]["mtimeNsAtFreeze"]:
        raise ValueError("RPE1 source size/mtime differs from frozen manifest")
    matrix_offset = int(manifest["source"]["matrixOffset"])
    query_count = len(metadata["query_ids"])
    rows = metadata["source_row_index"].astype(np.int64)
    if not np.all(rows[1:] > rows[:-1]):
        raise ValueError("RPE1 selected source rows must be strictly ascending")
    sums = np.zeros((len(metadata["gene_ids"]), query_count), dtype=np.float64)
    cells = np.zeros(len(metadata["gene_ids"]), dtype=np.int64)
    spans = 0
    opened_bytes = 0
    for start in range(0, len(rows), 2048):
        stop = min(start + 2048, len(rows))
        selected = rows[start:stop]
        first = int(selected[0])
        final = int(selected[-1]) + 1
        span_rows = final - first
        mapping = np.memmap(
            source,
            mode="r",
            dtype=np.float32,
            offset=matrix_offset + first * query_count * 4,
            shape=(span_rows, query_count),
            order="C",
        )
        raw_float = np.asarray(mapping[selected - first, :]).copy()
        mapping._mmap.close()
        del mapping
        spans += 1
        opened_bytes += span_rows * query_count * 4
        if not np.isfinite(raw_float).all() or np.any(raw_float < 0) or not np.array_equal(raw_float, np.rint(raw_float)):
            raise ValueError("RPE1 selected matrix rows are not finite nonnegative integer counts")
        raw = raw_float.astype(np.int32)
        library = raw.sum(axis=1, dtype=np.int64)
        if np.any(library <= 0):
            raise ValueError("RPE1 selected development cell has zero full-panel library")
        EVAL.accumulate_cp10k(sums, cells, raw, library, metadata["gene_index"][start:stop])
    if not np.array_equal(cells, metadata["cell_count"]):
        raise ValueError("RPE1 aggregated cells differ from prepared metadata")
    return EVAL.aggregate_truth(sums, cells), cells, {
        "boundedSpansOpened": spans,
        "logicalMatrixBytesCopied": int(len(rows) * query_count * 4),
        "physicalSpanBytesMapped": int(opened_bytes),
        "rowsOpened": int(cells.sum()),
        "minimumSourceRow": int(rows.min()),
        "maximumSourceRow": int(rows.max()),
        "testRowsOpened": 0,
        "unresolvedRowsOpened": 0,
    }


def score(artifact_dir: Path, output: Path) -> dict:
    metadata, forecasts, preaccess = _validate_before_access(artifact_dir, output)
    truth: dict[str, np.ndarray] = {}
    access: dict[str, dict] = {}
    truth["k562"], _, access["k562"] = _aggregate_k562(metadata["k562"])
    truth["rpe1"], _, access["rpe1"] = _aggregate_rpe1(metadata["rpe1"])
    truth_files = {}
    for source in ("k562", "rpe1"):
        arrays = metadata[source]
        path = output / f"development-truth-{source}.npz"
        _write_new(path, deterministic_npz({
            "schema": np.array("slp.human-essential-count-development-truth/v1"),
            "source_id": np.array(EXPECTED[source]["sourceId"]),
            "context_id": np.array(EXPECTED[source]["contextId"]),
            "gene_ids": arrays["gene_ids"],
            "query_ids": arrays["query_ids"],
            "cell_count": arrays["cell_count"],
            "gem_group_ids": arrays["gem_group_ids"],
            "gem_cell_count": arrays["gem_cell_count"],
            "truth_log1p_mean_cp10k": truth[source],
        }))
        truth_files[source] = {"path": path.name, "sha256": sha256(path)}
    metrics: dict[str, dict] = {}
    per_gene_arrays: dict[str, np.ndarray] = {"schema": np.array("slp.human-essential-count-development-per-gene-scores/v1")}
    for source in ("k562", "rpe1"):
        metrics[source] = {}
        per_gene_arrays[f"{source}_gene_ids"] = metadata[source]["gene_ids"]
        for key in EVAL.PREDICTION_KEYS:
            item, mse, correlation = EVAL.score_prediction(
                truth[source], forecasts[source][key], forecasts[source]["control_prediction"]
            )
            metrics[source][key] = item
            per_gene_arrays[f"{source}_{key}_mse"] = mse
            per_gene_arrays[f"{source}_{key}_independently_query_centered_residual_pearson"] = correlation
    per_gene_path = output / "per-gene-scores.npz"
    _write_new(per_gene_path, deterministic_npz(per_gene_arrays))
    report = {
        "schema": "slp.human-essential-count-shared-context-development-evaluation/v1",
        "status": "complete",
        "interpretation": "development evaluation of forecasts frozen before count access; K562 and RPE1 native query panels remain distinct",
        "forecastFreeze": preaccess["forecastFreeze"],
        "preaccessValidation": {"path": "FORECASTS-VALIDATED-BEFORE-DEVELOPMENT.json", "sha256": sha256(output / "FORECASTS-VALIDATED-BEFORE-DEVELOPMENT.json")},
        "truth": truth_files,
        "perGeneScores": {"path": per_gene_path.name, "sha256": sha256(per_gene_path)},
        "metrics": metrics,
        "advancement": EVAL.advancement(metrics),
        "developmentAccess": access,
        "testRowsOpened": 0,
        "unresolvedRpe1RowsOpened": 0,
        "reconstructionPreservation": "not evaluated here; K562 reconstruction diagnostics remain a separate training-side artifact",
        "limitations": [
            "This is adaptive molecular development evidence, not independent confirmation.",
            "RPE1 development perturbation counts are used only after forecast and dependency hashes are frozen and validated.",
            "The correlation removes the matched control anchor and then independently centers the held-gene cohort; it does not measure absolute control-referenced agreement.",
        ],
        "runtimeContract": {"maximumCpuThreads": 2, "maximumSeconds": 600},
    }
    report_path = output / "report.json"
    _write_new(report_path, canonical_json(report))
    complete = {
        "schema": "slp.human-essential-count-development-evaluation-receipt/v1",
        "status": "complete",
        "report": {"path": report_path.name, "sha256": sha256(report_path)},
        "testRowsOpened": 0,
    }
    _write_new(output / "COMPLETE.json", canonical_json(complete))
    return complete


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run = sub.add_parser("score")
    run.add_argument("--artifact-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output.resolve())
    else:
        result = score(args.artifact_dir.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
