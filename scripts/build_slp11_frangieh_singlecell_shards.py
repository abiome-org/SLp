"""Build leakage-bounded paired single-cell RNA/ADT fitting shards.

The source matrices are CSC.  Row eligibility is loaded from the previously
frozen metadata-only access artifact and reduced to train/control before this
script requests any matrix values.  RNA columns are scanned once and admitted
values are dispatched to bounded row shards as temporary column-major records.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/sources/frangieh-2021-scp1064-v1"
DEFAULT_PARENT = ROOT / "data/derived/slp11-frangieh/paired-development-v1"
DEFAULT_OUTPUT = ROOT / "data/derived/slp11-frangieh/paired-singlecell-train-control-v1"
SCHEMA = "slp.frangieh-paired-singlecell-shards/v1"
RECON_PREFIX = "slp11-cell-state-v1|731|"
ALLOWED_SOURCE_SPLITS = frozenset({"train", "control"})
RECORD_DTYPE = np.dtype([("row", "<u2"), ("column", "<u2"), ("value", "<f4")])


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def decode(values: object) -> np.ndarray:
    out = np.asarray(values)
    if out.dtype.kind == "S":
        return np.char.decode(out, "utf-8")
    if out.dtype.kind == "O":
        return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in out])
    return out


def column(group: h5py.Group, key: str) -> np.ndarray:
    node = group[key]
    if isinstance(node, h5py.Group) and {"codes", "categories"} <= set(node):
        categories = decode(node["categories"])
        codes = np.asarray(node["codes"])
        out = np.empty(codes.shape, dtype=object)
        out[codes < 0] = ""
        out[codes >= 0] = categories[codes[codes >= 0]]
        return out
    return decode(node)


def load_access(path: Path) -> dict[str, np.ndarray]:
    required = {
        "source_row_index",
        "cell_ids",
        "action_ids",
        "split",
        "context_ids",
        "full_guide_ids",
        "target_guide_sets",
        "rna_denominator",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("unexpected paired-cell access schema")
        access = {name: np.asarray(archive[name]) for name in required}
    if len({len(value) for value in access.values()}) != 1:
        raise ValueError("misaligned paired-cell access arrays")
    return access


def select_fitting_access(access: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Freeze the train/control row allowlist before any source value access."""
    source_split = np.asarray(access["split"], dtype=str)
    if not np.all(np.isin(source_split, ["train", "validation", "control"])):
        raise ValueError("access artifact contains an unknown source split")
    keep = np.isin(source_split, sorted(ALLOWED_SOURCE_SPLITS))
    selected = {name: np.asarray(values)[keep] for name, values in access.items()}
    rows = np.asarray(selected["source_row_index"], dtype=np.int64)
    if len(np.unique(rows)) != len(rows) or np.any(np.diff(rows) <= 0):
        raise ValueError("selected source rows must be strictly increasing and unique")
    if not np.all(np.isin(selected["split"].astype(str), sorted(ALLOWED_SOURCE_SPLITS))):
        raise AssertionError("forbidden row survived fitting selection")
    if np.any(np.asarray(selected["rna_denominator"], dtype=np.float64) <= 0):
        raise ValueError("selected RNA denominator is nonpositive")
    return selected


def reconstruction_split(cell_ids: np.ndarray) -> np.ndarray:
    result = []
    for cell_id in np.asarray(cell_ids, dtype=str):
        key = f"{RECON_PREFIX}{cell_id}".encode("utf-8")
        bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 100
        result.append("train" if bucket < 90 else "validation")
    return np.asarray(result, dtype="U10")


def transform_rna_values(values: np.ndarray, denominators: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    denominators = np.asarray(denominators, dtype=np.float64)
    if values.shape != denominators.shape or np.any(values < 0) or np.any(denominators <= 0):
        raise ValueError("invalid RNA counts or denominators")
    return np.log1p(10_000.0 * values / denominators).astype(np.float32)


def matched_isotype_transform(target: np.ndarray, control: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    if target.shape != control.shape or np.any(target < 0) or np.any(control < 0):
        raise ValueError("invalid paired ADT counts")
    return np.maximum(0.0, np.log((target + 1.0) / (control + 1.0))).astype(np.float32)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o600 << 16
    return info


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Stream a deterministic, pickle-free, uncompressed NPZ."""
    with zipfile.ZipFile(path, "w", allowZip64=True) as archive:
        for name in sorted(arrays):
            array = np.asarray(arrays[name])
            if array.dtype.hasobject:
                raise ValueError("object arrays are forbidden")
            with archive.open(_zip_info(name), "w", force_zip64=True) as member:
                np.lib.format.write_array(member, array, allow_pickle=False)


def _selected_dense_csc(handle: h5py.File, selected_rows: np.ndarray) -> np.ndarray:
    matrix = handle["X"]
    shape = tuple(int(x) for x in matrix.attrs["shape"])
    row_map = np.full(shape[0], -1, dtype=np.int32)
    row_map[selected_rows] = np.arange(len(selected_rows), dtype=np.int32)
    result = np.zeros((len(selected_rows), shape[1]), dtype=np.float32)
    indptr = np.asarray(matrix["indptr"])
    for source_column in range(shape[1]):
        start, stop = int(indptr[source_column]), int(indptr[source_column + 1])
        destination = row_map[np.asarray(matrix["indices"][start:stop])]
        keep = destination >= 0
        if np.any(keep):
            positions = np.flatnonzero(keep).astype(np.int64) + start
            result[destination[keep], source_column] = np.asarray(matrix["data"][positions])
    return result


def _profile_scan(
    matrix: h5py.Group,
    source_columns: np.ndarray,
    row_map: np.ndarray,
    denominators: np.ndarray,
    count: int,
) -> dict:
    columns = np.asarray(source_columns[: min(count, len(source_columns))], dtype=np.int64)
    indptr = np.asarray(matrix["indptr"])
    started = time.perf_counter()
    scanned = kept = 0
    checksum = 0.0
    for source_column in columns:
        start, stop = int(indptr[source_column]), int(indptr[source_column + 1])
        destination = row_map[np.asarray(matrix["indices"][start:stop])]
        use = destination >= 0
        if np.any(use):
            positions = np.flatnonzero(use).astype(np.int64) + start
            values = np.asarray(matrix["data"][positions])
            checksum += float(transform_rna_values(values, denominators[destination[use]]).sum())
            kept += len(values)
        scanned += stop - start
    elapsed = time.perf_counter() - started
    return {
        "columns": int(len(columns)),
        "source_entries_scanned": int(scanned),
        "admitted_values_read": int(kept),
        "elapsed_seconds": elapsed,
        "projected_rna_scan_seconds": elapsed * len(source_columns) / max(1, len(columns)),
        "transformed_checksum": checksum,
    }


def _dispatch_rna(
    matrix: h5py.Group,
    source_columns: np.ndarray,
    row_map: np.ndarray,
    denominators: np.ndarray,
    shard_size: int,
    temp_paths: list[Path],
) -> tuple[int, float]:
    indptr = np.asarray(matrix["indptr"])
    handles = [path.open("wb") for path in temp_paths]
    total = 0
    started = time.perf_counter()
    try:
        for output_column, source_column in enumerate(np.asarray(source_columns, dtype=np.int64)):
            start, stop = int(indptr[source_column]), int(indptr[source_column + 1])
            destination = row_map[np.asarray(matrix["indices"][start:stop])]
            keep = destination >= 0
            if not np.any(keep):
                continue
            positions = np.flatnonzero(keep).astype(np.int64) + start
            selected = destination[keep]
            values = transform_rna_values(
                np.asarray(matrix["data"][positions]), denominators[selected]
            )
            shard_ids = selected // shard_size
            cuts = np.r_[0, np.flatnonzero(np.diff(shard_ids)) + 1, len(selected)]
            for left, right in zip(cuts[:-1], cuts[1:], strict=True):
                shard_id = int(shard_ids[left])
                records = np.empty(right - left, dtype=RECORD_DTYPE)
                records["row"] = (selected[left:right] - shard_id * shard_size).astype(np.uint16)
                records["column"] = np.uint16(output_column)
                records["value"] = values[left:right]
                handles[shard_id].write(records.tobytes())
            total += len(selected)
    finally:
        for handle in handles:
            handle.close()
    return total, time.perf_counter() - started


def _choose_reconstruction_groups(development: dict[str, np.ndarray]) -> list[dict]:
    train = np.flatnonzero(np.asarray(development["split_train"], dtype=bool))
    positions = sorted(set([int(train[0]), int(train[len(train) // 2]), int(train[-1])]))
    groups = [
        {
            "kind": "target",
            "index": index,
            "action_id": str(development["action_ids"][index]),
            "context_id": str(development["context_ids"][index]),
            "target_guide_set": str(development["source_target_guide_sets"][index]),
        }
        for index in positions
    ]
    groups.extend(
        {
            "kind": "control",
            "index": int(index),
            "action_id": "",
            "context_id": str(context),
            "target_guide_set": "__VERIFIED_NONTARGETING_CONTROL__",
        }
        for index, context in enumerate(np.asarray(development["control_context_ids"], dtype=str))
    )
    return groups


def _group_mask(access: dict[str, np.ndarray], group: dict) -> np.ndarray:
    context = np.asarray(access["context_ids"], dtype=str) == group["context_id"]
    if group["kind"] == "control":
        return context & (np.asarray(access["split"], dtype=str) == "control")
    return (
        context
        & (np.asarray(access["split"], dtype=str) == "train")
        & (np.asarray(access["action_ids"], dtype=str) == group["action_id"])
        & (np.asarray(access["target_guide_sets"], dtype=str) == group["target_guide_set"])
    )


def build(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"immutable output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    temp_dir = output_dir / ".building"
    temp_dir.mkdir()

    rna_path = args.source_dir / "FrangiehIzar2021_RNA.h5ad"
    protein_path = args.source_dir / "FrangiehIzar2021_protein.h5ad"
    access_path = args.parent_dir / "paired-cell-access.npz"
    query_roster_path = args.parent_dir / "rna-query-ensembl-ids.txt"
    channel_roster_path = args.parent_dir / "adt-channel-roster.json"
    development_path = args.parent_dir / "development.npz"
    inputs = [rna_path, protein_path, access_path, query_roster_path, channel_roster_path, development_path]
    input_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in inputs}

    access_all = load_access(access_path)
    access = select_fitting_access(access_all)
    n_cells = len(access["cell_ids"])
    n_shards = (n_cells + args.shard_size - 1) // args.shard_size
    recon = reconstruction_split(access["cell_ids"])
    protocol = {
        "schema": SCHEMA,
        "status": "frozen-before-source-value-access",
        "hypothesis": "paired train/control cells can support a bounded observed-state reconstruction pilot without exposing held-gene cells",
        "advancement_rule": "all source rows are train or verified control; exact RNA/ADT barcode alignment; selected aggregate endpoints reproduce the frozen development artifact within float32 roundoff",
        "source_hashes": input_hashes,
        "source_row_policy": "metadata-only paired-cell-access selection; retain split=train or control only before any X/data request",
        "excluded_access_rows": {
            str(label): int(np.sum(np.asarray(access_all["split"], dtype=str) == label))
            for label in sorted(set(np.asarray(access_all["split"], dtype=str)) - ALLOWED_SOURCE_SPLITS)
        },
        "rna": {
            "query_roster": "rna-query-ensembl-ids.txt",
            "formula": "ln(1 + 10000 * count / row_sum_all_23712_source_RNA_columns)",
            "storage": "CSR float32, deterministic uncompressed NPZ, shard rows <= 2048",
        },
        "protein": {
            "channel_roster": "adt-channel-roster.json",
            "formula": "max(0, ln((target_count+1)/(matched_isotype_count+1)))",
            "storage": "dense float32 [cells,20]",
        },
        "reconstruction_split": {
            "formula": "first8bigendian(SHA256('slp11-cell-state-v1|731|'+barcode)) % 100; <90 train else validation",
            "scope": "cell reconstruction within original fitting genes and controls; not held-gene evidence",
        },
        "shard_size": args.shard_size,
        "cell_count": n_cells,
        "shard_count": n_shards,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    query_ids_expected = np.asarray(query_roster_path.read_text(encoding="ascii").splitlines(), dtype=str)
    channel_roster = json.loads(channel_roster_path.read_text(encoding="utf-8"))
    temp_paths = [temp_dir / f"rna-{index:05d}.bin" for index in range(n_shards)]
    for path in temp_paths:
        path.touch()

    started = time.perf_counter()
    with h5py.File(rna_path, "r") as rna, h5py.File(protein_path, "r") as protein:
        selected_rows = np.asarray(access["source_row_index"], dtype=np.int64)
        rna_cells = column(rna["obs"], "cell_name").astype(str)
        protein_cells = column(protein["obs"], "cell_name").astype(str)
        if not np.array_equal(rna_cells, protein_cells):
            raise ValueError("source RNA and ADT cell axes differ")
        if not np.array_equal(rna_cells[selected_rows], np.asarray(access["cell_ids"], dtype=str)):
            raise ValueError("access barcodes do not match source rows")

        ensembl = column(rna["var"], "ensembl_id").astype(str)
        stable = np.asarray([x.startswith("ENSG") and len(x) == 15 and x[4:].isdigit() for x in ensembl])
        source_columns = np.flatnonzero(stable)
        query_ids = ensembl[source_columns]
        if not np.array_equal(query_ids, query_ids_expected):
            raise ValueError("RNA query roster differs from frozen parent roster")

        matrix_shape = tuple(int(x) for x in rna["X"].attrs["shape"])
        row_map = np.full(matrix_shape[0], -1, dtype=np.int32)
        row_map[selected_rows] = np.arange(n_cells, dtype=np.int32)
        denominators = np.asarray(access["rna_denominator"], dtype=np.float64)
        profile = _profile_scan(rna["X"], source_columns, row_map, denominators, args.profile_columns)
        if profile["projected_rna_scan_seconds"] > args.max_projected_seconds:
            protocol["status"] = "profile-only-projection-exceeded-bound"
            protocol["profile"] = profile
            (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            raise RuntimeError("profile projection exceeds fixed runtime bound")

        total_nnz, rna_seconds = _dispatch_rna(
            rna["X"], source_columns, row_map, denominators, args.shard_size, temp_paths
        )
        raw_adt = _selected_dense_csc(protein, selected_rows)
        protein_names = column(protein["var"], "protein").astype(str)
        channel_ids = column(protein["var"], "Barcode").astype(str)
        matched = column(protein["var"], "Isotype_control").astype(str)
        roster_channels = channel_roster["channels"]
        if channel_ids.tolist() != [item["channel_id"] for item in roster_channels]:
            raise ValueError("ADT barcode roster differs from frozen parent roster")
        molecular = np.flatnonzero(matched != "nan")
        name_to_column = {name: index for index, name in enumerate(protein_names)}
        protein_values = np.empty((n_cells, len(molecular)), dtype=np.float32)
        for output_column, target_column in enumerate(molecular):
            control_name = matched[target_column]
            protein_values[:, output_column] = matched_isotype_transform(
                raw_adt[:, target_column], raw_adt[:, name_to_column[control_name]]
            )

    with np.load(development_path, allow_pickle=False) as archive:
        development = {name: np.asarray(archive[name]) for name in archive.files}
    reconstruction_groups = _choose_reconstruction_groups(development)
    reconstruction_accumulators = [
        {
            "count": 0,
            "rna_sum": np.zeros(len(query_ids), dtype=np.float64),
            "protein_sum": np.zeros(protein_values.shape[1], dtype=np.float64),
        }
        for _ in reconstruction_groups
    ]

    shutil.copyfile(query_roster_path, output_dir / query_roster_path.name)
    shutil.copyfile(channel_roster_path, output_dir / channel_roster_path.name)
    shards = []
    for shard_index, temp_path in enumerate(temp_paths):
        left = shard_index * args.shard_size
        right = min(n_cells, left + args.shard_size)
        records = np.fromfile(temp_path, dtype=RECORD_DTYPE)
        matrix = sparse.coo_matrix(
            (records["value"], (records["row"].astype(np.int32), records["column"].astype(np.int32))),
            shape=(right - left, len(query_ids)),
            dtype=np.float32,
        ).tocsr()
        matrix.sort_indices()
        shard_access = {name: values[left:right] for name, values in access.items()}
        shard_arrays = {
            "rna_data": matrix.data.astype(np.float32, copy=False),
            "rna_indices": matrix.indices.astype(np.int32, copy=False),
            "rna_indptr": matrix.indptr.astype(np.int64),
            "rna_shape": np.asarray(matrix.shape, dtype=np.int64),
            "protein_values": protein_values[left:right].astype(np.float32, copy=False),
            "source_row_index": np.asarray(shard_access["source_row_index"], dtype=np.int64),
            "cell_ids": np.asarray(shard_access["cell_ids"], dtype=str),
            "action_ids": np.asarray(shard_access["action_ids"], dtype=str),
            "source_split": np.asarray(shard_access["split"], dtype=str),
            "context_ids": np.asarray(shard_access["context_ids"], dtype=str),
            "full_guide_ids": np.asarray(shard_access["full_guide_ids"], dtype=str),
            "target_guide_sets": np.asarray(shard_access["target_guide_sets"], dtype=str),
            "reconstruction_split": recon[left:right],
        }
        for group, accumulator in zip(reconstruction_groups, reconstruction_accumulators, strict=True):
            mask = _group_mask(shard_access, group)
            if np.any(mask):
                accumulator["count"] += int(mask.sum())
                accumulator["rna_sum"] += np.asarray(matrix[mask].astype(np.float64).sum(axis=0)).ravel()
                accumulator["protein_sum"] += protein_values[left:right][mask].astype(np.float64).sum(axis=0)
        shard_path = output_dir / f"paired-cells-{shard_index:05d}.npz"
        write_deterministic_npz(shard_path, shard_arrays)
        shards.append(
            {
                "path": shard_path.name,
                "sha256": digest(shard_path),
                "bytes": shard_path.stat().st_size,
                "rows": right - left,
                "row_start": left,
                "row_stop": right,
                "rna_nnz": int(matrix.nnz),
            }
        )
        temp_path.unlink()
    temp_dir.rmdir()

    checks = []
    for group, accumulator in zip(reconstruction_groups, reconstruction_accumulators, strict=True):
        if accumulator["count"] <= 0:
            raise AssertionError("selected reconstruction group has no shard cells")
        rna_mean = (accumulator["rna_sum"] / accumulator["count"]).astype(np.float32)
        protein_mean = (accumulator["protein_sum"] / accumulator["count"]).astype(np.float32)
        if group["kind"] == "target":
            expected_rna = development["rna_targets"][group["index"]]
            expected_protein = development["protein_targets"][group["index"]]
            expected_count = int(development["num_cells"][group["index"]])
        else:
            expected_rna = development["control_rna_targets"][group["index"]]
            expected_protein = development["control_protein_targets"][group["index"]]
            expected_count = int(development["control_num_cells"][group["index"]])
        rna_error = float(np.max(np.abs(rna_mean - expected_rna)))
        protein_error = float(np.max(np.abs(protein_mean - expected_protein)))
        checks.append({**group, "cells": accumulator["count"], "expected_cells": expected_count, "rna_max_abs_error": rna_error, "protein_max_abs_error": protein_error})
        if accumulator["count"] != expected_count or rna_error > 2e-6 or protein_error > 2e-6:
            raise AssertionError(f"aggregate reconstruction mismatch: {checks[-1]}")

    elapsed = time.perf_counter() - started
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "source_hashes": input_hashes,
        "identity": {
            "taxonomy": 9606,
            "cell_axis": "source RNA/ADT cell_name barcode, exact paired source row",
            "rna_query_axis": "source-order stable unversioned ENSG roster in rna-query-ensembl-ids.txt",
            "protein_axis": "20 molecular TotalSeq-A barcode channels in source order; no forced gene join",
        },
        "counts": {
            "cells": n_cells,
            "source_train_cells": int(np.sum(np.asarray(access["split"], dtype=str) == "train")),
            "verified_control_cells": int(np.sum(np.asarray(access["split"], dtype=str) == "control")),
            "excluded_validation_cells": int(np.sum(np.asarray(access_all["split"], dtype=str) == "validation")),
            "reconstruction_train_cells": int(np.sum(recon == "train")),
            "reconstruction_validation_cells": int(np.sum(recon == "validation")),
            "rna_queries": len(query_ids),
            "protein_molecular_channels": protein_values.shape[1],
            "rna_nnz": int(total_nnz),
            "shards": len(shards),
        },
        "profile": profile,
        "runtime": {"rna_dispatch_seconds": rna_seconds, "total_seconds": elapsed},
        "shards": shards,
        "aggregate_reconstruction_checks": checks,
        "rosters": {
            "rna-query-ensembl-ids.txt": digest(output_dir / "rna-query-ensembl-ids.txt"),
            "adt-channel-roster.json": digest(output_dir / "adt-channel-roster.json"),
        },
        "limitations": [
            "reconstruction validation partitions cells within fitting intervention genes and controls; it is not gene-transfer evidence",
            "cells are independent observed states and are not paired before/after observations",
            "no scaler, learned basis, or quantitative model is fit in this artifact",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--profile-columns", type=int, default=256)
    parser.add_argument("--max-projected-seconds", type=float, default=600.0)
    args = parser.parse_args()
    if not 0 < args.shard_size <= 2048:
        parser.error("shard size must be in [1,2048]")
    if args.profile_columns <= 0:
        parser.error("profile columns must be positive")
    return args


if __name__ == "__main__":
    result = build(parse_args())
    print(json.dumps({"status": result["status"], "counts": result["counts"], "runtime": result["runtime"]}, indent=2))
