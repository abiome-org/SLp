"""Build leakage-bounded raw-count CSR shards for Replogle K562 essential cells."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/sources/replogle-2022-k562-essential-singlecell-v1/K562_essential_raw_singlecell_01.h5ad"
RECEIPT = SOURCE.parent / "complete.json"
ROUTING = ROOT / "data/derived/slp11-human-k562-essential-singlecell-metadata-v1/cell-routing-metadata.npz"
OUTPUT = ROOT / "data/derived/slp11-human-k562-essential-raw-cells-v2"
SOURCE_SHA256 = "3e5a63a9e892b21029bb55fca4e12517a49aad7af6c14133ca63d12cf68c6cee"
ROUTING_SHA256 = "47c89c5082c0a9d4008c6b567407c530933a36fb7603621c37cbe913143f15ad"
SOURCE_BYTES = 10_661_879_995
SOURCE_SHAPE = (310_385, 8_563)
SCHEMA = "slp.replogle-k562-essential-raw-cell-shards/v2"
ROLE_NAMES = ("fit", "control", "reconstruction-held", "development-validation")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _strings(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=str)
    width = max(1, *(len(x) for x in values))
    return values.astype(f"<U{width}")


def load_routing(path: Path = ROUTING) -> dict[str, np.ndarray]:
    required = {
        "schema", "source_sha256", "source_row_index", "cell_ids", "context_id",
        "entity_taxon", "action_ids", "intervention_role", "reconstruction_role",
        "is_control", "gene_symbols", "gene_transcript", "transcript_labels",
        "guide_pair_ids", "gem_group", "umi_count", "core_adjusted_umi_count",
        "core_scale_factor", "z_gemgroup_umi", "mitochondrial_fraction",
        "query_ids", "query_taxon", "query_names", "query_in_matrix",
        "matrix_value_space", "library_size_definition",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("routing sidecar schema drift")
        result = {name: np.asarray(archive[name]) for name in archive.files}
    if str(result["source_sha256"]) != SOURCE_SHA256:
        raise ValueError("routing source hash drift")
    if not np.array_equal(result["source_row_index"], np.arange(SOURCE_SHAPE[0], dtype=np.int64)):
        raise ValueError("routing source row axis drift")
    if len(np.unique(result["query_ids"].astype(str))) != SOURCE_SHAPE[1] or not np.all(result["query_in_matrix"]):
        raise ValueError("query roster is not unique and fully measured")
    return result


def role_masks(routing: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    intervention = routing["intervention_role"].astype(str)
    reconstruction = routing["reconstruction_role"].astype(str)
    control = np.asarray(routing["is_control"], dtype=bool)
    masks = {
        "fit": (intervention == "train") & (reconstruction == "train") & ~control,
        "control": (intervention == "control") & (reconstruction == "train") & control,
        "reconstruction-held": np.isin(intervention, ["train", "control"]) & (reconstruction == "validation"),
        "development-validation": (intervention == "validation") & (reconstruction == "none") & ~control,
    }
    excluded = intervention == "test-excluded"
    covered = excluded.copy()
    for mask in masks.values():
        if np.any(covered & mask):
            raise ValueError("routing roles overlap")
        covered |= mask
    if not np.all(covered):
        raise ValueError("routing role is neither allowed nor test-excluded")
    if any(np.any(mask & excluded) for mask in masks.values()):
        raise AssertionError("test-excluded row entered an allowlist")
    return masks


def validate_raw_block(values: np.ndarray, expected_umi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    if values.ndim != 2 or values.dtype != np.float32:
        raise ValueError("raw block must be two-dimensional float32")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("raw counts must be finite and nonnegative")
    rounded = np.rint(values)
    if not np.array_equal(values, rounded) or float(rounded.max(initial=0)) > np.iinfo(np.int32).max:
        raise ValueError("raw values are not exact int32 counts")
    counts = rounded.astype(np.int32)
    library = counts.astype(np.int64).sum(axis=1)
    expected = np.asarray(expected_umi, dtype=np.float64)
    if expected.shape != library.shape or not np.all(np.isfinite(expected)):
        raise ValueError("invalid obs UMI_count")
    return counts, library


def umi_comparison(library: np.ndarray, expected_umi: np.ndarray) -> dict[str, float | int]:
    library = np.asarray(library, dtype=np.int64)
    expected = np.asarray(expected_umi, dtype=np.float64)
    if library.shape != expected.shape or not np.all(np.isfinite(expected)) or np.any(expected < 0):
        raise ValueError("invalid UMI comparison vectors")
    difference = expected - library
    positive = expected > 0
    ratio = np.divide(library, expected, out=np.full(expected.shape, np.nan), where=positive)
    return {
        "rows": len(library), "exactRows": int(np.sum(difference == 0)),
        "obsGreaterRows": int(np.sum(difference > 0)), "obsLowerRows": int(np.sum(difference < 0)),
        "minimumObsMinusRetained": float(difference.min(initial=0)),
        "maximumObsMinusRetained": float(difference.max(initial=0)),
        "sumObsMinusRetained": float(difference.sum(dtype=np.float64)),
        "sumAbsoluteDifference": float(np.abs(difference).sum(dtype=np.float64)),
        "minimumRetainedFractionOfObs": float(np.nanmin(ratio)) if np.any(positive) else float("nan"),
        "maximumRetainedFractionOfObs": float(np.nanmax(ratio)) if np.any(positive) else float("nan"),
    }


def merge_umi_comparison(total: dict[str, float | int], part: dict[str, float | int]) -> None:
    for key in ("rows", "exactRows", "obsGreaterRows", "obsLowerRows", "sumObsMinusRetained", "sumAbsoluteDifference"):
        total[key] += part[key]
    total["minimumObsMinusRetained"] = min(total["minimumObsMinusRetained"], part["minimumObsMinusRetained"])
    total["maximumObsMinusRetained"] = max(total["maximumObsMinusRetained"], part["maximumObsMinusRetained"])
    total["minimumRetainedFractionOfObs"] = min(total["minimumRetainedFractionOfObs"], part["minimumRetainedFractionOfObs"])
    total["maximumRetainedFractionOfObs"] = max(total["maximumRetainedFractionOfObs"], part["maximumRetainedFractionOfObs"])


def cp10k_group_moments(
    counts: sparse.csr_matrix,
    library_size: np.ndarray,
    action_ids: np.ndarray,
    gem_group: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return additive, shard-local action x GEM CP10k sufficient statistics."""
    library = np.asarray(library_size, dtype=np.float64)
    if counts.shape[0] != len(library) or np.any(library <= 0):
        raise ValueError("moment rows require positive libraries")
    actions = np.asarray(action_ids, dtype=str)
    gems = np.asarray(gem_group, dtype=np.int64)
    pairs = np.asarray([f"{action}\t{gem}" for action, gem in zip(actions, gems, strict=True)])
    keys, inverse = np.unique(pairs, return_inverse=True)
    membership = sparse.csr_matrix(
        (np.ones(len(inverse), dtype=np.float64), (inverse, np.arange(len(inverse)))),
        shape=(len(keys), len(inverse)),
    )
    rates = counts.astype(np.float64).multiply((10_000.0 / library)[:, None]).tocsr()
    sums = (membership @ rates).tocsr()
    sums2 = (membership @ rates.power(2)).tocsr()
    sums.sort_indices()
    sums2.sort_indices()
    key_action = np.asarray([key.rsplit("\t", 1)[0] for key in keys])
    key_gem = np.asarray([int(key.rsplit("\t", 1)[1]) for key in keys], dtype=np.int16)
    num_cells = np.bincount(inverse, minlength=len(keys)).astype(np.int64)
    return {
        "action_ids": _strings(key_action), "gem_group": key_gem, "num_cells": num_cells,
        "sum_data": sums.data.astype(np.float64), "sum_indices": sums.indices.astype(np.int32),
        "sum_indptr": sums.indptr.astype(np.int64), "sum_shape": np.asarray(sums.shape, np.int64),
        "sum_squares_data": sums2.data.astype(np.float64),
        "sum_squares_indices": sums2.indices.astype(np.int32),
        "sum_squares_indptr": sums2.indptr.astype(np.int64),
        "sum_squares_shape": np.asarray(sums2.shape, np.int64),
    }


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    return info


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for name in sorted(arrays):
            array = np.asarray(arrays[name])
            if array.dtype.hasobject:
                raise ValueError("object arrays are forbidden")
            with archive.open(_zip_info(name), "w", force_zip64=True) as member:
                np.lib.format.write_array(member, array, allow_pickle=False)


def _script_sha() -> str:
    return digest(Path(__file__).resolve())


def _source_contract() -> dict[str, object]:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    if SOURCE.stat().st_size != SOURCE_BYTES or receipt.get("sha256") != SOURCE_SHA256:
        raise ValueError("source receipt or byte size drift")
    with h5py.File(SOURCE, "r") as handle:
        matrix = handle["X"]
        if matrix.shape != SOURCE_SHAPE or matrix.dtype != np.float32:
            raise ValueError("source matrix shape or dtype drift")
        if matrix.chunks is not None or matrix.compression is not None:
            raise ValueError("source matrix is no longer contiguous/uncompressed")
        offset = int(matrix.id.get_offset())
        storage = int(matrix.id.get_storage_size())
    if storage != int(np.prod(SOURCE_SHAPE)) * np.dtype(np.float32).itemsize:
        raise ValueError("source matrix storage size drift")
    return {
        "path": str(SOURCE.relative_to(ROOT)).replace("\\", "/"), "bytes": SOURCE_BYTES,
        "sha256": SOURCE_SHA256, "receipt": str(RECEIPT.relative_to(ROOT)).replace("\\", "/"),
        "receiptSha256": digest(RECEIPT), "matrixOffset": offset, "matrixStorageBytes": storage,
        "mtimeNsAtFreeze": SOURCE.stat().st_mtime_ns,
    }


def prepare(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.output_dir}")
    routing = load_routing(args.routing)
    masks = role_masks(routing)
    args.output_dir.mkdir(parents=True)
    query_ids = routing["query_ids"].astype(str)
    protocol = {
        "schema": SCHEMA, "status": "frozen-before-quantitative-X-access",
        "hypothesis": "the official raw K562 essential matrix can be losslessly routed into bounded role-separated integer CSR shards and additive CP10k moments without accessing test-excluded rows",
        "advancementRule": "all selected values are finite nonnegative exact int32 counts; test-excluded rows selected equals zero; every GEM group has positive reconstruction-training control support; projected build <=900 seconds and peak RSS <=6 GiB",
        "accessibleModalities": ["raw UMI counts for globally hashed train actions and verified non-targeting controls", "raw UMI counts for separate development-validation actions", "metadata for all source rows"],
        "forbidden": ["test-excluded action X rows", "HepG2", "Jurkat", "synthetic-lethality outcomes"],
        "source": _source_contract(),
        "routing": {"path": str(args.routing.relative_to(ROOT)).replace("\\", "/"), "sha256": ROUTING_SHA256},
        "queryRoster": {"count": len(query_ids), "orderedSha256": hashlib.sha256(("\n".join(query_ids) + "\n").encode()).hexdigest(), "taxonomy": 9606, "allSourceColumnsObserved": True},
        "roleCounts": {name: int(mask.sum()) for name, mask in masks.items()},
        "testExcludedRows": int(np.sum(routing["intervention_role"].astype(str) == "test-excluded")),
        "shard": {"maximumRows": args.shard_size, "format": "pickle-free NPZ containing raw int32 CSR", "directories": list(ROLE_NAMES)},
        "normalization": {"storedValues": "raw integer UMI counts", "libraryDefinition": "sum across the exact ordered 8563 source columns", "momentRate": "CP10k = 10000 * raw_count / full-8563-column library sum", "obsUmiCountUse": "diagnostic only; obs UMI_count may precede filtering to the retained 8563-gene matrix and never replaces the measured retained-panel denominator"},
        "moments": {"fit": "additive per-shard action_id x gem_group CP10k sums, squared sums, and cell counts; reconstruction-training fitting cells only", "control": "global GEM-group CP10k sums/squared sums plus pooled raw counts and total libraries; reconstruction-training verified non-targeting cells only"},
        "profileRule": {"rowsPerRole": min(args.profile_rows, args.shard_size), "maximumProjectedSeconds": args.max_seconds, "maximumPeakRssBytes": args.max_rss_bytes},
        "implementation": {"path": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"), "sha256": _script_sha()},
        "supersedes": {"path": "data/derived/slp11-human-k562-essential-raw-cells-v1/protocol.json", "reason": "the first allowlisted fit profile found obs UMI_count exceeds the retained-8563 raw row sum by as much as 973; equality was an overstrict metadata diagnostic, not a valid count gate", "firstProfileRowsRead": 2048, "testExcludedRowsRead": 0},
        "quantitativeXRowsReadAtFreeze": 0,
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "query-ids.txt").write_text("\n".join(query_ids) + "\n", encoding="ascii")
    return protocol


def _load_protocol(args: argparse.Namespace) -> dict[str, object]:
    path = args.output_dir / "protocol.json"
    if not path.exists():
        raise FileNotFoundError("freeze protocol before accessing X")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen-before-quantitative-X-access":
        raise ValueError("frozen protocol drift")
    current_sha = _script_sha()
    if protocol["implementation"]["sha256"] != current_sha:
        amendment_path = args.output_dir / "execution-amendment-v1.json"
        if not amendment_path.exists():
            raise ValueError("implementation drift without a frozen execution amendment")
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        if amendment["originalProtocolSha256"] != digest(path) or amendment["implementationSha256"] != current_sha:
            raise ValueError("execution amendment drift")
        if args.max_seconds != amendment["maximumSeconds"] or args.max_rss_bytes != amendment["maximumRssBytes"]:
            raise ValueError("runtime bounds differ from frozen execution amendment")
    if SOURCE.stat().st_size != SOURCE_BYTES or SOURCE.stat().st_mtime_ns != protocol["source"]["mtimeNsAtFreeze"]:
        raise ValueError("source changed after protocol freeze")
    if digest(args.routing) != ROUTING_SHA256:
        raise ValueError("routing sidecar hash drift")
    return protocol


def amend(args: argparse.Namespace) -> dict[str, object]:
    """Freeze a resumable execution amendment without reading any new source values."""
    protocol_path = args.output_dir / "protocol.json"
    profile_path = args.output_dir / "profile.json"
    staging = args.output_dir / ".building"
    amendment_path = args.output_dir / "execution-amendment-v1.json"
    if amendment_path.exists() or not protocol_path.exists() or not profile_path.exists() or not staging.exists():
        raise FileExistsError("execution amendment state is invalid or already frozen")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen-before-quantitative-X-access":
        raise ValueError("protocol is not frozen")
    files = []
    for path in sorted(staging.rglob("*.npz")):
        files.append({
            "path": str(path.relative_to(staging)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        })
    raw = [item for item in files if item["path"].startswith("fit/cells-")]
    moments = [item for item in files if item["path"].startswith("fit-moments/moments-")]
    raw_indices = [int(Path(item["path"]).stem.rsplit("-", 1)[1]) for item in raw]
    moment_indices = [int(Path(item["path"]).stem.rsplit("-", 1)[1]) for item in moments]
    if raw_indices != list(range(len(raw))) or moment_indices != raw_indices:
        raise ValueError("partial fitting shards are not a contiguous verified prefix")
    result = {
        "schema": f"{SCHEMA}.execution-amendment/v1", "status": "frozen-before-resumed-X-access",
        "reason": "the first process reached the 6 GiB RSS guard after 75 fitting shards because repeated sparse-moment compression retained allocator memory; completed shard pairs remain valid",
        "scientificProtocolChanged": False, "sourceRowsRepeated": False,
        "originalProtocolSha256": digest(protocol_path), "originalProfileSha256": digest(profile_path),
        "originalImplementationSha256": protocol["implementation"]["sha256"],
        "implementationSha256": _script_sha(), "maximumSeconds": args.max_seconds,
        "maximumRssBytes": args.max_rss_bytes, "verifiedPartialFiles": files,
        "verifiedFitShardPrefix": len(raw), "testExcludedRowsRead": 0,
        "resumeBehavior": "hash every partial file, reconstruct counters from compact metadata, skip the verified prefix, collect cyclic garbage after each new shard",
    }
    amendment_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _matrix_memmap(protocol: dict[str, object]) -> np.memmap:
    return np.memmap(SOURCE, mode="r", dtype=np.float32, offset=int(protocol["source"]["matrixOffset"]), shape=SOURCE_SHAPE, order="C")


def _rss() -> int:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return 0


def profile(args: argparse.Namespace) -> dict[str, object]:
    protocol = _load_protocol(args)
    if (args.output_dir / "profile.json").exists():
        raise FileExistsError("profile already exists")
    routing = load_routing(args.routing)
    masks = role_masks(routing)
    matrix = _matrix_memmap(protocol)
    started = time.perf_counter()
    measured = []
    values_read = 0
    peak = _rss()
    output_bytes = 0
    for role in ROLE_NAMES:
        rows = np.flatnonzero(masks[role])[: args.profile_rows]
        block_started = time.perf_counter()
        raw = np.asarray(matrix[rows, :])
        counts, library = validate_raw_block(raw, routing["umi_count"][rows])
        comparison = umi_comparison(library, routing["umi_count"][rows])
        csr = sparse.csr_matrix(counts)
        sample = args.output_dir / f".profile-{role}.npz"
        write_npz(sample, {"data": csr.data, "indices": csr.indices.astype(np.int32), "indptr": csr.indptr.astype(np.int64), "shape": np.asarray(csr.shape, np.int64)})
        output_bytes += sample.stat().st_size
        sample.unlink()
        elapsed = time.perf_counter() - block_started
        measured.append({"role": role, "rows": len(rows), "seconds": elapsed, "nnz": int(csr.nnz), "zeroLibraries": int(np.sum(library == 0)), "obsUmiComparison": comparison})
        values_read += len(rows)
        peak = max(peak, _rss())
    elapsed = time.perf_counter() - started
    total_rows = sum(protocol["roleCounts"].values())
    seconds_per_row = elapsed / max(values_read, 1)
    # Moment construction and hashing add a conservative 50% margin beyond sampled conversion/write.
    projected = seconds_per_row * total_rows * 1.5
    result = {
        "schema": f"{SCHEMA}.profile", "rowsRead": values_read, "testExcludedRowsRead": 0,
        "measured": measured, "elapsedSeconds": elapsed, "projectedBuildSecondsConservative": projected,
        "sampleOutputBytes": output_bytes, "projectedRawShardBytes": int(output_bytes / max(values_read, 1) * total_rows),
        "peakRssBytes": peak, "maximumSeconds": args.max_seconds, "maximumRssBytes": args.max_rss_bytes,
        "accepted": bool(projected <= args.max_seconds and peak <= args.max_rss_bytes),
    }
    (args.output_dir / "profile.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    if not result["accepted"]:
        raise RuntimeError("resource profile exceeds frozen bounds")
    return result


def _raw_arrays(routing: dict[str, np.ndarray], rows: np.ndarray, counts: np.ndarray, library: np.ndarray) -> dict[str, np.ndarray]:
    csr = sparse.csr_matrix(counts)
    csr.sort_indices()
    return {
        "schema": np.asarray(SCHEMA), "source_sha256": np.asarray(SOURCE_SHA256),
        "source_row_index": rows.astype(np.int64), "cell_ids": _strings(routing["cell_ids"][rows]),
        "entity_taxon": np.full(len(rows), 9606, np.int64), "action_ids": _strings(routing["action_ids"][rows]),
        "guide_pair_ids": _strings(routing["guide_pair_ids"][rows]),
        "population_ids": _strings(routing["gene_transcript"][rows]),
        "gem_group": routing["gem_group"][rows].astype(np.int16),
        "intervention_role": _strings(routing["intervention_role"][rows]),
        "reconstruction_role": _strings(routing["reconstruction_role"][rows]),
        "is_control": routing["is_control"][rows].astype(bool), "library_size": library.astype(np.int64),
        "raw_data": csr.data.astype(np.int32), "raw_indices": csr.indices.astype(np.int32),
        "raw_indptr": csr.indptr.astype(np.int64), "raw_shape": np.asarray(csr.shape, np.int64),
    }


def _update_control(
    counts: sparse.csr_matrix, library: np.ndarray, gems: np.ndarray,
    sum_cp10k: np.ndarray, sumsq_cp10k: np.ndarray, raw_sum: np.ndarray,
    library_sum: np.ndarray, num_cells: np.ndarray,
) -> None:
    if np.any(library <= 0):
        return
    for gem in np.unique(gems):
        use = gems == gem
        pos = int(gem) - 1
        block = counts[use]
        rate = block.astype(np.float64).multiply((10_000.0 / library[use])[:, None])
        sum_cp10k[pos] += np.asarray(rate.sum(axis=0)).ravel()
        sumsq_cp10k[pos] += np.asarray(rate.power(2).sum(axis=0)).ravel()
        raw_sum[pos] += np.asarray(block.astype(np.int64).sum(axis=0)).ravel()
        library_sum[pos] += int(library[use].sum(dtype=np.int64))
        num_cells[pos] += int(use.sum())


def _preload_existing(
    staging: Path,
    routing: dict[str, np.ndarray],
    umi_total: dict[str, float | int],
    zero_library: dict[str, int],
    count_sum: dict[str, int],
    control_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    shards: list[dict[str, object]] = []
    fit_moments: list[dict[str, object]] = []
    for role in ROLE_NAMES:
        for path in sorted((staging / role).glob("cells-*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                rows = np.asarray(archive["source_row_index"], dtype=np.int64)
                library = np.asarray(archive["library_size"], dtype=np.int64)
                indptr = np.asarray(archive["raw_indptr"], dtype=np.int64)
                shape = tuple(np.asarray(archive["raw_shape"], dtype=np.int64))
                if shape != (len(rows), SOURCE_SHAPE[1]):
                    raise ValueError("partial raw shard shape drift")
                merge_umi_comparison(umi_total, umi_comparison(library, routing["umi_count"][rows]))
                zero_library[role] += int(np.sum(library == 0))
                count_sum[role] += len(rows)
                if role == "control" and np.any(library > 0):
                    data = np.asarray(archive["raw_data"], dtype=np.int32)
                    indices = np.asarray(archive["raw_indices"], dtype=np.int32)
                    csr = sparse.csr_matrix((data, indices, indptr), shape=shape)
                    positive = library > 0
                    _update_control(csr[positive], library[positive], routing["gem_group"][rows][positive], *control_arrays)
            index = int(path.stem.rsplit("-", 1)[1])
            left = index * 2048
            shards.append({
                "role": role, "path": f"{role}/{path.name}", "rows": len(rows), "rowStart": left,
                "rowStop": left + len(rows), "sourceRowMinimum": int(rows[0]), "sourceRowMaximum": int(rows[-1]),
                "nnz": int(indptr[-1]), "bytes": path.stat().st_size, "sha256": digest(path),
            })
    for path in sorted((staging / "fit-moments").glob("moments-*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            groups = len(archive["action_ids"])
            cells = int(np.asarray(archive["num_cells"], np.int64).sum())
        fit_moments.append({"path": f"fit-moments/{path.name}", "groups": groups, "cells": cells, "bytes": path.stat().st_size, "sha256": digest(path), "additiveFragments": True})
    return shards, fit_moments


def build(args: argparse.Namespace) -> dict[str, object]:
    protocol = _load_protocol(args)
    profile_path = args.output_dir / "profile.json"
    if not profile_path.exists() or not json.loads(profile_path.read_text())["accepted"]:
        raise RuntimeError("accepted profile required before build")
    resume = (args.output_dir / ".building").exists()
    if (args.output_dir / "manifest.json").exists():
        raise FileExistsError("immutable build output already exists")
    if resume:
        amendment_path = args.output_dir / "execution-amendment-v1.json"
        if not amendment_path.exists():
            raise FileExistsError("staging exists without a frozen execution amendment")
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        for item in amendment["verifiedPartialFiles"]:
            path = args.output_dir / ".building" / item["path"]
            if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise ValueError("verified partial file drift")
    routing = load_routing(args.routing)
    masks = role_masks(routing)
    staging = args.output_dir / ".building"
    if not resume:
        staging.mkdir()
        for role in ROLE_NAMES:
            (staging / role).mkdir()
        (staging / "fit-moments").mkdir()
    matrix = _matrix_memmap(protocol)
    started = time.perf_counter()
    peak_rss = _rss()
    zero_library = {role: 0 for role in ROLE_NAMES}
    count_sum = {role: 0 for role in ROLE_NAMES}
    query_count = SOURCE_SHAPE[1]
    control_sum = np.zeros((48, query_count), dtype=np.float64)
    control_sumsq = np.zeros((48, query_count), dtype=np.float64)
    control_raw_sum = np.zeros((48, query_count), dtype=np.int64)
    control_library_sum = np.zeros(48, dtype=np.int64)
    control_num_cells = np.zeros(48, dtype=np.int64)
    umi_total: dict[str, float | int] = {
        "rows": 0, "exactRows": 0, "obsGreaterRows": 0, "obsLowerRows": 0,
        "minimumObsMinusRetained": float("inf"), "maximumObsMinusRetained": float("-inf"),
        "sumObsMinusRetained": 0.0, "sumAbsoluteDifference": 0.0,
        "minimumRetainedFractionOfObs": float("inf"), "maximumRetainedFractionOfObs": float("-inf"),
    }
    control_arrays = (control_sum, control_sumsq, control_raw_sum, control_library_sum, control_num_cells)
    if resume:
        shards, fit_moments = _preload_existing(staging, routing, umi_total, zero_library, count_sum, control_arrays)
    else:
        shards, fit_moments = [], []
    for role in ROLE_NAMES:
        allowed_rows = np.flatnonzero(masks[role])
        for shard_index, left in enumerate(range(0, len(allowed_rows), args.shard_size)):
            if time.perf_counter() - started > args.max_seconds:
                raise TimeoutError("build exceeded frozen 900 second bound")
            rows = allowed_rows[left : left + args.shard_size]
            shard_path = staging / role / f"cells-{shard_index:05d}.npz"
            moment_path = staging / "fit-moments" / f"moments-{shard_index:05d}.npz"
            if shard_path.exists() and (role != "fit" or moment_path.exists()):
                continue
            if shard_path.exists() and role == "fit" and not moment_path.exists():
                with np.load(shard_path, allow_pickle=False) as archive:
                    arrays = {name: np.asarray(archive[name]) for name in archive.files}
                counts = sparse.csr_matrix((arrays["raw_data"], arrays["raw_indices"], arrays["raw_indptr"]), shape=tuple(arrays["raw_shape"]))
                library = arrays["library_size"]
            else:
                raw = np.asarray(matrix[rows, :])
                counts_dense, library = validate_raw_block(raw, routing["umi_count"][rows])
                merge_umi_comparison(umi_total, umi_comparison(library, routing["umi_count"][rows]))
                positive = library > 0
                zero_library[role] += int((~positive).sum())
                count_sum[role] += len(rows)
                arrays = _raw_arrays(routing, rows, counts_dense, library)
                write_npz(shard_path, arrays)
                item = {"role": role, "path": f"{role}/{shard_path.name}", "rows": len(rows), "rowStart": left, "rowStop": left + len(rows), "sourceRowMinimum": int(rows[0]), "sourceRowMaximum": int(rows[-1]), "nnz": len(arrays["raw_data"]), "bytes": shard_path.stat().st_size, "sha256": digest(shard_path)}
                shards.append(item)
                counts = sparse.csr_matrix((arrays["raw_data"], arrays["raw_indices"], arrays["raw_indptr"]), shape=tuple(arrays["raw_shape"]))
            positive = library > 0
            if role == "fit" and np.any(positive):
                moments = cp10k_group_moments(counts[positive], library[positive], routing["action_ids"][rows][positive], routing["gem_group"][rows][positive])
                moments.update({"schema": np.asarray(f"{SCHEMA}.additive-fit-moments"), "query_count": np.asarray(query_count, np.int64), "source_cell_shard": np.asarray(item["path"])})
                write_npz(moment_path, moments)
                fit_moments.append({"path": f"fit-moments/{moment_path.name}", "groups": len(moments["action_ids"]), "cells": int(moments["num_cells"].sum()), "bytes": moment_path.stat().st_size, "sha256": digest(moment_path), "additiveFragments": True})
            elif role == "control" and np.any(positive):
                _update_control(counts[positive], library[positive], routing["gem_group"][rows][positive], *control_arrays)
            peak_rss = max(peak_rss, _rss())
            if peak_rss > args.max_rss_bytes:
                raise MemoryError("build exceeded frozen 6 GiB RSS bound")
            del arrays, counts, library, positive
            if "raw" in locals():
                del raw
            if "counts_dense" in locals():
                del counts_dense
            if "moments" in locals():
                del moments
            gc.collect()
    if any(count_sum[role] != int(masks[role].sum()) for role in ROLE_NAMES):
        raise AssertionError("written role count differs from allowlist")
    if np.any(control_num_cells <= 0):
        raise AssertionError("a GEM group lacks positive reconstruction-training control support")
    control_path = staging / "control-gem-moments.npz"
    write_npz(control_path, {
        "schema": np.asarray(f"{SCHEMA}.control-gem-moments"), "source_sha256": np.asarray(SOURCE_SHA256),
        "routing_sha256": np.asarray(ROUTING_SHA256), "query_ids": _strings(routing["query_ids"]),
        "query_taxon": np.full(query_count, 9606, np.int64), "gem_group": np.arange(1, 49, dtype=np.int16),
        "num_cells": control_num_cells, "sum_cp10k": control_sum, "sum_squares_cp10k": control_sumsq,
        "raw_count_sum": control_raw_sum, "library_count_sum": control_library_sum,
        "positive_basal_rate_formula": np.asarray("10000*(raw_count_sum+0.5)/(library_count_sum+0.5*8563)"),
    })
    control_item = {"path": control_path.name, "bytes": control_path.stat().st_size, "sha256": digest(control_path), "gemGroups": 48, "queries": query_count, "cells": int(control_num_cells.sum())}
    # Publish role directories only after every quantitative check succeeds.
    for role in ROLE_NAMES:
        (staging / role).replace(args.output_dir / role)
    (staging / "fit-moments").replace(args.output_dir / "fit-moments")
    control_path.replace(args.output_dir / control_path.name)
    staging.rmdir()
    elapsed = time.perf_counter() - started
    manifest = {
        "schema": SCHEMA, "status": "complete", "source": protocol["source"], "routing": protocol["routing"],
        "implementation": protocol["implementation"], "executionImplementationSha256": _script_sha(), "protocolSha256": digest(args.output_dir / "protocol.json"), "profileSha256": digest(profile_path),
        "identity": {"taxonomy": 9606, "contextId": str(routing["context_id"]), "queryCount": query_count, "queryRoster": "query-ids.txt", "queryRosterSha256": digest(args.output_dir / "query-ids.txt")},
        "counts": {"roleRows": count_sum, "testExcludedRowsRead": 0, "zeroLibraryRows": zero_library, "rawShards": len(shards), "fitMomentFragments": len(fit_moments)},
        "normalization": protocol["normalization"], "obsUmiCountComparison": umi_total, "shards": shards, "fitMomentFragments": fit_moments, "controlMoments": control_item,
        "runtime": {"resumedExecution": resume, "seconds": elapsed, "peakRssBytes": peak_rss, "maximumSeconds": args.max_seconds, "maximumRssBytes": args.max_rss_bytes},
        "limitations": ["development-validation raw cells are isolated from fitting and are not used in sufficient statistics", "reconstruction-held rows remain within fitting/control genes but are excluded from every fitted moment", "fit moment files are additive shard fragments keyed by exact action_id and GEM group; repeated keys must be summed before use", "raw cells are not before/after pairs and do not identify causal single-cell trajectories"],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "profile", "amend", "build"))
    parser.add_argument("--routing", type=Path, default=ROUTING)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--profile-rows", type=int, default=2048)
    parser.add_argument("--max-seconds", type=float, default=900.0)
    parser.add_argument("--max-rss-bytes", type=int, default=6 * 1024**3)
    args = parser.parse_args()
    args.routing = args.routing.resolve()
    args.output_dir = args.output_dir.resolve()
    if not 0 < args.shard_size <= 2048 or not 0 < args.profile_rows <= 2048:
        parser.error("shard-size and profile-rows must be in [1,2048]")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    result = {"prepare": prepare, "profile": profile, "amend": amend, "build": build}[arguments.mode](arguments)
    print(json.dumps({"mode": arguments.mode, "status": result.get("status", result.get("accepted")), "counts": result.get("counts", result.get("roleCounts")), "runtime": result.get("runtime", result.get("elapsedSeconds"))}, indent=2))
