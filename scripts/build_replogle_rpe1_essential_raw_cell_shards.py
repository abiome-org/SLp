"""Build bounded RPE1 raw-count shards, moments, and one training mmap."""

from __future__ import annotations

import argparse
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
SOURCE = ROOT / "data/sources/replogle-2022-rpe1-essential-singlecell-v1/rpe1_raw_singlecell_01.h5ad"
RECEIPT = SOURCE.parent / "complete.json"
ROUTING = ROOT / "data/derived/slp11-human-rpe1-essential-singlecell-metadata-v1/cell-routing-metadata.npz"
OUTPUT = ROOT / "data/derived/slp11-human-rpe1-essential-raw-cells-v1"
SOURCE_SHA256 = "9b05ef1f81526216fa008d677e9e0d03dce9a2f7a95499a4fb81e505e9d88ef1"
ROUTING_SHA256 = "10f3d313a5671122bde10a9bd586e3a2808d6f9b554f737ddcbbc28becc5e2f2"
SOURCE_BYTES = 8_700_873_216
SOURCE_SHAPE = (247_914, 8_749)
SCHEMA = "slp.replogle-rpe1-essential-raw-cell-shards/v1"
ROLE_NAMES = ("fit", "control", "reconstruction-held")
EXPECTED_ROWS = {"fit": 142_601, "control": 10_350, "reconstruction-held": 17_072}
TRAINING_ROWS = 152_951


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


def load_routing(path: Path = ROUTING) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        result = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "schema", "source_sha256", "source_row_index", "cell_ids", "context_id", "entity_taxon",
        "action_ids", "intervention_role", "reconstruction_role", "is_control", "unresolved_action",
        "gene_symbols", "gene_transcript", "transcript_labels", "guide_pair_ids", "gem_group", "umi_count",
        "core_adjusted_umi_count", "core_scale_factor", "z_gemgroup_umi", "mitochondrial_fraction",
        "query_ids", "query_taxon", "query_names", "query_in_matrix", "matrix_value_space", "library_size_definition",
    }
    if set(result) != required or str(result["source_sha256"]) != SOURCE_SHA256:
        raise ValueError("routing schema or source drift")
    if not np.array_equal(result["source_row_index"], np.arange(SOURCE_SHAPE[0], dtype=np.int64)):
        raise ValueError("routing row axis drift")
    if len(np.unique(result["query_ids"].astype(str))) != SOURCE_SHAPE[1] or not np.all(result["query_in_matrix"]):
        raise ValueError("query roster is not unique and fully observed")
    return result


def role_masks(routing: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    intervention = routing["intervention_role"].astype(str)
    reconstruction = routing["reconstruction_role"].astype(str)
    control = np.asarray(routing["is_control"], bool)
    unresolved = np.asarray(routing["unresolved_action"], bool)
    masks = {
        "fit": (intervention == "train") & (reconstruction == "train") & ~control,
        "control": (intervention == "control") & (reconstruction == "train") & control,
        "reconstruction-held": np.isin(intervention, ["train", "control"]) & (reconstruction == "validation"),
    }
    forbidden = np.isin(intervention, ["validation", "test-excluded", "unresolved-excluded"]) | unresolved
    covered = forbidden.copy()
    for name, mask in masks.items():
        if int(mask.sum()) != EXPECTED_ROWS[name] or np.any(mask & covered):
            raise ValueError(f"role mask drift or overlap: {name}")
        covered |= mask
    if not np.all(covered) or any(np.any(mask & forbidden) for mask in masks.values()):
        raise ValueError("a row is neither allowed nor explicitly forbidden")
    return masks


def read_rows_bounded(
    path: Path, matrix_offset: int, shape: tuple[int, int], rows: np.ndarray
) -> np.ndarray:
    """Copy exact sorted rows through a temporary row-span mmap, then close it."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0 or np.any(np.diff(rows) <= 0) or rows[0] < 0 or rows[-1] >= shape[0]:
        raise ValueError("requested rows must be nonempty, sorted, unique, and in bounds")
    row_bytes = shape[1] * np.dtype(np.float32).itemsize
    first, stop = int(rows[0]), int(rows[-1]) + 1
    mapping = np.memmap(
        path, mode="r", dtype=np.float32, offset=matrix_offset + first * row_bytes,
        shape=(stop - first, shape[1]), order="C",
    )
    result = np.asarray(mapping[rows - first, :]).copy()
    mapping._mmap.close()
    return result


def validate_raw_block(values: np.ndarray, expected_umi: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    values = np.asarray(values)
    if values.ndim != 2 or values.dtype != np.float32 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("raw block must be finite nonnegative float32")
    rounded = np.rint(values)
    if not np.array_equal(values, rounded) or float(rounded.max(initial=0)) > np.iinfo(np.uint16).max:
        raise ValueError("raw values are not exact uint16 counts")
    counts = rounded.astype(np.int32)
    library = counts.astype(np.int64).sum(axis=1)
    expected = np.asarray(expected_umi, dtype=np.float64)
    if expected.shape != library.shape or not np.all(np.isfinite(expected)) or np.any(expected < 0):
        raise ValueError("invalid obs UMI_count")
    difference = expected - library
    positive = expected > 0
    fraction = np.divide(library, expected, out=np.full(expected.shape, np.nan), where=positive)
    comparison = {
        "rows": len(library), "exactRows": int(np.sum(difference == 0)),
        "obsGreaterRows": int(np.sum(difference > 0)), "obsLowerRows": int(np.sum(difference < 0)),
        "minimumObsMinusRetained": float(min(0.0, difference.min())),
        "maximumObsMinusRetained": float(max(0.0, difference.max())),
        "sumObsMinusRetained": float(difference.sum(dtype=np.float64)),
        "minimumRetainedFractionOfObs": float(np.nanmin(fraction)),
        "maximumRetainedFractionOfObs": float(np.nanmax(fraction)),
    }
    return counts, library, comparison


def merge_comparison(total: dict[str, float | int], part: dict[str, float | int]) -> None:
    for key in ("rows", "exactRows", "obsGreaterRows", "obsLowerRows", "sumObsMinusRetained"):
        total[key] += part[key]
    total["minimumObsMinusRetained"] = min(total["minimumObsMinusRetained"], part["minimumObsMinusRetained"])
    total["maximumObsMinusRetained"] = max(total["maximumObsMinusRetained"], part["maximumObsMinusRetained"])
    total["minimumRetainedFractionOfObs"] = min(total["minimumRetainedFractionOfObs"], part["minimumRetainedFractionOfObs"])
    total["maximumRetainedFractionOfObs"] = max(total["maximumRetainedFractionOfObs"], part["maximumRetainedFractionOfObs"])


def _raw_arrays(
    routing: dict[str, np.ndarray], rows: np.ndarray, counts: np.ndarray, library: np.ndarray
) -> dict[str, np.ndarray]:
    matrix = sparse.csr_matrix(counts)
    matrix.sort_indices()
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
        "raw_data": matrix.data.astype(np.int32), "raw_indices": matrix.indices.astype(np.int32),
        "raw_indptr": matrix.indptr.astype(np.int64), "raw_shape": np.asarray(matrix.shape, np.int64),
    }


def _rss() -> int:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return 0


def _source_contract() -> dict[str, object]:
    receipt = json.loads(RECEIPT.read_text())
    if SOURCE.stat().st_size != SOURCE_BYTES or receipt.get("sha256") != SOURCE_SHA256:
        raise ValueError("source receipt drift")
    with h5py.File(SOURCE, "r") as handle:
        matrix = handle["X"]
        if matrix.shape != SOURCE_SHAPE or matrix.dtype != np.float32 or matrix.chunks is not None or matrix.compression is not None:
            raise ValueError("source matrix contract drift")
        offset = int(matrix.id.get_offset())
        storage = int(matrix.id.get_storage_size())
    if storage != int(np.prod(SOURCE_SHAPE)) * 4:
        raise ValueError("source storage size drift")
    return {
        "path": str(SOURCE.relative_to(ROOT)).replace("\\", "/"), "bytes": SOURCE_BYTES,
        "sha256": SOURCE_SHA256, "receiptSha256": digest(RECEIPT), "matrixOffset": offset,
        "matrixStorageBytes": storage, "mtimeNsAtFreeze": SOURCE.stat().st_mtime_ns,
    }


def prepare(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.output_dir}")
    routing = load_routing(args.routing)
    masks = role_masks(routing)
    source = _source_contract()
    args.output_dir.mkdir(parents=True)
    query_ids = routing["query_ids"].astype(str)
    training = np.sort(np.concatenate([np.flatnonzero(masks["fit"]), np.flatnonzero(masks["control"])]))
    protocol = {
        "schema": SCHEMA, "status": "frozen-before-quantitative-X-access",
        "hypothesis": "the official RPE1 raw matrix can be losslessly routed into bounded integer CSR shards and one random-access reconstruction-training mmap without accessing held intervention outcomes",
        "advancementRule": "all selected values are finite nonnegative exact uint16 counts; retained-panel libraries are positive; test, unresolved, and development-validation X rows read equal zero; all56 GEM groups have reconstruction-training control support; profile projects <=2400 seconds and peak RSS <=6GiB",
        "source": source, "routing": {"path": str(args.routing.relative_to(ROOT)).replace("\\", "/"), "sha256": ROUTING_SHA256},
        "queryRoster": {"count": len(query_ids), "orderedSha256": hashlib.sha256(("\n".join(query_ids) + "\n").encode()).hexdigest(), "taxonomy": 9606, "allSourceColumnsObserved": True},
        "roleCounts": {name: int(mask.sum()) for name, mask in masks.items()},
        "forbiddenRoleCounts": {role: int(np.sum(routing["intervention_role"].astype(str) == role)) for role in ["validation", "test-excluded", "unresolved-excluded"]},
        "normalization": {"storedValues": "raw integer UMI counts", "libraryDefinition": "sum across exact ordered8749 source columns", "momentRate": "CP10k=10000*raw_count/full-8749-column-library", "obsUmiCountUse": "diagnostic only; never a normalization denominator"},
        "shards": {"maximumRows": args.shard_size, "roles": list(ROLE_NAMES), "format": "pickle-free compressed NPZ int32 CSR"},
        "trainingMmap": {"shape": [len(training), SOURCE_SHAPE[1]], "dtype": "uint16", "order": "ascending source_row_index then query source order", "includedRoles": ["fit", "control"]},
        "moments": {"fitting": "pooled action CP10k sums, cell counts and action x GEM cell counts; reconstruction-training fitting rows only", "control": "GEM CP10k sums/squares plus pooled raw counts/libraries; reconstruction-training verified controls only"},
        "reader": "temporary row-span mmap per sorted block; exact requested offsets copied; mapping closed before next block",
        "profile": {"rows": 8192, "allocation": "two fit blocks, one control block, one reconstruction-held block", "maximumSeconds": args.max_seconds, "maximumRssBytes": args.max_rss_bytes},
        "implementation": {"path": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"), "sha256": digest(Path(__file__).resolve())},
        "quantitativeXRowsReadAtFreeze": 0,
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.output_dir / "query-ids.txt").write_text("\n".join(query_ids) + "\n", encoding="ascii")
    return protocol


def _load_protocol(args: argparse.Namespace) -> dict[str, object]:
    path = args.output_dir / "protocol.json"
    if not path.exists():
        raise FileNotFoundError("freeze protocol before X access")
    result = json.loads(path.read_text())
    if result["status"] != "frozen-before-quantitative-X-access" or result["implementation"]["sha256"] != digest(Path(__file__).resolve()):
        raise ValueError("protocol or implementation drift")
    if SOURCE.stat().st_size != SOURCE_BYTES or SOURCE.stat().st_mtime_ns != result["source"]["mtimeNsAtFreeze"]:
        raise ValueError("source changed after freeze")
    if digest(args.routing) != ROUTING_SHA256:
        raise ValueError("routing drift")
    return result


def profile(args: argparse.Namespace) -> dict[str, object]:
    protocol = _load_protocol(args)
    path = args.output_dir / "profile.json"
    if path.exists():
        raise FileExistsError("profile already exists")
    routing = load_routing(args.routing)
    masks = role_masks(routing)
    allocation = [("fit", 0), ("fit", args.shard_size), ("control", 0), ("reconstruction-held", 0)]
    began = time.perf_counter()
    measured = []
    peak = _rss()
    output_bytes = 0
    for role, left in allocation:
        allowed = np.flatnonzero(masks[role])
        rows = allowed[left : left + args.shard_size]
        started = time.perf_counter()
        raw = read_rows_bounded(SOURCE, int(protocol["source"]["matrixOffset"]), SOURCE_SHAPE, rows)
        counts, library, comparison = validate_raw_block(raw, routing["umi_count"][rows])
        arrays = _raw_arrays(routing, rows, counts, library)
        temporary = args.output_dir / f".profile-{role}-{left}.npz"
        write_npz(temporary, arrays)
        output_bytes += temporary.stat().st_size
        temporary.unlink()
        measured.append({"role": role, "rows": len(rows), "seconds": time.perf_counter() - started, "nnz": len(arrays["raw_data"]), "zeroLibraries": int(np.sum(library == 0)), "obsUmiComparison": comparison})
        peak = max(peak, _rss())
    elapsed = time.perf_counter() - began
    total_rows = sum(EXPECTED_ROWS.values())
    # Includes mmap allocation, moment consolidation, checksums, and a 50% conversion margin.
    projected = elapsed / 8192 * total_rows * 1.5 + 240.0
    result = {
        "schema": f"{SCHEMA}.profile", "rowsRead": 8192, "testRowsRead": 0,
        "developmentValidationRowsRead": 0, "unresolvedRowsRead": 0, "measured": measured,
        "elapsedSeconds": elapsed, "peakRssBytes": peak, "sampleOutputBytes": output_bytes,
        "projectedBuildSecondsConservative": projected,
        "projectedRawShardBytes": int(output_bytes / 8192 * total_rows),
        "accepted": bool(projected <= args.max_seconds and peak <= args.max_rss_bytes),
        "maximumSeconds": args.max_seconds, "maximumRssBytes": args.max_rss_bytes,
    }
    path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if not result["accepted"]:
        raise RuntimeError("profile exceeds frozen bounds")
    return result


def _empty_comparison() -> dict[str, float | int]:
    return {"rows": 0, "exactRows": 0, "obsGreaterRows": 0, "obsLowerRows": 0, "sumObsMinusRetained": 0.0, "minimumObsMinusRetained": float("inf"), "maximumObsMinusRetained": float("-inf"), "minimumRetainedFractionOfObs": float("inf"), "maximumRetainedFractionOfObs": float("-inf")}


def build(args: argparse.Namespace) -> dict[str, object]:
    protocol = _load_protocol(args)
    profile_path = args.output_dir / "profile.json"
    if not profile_path.exists() or not json.loads(profile_path.read_text())["accepted"]:
        raise RuntimeError("accepted profile required")
    if (args.output_dir / "manifest.json").exists():
        raise FileExistsError("immutable output complete")
    routing = load_routing(args.routing)
    masks = role_masks(routing)
    staging = args.output_dir / ".building"
    progress_path = args.output_dir / "build-progress.json"
    if staging.exists():
        progress = json.loads(progress_path.read_text())
        for item in progress["shards"]:
            path = staging / item["path"]
            if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise ValueError("completed shard drift")
    else:
        staging.mkdir()
        for role in ROLE_NAMES:
            (staging / role).mkdir()
        progress = {"schema": f"{SCHEMA}.progress", "shards": [], "sourceRowsRepeated": 0}
        progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
    completed = {item["path"]: item for item in progress["shards"]}
    training_rows = np.sort(np.concatenate([np.flatnonzero(masks["fit"]), np.flatnonzero(masks["control"])]))
    local_for_source = np.full(SOURCE_SHAPE[0], -1, dtype=np.int64)
    local_for_source[training_rows] = np.arange(len(training_rows), dtype=np.int64)
    mmap_path = staging / "reconstruction-train-counts.uint16"
    if mmap_path.exists():
        if mmap_path.stat().st_size != TRAINING_ROWS * SOURCE_SHAPE[1] * 2:
            raise ValueError("partial mmap size drift")
        training_mmap = np.memmap(mmap_path, mode="r+", dtype=np.uint16, shape=(TRAINING_ROWS, SOURCE_SHAPE[1]))
    else:
        training_mmap = np.memmap(mmap_path, mode="w+", dtype=np.uint16, shape=(TRAINING_ROWS, SOURCE_SHAPE[1]))
    filled = np.zeros(TRAINING_ROWS, bool)
    training_library = np.zeros(TRAINING_ROWS, dtype=np.int64)
    fit_genes = np.asarray(sorted(set(routing["action_ids"][masks["fit"]].astype(str))))
    gene_index = {gene: index for index, gene in enumerate(fit_genes)}
    fit_sum = np.zeros((len(fit_genes), SOURCE_SHAPE[1]), dtype=np.float64)
    fit_count = np.zeros(len(fit_genes), dtype=np.int64)
    fit_gem_count = np.zeros((len(fit_genes), 56), dtype=np.int64)
    control_sum = np.zeros((56, SOURCE_SHAPE[1]), dtype=np.float64)
    control_sumsq = np.zeros_like(control_sum)
    control_raw_sum = np.zeros((56, SOURCE_SHAPE[1]), dtype=np.int64)
    control_library_sum = np.zeros(56, dtype=np.int64)
    control_count = np.zeros(56, dtype=np.int64)
    comparison = _empty_comparison()
    zero_libraries = {role: 0 for role in ROLE_NAMES}
    maximum_count = 0
    peak = _rss()
    started = time.perf_counter()

    def accumulate(role: str, rows: np.ndarray, counts: sparse.csr_matrix, library: np.ndarray) -> None:
        nonlocal maximum_count
        maximum_count = max(maximum_count, int(counts.data.max(initial=0)))
        if role in {"fit", "control"}:
            positions = local_for_source[rows]
            if np.any(positions < 0) or np.any(filled[positions]):
                raise ValueError("training mmap row duplicated or outside order")
            training_mmap[positions] = counts.toarray().astype(np.uint16)
            training_library[positions] = library
            filled[positions] = True
        rates = counts.astype(np.float64).multiply((10_000.0 / library)[:, None]).tocsr()
        if role == "fit":
            actions = routing["action_ids"][rows].astype(str)
            genes, inverse = np.unique(actions, return_inverse=True)
            membership = sparse.csr_matrix((np.ones(len(rows)), (inverse, np.arange(len(rows)))), shape=(len(genes), len(rows)))
            sums = (membership @ rates).tocsr()
            for local, gene in enumerate(genes):
                destination = gene_index[gene]
                left, right = sums.indptr[local : local + 2]
                fit_sum[destination, sums.indices[left:right]] += sums.data[left:right]
                use = inverse == local
                fit_count[destination] += int(use.sum())
                fit_gem_count[destination] += np.bincount(routing["gem_group"][rows][use], minlength=57)[1:57]
        elif role == "control":
            gems = routing["gem_group"][rows]
            for gem in np.unique(gems):
                use = gems == gem
                destination = int(gem) - 1
                block_rate = rates[use]
                block_count = counts[use]
                control_sum[destination] += np.asarray(block_rate.sum(axis=0)).ravel()
                control_sumsq[destination] += np.asarray(block_rate.power(2).sum(axis=0)).ravel()
                control_raw_sum[destination] += np.asarray(block_count.astype(np.int64).sum(axis=0)).ravel()
                control_library_sum[destination] += int(library[use].sum())
                control_count[destination] += int(use.sum())

    # Reconstruct accumulators from completed derived shards without rereading source values.
    for item in progress["shards"]:
        path = staging / item["path"]
        with np.load(path, allow_pickle=False) as archive:
            rows = np.asarray(archive["source_row_index"], np.int64)
            library = np.asarray(archive["library_size"], np.int64)
            matrix = sparse.csr_matrix((archive["raw_data"], archive["raw_indices"], archive["raw_indptr"]), shape=tuple(archive["raw_shape"]))
        _, _, part = validate_raw_block(matrix.toarray().astype(np.float32), routing["umi_count"][rows])
        merge_comparison(comparison, part)
        zero_libraries[item["role"]] += int(np.sum(library == 0))
        accumulate(item["role"], rows, matrix, library)
    for role in ROLE_NAMES:
        allowed = np.flatnonzero(masks[role])
        for shard_index, left in enumerate(range(0, len(allowed), args.shard_size)):
            if time.perf_counter() - started > args.max_seconds:
                raise TimeoutError("build exceeds frozen wall bound")
            relative = f"{role}/cells-{shard_index:05d}.npz"
            if relative in completed:
                continue
            rows = allowed[left : left + args.shard_size]
            raw = read_rows_bounded(SOURCE, int(protocol["source"]["matrixOffset"]), SOURCE_SHAPE, rows)
            counts_dense, library, part = validate_raw_block(raw, routing["umi_count"][rows])
            merge_comparison(comparison, part)
            zero_libraries[role] += int(np.sum(library == 0))
            if np.any(library <= 0):
                raise ValueError("selected row has zero retained-panel library")
            arrays = _raw_arrays(routing, rows, counts_dense, library)
            path = staging / relative
            write_npz(path, arrays)
            item = {"role": role, "path": relative, "rows": len(rows), "sourceRowMinimum": int(rows[0]), "sourceRowMaximum": int(rows[-1]), "nnz": len(arrays["raw_data"]), "bytes": path.stat().st_size, "sha256": digest(path)}
            progress["shards"].append(item)
            progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True, allow_nan=False) + "\n")
            matrix = sparse.csr_matrix((arrays["raw_data"], arrays["raw_indices"], arrays["raw_indptr"]), shape=tuple(arrays["raw_shape"]))
            accumulate(role, rows, matrix, library)
            training_mmap.flush()
            peak = max(peak, _rss())
            if peak > args.max_rss_bytes:
                raise MemoryError("build exceeds frozen RSS bound")
    if not np.all(filled) or len(fit_genes) != 1_666 or int(fit_count.sum()) != EXPECTED_ROWS["fit"] or np.any(control_count <= 0):
        raise AssertionError("training mmap or moments do not close")
    training_mmap.flush()
    row_metadata_path = staging / "reconstruction-train-row-metadata.npz"
    write_npz(row_metadata_path, {
        "schema": np.asarray(f"{SCHEMA}.training-mmap"), "source_sha256": np.asarray(SOURCE_SHA256),
        "local_row_index": np.arange(TRAINING_ROWS, dtype=np.int64), "source_row_index": training_rows,
        "cell_ids": _strings(routing["cell_ids"][training_rows]), "action_ids": _strings(routing["action_ids"][training_rows]),
        "guide_pair_ids": _strings(routing["guide_pair_ids"][training_rows]), "population_ids": _strings(routing["gene_transcript"][training_rows]),
        "gem_group": routing["gem_group"][training_rows].astype(np.int16), "is_control": routing["is_control"][training_rows].astype(bool),
        "library_size": training_library,
        "query_ids": _strings(routing["query_ids"]), "query_taxon": np.full(SOURCE_SHAPE[1], 9606, np.int64),
    })
    fit_path = staging / "fitting-action-moments.npz"
    write_npz(fit_path, {
        "schema": np.asarray(f"{SCHEMA}.fitting-action-moments"), "source_sha256": np.asarray(SOURCE_SHA256),
        "routing_sha256": np.asarray(ROUTING_SHA256), "action_ids": _strings(fit_genes),
        "query_ids": _strings(routing["query_ids"]), "query_taxon": np.full(SOURCE_SHAPE[1], 9606, np.int64),
        "gem_group": np.arange(1, 57, dtype=np.int16), "cp10k_sum": fit_sum,
        "cell_count": fit_count, "gem_cell_count": fit_gem_count,
        "rate_definition": np.asarray(protocol["normalization"]["momentRate"]),
    })
    control_path = staging / "control-gem-moments.npz"
    write_npz(control_path, {
        "schema": np.asarray(f"{SCHEMA}.control-gem-moments"), "source_sha256": np.asarray(SOURCE_SHA256),
        "routing_sha256": np.asarray(ROUTING_SHA256), "query_ids": _strings(routing["query_ids"]),
        "query_taxon": np.full(SOURCE_SHAPE[1], 9606, np.int64), "gem_group": np.arange(1, 57, dtype=np.int16),
        "num_cells": control_count, "sum_cp10k": control_sum, "sum_squares_cp10k": control_sumsq,
        "raw_count_sum": control_raw_sum, "library_count_sum": control_library_sum,
        "positive_basal_rate_formula": np.asarray("10000*(raw_count_sum+0.5)/(library_count_sum+0.5*8749)"),
    })
    training_mmap._mmap.close()
    for role in ROLE_NAMES:
        (staging / role).replace(args.output_dir / role)
    for name in [mmap_path.name, row_metadata_path.name, fit_path.name, control_path.name]:
        (staging / name).replace(args.output_dir / name)
    staging.rmdir()
    elapsed = time.perf_counter() - started
    mmap_final = args.output_dir / mmap_path.name
    row_final = args.output_dir / row_metadata_path.name
    fit_final = args.output_dir / fit_path.name
    control_final = args.output_dir / control_path.name
    manifest = {
        "schema": SCHEMA, "status": "complete", "source": protocol["source"], "routing": protocol["routing"],
        "protocolSha256": digest(args.output_dir / "protocol.json"), "profileSha256": digest(profile_path),
        "implementation": protocol["implementation"], "counts": {"roleRows": EXPECTED_ROWS, "queries": SOURCE_SHAPE[1], "fitActions": len(fit_genes), "GEMGroups": 56, "zeroLibraryRows": zero_libraries, "testRowsRead": 0, "developmentValidationRowsRead": 0, "unresolvedRowsRead": 0, "maximumRawCount": maximum_count},
        "obsUmiCountComparison": comparison, "shards": progress["shards"],
        "trainingMmap": {"path": mmap_final.name, "shape": [TRAINING_ROWS, SOURCE_SHAPE[1]], "dtype": "uint16", "order": "C; ascending source_row_index", "bytes": mmap_final.stat().st_size, "sha256": digest(mmap_final), "rowMetadata": row_final.name, "rowMetadataSha256": digest(row_final)},
        "fittingActionMoments": {"path": fit_final.name, "shape": [len(fit_genes), SOURCE_SHAPE[1]], "bytes": fit_final.stat().st_size, "sha256": digest(fit_final)},
        "controlMoments": {"path": control_final.name, "shape": [56, SOURCE_SHAPE[1]], "bytes": control_final.stat().st_size, "sha256": digest(control_final)},
        "runtime": {"seconds": elapsed, "peakRssBytes": peak, "maximumSeconds": args.max_seconds, "maximumRssBytes": args.max_rss_bytes},
        "limitations": ["development-validation, test-excluded and unresolved-action X rows remain unopened", "reconstruction-held rows are isolated from the training mmap and all fitted moments", "raw cells are not paired before/after trajectories", "obs UMI_count is retained only as a diagnostic and does not replace the measured8749-column denominator"],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    progress_path.unlink()
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "profile", "build"))
    parser.add_argument("--routing", type=Path, default=ROUTING)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--max-seconds", type=float, default=2400.0)
    parser.add_argument("--max-rss-bytes", type=int, default=6 * 1024**3)
    args = parser.parse_args()
    args.routing = args.routing.resolve()
    args.output_dir = args.output_dir.resolve()
    if not 0 < args.shard_size <= 2048:
        parser.error("shard-size must be in [1,2048]")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    result = {"prepare": prepare, "profile": profile, "build": build}[arguments.mode](arguments)
    print(json.dumps({"mode": arguments.mode, "status": result.get("status", result.get("accepted")), "counts": result.get("counts", result.get("roleCounts")), "runtime": result.get("runtime", result.get("elapsedSeconds"))}, indent=2))
