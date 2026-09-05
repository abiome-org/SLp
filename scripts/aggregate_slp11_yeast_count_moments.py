"""Aggregate frozen selected yeast raw counts into batch-local moments shards."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "modules/slp-1-1-count-moments-v1/count_moments.py"
SPEC = importlib.util.spec_from_file_location("slp11_count_moments", CORE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load count moments core")
core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)

RAW_ROOT = (
    ROOT / "data/derived/slp11-yeast-atlas-counts/nadal-ribelles-raw-rna-development-v1"
)
SELECTION_ROOT = (
    ROOT / "results/slp11-transition/yeast-seurat-metadata-inventory-v1/selection"
)
CONTEXTS = ("Control", "NaCl")
MAX_RUNTIME_SECONDS = 900.0
MAX_RSS_BYTES = 6 * (1 << 30)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _check_budget(deadline: float, process: psutil.Process) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("count-moments execution exceeded its frozen 900 second cap")
    rss = process.memory_info().rss
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f"count-moments RSS {rss} exceeds frozen 6 GiB cap")


def aggregate_batch(
    counts: sparse.csc_matrix,
    positions: np.ndarray,
    stable_actions: np.ndarray,
    development_roles: np.ndarray,
    controls: np.ndarray,
    query_index: np.ndarray,
    denominator_mask: np.ndarray,
    query_ids: np.ndarray,
    *,
    context: str,
    batch_id: str,
    output_path: Path,
    block_cells: int = 4096,
    deadline: float | None = None,
    process: psutil.Process | None = None,
) -> dict[str, object]:
    """Aggregate one exact batch; input positions and groups are metadata-only."""
    positions = np.asarray(positions, dtype=np.int64)
    if positions.ndim != 1 or len(positions) == 0 or np.any(np.diff(positions) <= 0):
        raise ValueError("batch positions must be sorted, unique and nonempty")
    action = np.asarray(stable_actions)[positions]
    role = np.asarray(development_roles)[positions]
    is_control_cell = np.asarray(controls, dtype=np.bool_)[positions]
    if np.any(is_control_cell & (action != "")) or np.any(
        ~is_control_cell & (action == "")
    ):
        raise ValueError("control/action identity mismatch")
    mutant_ids = sorted(set(action[~is_control_cell].tolist()))
    group_ids = np.asarray(["CONTROL:WT", *mutant_ids])
    mutant_roles: list[str] = []
    for stable_id in mutant_ids:
        unique_roles = set(role[action == stable_id].tolist())
        if len(unique_roles) != 1:
            raise ValueError(f"action {stable_id} spans development roles")
        mutant_roles.append(next(iter(unique_roles)))
    group_roles = np.asarray(["control", *mutant_roles])
    if not set(group_roles.tolist()).issubset({"control", "train", "validation"}):
        raise ValueError("unexpected development role")
    group_lookup = {stable_id: index + 1 for index, stable_id in enumerate(mutant_ids)}
    groups = np.asarray(
        [
            0 if control else group_lookup[stable_id]
            for stable_id, control in zip(action, is_control_cell, strict=True)
        ],
        dtype=np.int64,
    )
    moments = core.CountMoments(
        query_index,
        denominator_mask,
        len(query_ids),
        len(group_ids),
    )
    if deadline is not None and process is not None:
        _check_budget(deadline, process)
    for start in range(0, len(positions), block_cells):
        stop = min(start + block_cells, len(positions))
        block = counts[:, positions[start:stop]].T.tocsr()
        moments.update(block, groups[start:stop])
        if deadline is not None and process is not None:
            _check_budget(deadline, process)
    summary = moments.summary()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        schema=np.asarray("slp.yeast-batch-count-moments/v1"),
        context=np.asarray(context),
        batch_id=np.asarray(batch_id),
        query_ids=query_ids,
        group_action_id=group_ids,
        development_role=group_roles,
        is_control=np.arange(len(group_ids)) == 0,
        sum=moments.sums,
        sum_squares=moments.squares,
        num_cells=summary["num_cells"],
        total_cells=summary["total_cells"],
        zero_library_cells=summary["zero_library_cells"],
        mean_observed=summary["mean_observed"],
        variance_observed=summary["variance_observed"],
    )
    return {
        "context": context,
        "batchId": batch_id,
        "path": str(output_path.resolve()),
        "groups": len(group_ids),
        "fittingGroups": int(np.count_nonzero(group_roles == "train")),
        "validationGroups": int(np.count_nonzero(group_roles == "validation")),
        "controlGroups": 1,
        "sourceCells": len(positions),
        "positiveLibraryCells": int(summary["num_cells"].sum()),
        "zeroLibraryCells": int(summary["zero_library_cells"].sum()),
    }


def main() -> None:
    started = time.monotonic()
    deadline = started + MAX_RUNTIME_SECONDS
    process = psutil.Process()
    raw_manifest_path = RAW_ROOT / "manifest.json"
    raw_manifest = json.loads(raw_manifest_path.read_text())
    query = np.load(SELECTION_ROOT / "query-map.npz")
    query_ids = query["query_ids"]
    query_index = query["source_to_query_index"]
    denominator = query["denominator_mask"]
    shards: list[dict[str, object]] = []
    for frame_index, context in enumerate(CONTEXTS):
        context_root = RAW_ROOT / context.lower()
        moments_root = context_root / "moments"
        if moments_root.exists():
            raise RuntimeError(f"refusing to overwrite {moments_root}")
        moments_root.mkdir(parents=True)
        raw = context_root / "raw-csc"
        indptr = np.load(raw / "p.npy", mmap_mode="r")
        indices = np.load(raw / "i.npy", mmap_mode="r")
        values = np.load(raw / "x.npy", mmap_mode="r")
        selected = np.load(SELECTION_ROOT / f"frame-{frame_index}-selection.npz")
        source_columns = np.load(raw / "source_columns.npy")
        if not np.array_equal(source_columns, selected["source_columns"]):
            raise RuntimeError("raw CSC and metadata selection column order mismatch")
        counts = sparse.csc_matrix(
            (values, indices, indptr),
            shape=(len(query_index), len(source_columns)),
            copy=False,
        )
        batch = selected["batch"]
        for batch_id in sorted(set(batch.tolist())):
            positions = np.flatnonzero(batch == batch_id)
            output = moments_root / f"{batch_id}.npz"
            shard = aggregate_batch(
                counts,
                positions,
                selected["stable_action_id"],
                selected["development_role"],
                selected["is_control"],
                query_index,
                denominator,
                query_ids,
                context=context,
                batch_id=batch_id,
                output_path=output,
                deadline=deadline,
                process=process,
            )
            shard["bytes"] = output.stat().st_size
            shard["sha256"] = _sha256(output)
            if shard["controlGroups"] != 1:
                raise RuntimeError("batch lacks its explicit WT population")
            if shard["validationGroups"] and not shard["fittingGroups"]:
                raise RuntimeError(
                    "validation population occurs in a batch without fitting populations"
                )
            shards.append(shard)
            _check_budget(deadline, process)
    report = {
        "schema": "slp.yeast-batch-count-moments-manifest/v1",
        "rawExtractionManifestSha256": _sha256(raw_manifest_path),
        "queryMapSha256": _sha256(SELECTION_ROOT / "query-map.npz"),
        "countCoreSha256": _sha256(CORE_PATH),
        "transform": "per-cell ln1p(10000 * strict-query-count / sum-all-6951-source-rows)",
        "grouping": "verbatim context x batch x stable action; explicit WT group",
        "storedMoments": "float64 sum and sum_squares; zeros included",
        "meanVarianceSerialized": False,
        "minimumCellThreshold": None,
        "cloneUsedAsInput": False,
        "maximumRuntimeSeconds": MAX_RUNTIME_SECONDS,
        "maximumRssBytes": MAX_RSS_BYTES,
        "shards": shards,
        "runtimeSeconds": time.monotonic() - started,
        "rawContextCount": len(raw_manifest["contexts"]),
    }
    (RAW_ROOT / "moments-manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
