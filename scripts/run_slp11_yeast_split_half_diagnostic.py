"""Fitting-only split-half reproducibility diagnostic for rebuilt yeast RNA."""

from __future__ import annotations

import argparse
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
RAW_ROOT = (
    ROOT / "data/derived/slp11-yeast-atlas-counts/nadal-ribelles-raw-rna-development-v1"
)
SELECTION_ROOT = (
    ROOT / "results/slp11-transition/yeast-seurat-metadata-inventory-v1/selection"
)
OUTPUT_ROOT = ROOT / "results/slp11-transition/yeast-rna-fitting-split-half-v1"
CORE_PATH = ROOT / "modules/slp-1-1-count-moments-v1/count_moments.py"
CONTEXTS = ("Control", "NaCl")
SEED = 731
MAX_SECONDS = 900.0
MAX_RSS = 6 * (1 << 30)
BLOCK_CELLS = 4096

SPEC = importlib.util.spec_from_file_location("slp11_split_count_moments", CORE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load count-moments core")
core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def stable_half_split(
    barcodes: np.ndarray,
    batches: np.ndarray,
    action_ids: np.ndarray,
    roles: np.ndarray,
    controls: np.ndarray,
    *,
    seed: int = SEED,
) -> tuple[np.ndarray, dict[str, int]]:
    """Rank barcode hashes within each fitting/control action-by-batch group."""
    arrays = [
        np.asarray(value) for value in (barcodes, batches, action_ids, roles, controls)
    ]
    n = len(arrays[0])
    if any(value.shape != (n,) for value in arrays):
        raise ValueError("split metadata must be aligned vectors")
    barcodes, batches, action_ids, roles, controls = arrays
    controls = controls.astype(np.bool_, copy=False)
    if len(set(barcodes.tolist())) != n:
        raise ValueError("barcodes must be unique within context")
    candidate = controls | (roles == "train")
    if np.any(controls & (roles != "control")):
        raise ValueError("WT cells must have the control role")
    if np.any(candidate & ~controls & (action_ids == "")):
        raise ValueError("fitting mutant lacks stable action identity")
    population = np.where(controls, "CONTROL:WT", action_ids)
    groups: dict[tuple[str, str], list[int]] = {}
    for index in np.flatnonzero(candidate):
        groups.setdefault((str(batches[index]), str(population[index])), []).append(
            int(index)
        )
    half = np.full(n, -2, dtype=np.int8)  # -2: outside fitting/control
    singleton_cells = 0
    paired_groups = 0
    for indices in groups.values():
        if len(indices) < 2:
            half[indices] = -1  # -1: fitting/control singleton population excluded
            singleton_cells += len(indices)
            continue
        ranked = sorted(
            indices,
            key=lambda index: (
                hashlib.sha256(f"{seed}\0{barcodes[index]}".encode()).digest(),
                str(barcodes[index]),
            ),
        )
        half[ranked[::2]] = 0
        half[ranked[1::2]] = 1
        paired_groups += 1
        if not ({0, 1} <= set(half[indices].tolist())):
            raise AssertionError("paired population does not retain both halves")
    stats = {
        "selectedCells": n,
        "validationCellsExcluded": int(np.count_nonzero(~candidate)),
        "fittingOrControlCells": int(np.count_nonzero(candidate)),
        "pairedCellsIncluded": int(np.count_nonzero(half >= 0)),
        "singletonCellsExcluded": singleton_cells,
        "candidatePopulations": len(groups),
        "pairedPopulations": paired_groups,
        "singletonPopulations": len(groups) - paired_groups,
        "halfACells": int(np.count_nonzero(half == 0)),
        "halfBCells": int(np.count_nonzero(half == 1)),
    }
    if stats["pairedCellsIncluded"] + singleton_cells != stats["fittingOrControlCells"]:
        raise AssertionError("split cell accounting mismatch")
    return half, stats


def prepare_plan() -> None:
    if OUTPUT_ROOT.exists():
        raise RuntimeError(f"refusing to overwrite {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    frames: list[dict[str, object]] = []
    for frame_index, context in enumerate(CONTEXTS):
        selection_path = SELECTION_ROOT / f"frame-{frame_index}-selection.npz"
        with np.load(selection_path) as selected:
            half, stats = stable_half_split(
                selected["barcode"],
                selected["batch"],
                selected["stable_action_id"],
                selected["development_role"],
                selected["is_control"],
            )
            plan_path = OUTPUT_ROOT / f"frame-{frame_index}-split-plan.npz"
            np.savez(
                plan_path,
                schema=np.asarray("slp.yeast-fitting-cell-split/v1"),
                context=np.asarray(context),
                seed=np.asarray(SEED, dtype=np.int64),
                source_columns=selected["source_columns"],
                half=half,
            )
        frames.append(
            {
                "context": context,
                "selectionPath": str(selection_path.resolve()),
                "selectionSha256": sha256(selection_path),
                "planPath": str(plan_path.resolve()),
                "planSha256": sha256(plan_path),
                **stats,
            }
        )
    report = {
        "schema": "slp.yeast-fitting-cell-split-plan/v1",
        "status": "metadata-only-frozen-before-raw-values",
        "seed": SEED,
        "hash": "SHA-256(seed + NUL + verbatim barcode), lexicographic digest/barcode rank",
        "stratum": "verbatim source context x batch x stable action; WT is CONTROL:WT",
        "allocation": "alternating ranked cells A/B; all cells retained when stratum n>=2",
        "support": "strata n=1 excluded before quantitative access",
        "roles": "development fitting mutants and WT only; development validation excluded",
        "frames": frames,
    }
    (OUTPUT_ROOT / "split-plan.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def selected_columns_as_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
    columns: np.ndarray,
    n_rows: int,
) -> sparse.csr_matrix:
    """Read only named CSC columns and return cells by source rows."""
    columns = np.asarray(columns, dtype=np.int64)
    if columns.ndim != 1 or len(columns) == 0 or np.any(np.diff(columns) <= 0):
        raise ValueError("selected CSC columns must be sorted, unique and nonempty")
    lengths = np.asarray(indptr[columns + 1] - indptr[columns], dtype=np.int64)
    local_p = np.empty(len(columns) + 1, dtype=np.int64)
    local_p[0] = 0
    np.cumsum(lengths, out=local_p[1:])
    local_i = np.empty(int(local_p[-1]), dtype=indices.dtype)
    local_x = np.empty(int(local_p[-1]), dtype=values.dtype)
    cursor = 0
    for column, length in zip(columns, lengths, strict=True):
        start = int(indptr[column])
        stop = start + int(length)
        local_i[cursor : cursor + length] = indices[start:stop]
        local_x[cursor : cursor + length] = values[start:stop]
        cursor += int(length)
    matrix = sparse.csc_matrix(
        (local_x, local_i, local_p), shape=(n_rows, len(columns)), copy=False
    )
    return matrix.T.tocsr()


def _check_budget(deadline: float, process: psutil.Process) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("split-half execution exceeded frozen 900 second cap")
    rss = process.memory_info().rss
    if rss > MAX_RSS:
        raise MemoryError(f"split-half RSS {rss} exceeds frozen 6 GiB cap")


def _row_pearson(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int]:
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    numerator = np.einsum("ij,ij->i", ac, bc)
    denominator = np.sqrt(np.einsum("ij,ij->i", ac, ac) * np.einsum("ij,ij->i", bc, bc))
    tolerance = np.finfo(np.float64).eps * a.shape[1] * 32
    defined = denominator > tolerance
    result = np.full(len(a), np.nan)
    result[defined] = numerator[defined] / denominator[defined]
    return result, int(np.count_nonzero(~defined))


def _metric_summary(a: np.ndarray, b: np.ndarray) -> dict[str, object]:
    mse = np.mean((a - b) ** 2, axis=1)
    correlation, undefined = _row_pearson(a, b)
    finite = correlation[np.isfinite(correlation)]
    return {
        "genes": len(a),
        "equalGeneMeanMse": float(mse.mean()),
        "medianGeneMse": float(np.median(mse)),
        "meanGeneProfilePearson": float(finite.mean()) if len(finite) else None,
        "medianGeneProfilePearson": float(np.median(finite)) if len(finite) else None,
        "undefinedGeneProfilePearson": undefined,
    }


def _center_queries(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return a - a.mean(axis=0, keepdims=True), b - b.mean(axis=0, keepdims=True)


def execute() -> None:
    started = time.monotonic()
    deadline = started + MAX_SECONDS
    process = psutil.Process()
    plan_report_path = OUTPUT_ROOT / "split-plan.json"
    protocol_path = OUTPUT_ROOT / "protocol.json"
    if not plan_report_path.exists() or not protocol_path.exists():
        raise RuntimeError("frozen split plan and protocol are required")
    report_path = OUTPUT_ROOT / "report.json"
    profile_path = OUTPUT_ROOT / "split-half-sufficient-statistics.npz"
    if report_path.exists() or profile_path.exists():
        raise RuntimeError("refusing to overwrite completed diagnostic outputs")
    query_map = np.load(SELECTION_ROOT / "query-map.npz")
    query_ids = query_map["query_ids"]
    query_index = query_map["source_to_query_index"]
    denominator = query_map["denominator_mask"]

    frame_data: list[tuple[np.lib.npyio.NpzFile, np.lib.npyio.NpzFile]] = []
    gene_union: set[str] = set()
    for frame_index in range(len(CONTEXTS)):
        selected = np.load(SELECTION_ROOT / f"frame-{frame_index}-selection.npz")
        plan = np.load(OUTPUT_ROOT / f"frame-{frame_index}-split-plan.npz")
        if not np.array_equal(selected["source_columns"], plan["source_columns"]):
            raise RuntimeError("split plan and metadata selection differ")
        eligible = (plan["half"] >= 0) & ~selected["is_control"]
        if np.any(selected["development_role"][eligible] != "train"):
            raise RuntimeError("non-fitting mutant entered split-half execution")
        gene_union.update(selected["stable_action_id"][eligible].tolist())
        frame_data.append((selected, plan))
    gene_ids = np.asarray(sorted(gene_union))
    gene_lookup = {gene: index for index, gene in enumerate(gene_ids)}
    sums = np.zeros((2, 2, len(gene_ids), len(query_ids)), dtype=np.float64)
    cells = np.zeros((2, 2, len(gene_ids)), dtype=np.int64)
    control_sums = np.zeros((2, 2, len(query_ids)), dtype=np.float64)
    control_cells = np.zeros((2, 2), dtype=np.int64)
    processed_cells = np.zeros(2, dtype=np.int64)

    for frame_index, context in enumerate(CONTEXTS):
        selected, plan = frame_data[frame_index]
        raw = RAW_ROOT / context.lower() / "raw-csc"
        indptr = np.load(raw / "p.npy", mmap_mode="r")
        indices = np.load(raw / "i.npy", mmap_mode="r")
        values = np.load(raw / "x.npy", mmap_mode="r")
        raw_columns = np.load(raw / "source_columns.npy")
        if not np.array_equal(raw_columns, selected["source_columns"]):
            raise RuntimeError("raw CSC and metadata selection column order mismatch")
        batch_values = selected["batch"]
        for batch_id in sorted(set(batch_values[plan["half"] >= 0].tolist())):
            positions = np.flatnonzero((batch_values == batch_id) & (plan["half"] >= 0))
            action = np.where(
                selected["is_control"][positions],
                "CONTROL:WT",
                selected["stable_action_id"][positions],
            )
            local_actions = np.asarray(sorted(set(action.tolist())))
            local_lookup = {value: index for index, value in enumerate(local_actions)}
            groups = np.asarray(
                [
                    2 * local_lookup[value] + int(half)
                    for value, half in zip(action, plan["half"][positions], strict=True)
                ],
                dtype=np.int64,
            )
            moments = core.CountMoments(
                query_index, denominator, len(query_ids), 2 * len(local_actions)
            )
            for block_start in range(0, len(positions), BLOCK_CELLS):
                block_positions = positions[block_start : block_start + BLOCK_CELLS]
                block = selected_columns_as_csr(
                    indptr, indices, values, block_positions, len(query_index)
                )
                valid = moments.update(
                    block, groups[block_start : block_start + len(block_positions)]
                )
                if not np.all(valid):
                    raise RuntimeError(
                        "unexpected zero-library cell in frozen split support"
                    )
                _check_budget(deadline, process)
            local_sum = moments.sums.reshape(len(local_actions), 2, len(query_ids))
            local_cells = moments.cells.reshape(len(local_actions), 2)
            for local_index, action_id in enumerate(local_actions):
                if action_id == "CONTROL:WT":
                    control_sums[frame_index] += local_sum[local_index]
                    control_cells[frame_index] += local_cells[local_index]
                else:
                    gene_index = gene_lookup[str(action_id)]
                    sums[frame_index, :, gene_index] += local_sum[local_index]
                    cells[frame_index, :, gene_index] += local_cells[local_index]
            processed_cells[frame_index] += len(positions)
            del moments
            _check_budget(deadline, process)

    np.savez(
        profile_path,
        schema=np.asarray("slp.yeast-fitting-split-half-sufficient-statistics/v1"),
        contexts=np.asarray(CONTEXTS),
        query_ids=query_ids,
        gene_ids=gene_ids,
        half_sum=sums,
        half_num_cells=cells,
        control_half_sum=control_sums,
        control_half_num_cells=control_cells,
    )

    context_metrics: list[dict[str, object]] = []
    for context_index, context in enumerate(CONTEXTS):
        observed = np.all(cells[context_index] > 0, axis=0)
        context_sums = sums[context_index][:, observed, :]
        context_cells = cells[context_index][:, observed]
        means = context_sums / context_cells[:, :, None]
        centered_a, centered_b = _center_queries(means[0], means[1])
        control_means = (
            control_sums[context_index] / control_cells[context_index, :, None]
        )
        control_corr, control_undefined = _row_pearson(
            control_means[0, None], control_means[1, None]
        )
        context_metrics.append(
            {
                "context": context,
                "pairedGenes": int(np.count_nonzero(observed)),
                "halfACells": int(cells[context_index, 0, observed].sum()),
                "halfBCells": int(cells[context_index, 1, observed].sum()),
                "rawAbsoluteSplitHalf": _metric_summary(means[0], means[1]),
                "independentlyQueryCenteredSplitHalf": _metric_summary(
                    centered_a, centered_b
                ),
                "wildType": {
                    "halfACells": int(control_cells[context_index, 0]),
                    "halfBCells": int(control_cells[context_index, 1]),
                    "rawMse": float(
                        np.mean((control_means[0] - control_means[1]) ** 2)
                    ),
                    "profilePearson": (
                        float(control_corr[0]) if np.isfinite(control_corr[0]) else None
                    ),
                    "undefinedProfilePearson": control_undefined,
                },
            }
        )

    shared = np.all(cells > 0, axis=(0, 1))
    full_means = np.empty((2, int(np.count_nonzero(shared)), len(query_ids)))
    for context_index in range(2):
        context_sums = sums[context_index][:, shared, :]
        context_cells = cells[context_index][:, shared]
        full_means[context_index] = (
            context_sums.sum(axis=0) / context_cells.sum(axis=0)[:, None]
        )
    cross_a, cross_b = _center_queries(full_means[0], full_means[1])
    report = {
        "schema": "slp.yeast-fitting-split-half-diagnostic/v1",
        "status": "complete-descriptive-no-model-advancement",
        "protocolSha256": sha256(protocol_path),
        "splitPlanSha256": sha256(plan_report_path),
        "rawManifestSha256": sha256(RAW_ROOT / "manifest.json"),
        "sufficientStatisticsPath": str(profile_path.resolve()),
        "sufficientStatisticsSha256": sha256(profile_path),
        "valueSpace": "per-cell ln1p(CP10k), denominator all 6951 audited RNA rows",
        "aggregation": "equal cell within action/context/half across paired batch strata",
        "queryCentering": "each half/context centroid computed independently as equal-gene per-query mean",
        "contextMetrics": context_metrics,
        "crossEnvironmentSameGene": {
            "sharedGenes": int(np.count_nonzero(shared)),
            "rawAbsoluteProfiles": _metric_summary(full_means[0], full_means[1]),
            "independentlyQueryCenteredProfiles": _metric_summary(cross_a, cross_b),
        },
        "processedCellsByContext": {
            context: int(processed_cells[index])
            for index, context in enumerate(CONTEXTS)
        },
        "runtimeSeconds": time.monotonic() - started,
        "limits": [
            "This is a fitting-only technical reproducibility diagnostic, not a biological noise ceiling.",
            "Shared batch, clone, genotype assignment, library preparation and normalization can correlate halves.",
            "Query centering removes the equal-gene average molecular profile; no batch-WT subtraction was applied.",
            "Cross-environment correlation describes stable versus environment-dependent centered patterns; it is not a perturbation forecast.",
        ],
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for selected, plan in frame_data:
        selected.close()
        plan.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-plan", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.prepare_plan == args.execute:
        parser.error("choose exactly one of --prepare-plan or --execute")
    if args.prepare_plan:
        prepare_plan()
    else:
        execute()


if __name__ == "__main__":
    main()
