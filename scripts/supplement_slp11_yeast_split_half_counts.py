"""Cell-support supplement for the frozen yeast fitting split-half diagnostic."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/slp11-transition/yeast-rna-fitting-split-half-v1"
SOURCE_PATH = RESULT_ROOT / "split-half-sufficient-statistics.npz"
ORIGINAL_REPORT = RESULT_ROOT / "report.json"
OUTPUT_PATH = RESULT_ROOT / "cell-count-bin-supplement.json"
DIAGNOSTIC_PATH = ROOT / "scripts/run_slp11_yeast_split_half_diagnostic.py"
BINS = ((2, 10), (10, 30), (30, 100), (100, None))

SPEC = importlib.util.spec_from_file_location("slp11_split_metrics", DIAGNOSTIC_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load frozen split-half metric implementation")
metrics = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = metrics
SPEC.loader.exec_module(metrics)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def centered_metrics(
    sums: np.ndarray, cells: np.ndarray, selected: np.ndarray
) -> dict[str, object]:
    """Report independently query-centered A/B profiles on one fixed subset."""
    indices = np.flatnonzero(selected)
    support = cells[:, indices]
    result: dict[str, object] = {
        "genes": len(indices),
        "halfACells": int(support[0].sum()),
        "halfBCells": int(support[1].sum()),
        "halfAMedianCellsPerGene": (
            float(np.median(support[0])) if len(indices) else None
        ),
        "halfBMedianCellsPerGene": (
            float(np.median(support[1])) if len(indices) else None
        ),
    }
    if not len(indices):
        result["metrics"] = None
        return result
    means = sums[:, indices, :] / support[:, :, None]
    centered_a, centered_b = metrics._center_queries(means[0], means[1])
    result["metrics"] = metrics._metric_summary(centered_a, centered_b)
    return result


def make_report(source: np.lib.npyio.NpzFile) -> dict[str, object]:
    contexts = source["contexts"]
    sums = source["half_sum"]
    cells = source["half_num_cells"]
    per_context: list[dict[str, object]] = []
    for context_index, context in enumerate(contexts):
        total = cells[context_index].sum(axis=0)
        paired = np.all(cells[context_index] > 0, axis=0)
        strata: list[dict[str, object]] = []
        for lower, upper in BINS:
            selected = paired & (total >= lower)
            if upper is not None:
                selected &= total < upper
            strata.append(
                {
                    "cellCountInterval": f"[{lower},{upper})"
                    if upper
                    else f"[{lower},infinity)",
                    **centered_metrics(
                        sums[context_index], cells[context_index], selected
                    ),
                }
            )
        per_context.append({"context": str(context), "strata": strata})

    total = cells.sum(axis=1)
    shared_100 = np.all(total >= 100, axis=0) & np.all(cells > 0, axis=(0, 1))
    shared_contexts = [
        {
            "context": str(context),
            **centered_metrics(sums[index], cells[index], shared_100),
        }
        for index, context in enumerate(contexts)
    ]
    shared_indices = np.flatnonzero(shared_100)
    cross_environment: dict[str, object] | None = None
    if len(shared_indices):
        pooled = np.empty((2, len(shared_indices), sums.shape[-1]), dtype=np.float64)
        for context_index in range(2):
            selected_sums = sums[context_index][:, shared_indices, :]
            selected_cells = cells[context_index][:, shared_indices]
            pooled[context_index] = (
                selected_sums.sum(axis=0) / selected_cells.sum(axis=0)[:, None]
            )
        centered_a, centered_b = metrics._center_queries(pooled[0], pooled[1])
        cross_environment = metrics._metric_summary(centered_a, centered_b)
    return {
        "schema": "slp.yeast-fitting-split-half-cell-count-supplement/v1",
        "status": "descriptive-supplement-no-new-quantitative-access",
        "sourceSufficientStatisticsSha256": sha256(SOURCE_PATH),
        "originalReportSha256": sha256(ORIGINAL_REPORT),
        "bins": ["[2,10)", "[10,30)", "[30,100)", "[100,infinity)"],
        "binAssignment": "sum of frozen A+B positive-library cells per gene and environment",
        "centering": "recomputed independently within each bin/context as equal-gene per-query mean",
        "perContext": per_context,
        "sharedAtLeast100Cells": {
            "selection": "same stable genes with at least 100 pooled A+B cells in each environment",
            "genes": len(shared_indices),
            "perContext": shared_contexts,
            "crossEnvironmentIndependentlyQueryCentered": cross_environment,
        },
        "limits": [
            "No raw counts or development-validation outcomes were opened for this supplement.",
            "Bins were fixed before reading the saved sufficient statistics; no cohort or metric rule changed.",
            "Cell count is measurement support, not a fitted predictive feature.",
            "These correlations are technical split-half descriptions, not biological noise ceilings.",
        ],
    }


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError(f"refusing to overwrite {OUTPUT_PATH}")
    with np.load(SOURCE_PATH) as source:
        report = make_report(source)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
