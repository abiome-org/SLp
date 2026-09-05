"""Application-neutral aggregate molecular forecast evaluation numerics."""

from __future__ import annotations

import math

import numpy as np


Array = np.ndarray


PREDICTION_KEYS = (
    "control_prediction",
    "anchored_mean_prediction",
    "static_ridge_prediction",
    "k562_only_prediction",
    "joint_prediction",
)


def validate_forecast_arrays(arrays: dict[str, Array]) -> tuple[int, int, int]:
    required = {
        "gene_ids", "query_ids", "cell_count", "gem_group_ids", "gem_cell_count",
        *PREDICTION_KEYS,
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"forecast bundle missing keys: {sorted(missing)}")
    genes = np.asarray(arrays["gene_ids"]).astype(str)
    queries = np.asarray(arrays["query_ids"]).astype(str)
    cells = np.asarray(arrays["cell_count"])
    gems = np.asarray(arrays["gem_group_ids"])
    gem_cells = np.asarray(arrays["gem_cell_count"])
    if genes.ndim != 1 or queries.ndim != 1 or gems.ndim != 1 or min(len(genes), len(queries), len(gems)) <= 0:
        raise ValueError("identity axes must be nonempty vectors")
    if len(set(genes.tolist())) != len(genes) or len(set(queries.tolist())) != len(queries):
        raise ValueError("gene and query identifiers must be unique")
    if list(genes) != sorted(genes.tolist()):
        raise ValueError("gene identifiers must be ascending")
    if cells.shape != (len(genes),) or gem_cells.shape != (len(genes), len(gems)):
        raise ValueError("cell-count arrays do not align")
    if not np.issubdtype(cells.dtype, np.integer) or not np.issubdtype(gem_cells.dtype, np.integer):
        raise ValueError("cell counts must be integers")
    if np.any(cells <= 0) or np.any(gem_cells < 0) or not np.array_equal(gem_cells.sum(1), cells):
        raise ValueError("cell counts must be positive and close exactly over GEMs")
    for key in PREDICTION_KEYS:
        values = np.asarray(arrays[key])
        if values.shape != (len(genes), len(queries)) or not np.isfinite(values).all():
            raise ValueError(f"{key} must be finite [G,Q]")
    return len(genes), len(queries), len(gems)


def control_prediction(basal_rate: Array, gem_cell_count: Array) -> Array:
    """GEM-composition-matched log1p control mean."""
    rate = np.asarray(basal_rate, dtype=np.float64)
    weight = np.asarray(gem_cell_count, dtype=np.float64)
    if rate.ndim != 2 or weight.ndim != 2 or weight.shape[1] != rate.shape[0]:
        raise ValueError("basal rates [C,Q] and GEM counts [G,C] required")
    if not np.isfinite(rate).all() or np.any(rate < 0) or not np.isfinite(weight).all() or np.any(weight < 0):
        raise ValueError("control inputs must be finite nonnegative")
    total = weight.sum(1)
    if np.any(total <= 0):
        raise ValueError("each gene requires positive GEM support")
    return np.log1p((weight / total[:, None]) @ rate)


def accumulate_cp10k(
    sums: Array,
    cell_count: Array,
    raw_counts: Array,
    library: Array,
    gene_index: Array,
) -> None:
    """Add exact per-cell CP10k profiles to fixed gene rows."""
    output = np.asarray(sums)
    counts = np.asarray(cell_count)
    raw = np.asarray(raw_counts)
    exposure = np.asarray(library, dtype=np.float64)
    index = np.asarray(gene_index, dtype=np.int64)
    if output.ndim != 2 or counts.shape != (len(output),) or raw.ndim != 2 or raw.shape[1] != output.shape[1]:
        raise ValueError("aggregation shape mismatch")
    if exposure.shape != (len(raw),) or index.shape != exposure.shape:
        raise ValueError("row metadata shape mismatch")
    if not np.isfinite(raw).all() or np.any(raw < 0) or not np.isfinite(exposure).all() or np.any(exposure <= 0):
        raise ValueError("counts and libraries must be finite nonnegative/positive")
    if index.size and (index.min() < 0 or index.max() >= len(output)):
        raise ValueError("gene index out of range")
    profile = np.asarray(raw, dtype=np.float64) * (10000.0 / exposure[:, None])
    for gene in np.unique(index):
        take = index == gene
        output[gene] += profile[take].sum(axis=0, dtype=np.float64)
        counts[gene] += int(take.sum())


def aggregate_truth(cp10k_sum: Array, cell_count: Array) -> Array:
    values = np.asarray(cp10k_sum, dtype=np.float64)
    count = np.asarray(cell_count, dtype=np.int64)
    if values.ndim != 2 or count.shape != (len(values),) or np.any(count <= 0):
        raise ValueError("positive aligned gene counts required")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("CP10k sums must be finite nonnegative")
    return np.log1p(values / count[:, None])


def independently_query_center(values: Array) -> Array:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("finite nonempty [G,Q] profiles required")
    anchored = values - values[:1]
    return anchored - anchored.mean(axis=0, dtype=np.float64)


def row_pearson(left: Array, right: Array) -> tuple[Array, Array]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("finite aligned profile matrices required")
    x = x - x.mean(1, keepdims=True)
    y = y - y.mean(1, keepdims=True)
    nx = np.linalg.norm(x, axis=1)
    ny = np.linalg.norm(y, axis=1)
    scale_x = np.maximum(1.0, np.max(np.abs(left), axis=1))
    scale_y = np.maximum(1.0, np.max(np.abs(right), axis=1))
    tolerance = 8.0 * np.finfo(np.float64).eps * math.sqrt(x.shape[1])
    defined = (nx > tolerance * scale_x) & (ny > tolerance * scale_y)
    correlation = np.full(len(x), np.nan, dtype=np.float64)
    correlation[defined] = np.einsum("ij,ij->i", x[defined], y[defined]) / (nx[defined] * ny[defined])
    return np.clip(correlation, -1.0, 1.0), defined


def score_prediction(truth: Array, prediction: Array, control: Array) -> tuple[dict[str, float | int | None], Array, Array]:
    target = np.asarray(truth, dtype=np.float64)
    forecast = np.asarray(prediction, dtype=np.float64)
    anchor = np.asarray(control, dtype=np.float64)
    if target.shape != forecast.shape or target.shape != anchor.shape or target.ndim != 2:
        raise ValueError("truth, prediction and control must align [G,Q]")
    if not np.isfinite(target).all() or not np.isfinite(forecast).all() or not np.isfinite(anchor).all():
        raise ValueError("profiles must be finite")
    per_gene_mse = np.mean(np.square(target - forecast), axis=1, dtype=np.float64)
    truth_centered = independently_query_center(target - anchor)
    prediction_centered = independently_query_center(forecast - anchor)
    correlation, defined = row_pearson(truth_centered, prediction_centered)
    report = {
        "geneProfileMse": float(per_gene_mse.mean()),
        "independentlyQueryCenteredResidualPearson": float(correlation[defined].mean()) if defined.any() else None,
        "finiteCorrelationGenes": int(defined.sum()),
        "undefinedCorrelationGenes": int((~defined).sum()),
        "genes": int(target.shape[0]),
        "queries": int(target.shape[1]),
    }
    return report, per_gene_mse, correlation


def advancement(metrics: dict[str, dict[str, dict[str, float | int | None]]]) -> dict[str, object]:
    """Apply the fixed two-source joint forecast rule."""
    checks: dict[str, bool] = {}
    for source in ("k562", "rpe1"):
        source_metrics = metrics[source]
        joint = source_metrics["joint_prediction"]
        mean = source_metrics["anchored_mean_prediction"]
        ridge = source_metrics["static_ridge_prediction"]
        jm = float(joint["geneProfileMse"])
        jr = joint["independentlyQueryCenteredResidualPearson"]
        rr = ridge["independentlyQueryCenteredResidualPearson"]
        checks[f"{source}JointMseOnePercentBelowMean"] = jm <= 0.99 * float(mean["geneProfileMse"])
        checks[f"{source}JointMseOnePercentBelowRidge"] = jm <= 0.99 * float(ridge["geneProfileMse"])
        checks[f"{source}JointCorrelationAtLeastPoint10"] = jr is not None and float(jr) >= 0.10
        checks[f"{source}JointCorrelationNonregressionVsRidge"] = jr is not None and rr is not None and float(jr) >= float(rr)
    checks["k562JointMseOnePercentBelowK562Only"] = (
        float(metrics["k562"]["joint_prediction"]["geneProfileMse"])
        <= 0.99 * float(metrics["k562"]["k562_only_prediction"]["geneProfileMse"])
    )
    return {"checks": checks, "passes": bool(all(checks.values()))}
