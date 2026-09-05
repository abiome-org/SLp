"""Ridge utilities for an explicitly action-aligned basal scalar.

The scalar and its observation indicator are caller-supplied control features.
Stable gene identity, splits, biological contexts and target access remain
outside this application-neutral numerical module.
"""

from __future__ import annotations

import math

import numpy as np


def fit_design_normalizer(
    static: np.ndarray,
    basal: np.ndarray,
    basal_observed: np.ndarray,
) -> dict[str, np.ndarray]:
    static = np.asarray(static, dtype=np.float32)
    basal = np.asarray(basal, dtype=np.float32)
    observed = np.asarray(basal_observed, dtype=bool)
    if static.ndim != 2 or basal.shape != (len(static),) or observed.shape != basal.shape:
        raise ValueError("static and basal fitting arrays do not align")
    if not np.isfinite(static).all() or not np.isfinite(basal[observed]).all():
        raise ValueError("observed fitting features must be finite")
    if not observed.any():
        raise ValueError("at least one fitting action needs measured basal abundance")
    static_mean = static.mean(axis=0, dtype=np.float64).astype(np.float32)
    static_scale = static.std(axis=0, dtype=np.float64).astype(np.float32)
    static_scale = np.where(static_scale > 1e-5, static_scale, 1.0).astype(np.float32)
    basal_mean = np.asarray(basal[observed].mean(dtype=np.float64), dtype=np.float32)
    basal_scale = np.asarray(basal[observed].std(dtype=np.float64), dtype=np.float32)
    if basal_scale <= 1e-5:
        basal_scale = np.asarray(1.0, dtype=np.float32)
    return {
        "static_mean": static_mean,
        "static_scale": static_scale,
        "basal_mean": basal_mean,
        "basal_scale": basal_scale,
    }


def transform_design(
    static: np.ndarray,
    basal: np.ndarray,
    basal_observed: np.ndarray,
    normalizer: dict[str, np.ndarray],
    *,
    include_basal_value: bool,
) -> np.ndarray:
    static = np.asarray(static, dtype=np.float32)
    basal = np.asarray(basal, dtype=np.float32)
    observed = np.asarray(basal_observed, dtype=bool)
    if static.ndim != 2 or basal.shape != (len(static),) or observed.shape != basal.shape:
        raise ValueError("static and basal arrays do not align")
    standardized = (static - normalizer["static_mean"]) / normalizer["static_scale"]
    scalar = np.zeros(len(static), dtype=np.float32)
    if include_basal_value:
        scalar[observed] = (
            basal[observed] - normalizer["basal_mean"]
        ) / normalizer["basal_scale"]
    return np.column_stack((standardized, scalar, observed.astype(np.float32))).astype(np.float32)


def fit_state(design: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    design = np.asarray(design, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if design.ndim != 2 or targets.ndim != 2 or len(design) != len(targets):
        raise ValueError("design and targets do not align")
    if not np.isfinite(design).all() or not np.isfinite(targets).all():
        raise ValueError("ridge fitting arrays must be finite")
    target_mean = targets.mean(axis=0, dtype=np.float64).astype(np.float32)
    gram = (design.T @ design).astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    keep = eigenvalues > 1e-8
    eigenvalues = eigenvalues[keep]
    eigenvectors = eigenvectors[:, keep].astype(np.float32)
    rotated = design @ eigenvectors
    rhs = rotated.T @ (targets - target_mean)
    return {
        "target_mean": target_mean,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "rhs": rhs.astype(np.float32),
    }


def predict_state(state: dict[str, np.ndarray], design: np.ndarray, candidate: str) -> np.ndarray:
    design = np.asarray(design, dtype=np.float32)
    if candidate == "mean-limit":
        return np.broadcast_to(state["target_mean"], (len(design), len(state["target_mean"]))).copy()
    rotated = design @ state["eigenvectors"]
    return (
        state["target_mean"]
        + (rotated / (state["eigenvalues"] + float(candidate))) @ state["rhs"]
    ).astype(np.float32)


def profile_pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    left_norm = np.linalg.norm(left_centered)
    right_norm = np.linalg.norm(right_centered)
    eps = np.finfo(np.float64).eps
    left_tol = 8 * eps * math.sqrt(left.size) * max(1.0, float(np.max(np.abs(left))))
    right_tol = 8 * eps * math.sqrt(right.size) * max(1.0, float(np.max(np.abs(right))))
    if left_norm <= left_tol or right_norm <= right_tol:
        return None
    return float((left_centered @ right_centered) / (left_norm * right_norm))


def independently_query_center(values: np.ndarray) -> np.ndarray:
    """Center columns after an exact first-row anchor to avoid cancellation."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("profiles must be a nonempty matrix")
    anchored = values - values[:1]
    return anchored - anchored.mean(axis=0, dtype=np.float64)
