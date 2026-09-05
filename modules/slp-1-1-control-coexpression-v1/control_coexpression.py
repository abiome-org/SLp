"""Target-free control-cell coexpression fingerprints.

This module contains only numerical operations. Dataset routing, identifiers, and
artifact I/O belong to the caller so the admitted module stays application neutral.
"""

from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np


Array = np.ndarray


def normalize_static_float32(
    values: Array, mean: Array, scale: Array, clip: float
) -> Array:
    """Replay the original count-model float32 static-feature transform."""
    x = np.asarray(values, dtype=np.float32)
    m = np.asarray(mean, dtype=np.float32)
    s = np.asarray(scale, dtype=np.float32)
    if x.ndim != 2 or m.shape != (x.shape[1],) or s.shape != m.shape:
        raise ValueError("static feature/normalizer shape mismatch")
    if not np.isfinite(m).all() or not np.isfinite(s).all() or np.any(s <= 0):
        raise ValueError("invalid static normalizer")
    safe = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip((safe - m) / s, -float(clip), float(clip)).astype(
        np.float32, copy=False
    )


def fixed_anchor_weights(
    normalized_static: Array, dimensions: int = 64, seed: int = 731
) -> tuple[Array, Array]:
    """Create centered, column-unit-norm Gaussian projections.

    Returns the anchor weights [genes, dimensions] and the uncentered Gaussian
    projection matrix [features, dimensions]. All calculations use float64 after
    the caller's frozen float32 normalization.
    """
    x = np.asarray(normalized_static)
    if x.ndim != 2 or dimensions <= 0 or not np.isfinite(x).all():
        raise ValueError("invalid normalized static features")
    gaussian = np.random.default_rng(int(seed)).standard_normal(
        (x.shape[1], int(dimensions))
    ) / np.sqrt(float(x.shape[1]))
    weights = np.asarray(x, dtype=np.float64) @ gaussian
    weights -= weights.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(weights, axis=0)
    if np.any(norm <= 0) or not np.isfinite(norm).all():
        raise ValueError("degenerate anchor projection")
    weights /= norm[None, :]
    return weights, gaussian


def barcode_half(barcode: str, seed: int = 731) -> int:
    """Deterministic control-cell half assignment."""
    payload = f"slp11-control-coexpression-v1|{int(seed)}|{barcode}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2


def empty_first_pass(groups: int, queries: int) -> dict[str, Array]:
    if groups <= 0 or queries <= 0:
        raise ValueError("groups and queries must be positive")
    return {
        "count": np.zeros(groups, dtype=np.int64),
        "sum_l": np.zeros(groups, dtype=np.float64),
        "sum_l2": np.zeros(groups, dtype=np.float64),
        "sum_x": np.zeros((groups, queries), dtype=np.float64),
        "sum_lx": np.zeros((groups, queries), dtype=np.float64),
    }


def update_first_pass(stats: dict[str, Array], x: Array, log_library: Array, group: Array) -> None:
    """Accumulate sufficient statistics for within-group affine regression."""
    values = np.asarray(x, dtype=np.float64)
    library = np.asarray(log_library, dtype=np.float64)
    group_index = np.asarray(group, dtype=np.int64)
    if values.ndim != 2 or library.shape != (len(values),) or group_index.shape != library.shape:
        raise ValueError("first-pass input shape mismatch")
    if not np.isfinite(values).all() or not np.isfinite(library).all():
        raise ValueError("nonfinite first-pass inputs")
    if len(values) == 0:
        return
    groups = len(stats["count"])
    if group_index.min() < 0 or group_index.max() >= groups:
        raise ValueError("group index out of range")
    for g in np.unique(group_index):
        take = group_index == g
        xv = values[take]
        lv = library[take]
        stats["count"][g] += len(lv)
        stats["sum_l"][g] += lv.sum(dtype=np.float64)
        stats["sum_l2"][g] += np.dot(lv, lv)
        stats["sum_x"][g] += xv.sum(axis=0, dtype=np.float64)
        stats["sum_lx"][g] += xv.T @ lv


def regression_parameters(stats: Mapping[str, Array]) -> tuple[Array, Array, Array]:
    """Return per-group query mean, centered-library slope, and library mean."""
    count = np.asarray(stats["count"], dtype=np.int64)
    sum_l = np.asarray(stats["sum_l"], dtype=np.float64)
    sum_l2 = np.asarray(stats["sum_l2"], dtype=np.float64)
    sum_x = np.asarray(stats["sum_x"], dtype=np.float64)
    sum_lx = np.asarray(stats["sum_lx"], dtype=np.float64)
    if np.any(count <= 0):
        raise ValueError("every group must contain cells")
    mean_l = sum_l / count
    mean_x = sum_x / count[:, None]
    denominator = sum_l2 - sum_l * mean_l
    numerator = sum_lx - sum_x * mean_l[:, None]
    tolerance = 64.0 * np.finfo(np.float64).eps * np.maximum(sum_l2, 1.0)
    slope = np.zeros_like(sum_x)
    good = denominator > tolerance
    slope[good] = numerator[good] / denominator[good, None]
    return mean_x, slope, mean_l


def residualize(
    x: Array,
    log_library: Array,
    group: Array,
    mean_x: Array,
    slope: Array,
    mean_library: Array,
) -> Array:
    values = np.asarray(x, dtype=np.float64)
    library = np.asarray(log_library, dtype=np.float64)
    group_index = np.asarray(group, dtype=np.int64)
    return values - mean_x[group_index] - (
        library - mean_library[group_index]
    )[:, None] * slope[group_index]


def empty_second_pass(queries: int, dimensions: int) -> dict[str, Array]:
    return {
        "cross": np.zeros((queries, dimensions), dtype=np.float64),
        "var_x": np.zeros(queries, dtype=np.float64),
        "var_z": np.zeros(dimensions, dtype=np.float64),
    }


def update_second_pass(
    moments: dict[str, Array], residual: Array, common_index: Array, anchor_weights: Array
) -> None:
    """Accumulate X'Z, diagonal X'X, and diagonal Z'Z."""
    r = np.asarray(residual, dtype=np.float64)
    common = np.asarray(common_index, dtype=np.int64)
    weights = np.asarray(anchor_weights, dtype=np.float64)
    if r.ndim != 2 or weights.shape[0] != len(common):
        raise ValueError("second-pass input shape mismatch")
    z = r[:, common] @ weights
    moments["cross"] += r.T @ z
    moments["var_x"] += np.einsum("ij,ij->j", r, r, optimize=True)
    moments["var_z"] += np.einsum("ij,ij->j", z, z, optimize=True)


def fingerprints_from_moments(
    cross: Array, var_x: Array, var_z: Array, native_anchor_weights: Array
) -> tuple[Array, Array, Array]:
    """Compute leave-self-out correlations and explicit per-coordinate presence."""
    cross = np.asarray(cross, dtype=np.float64)
    var_x = np.asarray(var_x, dtype=np.float64)
    var_z = np.asarray(var_z, dtype=np.float64)
    weight = np.asarray(native_anchor_weights, dtype=np.float64)
    if cross.shape != weight.shape or var_x.shape != (cross.shape[0],) or var_z.shape != (cross.shape[1],):
        raise ValueError("moment/weight shape mismatch")
    numerator = cross - weight * var_x[:, None]
    leave_var = var_z[None, :] - 2.0 * weight * cross + weight * weight * var_x[:, None]
    roundoff = 64.0 * np.finfo(np.float64).eps * np.maximum.reduce(
        [
            np.ones_like(leave_var),
            np.broadcast_to(np.abs(var_z)[None, :], leave_var.shape),
            np.abs(2.0 * weight * cross),
            np.abs(weight * weight * var_x[:, None]),
        ]
    )
    if np.any(leave_var < -roundoff):
        raise ValueError("negative leave-anchor variance beyond roundoff")
    leave_var = np.maximum(leave_var, 0.0)
    x_tolerance = 64.0 * np.finfo(np.float64).eps * max(float(var_x.max(initial=0.0)), 1.0)
    present = (var_x[:, None] > x_tolerance) & (leave_var > roundoff)
    denominator = np.sqrt(np.maximum(var_x[:, None] * leave_var, 0.0))
    feature = np.zeros_like(cross)
    feature[present] = numerator[present] / denominator[present]
    feature = np.clip(feature, -1.0, 1.0)
    if not np.isfinite(feature).all():
        raise ValueError("nonfinite fingerprints")
    return feature, present, leave_var


def split_half_cosine(a: Array, b: Array) -> tuple[Array, Array]:
    """Per-query cosine and defined mask for two fingerprint reconstructions."""
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("split-half shape mismatch")
    norm_a = np.linalg.norm(left, axis=1)
    norm_b = np.linalg.norm(right, axis=1)
    tolerance = 64.0 * np.finfo(np.float64).eps * np.sqrt(left.shape[1])
    defined = (norm_a > tolerance) & (norm_b > tolerance)
    cosine = np.zeros(len(left), dtype=np.float64)
    cosine[defined] = np.einsum("ij,ij->i", left[defined], right[defined]) / (
        norm_a[defined] * norm_b[defined]
    )
    return np.clip(cosine, -1.0, 1.0), defined
