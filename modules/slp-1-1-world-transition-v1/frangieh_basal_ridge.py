"""Context-local dual ridge utilities for Frangieh paired-development data."""

from __future__ import annotations

import hashlib

import numpy as np


def cv_fold(action_id: str, seed: int = 731, folds: int = 3) -> int:
    payload = f"slp11-frangieh-ridge-cv-v1|{seed}|9606|{action_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % folds


def development_split(action_id: str, seed: int = 731) -> str:
    payload = f"slp11-development-v1|{seed}|9606|{action_id}".encode()
    bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    raise ValueError(f"test-split action is forbidden in development input: {action_id}")


def collapse_gene_profiles(
    action_ids: np.ndarray,
    context_ids: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average guide-level pseudobulks equally for each gene and context."""
    action_ids = np.asarray(action_ids, dtype=str)
    context_ids = np.asarray(context_ids, dtype=str)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[0] != len(action_ids) or len(action_ids) != len(context_ids):
        raise ValueError("record axes do not align")
    keys = sorted(set(zip(action_ids, context_ids, strict=True)))
    action_out, context_out, target_out, counts = [], [], [], []
    for action, context in keys:
        rows = (action_ids == action) & (context_ids == context)
        action_out.append(action)
        context_out.append(context)
        target_out.append(np.mean(targets[rows], axis=0))
        counts.append(int(np.sum(rows)))
    return (
        np.asarray(action_out),
        np.asarray(context_out),
        np.asarray(target_out, dtype=np.float32),
        np.asarray(counts, dtype=np.int64),
    )


def _standardize_fit(values: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0, dtype=np.float64)
    scale = np.std(values, axis=0, ddof=0, dtype=np.float64)
    scale = np.maximum(scale, floor)
    return mean, scale


def dual_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    alpha: float,
    *,
    target_scale_floor: float = 0.05,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit standardized multi-output ridge through the sample-space kernel."""
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    x_eval = np.asarray(x_eval, dtype=np.float64)
    x_mean, x_scale = _standardize_fit(x_train, 1e-8)
    y_mean, y_scale = _standardize_fit(y_train, target_scale_floor)
    x_fit = (x_train - x_mean) / x_scale
    x_query = (x_eval - x_mean) / x_scale
    y_fit = (y_train - y_mean) / y_scale
    kernel = x_fit @ x_fit.T
    weights = np.linalg.solve(kernel + float(alpha) * np.eye(len(x_fit)), y_fit)
    prediction = (x_query @ x_fit.T @ weights) * y_scale + y_mean
    return prediction.astype(np.float32), {
        "feature_mean": x_mean,
        "feature_scale": x_scale,
        "target_mean": y_mean,
        "target_scale": y_scale,
    }


def mean_limit_predict(
    y_train: np.ndarray, n_eval: int, *, target_scale_floor: float = 0.05
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return the exact infinite-regularization limit and fitting target scale."""
    y_train = np.asarray(y_train, dtype=np.float64)
    if y_train.ndim != 2 or n_eval < 0:
        raise ValueError("invalid target array or evaluation size")
    target_mean, target_scale = _standardize_fit(y_train, target_scale_floor)
    prediction = np.broadcast_to(target_mean, (n_eval, y_train.shape[1])).astype(np.float32)
    return prediction, {"target_mean": target_mean, "target_scale": target_scale}


def query_centroid_adjusted_profile_pearson(
    prediction: np.ndarray, truth: np.ndarray
) -> tuple[float, np.ndarray]:
    """Remove query centroids across genes, then correlate row-centered profiles."""
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("prediction and truth must be aligned two-dimensional arrays")
    p = prediction - prediction.mean(axis=0, keepdims=True)
    t = truth - truth.mean(axis=0, keepdims=True)
    p -= p.mean(axis=1, keepdims=True)
    t -= t.mean(axis=1, keepdims=True)
    p_norm = np.sqrt(np.sum(p * p, axis=1))
    t_norm = np.sqrt(np.sum(t * t, axis=1))
    root_q = np.sqrt(prediction.shape[1])
    p_tolerance = 32 * np.finfo(np.float64).eps * np.maximum(1.0, np.max(np.abs(prediction), axis=1)) * root_q
    t_tolerance = 32 * np.finfo(np.float64).eps * np.maximum(1.0, np.max(np.abs(truth), axis=1)) * root_q
    valid = (p_norm > p_tolerance) & (t_norm > t_tolerance)
    denominator = p_norm * t_norm
    values = np.divide(
        np.sum(p * t, axis=1), denominator, out=np.full(len(p), np.nan), where=valid
    )
    score = float(np.mean(values[valid])) if np.any(valid) else float("nan")
    return score, values.astype(np.float32)


def ordinary_pearson(prediction: np.ndarray, truth: np.ndarray) -> float:
    p = np.asarray(prediction, dtype=np.float64).ravel()
    t = np.asarray(truth, dtype=np.float64).ravel()
    p -= p.mean()
    t -= t.mean()
    denominator = np.sqrt(np.sum(p * p) * np.sum(t * t))
    return float(np.sum(p * t) / denominator) if denominator > 0 else float("nan")


def metrics(prediction: np.ndarray, truth: np.ndarray, train_scale: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    adjusted_r, per_gene_r = query_centroid_adjusted_profile_pearson(prediction, truth)
    return {
        "raw_mse": float(np.mean(error * error)),
        "scaled_mse": float(np.mean((error / np.asarray(train_scale)) ** 2)),
        "query_centroid_adjusted_profile_pearson": adjusted_r,
        "query_centroid_adjusted_profile_pearson_undefined_genes": int(np.sum(~np.isfinite(per_gene_r))),
        "ordinary_pearson": ordinary_pearson(prediction, truth),
    }
