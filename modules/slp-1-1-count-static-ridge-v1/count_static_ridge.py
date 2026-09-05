"""Control-anchored static ridge baseline for aggregate count means.

This numerical module consumes externally prepared control rates, per-gene
GEM-group cell weights and stable-ID static features.  It does not read data,
define gene identities, or access perturbation measurements at inference.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np


ALPHAS = ("0.1", "1", "10", "100", "1000", "10000", "100000", "1e+06", "mean-limit")


def global_gene_fold(gene: str, seed: int = 731) -> int:
    """Existing source-human global fitting fold contract."""
    digest = hashlib.sha256(
        f"slp11-bp-ridge-v1|{seed}|global-inner-fold|9606|{gene}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % 3


def control_anchor(control_rate: np.ndarray, gem_cell_count: np.ndarray) -> np.ndarray:
    """Return log1p of a gene's observed-GEM-weighted control CP10k rate."""
    rate = np.asarray(control_rate, dtype=np.float64)
    counts = np.asarray(gem_cell_count, dtype=np.float64)
    if rate.ndim != 2 or counts.ndim != 2 or counts.shape[1] != rate.shape[0]:
        raise ValueError("control_rate [B,Q] and gem_cell_count [G,B] required")
    if (
        not np.isfinite(rate).all()
        or not np.isfinite(counts).all()
        or np.any(rate <= 0)
        or np.any(counts < 0)
    ):
        raise ValueError("control rates must be positive and cell counts nonnegative")
    total = counts.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError("each gene requires at least one fitting cell")
    weighted = (counts / total[:, None]) @ rate
    return np.log1p(weighted)


def response_from_cp10k_moments(
    cp10k_sum: np.ndarray, cell_count: np.ndarray
) -> np.ndarray:
    """Return ln1p of the equal-cell gene-average CP10k expression."""
    sums = np.asarray(cp10k_sum, dtype=np.float64)
    counts = np.asarray(cell_count, dtype=np.float64)
    if sums.ndim != 2 or counts.shape != (len(sums),):
        raise ValueError("cp10k_sum [G,Q] and cell_count [G] required")
    if (
        not np.isfinite(sums).all()
        or not np.isfinite(counts).all()
        or np.any(sums < 0)
        or np.any(counts <= 0)
    ):
        raise ValueError("moments must be finite, nonnegative, and supported")
    return np.log1p(sums / counts[:, None])


def fit_feature_normalizer(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite nonempty feature matrix required")
    mean = values.mean(axis=0, dtype=np.float64)
    sd = values.std(axis=0, dtype=np.float64, ddof=0)
    scale = np.where(sd > 1e-5, sd, 1.0)
    return {
        "feature_mean": mean,
        "feature_sd": sd,
        "feature_scale": scale,
    }


def transform_features(values: np.ndarray, state: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = np.asarray(state["feature_mean"], dtype=np.float64)
    scale = np.asarray(state["feature_scale"], dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != mean.shape or scale.shape != mean.shape:
        raise ValueError("feature normalization shape mismatch")
    result = (values.astype(np.float64) - mean) / scale
    if not np.isfinite(result).all():
        raise ValueError("feature transformation produced nonfinite values")
    return result


def fit_state(features: np.ndarray, residual_target: np.ndarray) -> dict[str, np.ndarray]:
    """Fit a ridge eigensystem with an exactly unpenalized intercept."""
    target = np.asarray(residual_target, dtype=np.float64)
    if target.ndim != 2 or len(target) != len(features) or not np.isfinite(target).all():
        raise ValueError("finite residual targets [G,Q] required")
    normalizer = fit_feature_normalizer(features)
    design = transform_features(features, normalizer)
    design_mean = design.mean(axis=0, dtype=np.float64)
    centered_design = design - design_mean
    target_mean = target.mean(axis=0, dtype=np.float64)
    centered = target - target_mean
    gram = centered_design.T @ centered_design
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    keep = eigenvalues > 1e-8
    eigenvalues = eigenvalues[keep]
    eigenvectors = eigenvectors[:, keep]
    rotated = centered_design @ eigenvectors
    rhs = rotated.T @ centered
    return {
        **normalizer,
        "design_mean": design_mean,
        "target_mean": target_mean,
        "eigenvalues": eigenvalues.astype(np.float64),
        "eigenvectors": eigenvectors,
        "rhs": rhs,
    }


def predict_residual(
    state: dict[str, np.ndarray], features: np.ndarray, candidate: str
) -> np.ndarray:
    if candidate not in ALPHAS:
        raise ValueError("unknown ridge candidate")
    target_mean = np.asarray(state["target_mean"], dtype=np.float64)
    if candidate == "mean-limit":
        return np.broadcast_to(target_mean, (len(features), len(target_mean))).copy()
    design = transform_features(features, state) - np.asarray(
        state["design_mean"], dtype=np.float64
    )
    eigenvectors = np.asarray(state["eigenvectors"], dtype=np.float64)
    eigenvalues = np.asarray(state["eigenvalues"], dtype=np.float64)
    rhs = np.asarray(state["rhs"], dtype=np.float64)
    if eigenvectors.shape != (design.shape[1], len(eigenvalues)) or rhs.shape != (
        len(eigenvalues),
        len(target_mean),
    ):
        raise ValueError("ridge state shape mismatch")
    rotated = design @ eigenvectors
    prediction = target_mean + (
        rotated / (eigenvalues + float(candidate))
    ) @ rhs
    if not np.isfinite(prediction).all():
        raise ValueError("ridge prediction is nonfinite")
    return prediction


def absolute_prediction(anchor: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Add anchor and residual only after promotion to float64."""
    anchor64 = np.asarray(anchor, dtype=np.float64)
    residual64 = np.asarray(residual, dtype=np.float64)
    if anchor64.shape != residual64.shape:
        raise ValueError("anchor and residual predictions must align")
    result = anchor64 + residual64
    if not np.isfinite(result).all():
        raise ValueError("absolute prediction is nonfinite")
    return result


def candidate_mse(
    state: dict[str, np.ndarray], features: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    """Score all ridge candidates with one set of multi-output moments."""
    target64 = np.asarray(target, dtype=np.float64)
    target_mean = np.asarray(state["target_mean"], dtype=np.float64)
    if target64.ndim != 2 or target64.shape[1:] != target_mean.shape:
        raise ValueError("held targets do not match ridge output")
    design = transform_features(features, state) - np.asarray(
        state["design_mean"], dtype=np.float64
    )
    vectors = np.asarray(state["eigenvectors"], dtype=np.float64)
    values = np.asarray(state["eigenvalues"], dtype=np.float64)
    rhs = np.asarray(state["rhs"], dtype=np.float64)
    rotated = design @ vectors
    centered = target64 - target_mean
    base = float(np.square(centered).sum(dtype=np.float64))
    cross = rotated.T @ centered
    linear = np.sum(cross * rhs, axis=1, dtype=np.float64)
    quadratic = (rotated.T @ rotated) * (rhs @ rhs.T)
    denominator = float(target64.size)
    result = {"mean-limit": base / denominator}
    for candidate in ALPHAS[:-1]:
        shrink = 1.0 / (values + float(candidate))
        sse = base - 2.0 * float(shrink @ linear) + float(
            shrink @ quadratic @ shrink
        )
        tolerance = 64 * np.finfo(np.float64).eps * max(1.0, base)
        if sse < -tolerance or not np.isfinite(sse):
            raise ValueError("invalid ridge SSE expansion")
        result[candidate] = max(0.0, sse) / denominator
    return {candidate: result[candidate] for candidate in ALPHAS}


def choose_alpha(
    genes: np.ndarray, features: np.ndarray, target: np.ndarray, seed: int = 731
) -> tuple[str, dict[str, float], list[dict[str, object]]]:
    """Choose one alpha by three global-gene folds and raw all-query MSE."""
    genes = np.asarray(genes).astype(str)
    if len(genes) != len(set(genes.tolist())) or len(genes) != len(features):
        raise ValueError("unique gene-aligned rows required")
    folds = np.asarray([global_gene_fold(gene, seed) for gene in genes], np.int8)
    totals = {candidate: 0.0 for candidate in ALPHAS}
    reports: list[dict[str, object]] = []
    for fold in range(3):
        fitting = folds != fold
        held = folds == fold
        if not fitting.any() or not held.any():
            raise ValueError("each inner fold requires fitting and held genes")
        state = fit_state(features[fitting], target[fitting])
        scores = candidate_mse(state, features[held], target[held])
        for candidate, score in scores.items():
            totals[candidate] += score * int(held.sum())
        reports.append(
            {
                "fold": fold,
                "fittingGenes": int(fitting.sum()),
                "heldGenes": int(held.sum()),
                "rawAllQueryMse": scores,
                "featureMeanSha256": hashlib.sha256(
                    state["feature_mean"].tobytes()
                ).hexdigest(),
                "featureScaleSha256": hashlib.sha256(
                    state["feature_scale"].tobytes()
                ).hexdigest(),
            }
        )
    mean_scores = {candidate: value / len(genes) for candidate, value in totals.items()}
    selected = min(ALPHAS, key=lambda item: (mean_scores[item], ALPHAS.index(item)))
    return selected, mean_scores, reports


def independently_query_center(values: np.ndarray) -> np.ndarray:
    """Center columns after an exact first-row anchor to reduce cancellation."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite nonempty profiles required")
    anchored = values - values[:1]
    return anchored - anchored.mean(axis=0, dtype=np.float64)


def profile_pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("finite aligned vectors required")
    x, y = left - left.mean(), right - right.mean()
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    eps = np.finfo(np.float64).eps
    tx = 8 * eps * math.sqrt(x.size) * max(1.0, float(np.max(np.abs(left))))
    ty = 8 * eps * math.sqrt(y.size) * max(1.0, float(np.max(np.abs(right))))
    if nx <= tx or ny <= ty:
        return None
    return float(x @ y / (nx * ny))


def centered_landscape_score(
    truth: np.ndarray, prediction: np.ndarray, anchor: np.ndarray
) -> dict[str, float | int | None]:
    """Score gene profiles after removing the same measured-control anchor."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    if truth.shape != prediction.shape or truth.shape != anchor.shape:
        raise ValueError("truth, prediction and anchor must align")
    truth_centered = independently_query_center(truth - anchor)
    prediction_centered = independently_query_center(prediction - anchor)
    correlations = [
        profile_pearson(left, right)
        for left, right in zip(truth_centered, prediction_centered, strict=True)
    ]
    finite = [value for value in correlations if value is not None]
    return {
        "rawGeneProfileMse": float(np.mean(np.square(truth - prediction))),
        "independentlyQueryCenteredResidualPearson": (
            float(np.mean(finite)) if finite else None
        ),
        "finiteCorrelationGenes": len(finite),
        "undefinedCorrelationGenes": len(correlations) - len(finite),
    }
