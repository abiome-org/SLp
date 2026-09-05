"""Faithful SLIM bilinear algebra adapted to SLp native residual panels."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SlimNativeModel:
    query_basis: np.ndarray
    weight: np.ndarray
    bias: np.ndarray
    rank: int
    lambda_reg: float

    def predict_residual(self, intervention_features: np.ndarray) -> np.ndarray:
        features = np.asarray(intervention_features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.weight.shape[1]:
            raise ValueError("intervention features do not match fitted SLIM operator")
        if not np.isfinite(features).all():
            raise ValueError("intervention features must be finite")
        prediction = (self.query_basis @ self.weight @ features.T + self.bias).T
        if not np.isfinite(prediction).all():
            raise ValueError("SLIM prediction is nonfinite")
        return prediction


def pca_basis(response_by_query: np.ndarray, rank: int) -> np.ndarray:
    """Match sklearn PCA.fit_transform used by official SLIM."""
    matrix = np.asarray(response_by_query, dtype=np.float64)
    if matrix.ndim != 2 or rank <= 0 or not np.isfinite(matrix).all():
        raise ValueError("invalid PCA basis input")
    components = min(rank, matrix.shape[0], matrix.shape[1] - 1)
    if components <= 0:
        raise ValueError("insufficient perturbations for a SLIM basis")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    return u[:, :components] * singular[:components]


def solve_weight(
    query_basis: np.ndarray,
    intervention_features: np.ndarray,
    centered_response_by_query: np.ndarray,
    lambda_reg: float,
) -> np.ndarray:
    """Official SLIM closed form: (G'G+lI)^-1 G'Y P (P'P+lI)^-1."""
    g = np.asarray(query_basis, dtype=np.float64)
    p = np.asarray(intervention_features, dtype=np.float64)
    y = np.asarray(centered_response_by_query, dtype=np.float64)
    if lambda_reg <= 0 or y.shape != (g.shape[0], p.shape[0]):
        raise ValueError("invalid SLIM solve matrices")
    left = np.linalg.solve(g.T @ g + lambda_reg * np.eye(g.shape[1]), g.T @ y)
    return np.linalg.solve(
        p.T @ p + lambda_reg * np.eye(p.shape[1]), (left @ p).T
    ).T


def fit(
    intervention_features: np.ndarray,
    residual_targets: np.ndarray,
    *,
    rank: int = 10,
    lambda_reg: float = 0.1,
) -> SlimNativeModel:
    """Fit official single-perturbation SLIM to control-anchored residuals."""
    p = np.asarray(intervention_features, dtype=np.float64)
    targets = np.asarray(residual_targets, dtype=np.float64)
    if (
        p.ndim != 2
        or targets.ndim != 2
        or p.shape[0] != targets.shape[0]
        or not np.isfinite(p).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("invalid SLIM fitting matrices")
    y = targets.T
    bias = y.mean(axis=1, keepdims=True)
    centered = y - bias
    basis = pca_basis(centered, rank)
    weight = solve_weight(basis, p, centered, lambda_reg)
    return SlimNativeModel(basis, weight, bias, basis.shape[1], float(lambda_reg))
