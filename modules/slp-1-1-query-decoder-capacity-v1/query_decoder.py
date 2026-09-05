"""Fitting-only query-feature decoder capacity diagnostic numerics."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class QueryDecoder(nn.Module):
    """Map normalized static query features to supervised response coordinates."""

    def __init__(self, feature_dim: int = 577, hidden_dim: int = 256, output_dim: int = 32):
        super().__init__()
        if min(feature_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("decoder dimensions must be positive")
        self.hidden = nn.Linear(feature_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.linear_residual = nn.Linear(feature_dim, output_dim, bias=False)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.linear_residual.weight)

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        if query_features.ndim != 2 or query_features.shape[1] != self.hidden.in_features:
            raise ValueError("query features must be [Q,F]")
        if not torch.isfinite(query_features).all():
            raise ValueError("query features must be finite")
        return self.output(torch.nn.functional.gelu(self.hidden(query_features))) + self.linear_residual(query_features)


def rank32_factors(state: dict[str, np.ndarray], alpha: float = 1000.0, rank: int = 32) -> dict[str, np.ndarray]:
    """Return the exact regularized rank-constrained response factorization."""
    values = np.asarray(state["eigenvalues"], dtype=np.float64)
    vectors = np.asarray(state["eigenvectors"], dtype=np.float64)
    rhs = np.asarray(state["rhs"], dtype=np.float64)
    if values.ndim != 1 or vectors.shape[1] != len(values) or rhs.shape[0] != len(values):
        raise ValueError("ridge eigensystem does not align")
    if rank < 1 or rank > len(values) or alpha <= 0 or np.any(values <= 0):
        raise ValueError("invalid rank or ridge eigenvalues")
    denominator = np.sqrt(values + float(alpha))
    whitened_rhs = rhs / denominator[:, None]
    _, output_vectors = np.linalg.eigh(whitened_rhs @ whitened_rhs.T)
    u = output_vectors[:, -rank:]
    state_projection = (vectors / denominator[None, :]) @ u
    query_loading = u.T @ whitened_rhs
    return {
        "output_vectors": u,
        "state_projection": state_projection,
        "query_loading": query_loading,
        "whitened_rhs": whitened_rhs,
    }


def action_state(core, state: dict[str, np.ndarray], features: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """Project raw action features through the fold's frozen ridge coordinates."""
    design = core.transform_features(features, state) - np.asarray(state["design_mean"], dtype=np.float64)
    matrix = np.asarray(projection, dtype=np.float64)
    if matrix.shape[0] != design.shape[1]:
        raise ValueError("state projection does not match features")
    result = design @ matrix
    if not np.isfinite(result).all():
        raise ValueError("action state is nonfinite")
    return result


def rms_scale(target: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Column RMS without centering; no held-action outcomes are required."""
    values = np.asarray(target, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or floor <= 0:
        raise ValueError("finite matrix and positive floor required")
    scale = np.sqrt(np.mean(np.square(values), axis=0, dtype=np.float64))
    return np.where(scale > floor, scale, 1.0)


def reconstruct(target_mean: np.ndarray, action: np.ndarray, loading_qd: np.ndarray) -> np.ndarray:
    mean = np.asarray(target_mean, dtype=np.float64)
    latent = np.asarray(action, dtype=np.float64)
    loading = np.asarray(loading_qd, dtype=np.float64)
    if mean.ndim != 1 or latent.ndim != 2 or loading.shape != (len(mean), latent.shape[1]):
        raise ValueError("mean, action state and query loading do not align")
    result = mean + latent @ loading.T
    if not np.isfinite(result).all():
        raise ValueError("reconstruction is nonfinite")
    return result


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def expected_parameter_count(feature_dim: int = 577, hidden_dim: int = 256, output_dim: int = 32) -> int:
    return feature_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim + feature_dim * output_dim
