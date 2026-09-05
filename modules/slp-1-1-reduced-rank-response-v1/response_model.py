"""Application-neutral reduced-rank feature-to-response model.

The fitted query loadings are quantitative descriptors of one measured panel.
They do not support unmeasured queries and are not static biological priors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReducedRankResponse:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    design_mean: np.ndarray
    state_projection: np.ndarray
    query_loading: np.ndarray
    intercept: np.ndarray
    alpha: float

    @property
    def rank(self) -> int:
        return self.state_projection.shape[1]

    def predict(self, features, query_indices=None):
        """Predict residual responses for raw static action features."""
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.feature_mean):
            raise ValueError("features must be [N,F]")
        if not np.isfinite(values).all():
            raise ValueError("features must be finite")
        selected = (
            np.arange(len(self.intercept), dtype=np.int64)
            if query_indices is None
            else np.asarray(query_indices)
        )
        if (
            selected.ndim != 1
            or not np.issubdtype(selected.dtype, np.integer)
            or np.any(selected < 0)
            or np.any(selected >= len(self.intercept))
        ):
            raise ValueError("query_indices must be valid one-dimensional integers")
        selected = selected.astype(np.int64, copy=False)
        design = (values.astype(np.float64) - self.feature_mean) / self.feature_scale
        state = (design - self.design_mean) @ self.state_projection
        prediction = self.intercept[selected] + state @ self.query_loading[:, selected]
        if not np.isfinite(prediction).all():
            raise ValueError("prediction is nonfinite")
        return prediction


def fit(features, targets, *, rank: int = 32, alpha: float = 1000.0):
    """Fit exact rank-constrained ridge with an unpenalized intercept."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float64)
    if (
        x.ndim != 2
        or y.ndim != 2
        or len(x) != len(y)
        or not len(x)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or rank <= 0
        or not np.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError("finite aligned matrices and positive rank/alpha required")
    mean = x.mean(0, dtype=np.float64)
    sd = x.std(0, dtype=np.float64)
    scale = np.where(sd > 1e-5, sd, 1.0)
    design = (x.astype(np.float64) - mean) / scale
    design_mean = design.mean(0, dtype=np.float64)
    centered_design = design - design_mean
    intercept = y.mean(0, dtype=np.float64)
    centered_target = y - intercept
    eigenvalues, eigenvectors = np.linalg.eigh(centered_design.T @ centered_design)
    keep = eigenvalues > 1e-8
    eigenvalues = eigenvalues[keep]
    eigenvectors = eigenvectors[:, keep]
    rhs = (centered_design @ eigenvectors).T @ centered_target
    root = np.sqrt(eigenvalues + alpha)
    whitened = rhs / root[:, None]
    _, response_vectors = np.linalg.eigh(whitened @ whitened.T)
    response_vectors = response_vectors[:, -min(rank, len(response_vectors)) :]
    state_projection = (eigenvectors / root[None, :]) @ response_vectors
    query_loading = response_vectors.T @ whitened
    model = ReducedRankResponse(
        feature_mean=mean,
        feature_scale=scale,
        design_mean=design_mean,
        state_projection=state_projection,
        query_loading=query_loading,
        intercept=intercept,
        alpha=float(alpha),
    )
    model.predict(x[:1])
    return model


def save(path, model: ReducedRankResponse, *, query_ids, source_id: str):
    query = np.asarray(query_ids).astype(str)
    if query.shape != (len(model.intercept),) or len(set(query.tolist())) != len(query):
        raise ValueError("query identity axis differs from fitted loading")
    np.savez_compressed(
        path,
        schema=np.asarray("slp.reduced-rank-response-model/v1"),
        source_id=np.asarray(source_id),
        rank=np.asarray(model.rank, np.int64),
        alpha=np.asarray(model.alpha, np.float64),
        query_ids=query,
        feature_mean=model.feature_mean,
        feature_scale=model.feature_scale,
        design_mean=model.design_mean,
        state_projection=model.state_projection,
        query_loading=model.query_loading,
        intercept=model.intercept,
    )


def load(path) -> ReducedRankResponse:
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["schema"]) != "slp.reduced-rank-response-model/v1":
            raise ValueError("unsupported reduced-rank model schema")
        model = ReducedRankResponse(
            feature_mean=np.asarray(archive["feature_mean"], np.float64),
            feature_scale=np.asarray(archive["feature_scale"], np.float64),
            design_mean=np.asarray(archive["design_mean"], np.float64),
            state_projection=np.asarray(archive["state_projection"], np.float64),
            query_loading=np.asarray(archive["query_loading"], np.float64),
            intercept=np.asarray(archive["intercept"], np.float64),
            alpha=float(archive["alpha"]),
        )
        rank = int(archive["rank"])
    f = len(model.feature_mean)
    if (
        model.feature_scale.shape != (f,)
        or model.design_mean.shape != (f,)
        or model.state_projection.shape != (f, rank)
        or model.query_loading.shape != (rank, len(model.intercept))
        or np.any(model.feature_scale <= 0)
        or not all(
            np.isfinite(value).all()
            for value in (
                model.feature_mean,
                model.feature_scale,
                model.design_mean,
                model.state_projection,
                model.query_loading,
                model.intercept,
            )
        )
    ):
        raise ValueError("invalid reduced-rank model arrays")
    return model


__all__ = ["ReducedRankResponse", "fit", "load", "save"]
