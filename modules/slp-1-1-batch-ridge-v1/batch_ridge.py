"""Streaming ridge with unpenalized observed-batch intercepts."""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


class BatchRidgeStatistics:
    """Fit y = features @ weights + batch_offset from fitting rows only.

    Feature normalization and statistical row weights are caller-owned. There
    are no gene IDs or missing-value imputations. Batches without fitting
    observations cannot be predicted. The intercept is a nuisance adjustment,
    not a biological control or an estimate of an unseen context.
    """

    def __init__(self, features: int, queries: int) -> None:
        if features < 1 or queries < 1:
            raise ValueError("positive feature and query dimensions required")
        self.xx = np.zeros((features, features), dtype=np.float64)
        self.xy = np.zeros((features, queries), dtype=np.float64)
        self.batches: dict[str, dict[str, object]] = {}

    def update(
        self, batch: str, x: np.ndarray, y: np.ndarray, weight: np.ndarray
    ) -> None:
        x, y, weight = (np.asarray(v, dtype=np.float64) for v in (x, y, weight))
        if (
            not isinstance(batch, str)
            or not batch
            or x.ndim != 2
            or y.ndim != 2
            or weight.ndim != 1
            or x.shape != (len(weight), self.xx.shape[0])
            or y.shape != (len(weight), self.xy.shape[1])
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or not np.isfinite(weight).all()
            or np.any(weight < 0)
        ):
            raise ValueError("invalid finite complete fitting block or weights")
        total = float(weight.sum())
        if not total > 0:
            raise ValueError("fitting block needs positive total weight")
        xw = x * weight[:, None]
        self.xx += x.T @ xw
        self.xy += xw.T @ y
        if batch not in self.batches:
            self.batches[batch] = {
                "weight": 0.0,
                "x_sum": np.zeros(self.xx.shape[0], dtype=np.float64),
                "y_sum": np.zeros(self.xy.shape[1], dtype=np.float64),
            }
        state = self.batches[batch]
        state["weight"] += total
        state["x_sum"] += xw.sum(axis=0)
        state["y_sum"] += weight @ y

    def solve(self, alpha: float) -> dict[str, np.ndarray]:
        """Minimize sum_i w_i ||y_i-X_i W-b_batch||² + alpha ||W||²."""
        if not np.isfinite(alpha) or alpha <= 0 or not self.batches:
            raise ValueError("positive finite ridge penalty and fitting data required")
        xx, xy = self.xx.copy(), self.xy.copy()
        labels = sorted(self.batches)
        means_x, means_y = [], []
        for label in labels:
            state = self.batches[label]
            total = state["weight"]
            mx, my = state["x_sum"] / total, state["y_sum"] / total
            xx -= total * np.outer(mx, mx)
            xy -= total * np.outer(mx, my)
            means_x.append(mx)
            means_y.append(my)
        xx = (xx + xx.T) / 2
        xx.flat[:: len(xx) + 1] += alpha
        coefficients = cho_solve(cho_factor(xx, lower=True), xy)
        means_x, means_y = np.asarray(means_x), np.asarray(means_y)
        return {
            "coefficients": coefficients,
            "batch_ids": np.asarray(labels, dtype=str),
            "batch_offsets": means_y - means_x @ coefficients,
            "batch_only_means": means_y,
            "alpha": np.asarray(alpha),
        }


def predict(model: dict[str, np.ndarray], batch: str, features: np.ndarray) -> np.ndarray:
    matches = np.flatnonzero(model["batch_ids"] == batch)
    if len(matches) != 1:
        raise ValueError("batch has no unique fitting-derived intercept")
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != model["coefficients"].shape[0] or not np.isfinite(x).all():
        raise ValueError("invalid prediction features")
    return x @ model["coefficients"] + model["batch_offsets"][matches[0]]
