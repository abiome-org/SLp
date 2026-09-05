"""Training-only response compression diagnostic.

This module separates two questions. ``oracle_reconstruct`` projects a supplied
response through a response basis learned only from fitting records; because it
consumes the response being reconstructed, it is a measurement-compression
diagnostic and never a forecast. ``predict`` maps static intervention features
to fitting latent scores with feature-linear ridge and is a genuine forecast.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.utils.extmath import randomized_svd
from transition_baselines import FeatureLinearRidgeBaseline, fit_ridge

Array = np.ndarray


class ResponseCompressionError(ValueError):
    """Raised when a response-compression contract is invalid."""


@dataclass(frozen=True)
class ResponseCompressionDiagnostic:
    """Nested training response basis and feature-to-score ridge map."""

    context_means_: Array
    query_scales_: Array
    components_: Array
    singular_values_: Array
    coefficient_model_: FeatureLinearRidgeBaseline
    total_standardized_energy_: float
    alpha: float
    seed: int
    scale_floor: float

    @property
    def maximum_rank(self) -> int:
        return int(self.components_.shape[0])

    def retained_training_variance(self, rank: int) -> float:
        """Fraction of fitting standardized residual energy retained at rank."""

        rank = self._rank(rank)
        return float(np.square(self.singular_values_[:rank]).sum() / self.total_standardized_energy_)

    def predict(self, action_features: Array, context_index: Array, rank: int) -> Array:
        """Forecast responses without consuming outcome values."""

        features, contexts = self._prediction_inputs(action_features, context_index)
        rank = self._rank(rank)
        scores = self.coefficient_model_.predict(features)[:, :rank]
        standardized = scores @ self.components_[:rank]
        return self.context_means_[contexts] + standardized * self.query_scales_

    def oracle_reconstruct(
        self,
        targets: Array,
        observed: Array,
        context_index: Array,
        rank: int,
    ) -> Array:
        """Project measured responses into the fitting basis and reconstruct.

        The supplied targets determine their own latent scores. This method is
        therefore suitable only for an explicitly labelled oracle compression
        diagnostic, never for held-out forecasting.
        """

        values, contexts = _response_inputs(targets, observed, context_index)
        if values.shape[1] != self.context_means_.shape[1]:
            raise ResponseCompressionError("targets have the wrong number of queries")
        if np.any(contexts >= self.context_means_.shape[0]):
            raise ResponseCompressionError("context_index is out of range")
        rank = self._rank(rank)
        standardized = (values - self.context_means_[contexts]) / self.query_scales_
        scores = standardized @ self.components_[:rank].T
        reconstruction = scores @ self.components_[:rank]
        return self.context_means_[contexts] + reconstruction * self.query_scales_

    def project_forecast(self, prediction: Array, context_index: Array, rank: int) -> Array:
        """Project an outcome-free forecast through the fitting response basis.

        For a fixed fitting design, ridge penalty, and context, ridge is linear
        in its response matrix. Projecting a full multioutput ridge forecast by
        this method is therefore equivalent to fitting ridge to the associated
        latent scores, up to numerical precision.
        """

        values = np.asarray(prediction, dtype=np.float64)
        contexts = np.asarray(context_index)
        if values.ndim != 2 or values.shape[1] != self.context_means_.shape[1]:
            raise ResponseCompressionError("prediction must have one column per query")
        if contexts.shape != (values.shape[0],) or contexts.dtype.kind not in "iu":
            raise ResponseCompressionError("context_index must contain one integer per record")
        contexts = contexts.astype(np.int64, copy=False)
        if np.any(contexts < 0) or np.any(contexts >= self.context_means_.shape[0]):
            raise ResponseCompressionError("context_index is out of range")
        if not np.isfinite(values).all():
            raise ResponseCompressionError("prediction must contain only finite values")
        rank = self._rank(rank)
        standardized = (values - self.context_means_[contexts]) / self.query_scales_
        projected = (standardized @ self.components_[:rank].T) @ self.components_[:rank]
        return self.context_means_[contexts] + projected * self.query_scales_

    def _rank(self, rank: int) -> int:
        if type(rank) is not int or rank < 1 or rank > self.maximum_rank:
            raise ResponseCompressionError(
                f"rank must be an integer from 1 through {self.maximum_rank}"
            )
        return rank

    def _prediction_inputs(self, action_features: Array, context_index: Array) -> tuple[Array, Array]:
        features = np.asarray(action_features, dtype=np.float64)
        contexts = np.asarray(context_index)
        if features.ndim != 2 or features.shape[0] == 0 or not np.isfinite(features).all():
            raise ResponseCompressionError("action_features must be a non-empty finite matrix")
        if contexts.shape != (features.shape[0],) or contexts.dtype.kind not in "iu":
            raise ResponseCompressionError("context_index must contain one integer per record")
        contexts = contexts.astype(np.int64, copy=False)
        if np.any(contexts < 0) or np.any(contexts >= self.context_means_.shape[0]):
            raise ResponseCompressionError("context_index is out of range")
        return features, contexts


def _response_inputs(targets: Array, observed: Array, context_index: Array) -> tuple[Array, Array]:
    values = np.asarray(targets, dtype=np.float64)
    mask = np.asarray(observed)
    contexts = np.asarray(context_index)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ResponseCompressionError("targets must be a non-empty response matrix")
    if mask.shape != values.shape or mask.dtype != np.bool_:
        raise ResponseCompressionError("observed must be a boolean matrix matching targets")
    if not np.all(mask):
        raise ResponseCompressionError("response compression requires a complete observed panel")
    if not np.isfinite(values).all():
        raise ResponseCompressionError("targets must contain only finite values")
    if contexts.shape != (values.shape[0],) or contexts.dtype.kind not in "iu":
        raise ResponseCompressionError("context_index must contain one integer per record")
    contexts = contexts.astype(np.int64, copy=False)
    if np.any(contexts < 0):
        raise ResponseCompressionError("context_index cannot be negative")
    return values, contexts


def fit_response_compression(
    action_features: Array,
    targets: Array,
    observed: Array,
    context_index: Array,
    *,
    maximum_rank: int = 128,
    alpha: float = 10000.0,
    seed: int = 731,
    scale_floor: float = 0.05,
) -> ResponseCompressionDiagnostic:
    """Fit one nested response basis using fitting records only.

    Per-context response means and pooled per-query residual RMS scales are fit
    before randomized SVD. The scale is floored in outcome units. The returned
    ridge maps static intervention features to all ``maximum_rank`` scores;
    lower ranks use leading nested components and score columns.
    """

    features = np.asarray(action_features, dtype=np.float64)
    values, contexts = _response_inputs(targets, observed, context_index)
    if features.ndim != 2 or features.shape[0] != values.shape[0] or not np.isfinite(features).all():
        raise ResponseCompressionError("action_features must be finite and align with targets")
    if type(maximum_rank) is not int or maximum_rank < 1 or maximum_rank > min(values.shape):
        raise ResponseCompressionError("maximum_rank exceeds the fitting response matrix")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ResponseCompressionError("alpha must be finite and non-negative")
    if type(seed) is not int:
        raise ResponseCompressionError("seed must be an integer")
    if not np.isfinite(scale_floor) or scale_floor <= 0.0:
        raise ResponseCompressionError("scale_floor must be finite and positive")
    context_count = int(contexts.max()) + 1
    if set(contexts.tolist()) != set(range(context_count)):
        raise ResponseCompressionError("fitting context indices must be contiguous from zero")
    means = np.stack([values[contexts == context].mean(axis=0) for context in range(context_count)])
    residuals = values - means[contexts]
    scales = np.maximum(np.sqrt(np.mean(np.square(residuals), axis=0)), scale_floor)
    standardized = (residuals / scales).astype(np.float32)
    _, singular_values, components = randomized_svd(
        standardized,
        n_components=maximum_rank,
        n_oversamples=10,
        n_iter=3,
        power_iteration_normalizer="auto",
        transpose="auto",
        flip_sign=True,
        random_state=seed,
    )
    if not np.isfinite(components).all() or not np.isfinite(singular_values).all():
        raise ResponseCompressionError("randomized SVD returned non-finite values")
    scores = standardized @ components.T
    score_mask = np.ones_like(scores, dtype=np.bool_)
    coefficient_model = fit_ridge(features, scores, score_mask, alpha, scale_floor=scale_floor)
    total_energy = float(np.square(standardized.astype(np.float64)).sum())
    if not np.isfinite(total_energy) or total_energy <= 0.0:
        raise ResponseCompressionError("standardized fitting residuals have no variation")
    return ResponseCompressionDiagnostic(
        context_means_=means,
        query_scales_=scales,
        components_=components.astype(np.float64),
        singular_values_=singular_values.astype(np.float64),
        coefficient_model_=coefficient_model,
        total_standardized_energy_=total_energy,
        alpha=float(alpha),
        seed=seed,
        scale_floor=float(scale_floor),
    )


def gene_macro_point_metrics(
    prediction: Array,
    truth: Array,
    observed: Array,
    action_ids: Sequence[str],
    source_centroid: Array,
) -> dict[str, float | int | None]:
    """Equal-gene point metrics with per-record molecular profile correlations."""

    means = np.asarray(prediction, dtype=np.float64)
    values = np.asarray(truth, dtype=np.float64)
    mask = np.asarray(observed)
    centroid = np.broadcast_to(np.asarray(source_centroid, dtype=np.float64), values.shape)
    if means.shape != values.shape or mask.shape != values.shape or mask.dtype != np.bool_:
        raise ResponseCompressionError("prediction, truth, and observed shapes must match")
    if len(action_ids) != values.shape[0]:
        raise ResponseCompressionError("action_ids must contain one item per record")
    if not np.all(np.isfinite(means[mask])) or not np.all(np.isfinite(values[mask])):
        raise ResponseCompressionError("observed predictions and targets must be finite")
    groups: dict[str, list[int]] = {}
    for row, action in enumerate(action_ids):
        groups.setdefault(str(action), []).append(row)
    per_gene: list[tuple[float, float | None, float | None]] = []
    for rows in groups.values():
        row_mse: list[float] = []
        ordinary: list[float] = []
        adjusted: list[float] = []
        for row in rows:
            selected = mask[row]
            if not np.any(selected):
                continue
            row_mse.append(float(np.mean(np.square(means[row, selected] - values[row, selected]))))
            ordinary_value = _pearson(means[row, selected], values[row, selected])
            adjusted_value = _pearson(
                means[row, selected] - centroid[row, selected],
                values[row, selected] - centroid[row, selected],
            )
            if ordinary_value is not None:
                ordinary.append(ordinary_value)
            if adjusted_value is not None:
                adjusted.append(adjusted_value)
        if row_mse:
            per_gene.append(
                (
                    float(np.mean(row_mse)),
                    float(np.mean(ordinary)) if ordinary else None,
                    float(np.mean(adjusted)) if adjusted else None,
                )
            )
    ordinary_genes = [item[1] for item in per_gene if item[1] is not None]
    adjusted_genes = [item[2] for item in per_gene if item[2] is not None]
    mse = float(np.mean([item[0] for item in per_gene])) if per_gene else None
    return {
        "intervention_genes": len(per_gene),
        "gene_macro_mse": mse,
        "gene_macro_source_centroid_adjusted_mse": mse,
        "gene_macro_profile_pearson_mean": float(np.mean(ordinary_genes)) if ordinary_genes else None,
        "gene_macro_profile_source_centroid_adjusted_pearson_mean": (
            float(np.mean(adjusted_genes)) if adjusted_genes else None
        ),
        "profile_pearson_defined_genes": len(ordinary_genes),
        "profile_source_centroid_adjusted_pearson_defined_genes": len(adjusted_genes),
    }


def _pearson(left: Array, right: Array) -> float | None:
    if left.size < 2:
        return None
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    if denominator <= np.finfo(np.float64).eps:
        return None
    return float((left @ right) / denominator)
