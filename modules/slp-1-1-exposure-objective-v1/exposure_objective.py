"""Fixed exposure-precision objective for pseudobulk molecular means.

This numerical helper does not estimate variance and has no mean-prediction
API.  Callers provide fitting/control-derived variance components, observed
cell counts, and fixed row weights.  Cell count therefore affects loss
precision only and cannot enter the molecular mean or state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

Array = np.ndarray


class ExposureObjectiveError(ValueError):
    """Raised when a fixed exposure objective violates its contract."""


@dataclass(frozen=True)
class PrecisionNormalization:
    multiplier: float
    old_weighted_mean_precision: float
    new_weighted_mean_precision_before: float
    new_weighted_mean_precision_after: float


def _components(biological_variance: Array, sampling_variance: Array) -> tuple[Array, Array]:
    biological = np.asarray(biological_variance, dtype=np.float64)
    sampling = np.asarray(sampling_variance, dtype=np.float64)
    if (
        biological.ndim != 2
        or biological.shape != sampling.shape
        or not biological.size
        or not np.isfinite(biological).all()
        or not np.isfinite(sampling).all()
        or np.any(biological < 0.0)
        or np.any(sampling < 0.0)
    ):
        raise ExposureObjectiveError("variance components must be aligned finite nonnegative context-query matrices")
    return biological, sampling


def exposure_precision(
    num_cells: Array,
    context_index: Array,
    biological_variance: Array,
    sampling_variance: Array,
    *,
    scale_floor: float = 0.05,
    multiplier: float = 1.0,
) -> Array:
    """Return record/query precision for ``tau² + sigma² / n``.

    The fixed scale floor is applied to variance before inversion.  The scalar
    multiplier may preserve a comparator objective's global precision without
    changing relative record/query weights.
    """

    biological, sampling = _components(biological_variance, sampling_variance)
    counts = np.asarray(num_cells, dtype=np.float64)
    contexts = np.asarray(context_index)
    if (
        counts.ndim != 1
        or contexts.shape != counts.shape
        or contexts.dtype.kind not in "iu"
        or not counts.size
        or not np.isfinite(counts).all()
        or np.any(counts <= 0.0)
        or np.any(contexts < 0)
        or np.any(contexts >= biological.shape[0])
        or not np.isfinite(scale_floor)
        or scale_floor <= 0.0
        or not np.isfinite(multiplier)
        or multiplier <= 0.0
    ):
        raise ExposureObjectiveError("invalid exposure, context, floor, or multiplier")
    variance = biological[contexts] + sampling[contexts] / counts[:, None]
    variance = np.maximum(variance, np.float64(scale_floor) ** 2)
    precision = np.float64(multiplier) / variance
    if not np.isfinite(precision).all() or np.any(precision <= 0.0):
        raise ExposureObjectiveError("fixed components produced invalid precision")
    return precision


def _weighted_observed_mean(values: Array, observed: Array, row_weight: Array) -> float:
    matrix = np.asarray(values, dtype=np.float64)
    mask = np.asarray(observed)
    weights = np.asarray(row_weight, dtype=np.float64)
    if (
        matrix.ndim != 2
        or mask.shape != matrix.shape
        or mask.dtype != np.bool_
        or weights.shape != (matrix.shape[0],)
        or not np.isfinite(matrix[mask]).all()
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or np.any(mask.sum(1) == 0)
    ):
        raise ExposureObjectiveError("weighted precision mean inputs do not align")
    row_means = np.where(mask, matrix, 0.0).sum(1) / mask.sum(1)
    return float(np.sum(row_means * weights) / np.sum(weights))


def match_global_precision(
    new_precision: Array,
    old_query_scale: Array,
    context_index: Array,
    observed: Array,
    row_weight: Array,
) -> PrecisionNormalization:
    """Match the new weighted-mean precision to the prior fixed objective.

    ``old_query_scale`` is context/query. The mean is over each fitting row's
    observed queries and then over fixed global row weights. One scalar is fit
    on the entire fitting snapshot and is never recomputed in a minibatch.
    """

    new = np.asarray(new_precision, dtype=np.float64)
    scale = np.asarray(old_query_scale, dtype=np.float64)
    contexts = np.asarray(context_index)
    if (
        new.ndim != 2
        or contexts.shape != (new.shape[0],)
        or contexts.dtype.kind not in "iu"
        or scale.ndim != 2
        or scale.shape[1] != new.shape[1]
        or np.any(contexts < 0)
        or np.any(contexts >= scale.shape[0])
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
    ):
        raise ExposureObjectiveError("old scale or context alignment is invalid")
    old = 1.0 / np.square(scale[contexts])
    old_mean = _weighted_observed_mean(old, observed, row_weight)
    new_mean = _weighted_observed_mean(new, observed, row_weight)
    multiplier = old_mean / new_mean
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ExposureObjectiveError("global precision normalization is invalid")
    return PrecisionNormalization(multiplier, old_mean, new_mean, new_mean * multiplier)


def masked_precision_mse(
    prediction_mean: torch.Tensor,
    target: torch.Tensor,
    observed: torch.Tensor,
    precision: torch.Tensor,
    row_weight: torch.Tensor,
) -> torch.Tensor:
    """Return fixed globally weighted precision MSE without batch renormalization."""

    if (
        prediction_mean.shape != target.shape
        or target.shape != observed.shape
        or target.shape != precision.shape
        or target.ndim != 2
        or observed.dtype != torch.bool
        or row_weight.shape != (target.shape[0],)
        or not torch.is_floating_point(row_weight)
    ):
        raise ExposureObjectiveError("precision objective tensors do not align")
    safe_target = torch.where(observed, target, torch.zeros_like(target))
    safe_precision = torch.where(observed, precision, torch.ones_like(precision))
    if (
        not torch.isfinite(safe_target).all()
        or not torch.isfinite(safe_precision).all()
        or not (safe_precision > 0).all()
        or not torch.isfinite(row_weight).all()
        or not (row_weight > 0).all()
    ):
        raise ExposureObjectiveError("precision objective requires finite positive inputs")
    counts = observed.sum(1)
    if not (counts > 0).all():
        raise ExposureObjectiveError("every fitting row requires an observed query")
    squared = torch.where(
        observed,
        torch.square(prediction_mean - safe_target) * safe_precision,
        torch.zeros_like(target),
    )
    per_record = squared.sum(1) / counts
    return (per_record * row_weight.to(per_record.dtype)).mean()
