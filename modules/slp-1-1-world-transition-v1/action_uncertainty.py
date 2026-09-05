"""Action-dependent biological variance for feature-linear ridge baselines.

This fitting-only utility models a scalar multiplier of the biological
variance already calibrated for each context and query.  It never accepts or
returns a molecular mean.  For fitting record ``i`` the frozen moment is

``sum_q mask[i,q] * (residual[i,q]^2 - sampling[c,q] / n[i])``
``--------------------------------------------------------------------``
``       sum_q mask[i,q] * biological[c,q]``.

The positive, floored moment is log transformed and regressed on standardized
static action features.  At application time the exponentiated feature-linear
prediction is clipped and multiplies only biological variance; sampling
variance remains divided by the requested cell exposure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


class ActionUncertaintyError(ValueError):
    """Raised when action-variance inputs violate the frozen contract."""


@dataclass(frozen=True)
class RecordVarianceMoments:
    """Fitting-only scalar residual moments and floor diagnostics."""

    raw_values: Array
    positive_values: Array
    identifiable: Array
    observed_counts: Array
    floor: float

    @property
    def identifiable_count(self) -> int:
        return int(np.count_nonzero(self.identifiable))

    @property
    def floor_fraction(self) -> float:
        selected = self.raw_values[self.identifiable]
        return float(np.mean(selected <= self.floor))


@dataclass(frozen=True)
class ActionBiologicalVarianceMultiplier:
    """Feature-linear log multiplier for biological observation variance."""

    feature_mean_: Array
    feature_scale_: Array
    coefficient_: Array
    intercept_: float
    alpha: float
    factor_min: float
    factor_max: float
    moment_floor: float
    fitting_records: int
    dropped_unidentifiable_records: int
    fitting_floor_fraction: float
    uncertainty_family: str = "static-feature-linear-biological-variance-multiplier"

    def multipliers(self, action_features: Array) -> Array:
        """Return clipped positive biological-variance factors per record."""

        features = _features(action_features)
        if features.shape[1] != self.feature_mean_.size:
            raise ActionUncertaintyError(
                f"action_features has {features.shape[1]} columns; expected "
                f"{self.feature_mean_.size}"
            )
        standardized = (features - self.feature_mean_) / self.feature_scale_
        log_factor = self.intercept_ + standardized @ self.coefficient_
        # Clip in log space before exp so extreme validation features cannot overflow.
        return np.exp(np.clip(log_factor, np.log(self.factor_min), np.log(self.factor_max)))

    def scales(
        self,
        action_features: Array,
        num_cells: Array,
        context_index: Array,
        biological_variance: Array,
        sampling_variance: Array,
        *,
        scale_floor: float = 0.05,
    ) -> Array:
        """Return scales after changing biological variance only.

        Cell exposure enters only the ``sampling_variance / num_cells`` term.
        Molecular means are intentionally absent from this API.
        """

        features = _features(action_features)
        counts = _exposures(num_cells, features.shape[0])
        biological, sampling = _components(biological_variance, sampling_variance)
        contexts = _contexts(context_index, features.shape[0], biological.shape[0])
        if not np.isfinite(scale_floor) or scale_floor <= 0.0:
            raise ActionUncertaintyError("scale_floor must be finite and positive")
        factor = self.multipliers(features)
        variance = (
            factor[:, None] * biological[contexts]
            + sampling[contexts] / counts[:, None]
        )
        variance = np.maximum(variance, float(scale_floor) ** 2)
        if not np.all(np.isfinite(variance)):
            raise ActionUncertaintyError("requested scales are non-finite")
        return np.sqrt(variance)


def _features(value: Array) -> Array:
    features = np.asarray(value, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ActionUncertaintyError("action_features must be a non-empty matrix")
    if not np.all(np.isfinite(features)):
        raise ActionUncertaintyError("action_features must contain only finite values")
    return features


def _exposures(value: Array, records: int) -> Array:
    counts = np.asarray(value, dtype=np.float64)
    if counts.shape != (records,) or not np.all(np.isfinite(counts)) or np.any(counts <= 0.0):
        raise ActionUncertaintyError("num_cells must be one finite positive value per record")
    return counts


def _components(biological_variance: Array, sampling_variance: Array) -> tuple[Array, Array]:
    biological = np.asarray(biological_variance, dtype=np.float64)
    sampling = np.asarray(sampling_variance, dtype=np.float64)
    if biological.ndim != 2 or biological.shape[0] == 0 or biological.shape[1] == 0:
        raise ActionUncertaintyError("biological_variance must be a non-empty matrix")
    if sampling.shape != biological.shape:
        raise ActionUncertaintyError("sampling_variance must match biological_variance")
    if (
        not np.all(np.isfinite(biological))
        or not np.all(np.isfinite(sampling))
        or np.any(biological < 0.0)
        or np.any(sampling < 0.0)
    ):
        raise ActionUncertaintyError("variance components must be finite and non-negative")
    return biological, sampling


def _contexts(value: Array, records: int, context_count: int) -> Array:
    contexts = np.asarray(value)
    if contexts.shape != (records,) or contexts.dtype.kind not in "iu":
        raise ActionUncertaintyError("context_index must be one integer per record")
    if np.any(contexts < 0) or np.any(contexts >= context_count):
        raise ActionUncertaintyError("context_index is out of range")
    return contexts.astype(np.int64, copy=False)


def estimate_record_variance_moments(
    oof_residuals: Array,
    observed: Array,
    num_cells: Array,
    context_index: Array,
    biological_variance: Array,
    sampling_variance: Array,
    *,
    moment_floor: float = 0.05,
) -> RecordVarianceMoments:
    """Estimate positive record moments from fitting-only OOF residuals."""

    residuals = np.asarray(oof_residuals, dtype=np.float64)
    mask = np.asarray(observed)
    if residuals.ndim != 2 or residuals.shape[0] == 0 or residuals.shape[1] == 0:
        raise ActionUncertaintyError("oof_residuals must be a non-empty matrix")
    if mask.shape != residuals.shape or mask.dtype != np.bool_:
        raise ActionUncertaintyError("observed must be a boolean matrix matching residuals")
    if not np.all(np.isfinite(residuals[mask])):
        raise ActionUncertaintyError("observed OOF residuals must be finite")
    counts = _exposures(num_cells, residuals.shape[0])
    biological, sampling = _components(biological_variance, sampling_variance)
    if biological.shape[1] != residuals.shape[1]:
        raise ActionUncertaintyError("variance components must share the residual query axis")
    contexts = _contexts(context_index, residuals.shape[0], biological.shape[0])
    if not np.isfinite(moment_floor) or moment_floor <= 0.0:
        raise ActionUncertaintyError("moment_floor must be finite and positive")

    observed_counts = mask.sum(axis=1, dtype=np.int64)
    squared = np.square(residuals, where=mask, out=np.zeros_like(residuals))
    sampling_term = sampling[contexts] / counts[:, None]
    numerator = np.where(mask, squared - sampling_term, 0.0).sum(axis=1)
    denominator = np.where(mask, biological[contexts], 0.0).sum(axis=1)
    tolerance = np.finfo(np.float64).eps * np.maximum(observed_counts, 1)
    identifiable = (observed_counts > 0) & (denominator > tolerance)
    raw = np.full(residuals.shape[0], np.nan, dtype=np.float64)
    raw[identifiable] = numerator[identifiable] / denominator[identifiable]
    positive = np.full_like(raw, np.nan)
    positive[identifiable] = np.maximum(raw[identifiable], float(moment_floor))
    if not np.any(identifiable):
        raise ActionUncertaintyError("no record has identifiable biological variance")
    return RecordVarianceMoments(
        raw_values=raw,
        positive_values=positive,
        identifiable=identifiable,
        observed_counts=observed_counts,
        floor=float(moment_floor),
    )


def fit_action_variance_multiplier(
    action_features: Array,
    moments: RecordVarianceMoments,
    *,
    alpha: float = 10_000.0,
    factor_min: float = 0.25,
    factor_max: float = 4.0,
) -> ActionBiologicalVarianceMultiplier:
    """Fit a standardized feature-linear model to fitting-only log moments."""

    features = _features(action_features)
    if moments.raw_values.shape != (features.shape[0],):
        raise ActionUncertaintyError("moments must contain one value per action feature row")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ActionUncertaintyError("alpha must be finite and non-negative")
    if (
        not np.isfinite(factor_min)
        or not np.isfinite(factor_max)
        or factor_min <= 0.0
        or factor_min >= factor_max
    ):
        raise ActionUncertaintyError("factor bounds must be finite, positive, and increasing")
    selected = np.asarray(moments.identifiable, dtype=bool)
    if selected.shape != (features.shape[0],) or not np.any(selected):
        raise ActionUncertaintyError("moments contain no identifiable fitting records")
    if not np.all(np.isfinite(moments.positive_values[selected])):
        raise ActionUncertaintyError("identifiable positive moments must be finite")

    fitting_features = features[selected]
    feature_mean = fitting_features.mean(axis=0)
    feature_scale = fitting_features.std(axis=0)
    feature_scale = np.where(feature_scale > np.finfo(np.float64).eps, feature_scale, 1.0)
    standardized = (fitting_features - feature_mean) / feature_scale
    response = np.log(moments.positive_values[selected])
    intercept = float(response.mean())
    centered = response - intercept
    gram = standardized.T @ standardized
    gram.flat[:: gram.shape[0] + 1] += float(alpha)
    rhs = standardized.T @ centered
    try:
        coefficient = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficient = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return ActionBiologicalVarianceMultiplier(
        feature_mean_=feature_mean,
        feature_scale_=feature_scale,
        coefficient_=coefficient,
        intercept_=intercept,
        alpha=float(alpha),
        factor_min=float(factor_min),
        factor_max=float(factor_max),
        moment_floor=float(moments.floor),
        fitting_records=int(np.count_nonzero(selected)),
        dropped_unidentifiable_records=int(np.count_nonzero(~selected)),
        fitting_floor_fraction=moments.floor_fraction,
    )
