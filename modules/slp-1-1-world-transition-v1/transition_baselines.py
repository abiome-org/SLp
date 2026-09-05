"""Fitting-only statistical controls for the SLp-1.1 transition study.

The ridge model is a feature-linear multioutput baseline: each readout has its
own intercept and linear coefficients over action features.  It is not a
bilinear action/query model and it does not consume query features.

All targets supplied here must belong to the fitting partition.  Residual
scales are frozen from either in-sample fitting residuals or caller-supplied
out-of-fold predictions for those same fitting records.  Evaluation data must
never be passed to a fitting or calibration function in this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Array = np.ndarray
_LOG_2PI = math.log(2.0 * math.pi)


class BaselineError(ValueError):
    """Raised when arrays cannot satisfy the frozen baseline contract."""


@dataclass(frozen=True)
class ResidualScale:
    """Per-readout Gaussian scales and their fitting-only provenance."""

    values: Array
    counts: Array
    provenance: str


@dataclass(frozen=True)
class MeanBaseline:
    """Per-readout fitting mean with no dependence on action features."""

    intercept_: Array
    residual_scale_: ResidualScale
    baseline_name: str = "fitting-mean"
    baseline_family: str = "mean"

    def predict(self, action_features: Array) -> Array:
        features = _feature_matrix(action_features, "action_features")
        return np.broadcast_to(self.intercept_, (features.shape[0], self.intercept_.size)).copy()


@dataclass(frozen=True)
class FeatureLinearRidgeBaseline:
    """Feature-linear multioutput ridge control with per-readout intercepts."""

    feature_mean_: Array
    feature_scale_: Array
    intercept_: Array
    coef_: Array
    residual_scale_: ResidualScale
    alpha: float
    baseline_name: str = "feature-linear-ridge"
    baseline_family: str = "feature-linear-multioutput"

    def predict(self, action_features: Array) -> Array:
        features = _feature_matrix(action_features, "action_features")
        if features.shape[1] != self.feature_mean_.size:
            raise BaselineError(
                f"action_features has {features.shape[1]} columns; expected "
                f"{self.feature_mean_.size}"
            )
        standardized = (features - self.feature_mean_) / self.feature_scale_
        return self.intercept_ + standardized @ self.coef_


def _feature_matrix(value: Array, label: str) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2:
        raise BaselineError(f"{label} must be a two-dimensional array")
    if result.shape[0] == 0:
        raise BaselineError(f"{label} must contain at least one record")
    if not np.all(np.isfinite(result)):
        raise BaselineError(f"{label} must contain only finite values")
    return result


def _targets_and_mask(targets: Array, observed: Array) -> tuple[Array, Array]:
    values = np.asarray(targets, dtype=np.float64)
    mask_input = np.asarray(observed)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise BaselineError("targets must be a non-empty two-dimensional array")
    if mask_input.shape != values.shape:
        raise BaselineError("observed must have the same shape as targets")
    if mask_input.dtype != np.bool_ and not np.all(np.isin(mask_input, (0, 1))):
        raise BaselineError("observed must be boolean or contain only zero and one")
    mask = mask_input.astype(bool, copy=False)
    if not np.all(np.isfinite(values[mask])):
        raise BaselineError("observed targets must be finite")
    return values, mask


def fit_residual_scale(
    predictions: Array,
    targets: Array,
    observed: Array,
    *,
    provenance: str = "fitting-residuals",
    floor: float = 1e-6,
) -> ResidualScale:
    """Fit per-readout Gaussian RMSE scales using fitting records only.

    ``provenance`` is deliberately restricted to in-sample fitting residuals or
    fitting-set out-of-fold residuals.  A readout with no observations receives
    a NaN scale and count zero.
    """

    if provenance not in {"fitting-residuals", "oof-fitting-residuals"}:
        raise BaselineError(
            "scale provenance must be 'fitting-residuals' or 'oof-fitting-residuals'"
        )
    if not np.isfinite(floor) or floor <= 0.0:
        raise BaselineError("floor must be finite and positive")
    values, mask = _targets_and_mask(targets, observed)
    means = np.asarray(predictions, dtype=np.float64)
    if means.shape != values.shape:
        raise BaselineError("predictions must have the same shape as targets")
    if not np.all(np.isfinite(means[mask])):
        raise BaselineError("observed predictions must be finite")

    counts = mask.sum(axis=0, dtype=np.int64)
    residual = np.zeros_like(values)
    np.subtract(means, values, out=residual, where=mask)
    squared = (residual**2).sum(axis=0)
    scales = np.full(values.shape[1], np.nan, dtype=np.float64)
    present = counts > 0
    scales[present] = np.maximum(np.sqrt(squared[present] / counts[present]), floor)
    return ResidualScale(values=scales, counts=counts, provenance=provenance)


def fit_mean(
    targets: Array,
    observed: Array,
    *,
    oof_predictions: Array | None = None,
    scale_floor: float = 1e-6,
) -> MeanBaseline:
    """Fit a missing-aware per-readout mean on fitting targets."""

    values, mask = _targets_and_mask(targets, observed)
    counts = mask.sum(axis=0)
    intercept = np.full(values.shape[1], np.nan, dtype=np.float64)
    present = counts > 0
    intercept[present] = np.where(mask, values, 0.0).sum(axis=0)[present] / counts[present]
    fitted = np.broadcast_to(intercept, values.shape)
    calibration = fitted if oof_predictions is None else np.asarray(oof_predictions)
    provenance = "fitting-residuals" if oof_predictions is None else "oof-fitting-residuals"
    scale = fit_residual_scale(
        calibration, values, mask, provenance=provenance, floor=scale_floor
    )
    return MeanBaseline(intercept_=intercept, residual_scale_=scale)


def _mask_groups(mask: Array) -> list[tuple[Array, Array]]:
    """Return (record mask, output indices) groups with identical missingness."""

    groups: dict[bytes, list[int]] = {}
    for output in range(mask.shape[1]):
        key = np.packbits(mask[:, output], bitorder="little").tobytes()
        groups.setdefault(key, []).append(output)
    return [(mask[:, outputs[0]], np.asarray(outputs, dtype=np.int64)) for outputs in groups.values()]


def fit_ridge(
    action_features: Array,
    targets: Array,
    observed: Array,
    alpha: float,
    *,
    oof_predictions: Array | None = None,
    scale_floor: float = 1e-6,
) -> FeatureLinearRidgeBaseline:
    """Fit a missing-aware feature-linear multioutput ridge baseline.

    Feature means and standard deviations use all fitting action rows and are
    then frozen.  For each readout, only records marked observed enter its
    regularized least-squares system.  Identical missingness patterns share one
    factorization; the intercept is never regularized.
    """

    features = _feature_matrix(action_features, "action_features")
    values, mask = _targets_and_mask(targets, observed)
    if features.shape[0] != values.shape[0]:
        raise BaselineError("action_features and targets must have the same number of records")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise BaselineError("alpha must be finite and non-negative")

    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale = np.where(feature_scale > np.finfo(np.float64).eps, feature_scale, 1.0)
    standardized = (features - feature_mean) / feature_scale
    intercept = np.full(values.shape[1], np.nan, dtype=np.float64)
    coef = np.zeros((features.shape[1], values.shape[1]), dtype=np.float64)

    full_design = np.column_stack(
        (np.ones(features.shape[0], dtype=np.float64), standardized)
    )
    penalty = np.zeros((full_design.shape[1], full_design.shape[1]), dtype=np.float64)
    penalty[1:, 1:] = float(alpha) * np.eye(features.shape[1])
    full_gram = full_design.T @ full_design + penalty
    safe_values = np.where(mask, values, 0.0)
    all_rhs = full_design.T @ safe_values

    for record_mask, outputs in _mask_groups(mask):
        count = int(record_mask.sum())
        if count == 0:
            continue
        if count <= features.shape[0] - count:
            observed_design = full_design[record_mask]
            gram = observed_design.T @ observed_design + penalty
        else:
            missing_design = full_design[~record_mask]
            gram = full_gram - missing_design.T @ missing_design
        rhs = all_rhs[:, outputs]
        try:
            solution = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        intercept[outputs] = solution[0]
        coef[:, outputs] = solution[1:]

    model_without_scale = FeatureLinearRidgeBaseline(
        feature_mean_=feature_mean,
        feature_scale_=feature_scale,
        intercept_=intercept,
        coef_=coef,
        residual_scale_=ResidualScale(
            values=np.full(values.shape[1], np.nan),
            counts=np.zeros(values.shape[1], dtype=np.int64),
            provenance="fitting-residuals",
        ),
        alpha=float(alpha),
    )
    fitted = model_without_scale.predict(features)
    calibration = fitted if oof_predictions is None else np.asarray(oof_predictions)
    provenance = "fitting-residuals" if oof_predictions is None else "oof-fitting-residuals"
    scale = fit_residual_scale(
        calibration, values, mask, provenance=provenance, floor=scale_floor
    )
    return FeatureLinearRidgeBaseline(
        feature_mean_=feature_mean,
        feature_scale_=feature_scale,
        intercept_=intercept,
        coef_=coef,
        residual_scale_=scale,
        alpha=float(alpha),
    )


def _broadcast(value: Array | float, shape: tuple[int, int], label: str) -> Array:
    try:
        result = np.broadcast_to(np.asarray(value, dtype=np.float64), shape)
    except ValueError as error:
        raise BaselineError(f"{label} cannot be broadcast to {shape}") from error
    return result


def _pearson(left: Array, right: Array) -> float | None:
    if left.size < 2:
        return None
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = math.sqrt(float(left_centered @ left_centered) * float(right_centered @ right_centered))
    if denominator <= np.finfo(np.float64).eps:
        return None
    return float((left_centered @ right_centered) / denominator)


def _macro_mean(rows: list[float]) -> float:
    return float(np.mean(rows)) if rows else math.nan


def evaluate(
    prediction: Array,
    truth: Array,
    observed: Array,
    reference: Array,
    scale: Array | float,
    *, value_space: str = "log2",
) -> dict[str, float | int | str]:
    """Evaluate fixed-scale Gaussian predictions in the fixed log2 value space.

    NLL and MSE are means per record followed by a macro mean across non-empty
    records.  Profile correlations are computed across observed readouts for
    each record, then averaged across records where they are defined.
    Centroid-adjusted profiles subtract ``reference`` from both prediction and
    truth before correlation.
    """

    values, mask = _targets_and_mask(truth, observed)
    means = np.asarray(prediction, dtype=np.float64)
    if means.shape != values.shape:
        raise BaselineError("prediction must have the same shape as truth")
    centroid = _broadcast(reference, values.shape, "reference")
    scales = _broadcast(scale, values.shape, "scale")
    for label, array in (("prediction", means), ("reference", centroid), ("scale", scales)):
        if not np.all(np.isfinite(array[mask])):
            raise BaselineError(f"observed {label} values must be finite")
    if np.any(scales[mask] <= 0.0):
        raise BaselineError("observed scales must be positive")

    nll_rows: list[float] = []
    mse_rows: list[float] = []
    ordinary: list[float] = []
    adjusted: list[float] = []
    ordinary_undefined = 0
    adjusted_undefined = 0
    nonempty = 0
    for row in range(values.shape[0]):
        selected = mask[row]
        if not np.any(selected):
            continue
        nonempty += 1
        residual = means[row, selected] - values[row, selected]
        row_scales = scales[row, selected]
        nll_rows.append(
            float(np.mean(0.5 * (_LOG_2PI + 2.0 * np.log(row_scales) + (residual / row_scales) ** 2)))
        )
        mse_rows.append(float(np.mean(residual**2)))
        correlation = _pearson(means[row, selected], values[row, selected])
        if correlation is None:
            ordinary_undefined += 1
        else:
            ordinary.append(correlation)
        adjusted_correlation = _pearson(
            means[row, selected] - centroid[row, selected],
            values[row, selected] - centroid[row, selected],
        )
        if adjusted_correlation is None:
            adjusted_undefined += 1
        else:
            adjusted.append(adjusted_correlation)

    observed_count = int(mask.sum())
    record_count = values.shape[0]
    return {
        "value_space": value_space,
        "nll_units": f"nats-per-observed-{value_space}-value",
        "nll": _macro_mean(nll_rows),
        "mse": _macro_mean(mse_rows),
        "profile_pearson_mean": _macro_mean(ordinary),
        "profile_centroid_adjusted_pearson_mean": _macro_mean(adjusted),
        "profile_pearson_undefined": ordinary_undefined,
        "profile_centroid_adjusted_pearson_undefined": adjusted_undefined,
        "record_count": record_count,
        "nonempty_record_count": nonempty,
        "empty_record_count": record_count - nonempty,
        "observed_count": observed_count,
        "coverage": observed_count / mask.size,
        "profile_coverage": nonempty / record_count,
        "profile_pearson_coverage": len(ordinary) / record_count,
        "profile_centroid_adjusted_pearson_coverage": len(adjusted) / record_count,
    }


def compare_paired_nll(
    candidate_prediction: Array,
    baseline_prediction: Array,
    truth: Array,
    observed: Array,
    candidate_scale: Array | float,
    baseline_scale: Array | float,
) -> dict[str, float | int | str]:
    """Compare paired record-level NLLs; positive delta favors the candidate."""

    values, mask = _targets_and_mask(truth, observed)
    candidate = np.asarray(candidate_prediction, dtype=np.float64)
    baseline = np.asarray(baseline_prediction, dtype=np.float64)
    if candidate.shape != values.shape or baseline.shape != values.shape:
        raise BaselineError("candidate and baseline predictions must match truth shape")
    candidate_scales = _broadcast(candidate_scale, values.shape, "candidate_scale")
    baseline_scales = _broadcast(baseline_scale, values.shape, "baseline_scale")
    for label, array in (
        ("candidate_prediction", candidate),
        ("baseline_prediction", baseline),
        ("candidate_scale", candidate_scales),
        ("baseline_scale", baseline_scales),
    ):
        if not np.all(np.isfinite(array[mask])):
            raise BaselineError(f"observed {label} values must be finite")
    if np.any(candidate_scales[mask] <= 0.0) or np.any(baseline_scales[mask] <= 0.0):
        raise BaselineError("observed scales must be positive")

    candidate_rows: list[float] = []
    baseline_rows: list[float] = []
    for row in range(values.shape[0]):
        selected = mask[row]
        if not np.any(selected):
            continue
        candidate_residual = candidate[row, selected] - values[row, selected]
        baseline_residual = baseline[row, selected] - values[row, selected]
        cs = candidate_scales[row, selected]
        bs = baseline_scales[row, selected]
        candidate_rows.append(
            float(np.mean(0.5 * (_LOG_2PI + 2.0 * np.log(cs) + (candidate_residual / cs) ** 2)))
        )
        baseline_rows.append(
            float(np.mean(0.5 * (_LOG_2PI + 2.0 * np.log(bs) + (baseline_residual / bs) ** 2)))
        )
    deltas = np.asarray(baseline_rows) - np.asarray(candidate_rows)
    return {
        "value_space": "log2",
        "delta_definition": "baseline-nll-minus-candidate-nll",
        "paired_record_count": int(deltas.size),
        "candidate_nll": _macro_mean(candidate_rows),
        "baseline_nll": _macro_mean(baseline_rows),
        "mean_nll_delta": float(deltas.mean()) if deltas.size else math.nan,
        "median_nll_delta": float(np.median(deltas)) if deltas.size else math.nan,
        "candidate_better_fraction": float(np.mean(deltas > 0.0)) if deltas.size else math.nan,
    }
