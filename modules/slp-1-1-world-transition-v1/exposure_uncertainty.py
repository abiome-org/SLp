"""Exposure-aware Gaussian uncertainty for pseudobulk molecular outcomes.

This module calibrates uncertainty only. It has no molecular mean-prediction
API, so cell count cannot become an intervention or target feature. For query
``q`` in context ``c`` at exposure ``n``, the variance is

``biological_variance[c, q] + sampling_variance[c, q] / n``.

Callers must supply training-only out-of-fold residuals. If core non-targeting
control pseudobulk means are supplied, their exposure-dependent variation
estimates the sampling component. Otherwise both components are estimated
jointly from perturbation residuals with the identifiability warning retained
on the returned object.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


class ExposureUncertaintyError(ValueError):
    """Raised when exposure calibration inputs violate the frozen contract."""


@dataclass(frozen=True)
class ExposureGaussianUncertainty:
    """Per-context/query variance components fit from training data only."""

    biological_variance_: Array
    sampling_variance_: Array
    residual_counts_: Array
    control_counts_: Array
    sampling_from_controls_: Array
    scale_floor: float
    component_provenance: str
    identifiability_warning: str | None
    uncertainty_family: str = "exposure-aware-gaussian-pseudobulk"

    def scales(self, num_cells: Array, context_index: Array) -> Array:
        """Return frozen Gaussian scales for planned or observed exposures."""

        counts = _exposures(num_cells, "num_cells")
        contexts = _contexts(
            context_index,
            counts.size,
            self.biological_variance_.shape[0],
            require_all=False,
        )
        variance = (
            self.biological_variance_[contexts]
            + self.sampling_variance_[contexts] / counts[:, None]
        )
        variance = np.maximum(variance, self.scale_floor**2)
        if not np.isfinite(variance).all():
            raise ExposureUncertaintyError("requested exposures produced non-finite variance")
        return np.sqrt(variance)


def _values_and_mask(values: Array, observed: Array, label: str) -> tuple[Array, Array]:
    matrix = np.asarray(values, dtype=np.float64)
    mask_input = np.asarray(observed)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ExposureUncertaintyError(f"{label} must be a non-empty matrix")
    if mask_input.shape != matrix.shape or mask_input.dtype != np.bool_:
        raise ExposureUncertaintyError(f"{label} observed mask must be a matching boolean matrix")
    if not np.isfinite(matrix[mask_input]).all():
        raise ExposureUncertaintyError(f"observed {label} values must be finite")
    return matrix, mask_input


def _exposures(value: Array, label: str) -> Array:
    counts = np.asarray(value, dtype=np.float64)
    if counts.ndim != 1 or counts.size == 0:
        raise ExposureUncertaintyError(f"{label} must be a non-empty vector")
    if not np.isfinite(counts).all() or np.any(counts <= 0.0):
        raise ExposureUncertaintyError(f"{label} must contain finite positive cell counts")
    return counts


def _contexts(
    value: Array,
    records: int,
    context_count: int | None = None,
    *,
    require_all: bool,
) -> Array:
    contexts = np.asarray(value)
    if contexts.shape != (records,) or contexts.dtype.kind not in "iu":
        raise ExposureUncertaintyError("context_index must be one integer per record")
    if np.any(contexts < 0):
        raise ExposureUncertaintyError("context_index cannot be negative")
    inferred = int(contexts.max()) + 1
    if context_count is None:
        context_count = inferred
    if np.any(contexts >= context_count):
        raise ExposureUncertaintyError("context_index is out of range")
    if require_all and set(contexts.tolist()) != set(range(context_count)):
        raise ExposureUncertaintyError("every contiguous context must have fitting records")
    return contexts.astype(np.int64, copy=False)


def _nnls_variance_components(
    squared: Array, mask: Array, inverse_exposure: Array
) -> tuple[Array, Array, Array, Array]:
    """Solve missing-aware two-column NNLS by checking all active sets."""

    weights = mask.astype(np.float64, copy=False)
    z = inverse_exposure[:, None]
    count = weights.sum(axis=0, dtype=np.float64)
    sum_z = (weights * z).sum(axis=0)
    sum_zz = (weights * z * z).sum(axis=0)
    sum_y = np.where(mask, squared, 0.0).sum(axis=0)
    sum_zy = np.where(mask, squared * z, 0.0).sum(axis=0)
    sum_yy = np.where(mask, squared * squared, 0.0).sum(axis=0)

    if np.any(count == 0):
        raise ExposureUncertaintyError("every query requires fitting residual support")
    biological_boundary = sum_y / count
    sampling_boundary = np.divide(
        sum_zy,
        sum_zz,
        out=np.zeros_like(sum_zy),
        where=sum_zz > 0.0,
    )
    candidates_a = np.stack(
        (biological_boundary, np.zeros_like(count), np.zeros_like(count)), axis=0
    )
    candidates_b = np.stack(
        (np.zeros_like(count), sampling_boundary, np.zeros_like(count)), axis=0
    )

    determinant = count * sum_zz - sum_z**2
    tolerance = np.finfo(np.float64).eps * np.maximum(count * sum_zz, 1.0) * 32.0
    identifiable = determinant > tolerance
    unconstrained_a = np.divide(
        sum_y * sum_zz - sum_z * sum_zy,
        determinant,
        out=np.zeros_like(sum_y),
        where=identifiable,
    )
    unconstrained_b = np.divide(
        count * sum_zy - sum_z * sum_y,
        determinant,
        out=np.zeros_like(sum_y),
        where=identifiable,
    )
    feasible = identifiable & (unconstrained_a >= 0.0) & (unconstrained_b >= 0.0)
    candidates_a = np.concatenate((candidates_a, unconstrained_a[None, :]), axis=0)
    candidates_b = np.concatenate((candidates_b, unconstrained_b[None, :]), axis=0)

    error = (
        sum_yy[None, :]
        - 2.0 * candidates_a * sum_y[None, :]
        - 2.0 * candidates_b * sum_zy[None, :]
        + candidates_a**2 * count[None, :]
        + 2.0 * candidates_a * candidates_b * sum_z[None, :]
        + candidates_b**2 * sum_zz[None, :]
    )
    error[3, ~feasible] = np.inf
    selected = np.argmin(error, axis=0)
    query = np.arange(squared.shape[1])
    return (
        candidates_a[selected, query],
        candidates_b[selected, query],
        count.astype(np.int64),
        identifiable,
    )


def _center_controls(values: Array, mask: Array) -> Array:
    counts = mask.sum(axis=0, dtype=np.int64)
    if np.any(counts == 0):
        raise ExposureUncertaintyError("every query requires core-control support")
    means = np.where(mask, values, 0.0).sum(axis=0) / counts
    return np.square(values - means, where=mask, out=np.zeros_like(values))


def fit_exposure_uncertainty(
    oof_residuals: Array,
    observed: Array,
    num_cells: Array,
    context_index: Array,
    *,
    control_targets: Array | None = None,
    control_observed: Array | None = None,
    control_num_cells: Array | None = None,
    control_context_index: Array | None = None,
    scale_floor: float = 0.05,
) -> ExposureGaussianUncertainty:
    """Fit exposure-aware variance from training-only OOF residuals.

    The four ``control_*`` arguments must be supplied together. Controls are
    centered within context/query, and their squared deviations estimate the
    ``1 / num_cells`` slope. Queries lacking identifiable control exposure
    variation fall back to joint NNLS on perturbation OOF residuals and retain
    an explicit warning.
    """

    residuals, mask = _values_and_mask(oof_residuals, observed, "oof_residuals")
    exposures = _exposures(num_cells, "num_cells")
    if exposures.shape != (residuals.shape[0],):
        raise ExposureUncertaintyError("num_cells must align with OOF residual rows")
    contexts = _contexts(context_index, residuals.shape[0], require_all=True)
    context_count = int(contexts.max()) + 1
    if not np.isfinite(scale_floor) or scale_floor <= 0.0:
        raise ExposureUncertaintyError("scale_floor must be finite and positive")

    control_arguments = (
        control_targets,
        control_observed,
        control_num_cells,
        control_context_index,
    )
    controls_supplied = all(item is not None for item in control_arguments)
    if any(item is not None for item in control_arguments) and not controls_supplied:
        raise ExposureUncertaintyError("all four control inputs must be supplied together")

    shape = (context_count, residuals.shape[1])
    biological = np.empty(shape, dtype=np.float64)
    sampling = np.empty(shape, dtype=np.float64)
    residual_counts = np.empty(shape, dtype=np.int64)
    control_counts = np.zeros(shape, dtype=np.int64)
    from_controls = np.zeros(shape, dtype=np.bool_)
    fallback_queries = 0

    if controls_supplied:
        controls, controls_mask = _values_and_mask(
            control_targets, control_observed, "control_targets"
        )
        if controls.shape[1] != residuals.shape[1]:
            raise ExposureUncertaintyError("control targets must share the residual query axis")
        control_exposures = _exposures(control_num_cells, "control_num_cells")
        if control_exposures.shape != (controls.shape[0],):
            raise ExposureUncertaintyError("control_num_cells must align with control rows")
        control_contexts = _contexts(
            control_context_index,
            controls.shape[0],
            context_count,
            require_all=True,
        )

    for context in range(context_count):
        selected = contexts == context
        squared = np.square(residuals[selected], where=mask[selected], out=np.zeros_like(residuals[selected]))
        joint_a, joint_b, counts, _ = _nnls_variance_components(
            squared, mask[selected], 1.0 / exposures[selected]
        )
        residual_counts[context] = counts
        if not controls_supplied:
            biological[context] = joint_a
            sampling[context] = joint_b
            continue

        control_selected = control_contexts == context
        local_controls = controls[control_selected]
        local_mask = controls_mask[control_selected]
        control_squared = _center_controls(local_controls, local_mask)
        _, control_b, local_control_counts, identifiable = _nnls_variance_components(
            control_squared,
            local_mask,
            1.0 / control_exposures[control_selected],
        )
        control_counts[context] = local_control_counts
        use_control = identifiable & (local_control_counts >= 3)
        from_controls[context] = use_control
        sampling[context] = np.where(use_control, control_b, joint_b)
        adjusted = np.where(
            mask[selected],
            squared - sampling[context][None, :] / exposures[selected, None],
            0.0,
        )
        biological[context] = np.maximum(adjusted.sum(axis=0) / counts, 0.0)
        fallback_queries += int(np.count_nonzero(~use_control))

    if not np.isfinite(biological).all() or not np.isfinite(sampling).all():
        raise ExposureUncertaintyError("variance component fitting produced non-finite values")
    warning: str | None
    if not controls_supplied:
        warning = (
            "Biological and sampling variance were jointly estimated from perturbation "
            "OOF residuals. Exposure association alone does not identify their causes."
        )
        provenance = "training-oof-residual-joint-nonnegative-regression"
    elif fallback_queries:
        warning = (
            f"Core controls lacked identifiable exposure variation for {fallback_queries} "
            "context/query components; those sampling components used joint OOF-residual "
            "regression and are not causally identifiable."
        )
        provenance = "core-control-slope-with-training-oof-residual-fallback"
    else:
        warning = None
        provenance = "core-control-sampling-slope-plus-training-oof-biological-intercept"
    return ExposureGaussianUncertainty(
        biological_variance_=biological,
        sampling_variance_=sampling,
        residual_counts_=residual_counts,
        control_counts_=control_counts,
        sampling_from_controls_=from_controls,
        scale_floor=float(scale_floor),
        component_provenance=provenance,
        identifiability_warning=warning,
    )
