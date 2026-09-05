"""Point-metric utilities for heterogeneous four-context development baselines."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


class FourContextBaselineError(ValueError):
    """Raised when a point-baseline metric contract is violated."""


def collapse_equal_records(
    action_ids: Array,
    values: Array,
    observed: Array,
) -> tuple[Array, Array, Array, Array]:
    """Collapse records equally into one missing-aware profile per intervention gene."""

    action_ids = np.asarray(action_ids).astype(str)
    values = np.asarray(values, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    if (
        values.ndim != 2
        or values.shape != observed.shape
        or values.shape[0] != len(action_ids)
    ):
        raise FourContextBaselineError("record axes do not align")
    if not np.isfinite(values[observed]).all():
        raise FourContextBaselineError("observed values must be finite")
    genes = np.asarray(sorted(set(action_ids.tolist())))
    output = np.zeros((len(genes), values.shape[1]), dtype=np.float64)
    mask = np.zeros((len(genes), values.shape[1]), dtype=bool)
    record_counts = np.zeros(len(genes), dtype=np.int64)
    for index, gene in enumerate(genes):
        rows = action_ids == gene
        record_counts[index] = int(rows.sum())
        counts = observed[rows].sum(axis=0)
        mask[index] = counts > 0
        output[index] = np.divide(
            np.where(observed[rows], values[rows], 0.0).sum(axis=0),
            counts,
            out=np.zeros(values.shape[1], dtype=np.float64),
            where=mask[index],
        )
    return genes, output.astype(np.float32), mask, record_counts


def fitting_query_scale(
    gene_profiles: Array, observed: Array, floor: float = 0.05
) -> Array:
    """Compute missing-aware fitting-gene query SD with a fixed floor."""

    values = np.asarray(gene_profiles, dtype=np.float64)
    mask = np.asarray(observed, dtype=bool)
    if values.shape != mask.shape or values.ndim != 2 or floor <= 0:
        raise FourContextBaselineError("invalid fitting profile scale inputs")
    counts = mask.sum(axis=0)
    mean = np.divide(
        np.where(mask, values, 0.0).sum(axis=0),
        counts,
        out=np.zeros(values.shape[1]),
        where=counts > 0,
    )
    variance = np.divide(
        np.where(mask, (values - mean) ** 2, 0.0).sum(axis=0),
        counts,
        out=np.zeros(values.shape[1]),
        where=counts > 0,
    )
    return np.maximum(np.sqrt(variance), floor)


def _pearson(left: Array, right: Array) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2:
        return None
    left_scale = max(1.0, float(np.max(np.abs(left))))
    right_scale = max(1.0, float(np.max(np.abs(right))))
    left = left - left.mean()
    right = right - right.mean()
    left_norm = np.sqrt(np.sum(left * left))
    right_norm = np.sqrt(np.sum(right * right))
    left_tolerance = 8 * np.finfo(np.float32).eps * left_scale * np.sqrt(left.size)
    right_tolerance = 8 * np.finfo(np.float32).eps * right_scale * np.sqrt(right.size)
    if left_norm <= left_tolerance or right_norm <= right_tolerance:
        return None
    return float(np.sum(left * right) / (left_norm * right_norm))


def independently_query_centered_profile_pearson(
    prediction: Array, truth: Array, observed: Array
) -> tuple[float | None, int]:
    """Center prediction/truth separately per query across genes, then correlate profiles."""

    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    mask = np.asarray(observed, dtype=bool)
    if (
        prediction.shape != truth.shape
        or prediction.shape != mask.shape
        or prediction.ndim != 2
    ):
        raise FourContextBaselineError("correlation matrices do not align")
    counts = mask.sum(axis=0)
    prediction_centroid = np.divide(
        np.where(mask, prediction, 0.0).sum(axis=0),
        counts,
        out=np.zeros(prediction.shape[1]),
        where=counts > 0,
    )
    truth_centroid = np.divide(
        np.where(mask, truth, 0.0).sum(axis=0),
        counts,
        out=np.zeros(truth.shape[1]),
        where=counts > 0,
    )
    correlations = []
    undefined = 0
    for row in range(len(prediction)):
        selected = mask[row]
        value = _pearson(
            prediction[row, selected] - prediction_centroid[selected],
            truth[row, selected] - truth_centroid[selected],
        )
        if value is None:
            undefined += 1
        else:
            correlations.append(value)
    return (float(np.mean(correlations)) if correlations else None), undefined


def training_centroid_adjusted_profile_pearson(
    prediction: Array,
    truth: Array,
    observed: Array,
    training_centroid: Array,
) -> tuple[float | None, int]:
    """Correlate per-gene profiles after subtracting one fitting-only target centroid."""

    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    mask = np.asarray(observed, dtype=bool)
    centroid = np.asarray(training_centroid, dtype=np.float64)
    if (
        prediction.shape != truth.shape
        or prediction.shape != mask.shape
        or centroid.shape != (prediction.shape[1],)
    ):
        raise FourContextBaselineError("training-centroid metric axes do not align")
    correlations = []
    undefined = 0
    for row in range(len(prediction)):
        selected = mask[row]
        value = _pearson(
            prediction[row, selected] - centroid[selected],
            truth[row, selected] - centroid[selected],
        )
        if value is None:
            undefined += 1
        else:
            correlations.append(value)
    return (float(np.mean(correlations)) if correlations else None), undefined


def point_metrics(
    prediction: Array,
    truth: Array,
    observed: Array,
    query_scale: Array,
    training_centroid: Array,
) -> dict[str, float | int | None]:
    """Score gene profiles in raw and fitting-query-SD units without likelihood claims."""

    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    mask = np.asarray(observed, dtype=bool)
    scale = np.asarray(query_scale, dtype=np.float64)
    if (
        prediction.shape != truth.shape
        or prediction.shape != mask.shape
        or scale.shape != (prediction.shape[1],)
    ):
        raise FourContextBaselineError("metric axes do not align")
    raw_rows = []
    scaled_rows = []
    for row in range(len(prediction)):
        selected = mask[row]
        if not np.any(selected):
            continue
        error = prediction[row, selected] - truth[row, selected]
        raw_rows.append(float(np.mean(error * error)))
        scaled_rows.append(float(np.mean((error / scale[selected]) ** 2)))
    independent_r, independent_undefined = independently_query_centered_profile_pearson(
        prediction, truth, mask
    )
    training_r, training_undefined = training_centroid_adjusted_profile_pearson(
        prediction, truth, mask, training_centroid
    )
    return {
        "gene_profile_raw_mse": float(np.mean(raw_rows)),
        "gene_profile_fitting_query_sd_scaled_mse": float(np.mean(scaled_rows)),
        "independently_query_centered_profile_pearson": independent_r,
        "independently_query_centered_profile_pearson_undefined_genes": independent_undefined,
        "training_centroid_adjusted_profile_pearson": training_r,
        "training_centroid_adjusted_profile_pearson_undefined_genes": training_undefined,
        "genes": len(prediction),
        "observed_values": int(mask.sum()),
    }
