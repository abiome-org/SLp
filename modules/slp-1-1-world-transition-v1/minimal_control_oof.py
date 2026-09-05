"""Pure contracts for fitting-only minimal-control neural OOF calibration."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

import numpy as np

Array = np.ndarray


class MinimalControlOofError(ValueError):
    """Raised when an OOF calibration contract is violated."""


def array_sha256(array: Array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def string_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{item}\n" for item in values).encode()).hexdigest()


def make_fold_plan(
    action_ids: Array,
    train: Array,
    outer_validation: Array,
    fold_ids: Array,
    *,
    folds: int = 3,
) -> list[dict[str, object]]:
    """Prove global held-gene folds exclude outer validation and partition fitting rows."""

    actions = np.asarray(action_ids).astype(str)
    train = np.asarray(train)
    validation = np.asarray(outer_validation)
    fold_ids = np.asarray(fold_ids)
    if train.ndim != 1 or validation.ndim != 1 or fold_ids.shape != train.shape:
        raise MinimalControlOofError("split and fold arrays must be one-dimensional")
    if any(item.dtype.kind not in "iu" for item in (train, validation, fold_ids)):
        raise MinimalControlOofError("split and fold arrays must contain integers")
    if not train.size or not validation.size:
        raise MinimalControlOofError("train and outer validation must be nonempty")
    if np.unique(train).size != train.size or np.unique(validation).size != validation.size:
        raise MinimalControlOofError("outer split indices must be unique")
    if np.intersect1d(train, validation).size:
        raise MinimalControlOofError("outer validation overlaps fitting rows")
    if np.any(train < 0) or np.any(train >= actions.size):
        raise MinimalControlOofError("training index is out of range")
    if set(fold_ids.tolist()) != set(range(folds)):
        raise MinimalControlOofError("every OOF fold must be represented")
    training_actions = actions[train]
    for action in np.unique(training_actions):
        if np.unique(fold_ids[training_actions == action]).size != 1:
            raise MinimalControlOofError(f"intervention gene crosses OOF folds: {action}")
    plan: list[dict[str, object]] = []
    for fold in range(folds):
        held = train[fold_ids == fold]
        fitting = train[fold_ids != fold]
        held_actions = sorted(set(actions[held]))
        fitting_actions = sorted(set(actions[fitting]))
        if not held.size or not fitting.size or set(held_actions) & set(fitting_actions):
            raise MinimalControlOofError("fold has empty rows or intervention leakage")
        if np.intersect1d(held, validation).size or np.intersect1d(fitting, validation).size:
            raise MinimalControlOofError("fold contains outer-validation rows")
        plan.append(
            {
                "fold": fold,
                "fittingRows": fitting,
                "heldRows": held,
                "fittingRecords": int(fitting.size),
                "heldRecords": int(held.size),
                "fittingInterventions": len(fitting_actions),
                "heldInterventions": len(held_actions),
                "fittingRowSha256": array_sha256(fitting.astype("<i8")),
                "heldRowSha256": array_sha256(held.astype("<i8")),
                "fittingGeneSha256": string_sha256(fitting_actions),
                "heldGeneSha256": string_sha256(held_actions),
            }
        )
    held_union = np.sort(np.concatenate([np.asarray(item["heldRows"]) for item in plan]))
    if not np.array_equal(held_union, np.sort(train)):
        raise MinimalControlOofError("held folds do not partition fitting rows")
    return plan


def collect_oof_predictions(
    record_count: int,
    query_count: int,
    plan: Sequence[dict[str, object]],
    predict_fold: Callable[[Array, Array, int], Array],
) -> Array:
    """Collect one independently fitted held-fold forecast without overlap."""

    result = np.full((record_count, query_count), np.nan, dtype=np.float32)
    for item in plan:
        fitting = np.asarray(item["fittingRows"], dtype=np.int64)
        held = np.asarray(item["heldRows"], dtype=np.int64)
        prediction = np.asarray(predict_fold(fitting, held, int(item["fold"])), dtype=np.float32)
        if prediction.shape != (held.size, query_count) or not np.isfinite(prediction).all():
            raise MinimalControlOofError("fold predictor returned invalid means")
        if np.isfinite(result[held]).any():
            raise MinimalControlOofError("a fitting row received multiple OOF forecasts")
        result[held] = prediction
    return result


def select_common_context_tokens(
    basal: Array,
    basal_observed: Array,
    *,
    tokens: int = 64,
    expected_common: int = 6789,
) -> tuple[Array, Array]:
    """Select stable highest-variance control tokens from the exact common mask."""

    values = np.asarray(basal, dtype=np.float64)
    observed = np.asarray(basal_observed)
    if values.ndim != 2 or observed.shape != values.shape or observed.dtype != np.bool_:
        raise MinimalControlOofError("basal values and boolean mask must align")
    common = observed.all(axis=0)
    if int(common.sum()) != expected_common:
        raise MinimalControlOofError("common basal query count drift")
    if type(tokens) is not int or tokens < 1 or tokens > expected_common:
        raise MinimalControlOofError("invalid context token count")
    common_indices = np.flatnonzero(common)
    selected = common_indices[
        np.argsort(-values[:, common].var(axis=0), kind="stable")[:tokens]
    ]
    means = np.asarray([values[row, common].mean() for row in range(values.shape[0])])[:, None]
    scales = np.maximum(
        np.asarray([values[row, common].std() for row in range(values.shape[0])])[:, None],
        1e-5,
    )
    normalized = np.where(observed, (values - means) / scales, 0.0).astype(np.float32)
    return selected.astype(np.int64), normalized


def pooled_delta_amplitude(targets: Array, oof_mean: Array, observed: Array, floor: float = 0.05) -> Array:
    """Fit one query amplitude from inner OOF mean residuals on fitting rows."""

    values = np.asarray(targets, dtype=np.float64)
    means = np.asarray(oof_mean, dtype=np.float64)
    mask = np.asarray(observed)
    if values.shape != means.shape or mask.shape != values.shape or mask.dtype != np.bool_:
        raise MinimalControlOofError("amplitude arrays must align")
    if not np.all(mask) or not np.isfinite(values).all() or not np.isfinite(means).all():
        raise MinimalControlOofError("amplitude requires complete finite fitting arrays")
    if not np.isfinite(floor) or floor <= 0.0:
        raise MinimalControlOofError("amplitude floor must be positive")
    return np.maximum(np.sqrt(np.mean(np.square(values - means), axis=0)), floor).astype(np.float32)


def advancement_checks(
    world_nll: float,
    mean_nll: float,
    ridge_nll: float,
    adjusted_pearson: float,
) -> dict[str, bool]:
    """Apply the frozen per-context calibration advancement rule."""

    values = np.asarray([world_nll, mean_nll, ridge_nll, adjusted_pearson], dtype=np.float64)
    if not np.isfinite(values).all():
        raise MinimalControlOofError("advancement metrics must be finite")
    return {
        "nllDeltaAgainstMeanAtLeast002": mean_nll - world_nll >= 0.02,
        "nllDeltaAgainstRidgeAtLeast002": ridge_nll - world_nll >= 0.02,
        "adjustedPearsonAtLeast010": adjusted_pearson >= 0.10,
    }
