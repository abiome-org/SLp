"""Gene-grouped fitting-only OOF scale calibration for statistical controls.

Every input record must come from the supplied fitting subset.  Repeated
records for one composite ``(NCBI taxonomy ID, action entity ID)`` are assigned
to one fold, so each OOF prediction comes from a model that saw no quantitative
outcome for that intervention gene.  The returned model is then fit on all
supplied records while retaining scales frozen from those OOF residuals.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from transition_baselines import (
    BaselineError,
    FeatureLinearRidgeBaseline,
    MeanBaseline,
    fit_mean,
    fit_ridge,
)

Array = np.ndarray
ActionKey = tuple[int, str]


class CalibrationError(BaselineError):
    """Raised when grouped fitting-only OOF calibration is not possible."""


def _normalize_action_keys(action_keys: Sequence[ActionKey], records: int) -> tuple[ActionKey, ...]:
    if isinstance(action_keys, (str, bytes)) or len(action_keys) != records:
        raise CalibrationError("action_keys must contain one composite identity per record")
    normalized: list[ActionKey] = []
    for index, key in enumerate(action_keys):
        if (
            not isinstance(key, (tuple, list))
            or len(key) != 2
            or type(key[0]) is not int
            or key[0] <= 0
            or not isinstance(key[1], str)
            or not key[1]
            or key[1] != key[1].strip()
        ):
            raise CalibrationError(f"action_keys[{index}] is not a composite identity")
        normalized.append((key[0], key[1]))
    return tuple(normalized)


def grouped_fold_ids(
    action_keys: Sequence[ActionKey],
    folds: int = 3,
    seed: int = 731,
) -> Array:
    """Return deterministic, balanced record-level fold IDs grouped by gene.

    Groups are ordered by a seed-salted SHA-256 digest, then assigned greedily
    to the fold with the fewest records.  The balancing step guarantees that
    all folds are non-empty when at least ``folds`` distinct genes are supplied.
    """

    if type(folds) is not int or folds < 2:
        raise CalibrationError("folds must be an integer of at least two")
    if type(seed) is not int:
        raise CalibrationError("seed must be an integer")
    try:
        records = len(action_keys)
    except TypeError as error:
        raise CalibrationError("action_keys must be a sized sequence") from error
    if records == 0:
        raise CalibrationError("action_keys must contain at least one record")
    keys = _normalize_action_keys(action_keys, records)

    group_rows: dict[ActionKey, list[int]] = {}
    for row, key in enumerate(keys):
        group_rows.setdefault(key, []).append(row)
    if len(group_rows) < folds:
        raise CalibrationError(
            f"grouped OOF requires at least {folds} distinct action genes; "
            f"received {len(group_rows)}"
        )

    prefix = f"slp11-fitting-oof-calibration-v1|{seed}|"
    ordered_groups: list[tuple[int, bytes, ActionKey, list[int]]] = []
    for key, rows in group_rows.items():
        digest = hashlib.sha256(f"{prefix}{key[0]}|{key[1]}".encode()).digest()
        ordered_groups.append((-len(rows), digest, key, rows))
    ordered_groups.sort(key=lambda item: (item[0], item[1], item[2]))

    fold_sizes = [0] * folds
    assignments = np.full(records, -1, dtype=np.int64)
    for _, digest, _, rows in ordered_groups:
        smallest = min(fold_sizes)
        candidates = [fold for fold, size in enumerate(fold_sizes) if size == smallest]
        offset = int.from_bytes(digest[:8], byteorder="big", signed=False) % len(candidates)
        fold = candidates[offset]
        assignments[rows] = fold
        fold_sizes[fold] += len(rows)

    if np.any(assignments < 0) or any(size == 0 for size in fold_sizes):
        raise CalibrationError("deterministic grouped assignment produced an empty fold")
    for key, rows in group_rows.items():
        if np.unique(assignments[rows]).size != 1:
            raise CalibrationError(f"action gene {key!r} crossed OOF folds")
    return assignments


def _validate_calibration_shapes(
    targets: Array,
    observed: Array,
    action_keys: Sequence[ActionKey],
) -> tuple[Array, Array, tuple[ActionKey, ...]]:
    values = np.asarray(targets, dtype=np.float64)
    mask = np.asarray(observed)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise CalibrationError("targets must be a non-empty two-dimensional array")
    if mask.shape != values.shape:
        raise CalibrationError("observed must have the same shape as targets")
    keys = _normalize_action_keys(action_keys, values.shape[0])
    return values, mask, keys


def _ensure_oof_support(predictions: Array, observed: Array) -> None:
    mask = np.asarray(observed, dtype=bool)
    if not np.all(np.isfinite(predictions[mask])):
        unsupported = np.flatnonzero(np.any(mask & ~np.isfinite(predictions), axis=0))
        preview = ", ".join(str(int(index)) for index in unsupported[:8])
        raise CalibrationError(
            "OOF calibration lacks training-fold support for observed readout indices: " + preview
        )


def fit_grouped_oof_ridge(
    action_features: Array,
    targets: Array,
    observed: Array,
    action_keys: Sequence[ActionKey],
    alpha: float,
    folds: int = 3,
    seed: int = 731,
    scale_floor: float = 0.05,
    return_oof: bool = False,
) -> FeatureLinearRidgeBaseline | tuple[FeatureLinearRidgeBaseline, Array]:
    """Fit ridge on all fitting records with gene-isolated OOF residual scales."""

    features = np.asarray(action_features, dtype=np.float64)
    values, mask, keys = _validate_calibration_shapes(targets, observed, action_keys)
    if features.ndim != 2 or features.shape[0] != values.shape[0]:
        raise CalibrationError(
            "action_features must be two-dimensional with one row per target record"
        )
    fold_ids = grouped_fold_ids(keys, folds=folds, seed=seed)
    oof_predictions = np.full(values.shape, np.nan, dtype=np.float64)
    for fold in range(folds):
        held = fold_ids == fold
        fitting = ~held
        if not np.any(held) or not np.any(fitting):
            raise CalibrationError("every OOF fold must have non-empty fitting and held records")
        fold_model = fit_ridge(
            features[fitting],
            values[fitting],
            mask[fitting],
            alpha,
            scale_floor=scale_floor,
        )
        oof_predictions[held] = fold_model.predict(features[held])
    _ensure_oof_support(oof_predictions, mask)
    model = fit_ridge(
        features,
        values,
        mask,
        alpha,
        oof_predictions=oof_predictions,
        scale_floor=scale_floor,
    )
    return (model, oof_predictions) if return_oof else model


def fit_grouped_oof_mean(
    targets: Array,
    observed: Array,
    action_keys: Sequence[ActionKey],
    folds: int = 3,
    seed: int = 731,
    scale_floor: float = 0.05,
    return_oof: bool = False,
) -> MeanBaseline | tuple[MeanBaseline, Array]:
    """Fit the mean on all fitting records with gene-isolated OOF scales."""

    values, mask, keys = _validate_calibration_shapes(targets, observed, action_keys)
    fold_ids = grouped_fold_ids(keys, folds=folds, seed=seed)
    oof_predictions = np.full(values.shape, np.nan, dtype=np.float64)
    for fold in range(folds):
        held = fold_ids == fold
        fitting = ~held
        if not np.any(held) or not np.any(fitting):
            raise CalibrationError("every OOF fold must have non-empty fitting and held records")
        fold_model = fit_mean(values[fitting], mask[fitting], scale_floor=scale_floor)
        oof_predictions[held] = fold_model.predict(np.zeros((int(held.sum()), 1)))
    _ensure_oof_support(oof_predictions, mask)
    model = fit_mean(
        values,
        mask,
        oof_predictions=oof_predictions,
        scale_floor=scale_floor,
    )
    return (model, oof_predictions) if return_oof else model
