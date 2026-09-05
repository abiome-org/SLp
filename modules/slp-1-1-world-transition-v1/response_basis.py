"""Training-only reduced-rank molecular response baseline.

The model removes per-context training means, learns a shared response basis
from the remaining molecular profiles, and maps static intervention features
to basis coefficients with feature-linear ridge. During OOF calibration every
held action gene is excluded from context means, response SVD, and coefficient
fitting. Complete response panels are required; outcome imputation is forbidden.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from human_baselines import action_oof_fold
from sklearn.utils.extmath import randomized_svd
from transition_baselines import FeatureLinearRidgeBaseline, fit_ridge

Array = np.ndarray


class ResponseBasisError(ValueError):
    """Raised when response-basis fitting would violate the frozen contract."""


@dataclass(frozen=True)
class ContextResidualScale:
    values: Array
    counts: Array
    provenance: str = "action-grouped-oof-training-residuals-with-fold-refit-basis"


@dataclass(frozen=True)
class ResponseBasisBaseline:
    """Context mean plus static-feature prediction in a learned response basis."""

    context_means_: Array
    components_: Array
    coefficient_model_: FeatureLinearRidgeBaseline
    residual_scale_: ContextResidualScale
    rank: int
    alpha: float
    seed: int
    fold_audit_: tuple[dict[str, object], ...]
    baseline_family: str = "training-response-basis-plus-feature-linear-ridge"

    def predict(self, action_features: Array, context_index: Array) -> Array:
        features = np.asarray(action_features, dtype=np.float64)
        contexts = np.asarray(context_index)
        if features.ndim != 2 or features.shape[0] == 0 or not np.isfinite(features).all():
            raise ResponseBasisError("action_features must be a non-empty finite matrix")
        if contexts.shape != (features.shape[0],) or contexts.dtype.kind not in "iu":
            raise ResponseBasisError("context_index must be one integer per action row")
        if np.any(contexts < 0) or np.any(contexts >= self.context_means_.shape[0]):
            raise ResponseBasisError("context_index is out of range")
        coefficients = self.coefficient_model_.predict(features)
        return self.context_means_[contexts] + coefficients @ self.components_

    def scales(self, context_index: Array) -> Array:
        contexts = np.asarray(context_index)
        if contexts.ndim != 1 or contexts.dtype.kind not in "iu":
            raise ResponseBasisError("context_index must be a one-dimensional integer array")
        if np.any(contexts < 0) or np.any(contexts >= self.context_means_.shape[0]):
            raise ResponseBasisError("context_index is out of range")
        return self.residual_scale_.values[contexts]


def _validate(
    action_features: Array,
    targets: Array,
    observed: Array,
    context_index: Array,
    action_ids: Sequence[str],
) -> tuple[Array, Array, Array, tuple[str, ...]]:
    features = np.asarray(action_features, dtype=np.float64)
    values = np.asarray(targets, dtype=np.float64)
    mask = np.asarray(observed)
    contexts = np.asarray(context_index)
    if features.ndim != 2 or features.shape[0] == 0 or not np.isfinite(features).all():
        raise ResponseBasisError("action_features must be a non-empty finite matrix")
    if values.ndim != 2 or values.shape[0] != features.shape[0] or values.shape[1] < 2:
        raise ResponseBasisError("targets must align with action rows and contain readouts")
    if mask.shape != values.shape or mask.dtype != np.bool_:
        raise ResponseBasisError("observed must be a boolean matrix matching targets")
    if not np.all(mask):
        raise ResponseBasisError("response basis requires a complete panel; no outcome imputation")
    if not np.isfinite(values).all():
        raise ResponseBasisError("targets must be finite")
    if contexts.shape != (features.shape[0],) or contexts.dtype.kind not in "iu":
        raise ResponseBasisError("context_index must be one integer per record")
    if np.any(contexts < 0) or set(contexts.tolist()) != set(range(int(contexts.max()) + 1)):
        raise ResponseBasisError("context indices must be contiguous from zero")
    if isinstance(action_ids, (str, bytes)) or len(action_ids) != features.shape[0]:
        raise ResponseBasisError("action_ids must contain one stable ID per record")
    actions = tuple(str(item) for item in action_ids)
    if any(not item or item != item.strip() for item in actions):
        raise ResponseBasisError("action_ids must be non-empty trimmed strings")
    for action in set(actions):
        rows = [index for index, item in enumerate(actions) if item == action]
        if not np.all(features[rows] == features[rows[0]]):
            raise ResponseBasisError(f"repeated action {action!r} has inconsistent static features")
    return features, values, contexts.astype(np.int64), actions


def _context_means(values: Array, contexts: Array) -> Array:
    means = np.empty((int(contexts.max()) + 1, values.shape[1]), dtype=np.float64)
    for context in range(means.shape[0]):
        rows = contexts == context
        if not np.any(rows):
            raise ResponseBasisError(f"context {context} has no fitting records")
        means[context] = values[rows].mean(axis=0)
    return means


def _response_components(residuals: Array, rank: int, seed: int) -> Array:
    maximum = min(residuals.shape)
    if rank > maximum:
        raise ResponseBasisError(f"rank {rank} exceeds response matrix limit {maximum}")
    _, _, components = randomized_svd(
        residuals,
        n_components=rank,
        n_oversamples=10,
        n_iter=3,
        power_iteration_normalizer="auto",
        transpose="auto",
        flip_sign=True,
        random_state=seed,
    )
    if components.shape != (rank, residuals.shape[1]) or not np.isfinite(components).all():
        raise ResponseBasisError("randomized response SVD produced invalid components")
    return components


def _action_set_sha256(actions: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(set(actions))) + "\n").encode("ascii")).hexdigest()


def fit_grouped_oof_response_basis_grid(
    action_features: Array,
    targets: Array,
    observed: Array,
    context_index: Array,
    action_ids: Sequence[str],
    *,
    ranks: Sequence[int] = (16, 32, 64),
    alphas: Sequence[float] = (100.0, 1000.0, 10000.0),
    folds: int = 3,
    seed: int = 731,
    scale_floor: float = 0.05,
) -> dict[tuple[int, float], ResponseBasisBaseline]:
    """Fit a nested-rank grid with fully refit action-grouped OOF calibration."""

    features, values, contexts, actions = _validate(
        action_features, targets, observed, context_index, action_ids
    )
    rank_values = tuple(int(rank) for rank in ranks)
    alpha_values = tuple(float(alpha) for alpha in alphas)
    if (
        not rank_values
        or len(set(rank_values)) != len(rank_values)
        or any(type(rank) is not int or rank < 1 for rank in ranks)
    ):
        raise ResponseBasisError("ranks must be distinct positive integers")
    if (
        not alpha_values
        or len(set(alpha_values)) != len(alpha_values)
        or any(not np.isfinite(alpha) or alpha < 0.0 for alpha in alpha_values)
    ):
        raise ResponseBasisError("alphas must be distinct finite non-negative values")
    if type(folds) is not int or folds < 2:
        raise ResponseBasisError("folds must be an integer of at least two")
    if type(seed) is not int:
        raise ResponseBasisError("seed must be an integer")
    if not np.isfinite(scale_floor) or scale_floor <= 0.0:
        raise ResponseBasisError("scale_floor must be finite and positive")
    max_rank = max(rank_values)

    fold_ids = np.asarray(
        [action_oof_fold(action, folds=folds, seed=seed) for action in actions],
        dtype=np.int64,
    )
    contexts_count = int(contexts.max()) + 1
    squared_error = {
        key: np.zeros((contexts_count, values.shape[1]), dtype=np.float64)
        for key in ((rank, alpha) for rank in rank_values for alpha in alpha_values)
    }
    counts = np.zeros((contexts_count, values.shape[1]), dtype=np.int64)
    fold_audits: list[dict[str, object]] = []
    for fold in range(folds):
        held = fold_ids == fold
        fitting = ~held
        if not np.any(held) or not np.any(fitting):
            raise ResponseBasisError(f"OOF fold {fold} has empty fitting or held records")
        held_actions = {actions[index] for index in np.flatnonzero(held)}
        fitting_actions = {actions[index] for index in np.flatnonzero(fitting)}
        if held_actions & fitting_actions:
            raise ResponseBasisError("an action gene crossed OOF fitting and held rows")
        means = _context_means(values[fitting], contexts[fitting])
        residuals = values[fitting] - means[contexts[fitting]]
        components = _response_components(residuals, max_rank, seed)
        audit = {
            "fold": fold,
            "fittingRecords": int(fitting.sum()),
            "heldRecords": int(held.sum()),
            "fittingActions": len(fitting_actions),
            "heldActions": len(held_actions),
            "actionOverlap": 0,
            "fittingActionSetSha256": _action_set_sha256(tuple(fitting_actions)),
            "heldActionSetSha256": _action_set_sha256(tuple(held_actions)),
            "basisFloat64Sha256": hashlib.sha256(
                components.astype("<f8", copy=False).tobytes(order="C")
            ).hexdigest(),
        }
        fold_audits.append(audit)
        for context in range(contexts_count):
            counts[context] += int(np.count_nonzero(held & (contexts == context)))
        for rank in rank_values:
            local_components = components[:rank]
            scores = residuals @ local_components.T
            score_mask = np.ones_like(scores, dtype=np.bool_)
            for alpha in alpha_values:
                coefficient_model = fit_ridge(
                    features[fitting], scores, score_mask, alpha, scale_floor=scale_floor
                )
                held_prediction = (
                    means[contexts[held]]
                    + coefficient_model.predict(features[held]) @ local_components
                )
                held_residual = held_prediction - values[held]
                for context in range(contexts_count):
                    local = contexts[held] == context
                    if np.any(local):
                        squared_error[(rank, alpha)][context] += np.square(
                            held_residual[local]
                        ).sum(axis=0)

    if np.any(counts == 0):
        raise ResponseBasisError("every context requires held records across OOF calibration")
    full_means = _context_means(values, contexts)
    full_residuals = values - full_means[contexts]
    full_components = _response_components(full_residuals, max_rank, seed)
    models: dict[tuple[int, float], ResponseBasisBaseline] = {}
    for rank in rank_values:
        local_components = full_components[:rank]
        scores = full_residuals @ local_components.T
        score_mask = np.ones_like(scores, dtype=np.bool_)
        for alpha in alpha_values:
            coefficient_model = fit_ridge(
                features, scores, score_mask, alpha, scale_floor=scale_floor
            )
            scales = np.maximum(
                np.sqrt(squared_error[(rank, alpha)] / counts), scale_floor
            )
            models[(rank, alpha)] = ResponseBasisBaseline(
                context_means_=full_means.copy(),
                components_=local_components.copy(),
                coefficient_model_=coefficient_model,
                residual_scale_=ContextResidualScale(scales, counts.copy()),
                rank=rank,
                alpha=alpha,
                seed=seed,
                fold_audit_=tuple(dict(item) for item in fold_audits),
            )
    return models


def fit_grouped_oof_response_basis(
    action_features: Array,
    targets: Array,
    observed: Array,
    context_index: Array,
    action_ids: Sequence[str],
    rank: int,
    alpha: float,
    *,
    folds: int = 3,
    seed: int = 731,
    scale_floor: float = 0.05,
) -> ResponseBasisBaseline:
    """Fit one response-basis configuration using the same nested-grid engine."""

    return fit_grouped_oof_response_basis_grid(
        action_features,
        targets,
        observed,
        context_index,
        action_ids,
        ranks=(rank,),
        alphas=(alpha,),
        folds=folds,
        seed=seed,
        scale_floor=scale_floor,
    )[(rank, float(alpha))]
