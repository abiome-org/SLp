"""Training-only null baselines and diagnostics for human Perturb-seq.

This module consumes the derived development bundle.  It never loads a path,
chooses a candidate, or accesses the sealed test-only artifact.  Duplicate
rows are guide-level per-perturbation cell-mean summaries, not biological
replicates, so their agreement is reported only as a development diagnostic.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

Array = np.ndarray
BaselineKind = Literal["basal_control", "training_perturbation_mean"]
_LOG_2PI = math.log(2.0 * math.pi)
_OOF_NAMESPACE = "slp11-human-baseline-oof-v1"


class HumanBaselineError(ValueError):
    """Raised when an array violates the development-baseline contract."""


@dataclass(frozen=True)
class ContextReferences:
    """Per-context null means and training-only OOF Gaussian scales."""

    context_ids: tuple[str, ...]
    perturbation_mean: Array
    perturbation_scale: Array
    basal_mean: Array
    basal_scale: Array
    scale_counts: Array
    folds: int
    seed: int
    scale_provenance: str = "action-grouped-oof-training-residuals"

    def _select(self, kind: BaselineKind, context_index: Array, *, scale: bool) -> Array:
        contexts = np.asarray(context_index)
        if contexts.ndim != 1 or contexts.dtype.kind not in "iu":
            raise HumanBaselineError("context_index must be a one-dimensional integer array")
        if np.any(contexts < 0) or np.any(contexts >= len(self.context_ids)):
            raise HumanBaselineError("context_index is out of range")
        if kind == "basal_control":
            values = self.basal_scale if scale else self.basal_mean
        elif kind == "training_perturbation_mean":
            values = self.perturbation_scale if scale else self.perturbation_mean
        else:
            raise HumanBaselineError(f"unknown baseline kind: {kind}")
        return values[contexts]

    def predict(self, kind: BaselineKind, context_index: Array) -> Array:
        """Materialize context-matched null means for requested records."""

        return self._select(kind, context_index, scale=False)

    def scales(self, kind: BaselineKind, context_index: Array) -> Array:
        """Materialize frozen context/query scales for requested records."""

        return self._select(kind, context_index, scale=True)


def action_oof_fold(action_id: str, *, folds: int = 5, seed: int = 731) -> int:
    """Assign one stable OOF fold to an action across every context."""

    if not isinstance(action_id, str) or not action_id:
        raise HumanBaselineError("action_id must be a non-empty string")
    if folds < 2:
        raise HumanBaselineError("folds must be at least two")
    payload = f"{_OOF_NAMESPACE}|{seed}|9606|{action_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % folds


def _validate(
    targets: Array,
    observed: Array,
    context_index: Array,
    action_ids: Array,
    indices: Array,
    basal_control: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    values = np.asarray(targets, dtype=np.float64)
    mask = np.asarray(observed)
    contexts = np.asarray(context_index)
    actions = np.asarray(action_ids)
    selected = np.asarray(indices)
    basal = np.asarray(basal_control, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise HumanBaselineError("targets must be a non-empty two-dimensional array")
    if mask.shape != values.shape or mask.dtype != np.bool_:
        raise HumanBaselineError("observed must be a boolean array matching targets")
    if contexts.shape != (values.shape[0],) or contexts.dtype.kind not in "iu":
        raise HumanBaselineError("context_index must be one-dimensional integer data")
    if actions.shape != (values.shape[0],) or actions.dtype.kind not in "US":
        raise HumanBaselineError("action_ids must be one-dimensional strings")
    if selected.ndim != 1 or selected.dtype.kind not in "iu" or not len(selected):
        raise HumanBaselineError("indices must be a non-empty integer array")
    if len(np.unique(selected)) != len(selected):
        raise HumanBaselineError("indices must be unique")
    if np.any(selected < 0) or np.any(selected >= values.shape[0]):
        raise HumanBaselineError("indices are out of range")
    if basal.ndim != 2 or basal.shape[1] != values.shape[1] or not basal.shape[0]:
        raise HumanBaselineError("basal_control must have shape contexts by queries")
    if np.any(contexts < 0) or np.any(contexts >= basal.shape[0]):
        raise HumanBaselineError("context_index is out of range")
    if not np.all(np.isfinite(values[mask])) or not np.all(np.isfinite(basal)):
        raise HumanBaselineError("observed targets and basal controls must be finite")
    return values, mask, contexts.astype(np.int64), actions, selected.astype(np.int64), basal


def _masked_mean(values: Array, mask: Array) -> tuple[Array, Array]:
    counts = mask.sum(axis=0, dtype=np.int64)
    means = np.full(values.shape[1], np.nan, dtype=np.float64)
    present = counts > 0
    means[present] = np.where(mask, values, 0.0).sum(axis=0)[present] / counts[present]
    return means, counts


def _residual_scale(
    prediction: Array, values: Array, mask: Array, floor: float
) -> tuple[Array, Array]:
    counts = mask.sum(axis=0, dtype=np.int64)
    residual = np.where(mask, prediction - values, 0.0)
    scale = np.full(values.shape[1], np.nan, dtype=np.float64)
    present = counts > 0
    scale[present] = np.maximum(
        np.sqrt(np.square(residual).sum(axis=0)[present] / counts[present]), floor
    )
    return scale, counts


def fit_context_references(
    targets: Array,
    observed: Array,
    context_index: Array,
    action_ids: Array,
    train_indices: Array,
    basal_control: Array,
    context_ids: Array | None = None,
    *,
    folds: int = 5,
    seed: int = 731,
    scale_floor: float = 1e-3,
) -> ContextReferences:
    """Fit context null means and action-grouped OOF scales on training rows.

    Every occurrence of an action receives one OOF fold across all contexts.
    A context/query training mean for a held fold excludes every row carrying
    any action in that fold.  The matched basal-control mean is supplied by the
    data adapter and is never re-estimated from perturbation outcomes.
    """

    if folds < 2:
        raise HumanBaselineError("folds must be at least two")
    if not np.isfinite(scale_floor) or scale_floor <= 0:
        raise HumanBaselineError("scale_floor must be finite and positive")
    values, mask, contexts, actions, train, basal = _validate(
        targets, observed, context_index, action_ids, train_indices, basal_control
    )
    if context_ids is None:
        names = tuple(str(index) for index in range(basal.shape[0]))
    else:
        names_array = np.asarray(context_ids)
        if names_array.shape != (basal.shape[0],) or names_array.dtype.kind not in "US":
            raise HumanBaselineError("context_ids must match basal_control contexts")
        names = tuple(str(item) for item in names_array)
    if len(set(names)) != len(names):
        raise HumanBaselineError("context_ids must be unique")

    fold_by_row = np.asarray(
        [action_oof_fold(str(actions[row]), folds=folds, seed=seed) for row in train],
        dtype=np.int64,
    )
    shape = basal.shape
    means = np.full(shape, np.nan, dtype=np.float64)
    perturbation_scale = np.full(shape, np.nan, dtype=np.float64)
    basal_scale = np.full(shape, np.nan, dtype=np.float64)
    scale_counts = np.zeros(shape, dtype=np.int64)
    for context in range(shape[0]):
        local = train[contexts[train] == context]
        if not len(local):
            raise HumanBaselineError(f"context {context} has no training rows")
        local_folds = fold_by_row[contexts[train] == context]
        means[context], counts = _masked_mean(values[local], mask[local])
        if np.any(counts == 0):
            raise HumanBaselineError(f"context {context} has unobserved training queries")

        oof = np.full((len(local), values.shape[1]), np.nan, dtype=np.float64)
        for fold in range(folds):
            held = local_folds == fold
            if not np.any(held):
                continue
            fitting = ~held
            if not np.any(fitting):
                raise HumanBaselineError(f"context {context} OOF fold has no fitting rows")
            fold_mean, fold_counts = _masked_mean(values[local[fitting]], mask[local[fitting]])
            required = mask[local[held]].any(axis=0)
            if np.any(required & (fold_counts == 0)):
                raise HumanBaselineError("an OOF fold has no fitting support for a held query")
            oof[held] = fold_mean
        if np.any(mask[local] & ~np.isfinite(oof)):
            raise HumanBaselineError("OOF predictions are incomplete")
        perturbation_scale[context], scale_counts[context] = _residual_scale(
            oof, values[local], mask[local], scale_floor
        )
        basal_prediction = np.broadcast_to(basal[context], values[local].shape)
        basal_scale[context], basal_counts = _residual_scale(
            basal_prediction, values[local], mask[local], scale_floor
        )
        if not np.array_equal(scale_counts[context], basal_counts):
            raise HumanBaselineError("baseline scale counts disagree")

    return ContextReferences(
        context_ids=names,
        perturbation_mean=means,
        perturbation_scale=perturbation_scale,
        basal_mean=basal.copy(),
        basal_scale=basal_scale,
        scale_counts=scale_counts,
        folds=folds,
        seed=seed,
    )


def _pearson(left: Array, right: Array) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2:
        return None
    left_tolerance = 8.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(left))))
    right_tolerance = 8.0 * np.finfo(np.float64).eps * max(
        1.0, float(np.max(np.abs(right)))
    )
    if np.ptp(left) <= left_tolerance or np.ptp(right) <= right_tolerance:
        return None
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    if denominator <= np.finfo(np.float64).eps:
        return None
    return float((left @ right) / denominator)


def evaluate_context_baselines(
    references: ContextReferences,
    targets: Array,
    observed: Array,
    context_index: Array,
    action_ids: Array,
    validation_indices: Array,
    basal_control: Array,
) -> dict[str, object]:
    """Evaluate fixed null references on validation rows by context."""

    values, mask, contexts, _, validation, _ = _validate(
        targets,
        observed,
        context_index,
        action_ids,
        validation_indices,
        basal_control,
    )
    report: dict[str, object] = {}
    for context, context_id in enumerate(references.context_ids):
        rows = validation[contexts[validation] == context]
        if not len(rows):
            raise HumanBaselineError(f"context {context} has no validation rows")
        truth = values[rows]
        local_mask = mask[rows]
        context_report: dict[str, object] = {}
        for kind in ("basal_control", "training_perturbation_mean"):
            prediction = references.predict(kind, contexts[rows])
            scales = references.scales(kind, contexts[rows])
            residual = np.where(local_mask, prediction - truth, 0.0)
            point_nll = 0.5 * (
                _LOG_2PI + 2.0 * np.log(scales) + np.square(residual / scales)
            )
            correlations: list[float] = []
            undefined = 0
            for query in range(values.shape[1]):
                present = local_mask[:, query]
                correlation = _pearson(prediction[present, query], truth[present, query])
                if correlation is None:
                    undefined += 1
                else:
                    correlations.append(correlation)
            context_report[kind] = {
                "records": len(rows),
                "observedValues": int(local_mask.sum()),
                "gaussianNllNatsPerObservedValue": float(point_nll[local_mask].mean()),
                "rmse": float(np.sqrt(np.square(residual)[local_mask].mean())),
                "perGeneCenteredPearsonMean": (
                    float(np.mean(correlations)) if correlations else None
                ),
                "perGeneCenteredPearsonDefinedGenes": len(correlations),
                "perGeneCenteredPearsonUndefinedGenes": undefined,
            }
        report[context_id] = context_report
    return report


def randomized_pca_explained_variance(
    residuals: Array,
    *,
    components: int = 10,
    oversamples: int = 10,
    power_iterations: int = 2,
    seed: int = 731,
) -> dict[str, object]:
    """Estimate leading PCA variance fractions with deterministic range finding."""

    matrix = np.asarray(residuals, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2 or not np.all(np.isfinite(matrix)):
        raise HumanBaselineError("PCA residuals must be a finite two-dimensional matrix")
    maximum = min(matrix.shape)
    count = min(components, maximum)
    width = min(maximum, count + oversamples)
    if count < 1 or oversamples < 0 or power_iterations < 0:
        raise HumanBaselineError("invalid randomized PCA parameters")
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    total = float(np.square(matrix).sum())
    if total <= np.finfo(np.float64).eps:
        raise HumanBaselineError("PCA residual matrix has zero variance")
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((matrix.shape[1], width))
    basis, _ = np.linalg.qr(matrix @ omega, mode="reduced")
    for _ in range(power_iterations):
        basis, _ = np.linalg.qr(matrix @ (matrix.T @ basis), mode="reduced")
    singular = np.linalg.svd(basis.T @ matrix, compute_uv=False)[:count]
    ratios = np.square(singular) / total
    return {
        "method": "deterministic-randomized-pca",
        "components": int(count),
        "oversamples": int(oversamples),
        "powerIterations": int(power_iterations),
        "seed": int(seed),
        "explainedVarianceRatio": [float(item) for item in ratios],
        "topComponentsCumulativeRatio": float(ratios.sum()),
    }


def duplicate_guide_consistency(
    targets: Array,
    action_ids: Array,
    basal_reference: Array,
    *,
    bootstrap_samples: int = 2_000,
    seed: int = 731,
) -> dict[str, object]:
    """Bootstrap agreement among duplicate guide/cell-mean summaries by action."""

    values = np.asarray(targets, dtype=np.float64)
    actions = np.asarray(action_ids)
    basal = np.asarray(basal_reference, dtype=np.float64)
    if values.ndim != 2 or actions.shape != (values.shape[0],):
        raise HumanBaselineError("duplicate-guide targets and action_ids do not align")
    if basal.shape != (values.shape[1],) or not np.all(np.isfinite(values)):
        raise HumanBaselineError("duplicate-guide values and basal reference must be finite")
    if bootstrap_samples < 1:
        raise HumanBaselineError("bootstrap_samples must be positive")
    grouped: dict[str, list[int]] = {}
    for row, action in enumerate(actions):
        grouped.setdefault(str(action), []).append(row)
    group_scores: list[float] = []
    pair_count = 0
    for rows in grouped.values():
        if len(rows) < 2:
            continue
        pair_scores: list[float] = []
        for offset, left in enumerate(rows[:-1]):
            for right in rows[offset + 1 :]:
                correlation = _pearson(values[left] - basal, values[right] - basal)
                if correlation is not None:
                    pair_scores.append(correlation)
        if pair_scores:
            group_scores.append(float(np.mean(pair_scores)))
            pair_count += len(pair_scores)
    if not group_scores:
        return {
            "unit": "within-action guide/cell-mean summary agreement",
            "profileDefinition": "fixed-transform profile minus matched basal control",
            "aggregation": "pair mean within action, then macro mean across actions",
            "isBiologicalReplicateNoiseCeiling": False,
            "duplicateActionGroups": 0,
            "guideSummaryPairs": 0,
            "meanPairPearson": None,
            "bootstrap95PercentileInterval": None,
            "bootstrapSamples": bootstrap_samples,
        }
    scores = np.asarray(group_scores)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(scores), size=(bootstrap_samples, len(scores)))
    bootstrap = scores[draws].mean(axis=1)
    interval = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "unit": "within-action guide/cell-mean summary agreement",
        "profileDefinition": "fixed-transform profile minus matched basal control",
        "aggregation": "pair mean within action, then macro mean across actions",
        "isBiologicalReplicateNoiseCeiling": False,
        "duplicateActionGroups": len(scores),
        "guideSummaryPairs": int(pair_count),
        "meanPairPearson": float(scores.mean()),
        "bootstrap95PercentileInterval": [float(item) for item in interval],
        "bootstrapSamples": bootstrap_samples,
        "bootstrapResamplingUnit": "action gene",
    }
