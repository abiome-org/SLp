"""Masked static-feature baselines for the Nadal-Ribelles yeast RNA summaries.

The linear model predicts each molecular readout from intervention-gene static
features.  The nonlinear model first forms training-only Nyström RBF features
and then applies the same feature-linear readout.  Neither is a world model,
and neither consumes query identity or quantitative outcomes as input features.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import pdist

Array = np.ndarray
FOLD_DOMAIN = "slp11-yeast-static-baseline-inner-v1"


class StaticBaselineError(ValueError):
    """Raised when an array violates the frozen baseline contract."""


def grouped_folds(action_ids: Array, *, folds: int = 3, seed: int = 731) -> Array:
    """Assign every occurrence of a stable action identity to one hash fold."""
    ids = np.asarray(action_ids).astype(str)
    if ids.ndim != 1 or not ids.size or type(folds) is not int or folds < 2:
        raise StaticBaselineError("action_ids and folds do not define grouped folds")
    mapping = {
        identity: int.from_bytes(
            hashlib.sha256(
                f"{FOLD_DOMAIN}|{seed}|4932|{identity}".encode(),
            ).digest()[:8],
            "big",
        )
        % folds
        for identity in set(ids)
    }
    result = np.asarray([mapping[identity] for identity in ids], dtype=np.int64)
    if set(result) != set(range(folds)):
        raise StaticBaselineError("grouped fold assignment contains an empty fold")
    return result


def _features(value: Array) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not result.shape[0] or not result.shape[1]:
        raise StaticBaselineError("features must be a non-empty matrix")
    if not np.isfinite(result).all():
        raise StaticBaselineError("features must be finite")
    return result


def _targets(value: Array, observed: Array) -> tuple[Array, Array]:
    target = np.asarray(value, dtype=np.float64)
    mask = np.asarray(observed)
    if target.ndim != 2 or not target.shape[0] or not target.shape[1]:
        raise StaticBaselineError("targets must be a non-empty matrix")
    if mask.shape != target.shape:
        raise StaticBaselineError("observed must match targets")
    if mask.dtype != np.bool_ and not np.isin(mask, (0, 1)).all():
        raise StaticBaselineError("observed must be boolean")
    mask = mask.astype(bool, copy=False)
    if not np.isfinite(target[mask]).all():
        raise StaticBaselineError("observed targets must be finite")
    return target, mask


def _mask_groups(mask: Array) -> list[tuple[Array, Array]]:
    groups: dict[bytes, list[int]] = {}
    for query in range(mask.shape[1]):
        key = np.packbits(mask[:, query], bitorder="little").tobytes()
        groups.setdefault(key, []).append(query)
    return [
        (mask[:, queries[0]], np.asarray(queries, dtype=np.int64))
        for queries in groups.values()
    ]


@dataclass(frozen=True)
class MeanModel:
    intercept: Array

    def predict(self, features: Array) -> Array:
        rows = _features(features).shape[0]
        return np.broadcast_to(self.intercept, (rows, self.intercept.size)).copy()


@dataclass(frozen=True)
class RidgeModel:
    feature_mean: Array
    feature_scale: Array
    intercept: Array
    coefficient: Array
    alpha: float

    def predict(self, features: Array) -> Array:
        value = _features(features)
        if value.shape[1] != self.feature_mean.size:
            raise StaticBaselineError("feature width differs from fitted model")
        return self.intercept + ((value - self.feature_mean) / self.feature_scale) @ self.coefficient

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            model_type=np.asarray("masked-feature-linear-ridge-v1"),
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            intercept=self.intercept,
            coefficient=self.coefficient,
            alpha=np.asarray(self.alpha, dtype=np.float64),
        )

    @classmethod
    def load(cls, path: Path) -> RidgeModel:
        with np.load(path, allow_pickle=False) as item:
            if str(item["model_type"]) != "masked-feature-linear-ridge-v1":
                raise StaticBaselineError("ridge model type mismatch")
            return cls(
                feature_mean=item["feature_mean"],
                feature_scale=item["feature_scale"],
                intercept=item["intercept"],
                coefficient=item["coefficient"],
                alpha=float(item["alpha"]),
            )


def fit_mean(targets: Array, observed: Array) -> MeanModel:
    """Fit exact per-query means from observed fitting values only."""
    value, mask = _targets(targets, observed)
    counts = mask.sum(axis=0)
    intercept = np.full(value.shape[1], np.nan, dtype=np.float64)
    present = counts > 0
    intercept[present] = np.where(mask, value, 0.0).sum(axis=0)[present] / counts[present]
    return MeanModel(intercept)


def fit_target_scale(
    targets: Array,
    observed: Array,
    *,
    floor: float = 0.05,
) -> Array:
    """Fit per-query population SD on fitting records only."""
    if not np.isfinite(floor) or floor <= 0:
        raise StaticBaselineError("scale floor must be finite and positive")
    value, mask = _targets(targets, observed)
    mean = fit_mean(value, mask).intercept
    count = mask.sum(axis=0)
    centered = np.where(mask, value - mean, 0.0)
    scale = np.full(value.shape[1], np.nan, dtype=np.float64)
    present = count > 0
    scale[present] = np.maximum(
        np.sqrt(np.sum(centered[:, present] ** 2, axis=0) / count[present]), floor,
    )
    return scale


def fit_ridge(
    features: Array,
    targets: Array,
    observed: Array,
    alpha: float,
) -> RidgeModel:
    """Fit an exact missing-aware ridge, sharing solves for identical masks."""
    x = _features(features)
    y, mask = _targets(targets, observed)
    if x.shape[0] != y.shape[0]:
        raise StaticBaselineError("feature and target row counts differ")
    if not np.isfinite(alpha) or alpha <= 0:
        raise StaticBaselineError("alpha must be finite and positive")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale <= np.finfo(np.float64).eps] = 1.0
    design = np.column_stack((np.ones(x.shape[0]), (x - mean) / scale))
    penalty = np.diag(np.r_[0.0, np.full(x.shape[1], float(alpha))])
    intercept = np.full(y.shape[1], np.nan, dtype=np.float64)
    coefficient = np.zeros((x.shape[1], y.shape[1]), dtype=np.float64)
    for rows, queries in _mask_groups(mask):
        if not rows.any():
            continue
        local_design = design[rows]
        gram = local_design.T @ local_design + penalty
        rhs = local_design.T @ y[np.ix_(rows, queries)]
        try:
            solution = cho_solve(
                cho_factor(gram, lower=True, check_finite=False), rhs, check_finite=False,
            )
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        intercept[queries] = solution[0]
        coefficient[:, queries] = solution[1:]
    return RidgeModel(mean, scale, intercept, coefficient, float(alpha))


@dataclass(frozen=True)
class NystromMap:
    input_mean: Array
    input_scale: Array
    landmarks: Array
    landmark_ids: Array
    bandwidth: float
    eigenvectors: Array
    eigenvalues: Array
    requested_landmarks: int
    seed: int

    def transform(self, features: Array) -> Array:
        value = _features(features)
        if value.shape[1] != self.input_mean.size:
            raise StaticBaselineError("Nyström input width mismatch")
        standardized = (value - self.input_mean) / self.input_scale
        squared = (
            np.sum(standardized**2, axis=1)[:, None]
            + np.sum(self.landmarks**2, axis=1)[None, :]
            - 2.0 * standardized @ self.landmarks.T
        )
        kernel = np.exp(-np.maximum(squared, 0.0) / (2.0 * self.bandwidth**2))
        return kernel @ (self.eigenvectors / np.sqrt(self.eigenvalues)[None, :])


@dataclass(frozen=True)
class NystromRidgeModel:
    mapping: NystromMap
    ridge: RidgeModel

    def predict(self, features: Array) -> Array:
        return self.ridge.predict(self.mapping.transform(features))

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            model_type=np.asarray("nystrom-rbf-plus-masked-feature-linear-ridge-v1"),
            input_mean=self.mapping.input_mean,
            input_scale=self.mapping.input_scale,
            landmarks=self.mapping.landmarks,
            landmark_ids=self.mapping.landmark_ids,
            bandwidth=np.asarray(self.mapping.bandwidth),
            eigenvectors=self.mapping.eigenvectors,
            eigenvalues=self.mapping.eigenvalues,
            requested_landmarks=np.asarray(self.mapping.requested_landmarks),
            seed=np.asarray(self.mapping.seed),
            ridge_feature_mean=self.ridge.feature_mean,
            ridge_feature_scale=self.ridge.feature_scale,
            intercept=self.ridge.intercept,
            coefficient=self.ridge.coefficient,
            alpha=np.asarray(self.ridge.alpha),
        )

    @classmethod
    def load(cls, path: Path) -> NystromRidgeModel:
        with np.load(path, allow_pickle=False) as item:
            if str(item["model_type"]) != "nystrom-rbf-plus-masked-feature-linear-ridge-v1":
                raise StaticBaselineError("Nyström model type mismatch")
            mapping = NystromMap(
                item["input_mean"], item["input_scale"], item["landmarks"],
                item["landmark_ids"], float(item["bandwidth"]), item["eigenvectors"],
                item["eigenvalues"], int(item["requested_landmarks"]), int(item["seed"]),
            )
            ridge = RidgeModel(
                item["ridge_feature_mean"], item["ridge_feature_scale"], item["intercept"],
                item["coefficient"], float(item["alpha"]),
            )
        return cls(mapping, ridge)


def fit_nystrom_map(
    features: Array,
    action_ids: Array,
    *,
    landmarks: int = 256,
    seed: int = 731,
) -> NystromMap:
    """Fit a median-bandwidth RBF Nyström map on fitting actions only."""
    x = _features(features)
    ids = np.asarray(action_ids).astype(str)
    if ids.shape != (x.shape[0],) or len(set(ids)) != len(ids):
        raise StaticBaselineError("Nyström fitting actions must be unique")
    if type(landmarks) is not int or landmarks < 2:
        raise StaticBaselineError("landmarks must be an integer of at least two")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale <= np.finfo(np.float64).eps] = 1.0
    standardized = (x - mean) / scale
    distances = pdist(standardized)
    positive = distances[distances > np.finfo(np.float64).eps]
    if not positive.size:
        raise StaticBaselineError("training features contain no positive pair distance")
    bandwidth = float(np.median(positive))
    distinct: dict[bytes, tuple[bytes, str, Array]] = {}
    for identity, row in zip(ids, standardized, strict=True):
        key = np.asarray(row, dtype="<f8").tobytes()
        candidate = (
            hashlib.sha256(f"slp11-yeast-nystrom-v1|{seed}|{identity}".encode()).digest(),
            identity,
            row,
        )
        if key not in distinct or candidate[:2] < distinct[key][:2]:
            distinct[key] = candidate
    ordered = sorted(distinct.values(), key=lambda item: item[:2])
    if len(ordered) < landmarks:
        raise StaticBaselineError("not enough distinct feature rows for requested landmarks")
    chosen = ordered[:landmarks]
    landmark_values = np.stack([item[2] for item in chosen])
    landmark_ids = np.asarray([item[1] for item in chosen])
    squared = (
        np.sum(landmark_values**2, axis=1)[:, None]
        + np.sum(landmark_values**2, axis=1)[None, :]
        - 2.0 * landmark_values @ landmark_values.T
    )
    kernel = np.exp(-np.maximum(squared, 0.0) / (2.0 * bandwidth**2))
    eigenvalues, eigenvectors = np.linalg.eigh((kernel + kernel.T) * 0.5)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    keep = eigenvalues > eigenvalues[0] * 1e-10
    return NystromMap(
        mean, scale, landmark_values, landmark_ids, bandwidth,
        eigenvectors[:, keep], eigenvalues[keep], landmarks, seed,
    )


def fit_nystrom_ridge(
    features: Array,
    action_ids: Array,
    targets: Array,
    observed: Array,
    alpha: float,
    *,
    landmarks: int = 256,
    seed: int = 731,
) -> NystromRidgeModel:
    mapping = fit_nystrom_map(features, action_ids, landmarks=landmarks, seed=seed)
    ridge = fit_ridge(mapping.transform(features), targets, observed, alpha)
    return NystromRidgeModel(mapping, ridge)


def _pearson(left: Array, right: Array, *, tolerance: float = 1e-12) -> float | None:
    if left.size < 2:
        return None
    a, b = left - left.mean(), right - right.mean()
    left_energy, right_energy = float(a @ a), float(b @ b)
    left_floor = tolerance * max(float(left @ left), float(left.size))
    right_floor = tolerance * max(float(right @ right), float(right.size))
    if left_energy <= left_floor or right_energy <= right_floor:
        return None
    denominator = math.sqrt(left_energy * right_energy)
    return float((a @ b) / denominator)


def evaluate_gene_profiles(
    prediction: Array,
    truth: Array,
    observed: Array,
    training_centroid: Array,
    training_scale: Array,
) -> dict[str, float | int | None]:
    """Evaluate equal-gene masked profiles, including independent query centering."""
    target, mask = _targets(truth, observed)
    pred = np.asarray(prediction, dtype=np.float64)
    centroid = np.asarray(training_centroid, dtype=np.float64)
    scale = np.asarray(training_scale, dtype=np.float64)
    if pred.shape != target.shape or centroid.shape != (target.shape[1],):
        raise StaticBaselineError("prediction or centroid shape mismatch")
    if scale.shape != (target.shape[1],):
        raise StaticBaselineError("training_scale shape mismatch")
    eligible = np.isfinite(centroid) & np.isfinite(scale) & (scale > 0)
    mask = mask & eligible[None, :] & np.isfinite(pred)
    active_queries = mask.any(axis=0)
    prediction_center = np.full(target.shape[1], np.nan)
    truth_center = np.full(target.shape[1], np.nan)
    counts = mask.sum(axis=0)
    prediction_center[active_queries] = np.where(mask, pred, 0.0).sum(axis=0)[active_queries] / counts[active_queries]
    truth_center[active_queries] = np.where(mask, target, 0.0).sum(axis=0)[active_queries] / counts[active_queries]
    mse: list[float] = []
    standardized_mse: list[float] = []
    ordinary: list[float] = []
    adjusted: list[float] = []
    independent: list[float] = []
    undefined = {"ordinary": 0, "trainingCentroid": 0, "independentQueryCentered": 0}
    for row in range(target.shape[0]):
        use = mask[row]
        if not use.any():
            continue
        residual = pred[row, use] - target[row, use]
        mse.append(float(np.mean(residual**2)))
        standardized_mse.append(float(np.mean((residual / scale[use]) ** 2)))
        values = (
            ("ordinary", pred[row, use], target[row, use], ordinary),
            (
                "trainingCentroid", pred[row, use] - centroid[use],
                target[row, use] - centroid[use], adjusted,
            ),
            (
                "independentQueryCentered", pred[row, use] - prediction_center[use],
                target[row, use] - truth_center[use], independent,
            ),
        )
        for name, left, right, destination in values:
            correlation = _pearson(left, right)
            if correlation is None:
                undefined[name] += 1
            else:
                destination.append(correlation)
    return {
        "gene_macro_mse": float(np.mean(mse)) if mse else None,
        "gene_macro_training_standardized_mse": (
            float(np.mean(standardized_mse)) if standardized_mse else None
        ),
        "gene_macro_profile_pearson": float(np.mean(ordinary)) if ordinary else None,
        "gene_macro_training_centroid_profile_pearson": (
            float(np.mean(adjusted)) if adjusted else None
        ),
        "gene_macro_independent_query_centered_profile_pearson": (
            float(np.mean(independent)) if independent else None
        ),
        "genes": len(mse),
        "eligible_queries": int(active_queries.sum()),
        "observed_values": int(mask.sum()),
        "undefined_ordinary": undefined["ordinary"],
        "undefined_training_centroid": undefined["trainingCentroid"],
        "undefined_independent_query_centered": undefined["independentQueryCentered"],
    }
