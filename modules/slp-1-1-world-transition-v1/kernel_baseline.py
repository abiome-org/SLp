"""Training-only Nyström RBF features for the SLp-1.1 ridge comparator.

This module supplies a nonlinear static-feature map.  Its downstream predictor
remains the existing feature-linear multioutput ridge, whose additional column
standardization means the composition is not represented as pure kernel ridge
regression.  Landmark selection and bandwidth fitting consume action features
and composite action identities only; no molecular outcomes are accepted.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist

Array = np.ndarray
ActionKey = tuple[int, str]


class KernelBaselineError(ValueError):
    """Raised when the frozen Nyström feature contract cannot be satisfied."""


@dataclass(frozen=True)
class NystromRbfFeatures:
    """Fitted training-only Nyström RBF feature transform."""

    input_mean_: Array
    input_scale_: Array
    landmarks_: Array
    landmark_keys_: tuple[ActionKey, ...]
    bandwidth_: float
    bandwidth_factor: float
    median_pair_distance_: float
    eigenvectors_: Array
    eigenvalues_: Array
    requested_landmarks: int
    seed: int
    baseline_family: str = "nystrom-rbf-features-plus-feature-linear-ridge"

    def transform(self, action_features: Array) -> Array:
        features = _feature_matrix(action_features)
        if features.shape[1] != self.input_mean_.size:
            raise KernelBaselineError(
                f"action feature width {features.shape[1]} does not match fitted width "
                f"{self.input_mean_.size}"
            )
        standardized = (features - self.input_mean_) / self.input_scale_
        squared_distance = _squared_distances(standardized, self.landmarks_)
        kernel = np.exp(-squared_distance / (2.0 * self.bandwidth_**2))
        whitening = self.eigenvectors_ / np.sqrt(self.eigenvalues_)[None, :]
        result = kernel @ whitening
        if not np.isfinite(result).all():
            raise KernelBaselineError("Nyström transform produced non-finite values")
        return result


def _feature_matrix(value: Array) -> Array:
    features = np.asarray(value, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise KernelBaselineError("action_features must be a non-empty two-dimensional array")
    if not np.isfinite(features).all():
        raise KernelBaselineError("action_features must contain only finite values")
    return features


def _action_keys(action_keys: Sequence[ActionKey], records: int) -> tuple[ActionKey, ...]:
    if isinstance(action_keys, (str, bytes)) or len(action_keys) != records:
        raise KernelBaselineError("action_keys must contain one composite identity per record")
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
            raise KernelBaselineError(f"action_keys[{index}] is not a composite identity")
        normalized.append((key[0], key[1]))
    return tuple(normalized)


def _squared_distances(left: Array, right: Array) -> Array:
    distances = (
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.maximum(distances, 0.0)


def fit_nystrom_rbf(
    action_features: Array,
    action_keys: Sequence[ActionKey],
    *,
    n_landmarks: int = 128,
    bandwidth_factor: float = 1.0,
    seed: int = 731,
    eigenvalue_tolerance: float = 1e-10,
) -> NystromRbfFeatures:
    """Fit a deterministic Nyström map using training action genes only.

    Input columns are standardized on the supplied training records.  The base
    bandwidth is the exact median positive distance among unique training genes.
    Landmark candidates are distinct static feature rows ordered by a salted
    hash of their representative composite identities.
    """

    features = _feature_matrix(action_features)
    keys = _action_keys(action_keys, features.shape[0])
    if type(n_landmarks) is not int or n_landmarks < 2:
        raise KernelBaselineError("n_landmarks must be an integer of at least two")
    if not np.isfinite(bandwidth_factor) or bandwidth_factor <= 0.0:
        raise KernelBaselineError("bandwidth_factor must be finite and positive")
    if type(seed) is not int:
        raise KernelBaselineError("seed must be an integer")
    if not np.isfinite(eigenvalue_tolerance) or not 0.0 < eigenvalue_tolerance < 1.0:
        raise KernelBaselineError("eigenvalue_tolerance must lie strictly between zero and one")

    input_mean = features.mean(axis=0)
    input_scale = features.std(axis=0)
    input_scale = np.where(input_scale > np.finfo(np.float64).eps, input_scale, 1.0)
    standardized = (features - input_mean) / input_scale

    per_gene: dict[ActionKey, Array] = {}
    for row, key in enumerate(keys):
        previous = per_gene.get(key)
        if previous is None:
            per_gene[key] = standardized[row]
        elif not np.array_equal(previous, standardized[row]):
            raise KernelBaselineError(f"repeated action gene {key!r} has inconsistent static features")
    unique_gene_features = np.stack([per_gene[key] for key in sorted(per_gene)])
    positive_distances = pdist(unique_gene_features, metric="euclidean")
    positive_distances = positive_distances[positive_distances > np.finfo(np.float64).eps]
    if positive_distances.size == 0:
        raise KernelBaselineError("training action genes have no positive pairwise feature distance")
    median_distance = float(np.median(positive_distances))
    bandwidth = median_distance * float(bandwidth_factor)

    prefix = f"slp11-nystrom-rbf-landmark-v1|{seed}|"
    distinct_rows: dict[bytes, tuple[bytes, ActionKey, Array]] = {}
    for key, values in per_gene.items():
        row_bytes = np.asarray(values, dtype="<f8").tobytes(order="C")
        digest = hashlib.sha256(f"{prefix}{key[0]}|{key[1]}".encode()).digest()
        candidate = (digest, key, values)
        current = distinct_rows.get(row_bytes)
        if current is None or candidate[:2] < current[:2]:
            distinct_rows[row_bytes] = candidate
    ordered = sorted(distinct_rows.values(), key=lambda item: (item[0], item[1]))
    if len(ordered) < n_landmarks:
        raise KernelBaselineError(
            f"requested {n_landmarks} landmarks but only {len(ordered)} distinct "
            "training action feature rows exist"
        )
    selected = ordered[:n_landmarks]
    landmark_keys = tuple(item[1] for item in selected)
    landmarks = np.stack([item[2] for item in selected])

    landmark_squared_distance = _squared_distances(landmarks, landmarks)
    landmark_kernel = np.exp(
        -landmark_squared_distance / (2.0 * bandwidth**2)
    )
    landmark_kernel = 0.5 * (landmark_kernel + landmark_kernel.T)
    eigenvalues, eigenvectors = np.linalg.eigh(landmark_kernel)
    descending = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[descending]
    eigenvectors = eigenvectors[:, descending]
    keep = eigenvalues > eigenvalues[0] * eigenvalue_tolerance
    if not np.any(keep):
        raise KernelBaselineError("landmark kernel has no stable positive eigenspace")
    return NystromRbfFeatures(
        input_mean_=input_mean,
        input_scale_=input_scale,
        landmarks_=landmarks,
        landmark_keys_=landmark_keys,
        bandwidth_=bandwidth,
        bandwidth_factor=float(bandwidth_factor),
        median_pair_distance_=median_distance,
        eigenvectors_=eigenvectors[:, keep],
        eigenvalues_=eigenvalues[keep],
        requested_landmarks=n_landmarks,
        seed=seed,
    )
