from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from kernel_baseline import KernelBaselineError, fit_nystrom_rbf

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_slp11_kernel_baseline.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("run_slp11_kernel_baseline", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


def _keys(count: int) -> list[tuple[int, str]]:
    return [(559292, f"SGD:S{index:09d}") for index in range(1, count + 1)]


def test_transform_is_deterministic_and_landmarks_are_training_genes_only() -> None:
    features = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 3.0]]
    )
    keys = _keys(len(features))

    first = fit_nystrom_rbf(features, keys, n_landmarks=4, seed=731)
    second = fit_nystrom_rbf(features, keys, n_landmarks=4, seed=731)

    np.testing.assert_array_equal(first.transform(features), second.transform(features))
    assert first.landmark_keys_ == second.landmark_keys_
    assert set(first.landmark_keys_) <= set(keys)
    assert first.baseline_family == "nystrom-rbf-features-plus-feature-linear-ridge"


def test_full_landmark_features_reconstruct_rbf_similarity() -> None:
    features = np.array([[0.0], [1.0], [3.0], [5.0]])
    model = fit_nystrom_rbf(features, _keys(4), n_landmarks=4, seed=731)

    transformed = model.transform(features)
    standardized = (features - model.input_mean_) / model.input_scale_
    squared = (standardized - standardized.T) ** 2
    expected = np.exp(-squared / (2.0 * model.bandwidth_**2))

    np.testing.assert_allclose(transformed @ transformed.T, expected, atol=1e-10)


def test_bandwidth_factor_is_fixed_multiple_of_training_pair_median() -> None:
    features = np.array([[0.0], [1.0], [2.0], [4.0]])
    half = fit_nystrom_rbf(
        features, _keys(4), n_landmarks=3, bandwidth_factor=0.5, seed=731
    )
    double = fit_nystrom_rbf(
        features, _keys(4), n_landmarks=3, bandwidth_factor=2.0, seed=731
    )

    assert half.bandwidth_ == pytest.approx(half.median_pair_distance_ * 0.5)
    assert double.bandwidth_ == pytest.approx(double.median_pair_distance_ * 2.0)
    assert double.bandwidth_ == pytest.approx(half.bandwidth_ * 4.0)


def test_repeated_gene_must_have_identical_static_features() -> None:
    keys = [(559292, "SGD:S000000001")] * 2 + _keys(2)[1:]
    with pytest.raises(KernelBaselineError, match="inconsistent static features"):
        fit_nystrom_rbf(np.array([[0.0], [1.0], [2.0]]), keys, n_landmarks=2)


def test_landmarks_require_distinct_training_static_rows() -> None:
    with pytest.raises(KernelBaselineError, match="distinct training action feature rows"):
        fit_nystrom_rbf(np.array([[0.0], [0.0], [1.0]]), _keys(3), n_landmarks=3)


def test_feature_join_selects_required_composite_keys_in_corpus_order(tmp_path: Path) -> None:
    path = tmp_path / "features.npz"
    np.savez_compressed(
        path,
        feature_values=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        entity_taxon=np.array([4932, 4932, 4932], dtype=np.int64),
        entity_id=np.array(["SGD:A", "SGD:B", "SGD:C"]),
    )

    selected, artifact_shape = RUNNER.load_aligned_features(
        path, [(4932, "SGD:C"), (4932, "SGD:A"), (4932, "SGD:C")]
    )

    np.testing.assert_array_equal(selected, [[5.0, 6.0], [1.0, 2.0], [5.0, 6.0]])
    assert artifact_shape == (3, 2)
    with pytest.raises(ValueError, match="missing required action"):
        RUNNER.load_aligned_features(path, [(4932, "SGD:missing")])
