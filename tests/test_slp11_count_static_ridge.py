from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_test_count_static_ridge",
    ROOT / "modules/slp-1-1-count-static-ridge-v1/count_static_ridge.py",
)
assert SPEC and SPEC.loader
RIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RIDGE)

CONTROL_SPEC = importlib.util.spec_from_file_location(
    "_test_build_slp11_k562_count_control_reference",
    ROOT / "scripts" / "build_slp11_k562_count_control_reference.py",
)
assert CONTROL_SPEC and CONTROL_SPEC.loader
CONTROL = importlib.util.module_from_spec(CONTROL_SPEC)
CONTROL_SPEC.loader.exec_module(CONTROL)


def test_control_anchor_uses_each_genes_exact_gem_cell_weights() -> None:
    rate = np.asarray([[1.0, 3.0], [5.0, 7.0]], np.float32)
    cells = np.asarray([[3, 1], [0, 4]], np.int64)
    expected = np.log1p([[2.0, 4.0], [5.0, 7.0]])
    np.testing.assert_allclose(RIDGE.control_anchor(rate, cells), expected, rtol=1e-7)


def test_response_moment_is_log_after_equal_cell_cp10k_mean() -> None:
    sums = np.asarray([[4.0, 8.0], [9.0, 3.0]])
    cells = np.asarray([2, 3])
    np.testing.assert_allclose(
        RIDGE.response_from_cp10k_moments(sums, cells), np.log1p([[2, 4], [3, 1]])
    )


def test_fold_local_normalizer_and_ridge_reconstruct_linear_residual() -> None:
    features = np.asarray([[0, 1], [1, 0], [2, 1], [3, 2]], np.float32)
    weights = np.asarray([[2, -1], [1, 3]], np.float32)
    target = features @ weights + np.asarray([4, 5], np.float32)
    state = RIDGE.fit_state(features, target)
    prediction = RIDGE.predict_residual(state, features, "0.1")
    assert float(np.mean(np.square(prediction - target))) < 1e-2
    shifted = features.copy()
    shifted[-1] = 1000
    fit_a = RIDGE.fit_feature_normalizer(features[:-1])
    fit_b = RIDGE.fit_feature_normalizer(shifted[:-1])
    np.testing.assert_array_equal(fit_a["feature_mean"], fit_b["feature_mean"])
    np.testing.assert_array_equal(fit_a["feature_scale"], fit_b["feature_scale"])


def test_ridge_matches_direct_augmented_intercept_solution() -> None:
    rng = np.random.default_rng(731)
    features = rng.normal(size=(24, 5)).astype(np.float32)
    target = rng.normal(size=(24, 7))
    state = RIDGE.fit_state(features, target)
    alpha = 10.0
    prediction = RIDGE.predict_residual(state, features, "10")
    design = RIDGE.transform_features(features, state)
    centered = design - design.mean(0)
    direct = target.mean(0) + centered @ np.linalg.solve(
        centered.T @ centered + alpha * np.eye(centered.shape[1]),
        centered.T @ (target - target.mean(0)),
    )
    np.testing.assert_allclose(prediction, direct, rtol=1e-10, atol=1e-10)
    scores = RIDGE.candidate_mse(state, features, target)
    for candidate in RIDGE.ALPHAS:
        explicit = RIDGE.predict_residual(state, features, candidate)
        expected = float(np.mean(np.square(explicit - target)))
        assert np.isclose(scores[candidate], expected, rtol=1e-10, atol=1e-12)


def test_centered_landscape_removes_common_anchor_and_profile_offsets() -> None:
    truth_residual = np.asarray([[1, 2, 4], [2, 4, 8], [3, 6, 12]], np.float64)
    prediction_residual = 2 * truth_residual + np.asarray([[3], [7], [11]])
    anchor = np.asarray([[5, 7, 9], [2, 4, 6], [8, 3, 1]], np.float64)
    score = RIDGE.centered_landscape_score(
        truth_residual + anchor, prediction_residual + anchor, anchor
    )
    assert np.isclose(score["independentlyQueryCenteredResidualPearson"], 1.0)
    assert score["finiteCorrelationGenes"] == 2
    assert score["undefinedCorrelationGenes"] == 1


def test_common_residual_mean_cannot_gain_anchor_dependent_correlation() -> None:
    rng = np.random.default_rng(731)
    anchor = rng.integers(0, 64, size=(305, 8563)).astype(np.float64) / 8.0
    residual = rng.integers(-8, 9, size=(1, 8563)).astype(np.float64) / 8.0
    absolute = RIDGE.absolute_prediction(anchor, np.broadcast_to(residual, anchor.shape))
    score = RIDGE.centered_landscape_score(absolute, absolute, anchor)
    assert score["finiteCorrelationGenes"] == 0
    assert score["undefinedCorrelationGenes"] == 305
    assert score["independentlyQueryCenteredResidualPearson"] is None


def test_global_fold_is_stable_and_seeded() -> None:
    ids = ["ENSG00000000001", "ENSG00000000002", "ENSG00000000003"]
    first = [RIDGE.global_gene_fold(item, 731) for item in ids]
    second = [RIDGE.global_gene_fold(item, 731) for item in reversed(ids)]
    assert first == list(reversed(second))
    assert all(0 <= value < 3 for value in first)


def test_control_reference_smoothing_is_positive_and_mass_preserving() -> None:
    raw = np.zeros((2, CONTROL.QUERY_COUNT), dtype=np.int64)
    raw[0, 0] = 10
    raw[1, 1] = 3
    rate, audit = CONTROL.positive_control_rate(
        raw, raw.sum(1), np.asarray([2, 1]), pseudocount=0.5
    )
    assert rate.shape == raw.shape
    assert np.all(rate > 0)
    np.testing.assert_allclose(rate.sum(1), 10000.0, rtol=2e-7)
    assert audit["maximumFloat64MassError"] < 1e-8
