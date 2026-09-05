import importlib.util
from pathlib import Path

import numpy as np

PATH = Path(__file__).parents[1] / "modules/slp-1-1-world-transition-v1/frangieh_basal_ridge.py"
SPEC = importlib.util.spec_from_file_location("frangieh_basal_ridge", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_guide_pseudobulks_are_collapsed_equally():
    action, context, target, count = MOD.collapse_gene_profiles(
        np.array(["A", "A", "B"]),
        np.array(["C", "C", "C"]),
        np.array([[0.0, 2.0], [2.0, 4.0], [8.0, 6.0]]),
    )
    np.testing.assert_array_equal(action, ["A", "B"])
    np.testing.assert_array_equal(context, ["C", "C"])
    np.testing.assert_array_equal(target, [[1.0, 3.0], [8.0, 6.0]])
    np.testing.assert_array_equal(count, [2, 1])


def test_dual_ridge_refits_fold_statistics_and_shapes():
    x = np.array([[0.0], [1.0], [2.0]])
    y = np.array([[1.0, 4.0], [2.0, 4.0], [3.0, 4.0]])
    prediction, stats = MOD.dual_ridge_predict(x, y, [[1.5]], 0.1)
    assert prediction.shape == (1, 2)
    np.testing.assert_array_equal(stats["feature_mean"], [1.0])
    assert stats["target_scale"][1] == 0.05


def test_query_centroid_adjustment_is_invariant_to_shared_query_landscapes():
    truth = np.array([[1.0, 2.0, 3.0], [5.0, 3.0, 1.0], [2.0, 8.0, 4.0]])
    prediction = truth + np.array([[0.2], [-0.1], [0.4]])
    score, per_gene = MOD.query_centroid_adjusted_profile_pearson(prediction, truth)
    shifted_score, shifted = MOD.query_centroid_adjusted_profile_pearson(
        prediction + np.array([[100.0, -5.0, 22.0]]),
        truth + np.array([[-4.0, 200.0, 9.0]]),
    )
    np.testing.assert_allclose(shifted_score, score)
    np.testing.assert_allclose(shifted, per_gene)


def test_constant_prediction_is_robustly_undefined():
    truth = np.array([[1.0, 2.0, 3.0], [5.0, 3.0, 1.0], [2.0, 8.0, 4.0]])
    prediction = np.broadcast_to(np.array([1e8, 1e8 + 1, 1e8 - 1]), truth.shape).copy()
    score, per_gene = MOD.query_centroid_adjusted_profile_pearson(prediction, truth)
    assert np.isnan(score)
    assert np.isnan(per_gene).all()


def test_development_split_fails_closed_for_test_gene():
    assert MOD.development_split("ENSG00000121410") == "train"
    assert MOD.development_split("ENSG00000204518") == "validation"
    with np.testing.assert_raises(ValueError):
        MOD.development_split("ENSG00000245105")


def test_exact_mean_limit_uses_fitting_rows_only():
    prediction, stats = MOD.mean_limit_predict(np.array([[1.0, 5.0], [3.0, 5.0]]), 3)
    np.testing.assert_array_equal(prediction, [[2.0, 5.0]] * 3)
    np.testing.assert_array_equal(stats["target_mean"], [2.0, 5.0])
    np.testing.assert_array_equal(stats["target_scale"], [1.0, 0.05])
