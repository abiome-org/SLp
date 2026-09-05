import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
PATH = ROOT / "modules/slp-1-1-yeast-static-baseline-v1/static_baseline.py"
SPEC = importlib.util.spec_from_file_location("slp11_yeast_static_baseline_test", PATH)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_masked_ridge_matches_independent_observed_only_solves():
    x = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    y = np.asarray([[1.0, 9.0], [3.0, 4.0], [5.0, 5.0], [7.0, 10.0]])
    observed = np.asarray([[1, 0], [1, 1], [1, 1], [1, 0]], dtype=bool)
    model = MOD.fit_ridge(x, y, observed, alpha=0.5)
    design = np.column_stack((np.ones(4), (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()))
    penalty = np.diag([0.0, 0.5])
    for query in range(2):
        keep = observed[:, query]
        expected = np.linalg.solve(
            design[keep].T @ design[keep] + penalty,
            design[keep].T @ y[keep, query],
        )
        np.testing.assert_allclose(model.intercept[query], expected[0])
        np.testing.assert_allclose(model.coefficient[:, query], expected[1:])


def test_training_normalization_and_model_reload_are_target_free(tmp_path: Path):
    x = np.asarray([[1.0, 5.0], [3.0, 5.0], [9.0, 5.0]])
    y = np.asarray([[2.0], [4.0], [8.0]])
    model = MOD.fit_ridge(x[:2], y[:2], np.ones((2, 1), dtype=bool), alpha=1.0)
    np.testing.assert_allclose(model.feature_mean, [2.0, 5.0])
    np.testing.assert_allclose(model.feature_scale, [1.0, 1.0])
    path = tmp_path / "model.npz"
    model.save(path)
    np.testing.assert_allclose(MOD.RidgeModel.load(path).predict(x), model.predict(x))


def test_nystrom_reload_and_gene_hash_inputs_do_not_use_targets(tmp_path: Path):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(9, 3))
    y = rng.normal(size=(9, 4))
    observed = rng.random((9, 4)) > 0.2
    ids = np.asarray([f"SGD:S{index:09d}" for index in range(9)])
    model = MOD.fit_nystrom_ridge(x, ids, y, observed, 2.0, landmarks=4, seed=731)
    path = tmp_path / "rbf.npz"
    model.save(path)
    reloaded = MOD.NystromRidgeModel.load(path)
    np.testing.assert_allclose(reloaded.predict(x), model.predict(x))
    changed = MOD.fit_nystrom_map(x, ids, landmarks=4, seed=731)
    np.testing.assert_allclose(changed.landmarks, model.mapping.landmarks)


def test_independent_query_centering_and_all_masked_query_exclusion():
    truth = np.asarray([[1.0, 2.0, 99.0], [2.0, 4.0, 99.0]])
    prediction = np.asarray([[2.0, 4.0, np.nan], [4.0, 8.0, np.nan]])
    observed = np.asarray([[1, 1, 0], [1, 1, 0]], dtype=bool)
    report = MOD.evaluate_gene_profiles(
        prediction,
        truth,
        observed,
        training_centroid=np.asarray([0.0, 0.0, np.nan]),
        training_scale=np.asarray([1.0, 1.0, np.nan]),
    )
    assert report["eligible_queries"] == 2
    assert report["genes"] == 2
    assert report["gene_macro_independent_query_centered_profile_pearson"] == 1.0


def test_mean_prediction_has_undefined_independently_centered_profiles():
    truth = np.asarray([[0.0, 1.0], [2.0, 3.0]])
    prediction = np.ones((2, 2))
    report = MOD.evaluate_gene_profiles(
        prediction,
        truth,
        np.ones((2, 2), dtype=bool),
        training_centroid=np.asarray([1.0, 1.0]),
        training_scale=np.asarray([1.0, 1.0]),
    )
    assert report["gene_macro_independent_query_centered_profile_pearson"] is None
    assert report["undefined_independent_query_centered"] == 2

    noisy_constant = prediction.copy()
    noisy_constant[0, 0] += 1e-13
    report = MOD.evaluate_gene_profiles(
        noisy_constant,
        truth,
        np.ones((2, 2), dtype=bool),
        training_centroid=np.asarray([1.0, 1.0]),
        training_scale=np.asarray([1.0, 1.0]),
    )
    assert report["gene_macro_independent_query_centered_profile_pearson"] is None


def test_grouped_folds_keep_action_identity_together():
    ids = np.asarray([f"SGD:S{index:09d}" for index in range(30)] * 2)
    folds = MOD.grouped_folds(ids, folds=3, seed=731)
    np.testing.assert_array_equal(folds[:30], folds[30:])
    assert set(folds) == {0, 1, 2}
