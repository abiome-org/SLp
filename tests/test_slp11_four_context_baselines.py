import importlib.util
from pathlib import Path

import numpy as np

PATH = (
    Path(__file__).parents[1]
    / "modules/slp-1-1-world-transition-v1/four_context_baselines.py"
)
SPEC = importlib.util.spec_from_file_location("four_context_baselines_test", PATH)
BASELINES = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASELINES)


def test_equal_record_collapse_is_missing_aware():
    genes, values, observed, counts = BASELINES.collapse_equal_records(
        np.array(["B", "A", "A"]),
        np.array([[8.0, 1.0], [2.0, 9.0], [4.0, 7.0]]),
        np.array([[True, False], [True, False], [True, True]]),
    )
    np.testing.assert_array_equal(genes, ["A", "B"])
    np.testing.assert_array_equal(values, [[3.0, 7.0], [8.0, 0.0]])
    np.testing.assert_array_equal(observed, [[True, True], [True, False]])
    np.testing.assert_array_equal(counts, [2, 1])


def test_independent_query_centering_removes_shared_query_landscape():
    truth = np.array([[1.0, -1.0, 2.0], [-1.0, 1.0, -2.0], [2.0, -2.0, 1.0]])
    prediction = truth + np.array([100.0, -50.0, 20.0])[None, :]
    score, undefined = BASELINES.independently_query_centered_profile_pearson(
        prediction, truth, np.ones_like(truth, dtype=bool)
    )
    assert np.isclose(score, 1.0)
    assert undefined == 0


def test_constant_gene_prediction_has_undefined_independent_correlation():
    truth = np.array([[1.0, -1.0], [-1.0, 1.0]])
    prediction = np.ones_like(truth)
    score, undefined = BASELINES.independently_query_centered_profile_pearson(
        prediction, truth, np.ones_like(truth, dtype=bool)
    )
    assert score is None
    assert undefined == 2


def test_float32_rounding_after_training_centroid_is_undefined():
    centroid = np.array([0.123456789, -0.345678912, 0.777777777], dtype=np.float64)
    prediction = np.broadcast_to(centroid.astype(np.float32), (2, 3))
    truth = np.array([[1.0, -1.0, 2.0], [-1.0, 1.0, -2.0]])
    score, undefined = BASELINES.training_centroid_adjusted_profile_pearson(
        prediction, truth, np.ones_like(truth, dtype=bool), centroid
    )
    assert score is None
    assert undefined == 2


def test_fitting_scale_uses_gene_profiles_and_floor():
    values = np.array([[0.0, 1.0], [0.02, 3.0]])
    scale = BASELINES.fitting_query_scale(values, np.ones_like(values, dtype=bool))
    np.testing.assert_allclose(scale, [0.05, 1.0])
