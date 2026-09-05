import importlib.util
from pathlib import Path

import numpy as np

PATH = Path(__file__).parents[1] / "scripts/run_slp11_frangieh_specieswide_physical_ridge.py"
SPEC = importlib.util.spec_from_file_location("frangieh_specieswide_ridge", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_candidate_grid_has_exact_mean_limit_and_fixed_alphas():
    assert MOD.ALPHAS == (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0)
    prediction, stats = MOD._predict_candidate(
        "mean_limit", np.zeros((2, 1)), np.array([[1.0], [3.0]]), np.zeros((3, 1))
    )
    np.testing.assert_array_equal(prediction, [[2.0], [2.0], [2.0]])
    np.testing.assert_array_equal(stats["target_mean"], [2.0])


def test_feature_arms_share_availability_and_zero_fill_missing():
    ids = np.array(["A", "B"])
    values = np.arange(12, dtype=np.float32).reshape(2, 6)
    got, available = MOD._features_for_actions(np.array(["B", "C"]), ids, values, slice(0, 3))
    np.testing.assert_array_equal(available, [True, False])
    np.testing.assert_array_equal(got[0], [6, 7, 8, 1])
    np.testing.assert_array_equal(got[1], [0, 0, 0, 0])
