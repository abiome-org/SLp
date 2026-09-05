from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "modules/slp-1-1-basal-action-ridge-v1/basal_action_ridge.py"


def load_module():
    spec = importlib.util.spec_from_file_location("basal_action_ridge_test", PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_basal_stays_zero_and_presence_is_shared_between_arms() -> None:
    module = load_module()
    static = np.asarray([[0.0], [2.0], [4.0]], dtype=np.float32)
    basal = np.asarray([1.0, 1000.0, 3.0], dtype=np.float32)
    observed = np.asarray([True, False, True])
    normalizer = module.fit_design_normalizer(static, basal, observed)
    control = module.transform_design(static, basal, observed, normalizer, include_basal_value=False)
    augmented = module.transform_design(static, basal, observed, normalizer, include_basal_value=True)
    np.testing.assert_array_equal(control[:, -2], 0.0)
    np.testing.assert_array_equal(control[:, -1], augmented[:, -1])
    assert augmented[1, -2] == 0.0
    np.testing.assert_allclose(augmented[[0, 2], -2], [-1.0, 1.0])


def test_fold_normalizer_does_not_use_held_extreme() -> None:
    module = load_module()
    fitting = module.fit_design_normalizer(
        np.asarray([[0.0], [2.0]], dtype=np.float32),
        np.asarray([1.0, 3.0], dtype=np.float32),
        np.asarray([True, True]),
    )
    held = module.transform_design(
        np.asarray([[1e6]], dtype=np.float32),
        np.asarray([1e6], dtype=np.float32),
        np.asarray([True]), fitting, include_basal_value=True,
    )
    np.testing.assert_allclose(fitting["static_mean"], [1.0])
    assert fitting["basal_mean"] == 2.0
    assert held[0, -2] == 999998.0


def test_anchored_query_centering_keeps_constant_prediction_undefined() -> None:
    module = load_module()
    constant = np.full((3, 5), 1e12)
    truth = np.arange(15, dtype=np.float64).reshape(3, 5)
    centered_constant = module.independently_query_center(constant)
    centered_truth = module.independently_query_center(truth)
    assert module.profile_pearson(centered_constant[1], centered_truth[1]) is None
