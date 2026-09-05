from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from action_uncertainty import (
    ActionUncertaintyError,
    estimate_record_variance_moments,
    fit_action_variance_multiplier,
)


def test_frozen_moment_subtracts_sampling_and_handles_missing_queries() -> None:
    biological = np.array([[2.0, 4.0, 8.0]])
    sampling = np.array([[10.0, 20.0, 30.0]])
    exposures = np.array([10.0, 20.0])
    observed = np.array([[True, True, False], [False, True, True]])
    desired = np.array([2.0, 0.5])
    residuals = np.zeros((2, 3))
    for row in range(2):
        present = observed[row]
        # Give every observed query the same desired biological multiplier.
        residuals[row, present] = np.sqrt(
            desired[row] * biological[0, present]
            + sampling[0, present] / exposures[row]
        )
    moments = estimate_record_variance_moments(
        residuals,
        observed,
        exposures,
        np.zeros(2, dtype=np.int64),
        biological,
        sampling,
    )
    np.testing.assert_allclose(moments.raw_values, desired, rtol=0.0, atol=1e-14)
    np.testing.assert_array_equal(moments.observed_counts, [2, 2])


def test_scales_change_biological_variance_without_mean_or_sampling_leakage() -> None:
    features = np.array([[-1.0], [1.0]])
    moments = estimate_record_variance_moments(
        np.array([[1.0], [2.0]]),
        np.ones((2, 1), dtype=bool),
        np.array([10.0, 10.0]),
        np.zeros(2, dtype=np.int64),
        np.ones((1, 1)),
        np.zeros((1, 1)),
    )
    model = fit_action_variance_multiplier(features, moments, alpha=0.0)
    requested_features = np.array([[-1.0], [-1.0]])
    scale = model.scales(
        requested_features,
        np.array([10.0, 20.0]),
        np.zeros(2, dtype=np.int64),
        np.array([[2.0]]),
        np.array([[30.0]]),
    )
    factor = model.multipliers(requested_features)
    np.testing.assert_allclose(
        scale[:, 0] ** 2, factor * 2.0 + 30.0 / np.array([10.0, 20.0])
    )
    assert "prediction" not in inspect.signature(model.scales).parameters
    assert "target" not in inspect.signature(model.scales).parameters


def test_fit_is_frozen_before_application_features_and_clips_predictions() -> None:
    fitting_features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    raw = np.exp(np.array([-8.0, -4.0, 4.0, 8.0]))
    moments = estimate_record_variance_moments(
        np.sqrt(raw)[:, None],
        np.ones((4, 1), dtype=bool),
        np.ones(4),
        np.zeros(4, dtype=np.int64),
        np.ones((1, 1)),
        np.zeros((1, 1)),
        moment_floor=1e-12,
    )
    model = fit_action_variance_multiplier(fitting_features, moments, alpha=0.0)
    frozen = (model.feature_mean_.copy(), model.feature_scale_.copy(), model.coefficient_.copy())
    factors = model.multipliers(np.array([[-1e6], [1e6]]))
    np.testing.assert_allclose(factors, [0.25, 4.0])
    for actual, expected in zip(
        (model.feature_mean_, model.feature_scale_, model.coefficient_), frozen, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_unidentifiable_biological_denominator_is_reported() -> None:
    with pytest.raises(ActionUncertaintyError, match="no record has identifiable"):
        estimate_record_variance_moments(
            np.ones((2, 2)),
            np.ones((2, 2), dtype=bool),
            np.ones(2),
            np.zeros(2, dtype=np.int64),
            np.zeros((1, 2)),
            np.ones((1, 2)),
        )
