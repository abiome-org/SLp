"""Focused synthetic tests for fitting-only transition baselines and metrics."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from transition_baselines import (
    BaselineError,
    compare_paired_nll,
    evaluate,
    fit_mean,
    fit_residual_scale,
    fit_ridge,
)


class TransitionBaselineTests(unittest.TestCase):
    def test_mean_is_missing_aware_and_reports_fitting_calibration(self) -> None:
        targets = np.array([[1.0, 10.0], [3.0, 99.0], [5.0, 14.0]])
        observed = np.array([[1, 1], [1, 0], [0, 1]], dtype=bool)
        model = fit_mean(targets, observed)

        np.testing.assert_allclose(model.intercept_, [2.0, 12.0])
        np.testing.assert_allclose(model.predict(np.zeros((2, 7))), [[2.0, 12.0]] * 2)
        np.testing.assert_allclose(model.residual_scale_.values, [1.0, 2.0])
        self.assertEqual(model.residual_scale_.counts.tolist(), [2, 2])
        self.assertEqual(model.residual_scale_.provenance, "fitting-residuals")

    def test_ridge_recovers_per_readout_intercepts_under_distinct_masks(self) -> None:
        features = np.array(
            [[-2.0, 4.0, 1.0], [-1.0, 4.0, 1.0], [0.0, 4.0, 1.0],
             [1.0, 4.0, 1.0], [2.0, 4.0, 1.0], [3.0, 4.0, 1.0]]
        )
        targets = np.column_stack((3.0 + 2.0 * features[:, 0], -5.0 - features[:, 0]))
        observed = np.ones_like(targets, dtype=bool)
        observed[0, 1] = False
        observed[-1, 0] = False

        model = fit_ridge(features, targets, observed, alpha=1e-10)
        prediction = model.predict(features)

        np.testing.assert_allclose(prediction[observed], targets[observed], atol=1e-9)
        self.assertEqual(model.baseline_name, "feature-linear-ridge")
        self.assertEqual(model.baseline_family, "feature-linear-multioutput")
        self.assertNotIn("bilinear", model.baseline_family)
        np.testing.assert_allclose(model.feature_scale_[1:], [1.0, 1.0])

    def test_oof_predictions_freeze_scale_with_honest_provenance(self) -> None:
        targets = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        observed = np.ones_like(targets, dtype=bool)
        oof = targets + np.array([[1.0, -2.0], [-1.0, 2.0], [1.0, -2.0]])
        model = fit_ridge(
            np.arange(3.0)[:, None], targets, observed, alpha=1.0, oof_predictions=oof
        )

        np.testing.assert_allclose(model.residual_scale_.values, [1.0, 2.0])
        self.assertEqual(model.residual_scale_.provenance, "oof-fitting-residuals")

    def test_scale_rejects_validation_or_ambiguous_provenance(self) -> None:
        with self.assertRaisesRegex(BaselineError, "provenance"):
            fit_residual_scale(
                np.zeros((2, 1)),
                np.zeros((2, 1)),
                np.ones((2, 1), dtype=bool),
                provenance="validation-residuals",
            )

    def test_evaluate_uses_record_macro_nll_and_mse(self) -> None:
        truth = np.array([[0.0, 2.0, np.nan], [10.0, np.nan, np.nan]])
        prediction = np.array([[0.0, 0.0, 123.0], [12.0, 123.0, 123.0]])
        observed = np.array([[1, 1, 0], [1, 0, 0]], dtype=bool)

        metrics = evaluate(prediction, truth, observed, np.zeros(3), scale=1.0)

        self.assertAlmostEqual(metrics["mse"], 3.0)
        self.assertAlmostEqual(metrics["nll"], 0.5 * math.log(2.0 * math.pi) + 1.5)
        self.assertEqual(metrics["value_space"], "log2")
        self.assertEqual(metrics["observed_count"], 3)
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["profile_pearson_undefined"], 2)
        self.assertTrue(math.isnan(metrics["profile_pearson_mean"]))

    def test_profile_metrics_adjust_by_supplied_fitting_centroid(self) -> None:
        reference = np.array([10.0, 20.0, 30.0, 40.0])
        truth = np.array([reference + [1.0, -1.0, 2.0, -2.0]])
        prediction = truth.copy()
        observed = np.ones_like(truth, dtype=bool)

        metrics = evaluate(prediction, truth, observed, reference, scale=np.ones(4))

        self.assertAlmostEqual(metrics["profile_pearson_mean"], 1.0)
        self.assertAlmostEqual(metrics["profile_centroid_adjusted_pearson_mean"], 1.0)
        self.assertEqual(metrics["profile_centroid_adjusted_pearson_undefined"], 0)
        self.assertEqual(metrics["profile_centroid_adjusted_pearson_coverage"], 1.0)

    def test_paired_nll_delta_is_macro_and_positive_for_better_candidate(self) -> None:
        truth = np.array([[0.0, 0.0], [0.0, 0.0]])
        candidate = np.zeros_like(truth)
        baseline = np.array([[2.0, 2.0], [1.0, 0.0]])
        observed = np.array([[1, 1], [1, 0]], dtype=bool)

        result = compare_paired_nll(candidate, baseline, truth, observed, 1.0, 1.0)

        self.assertEqual(result["delta_definition"], "baseline-nll-minus-candidate-nll")
        self.assertEqual(result["paired_record_count"], 2)
        self.assertAlmostEqual(result["mean_nll_delta"], 1.25)
        self.assertEqual(result["candidate_better_fraction"], 1.0)

    def test_unobserved_nan_is_allowed_but_observed_nan_is_rejected(self) -> None:
        model = fit_mean(np.array([[1.0, np.nan], [3.0, 5.0]]), [[1, 0], [1, 1]])
        np.testing.assert_allclose(model.intercept_, [2.0, 5.0])
        with self.assertRaisesRegex(BaselineError, "observed targets"):
            fit_mean(np.array([[np.nan]]), np.array([[True]]))


if __name__ == "__main__":
    unittest.main()
