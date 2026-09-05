"""Synthetic checks for gene-grouped fitting-only scale calibration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from transition_baselines import fit_mean, fit_ridge
from transition_calibration import (
    CalibrationError,
    fit_grouped_oof_mean,
    fit_grouped_oof_ridge,
    grouped_fold_ids,
)


class GroupedOofCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.keys = [
            (559292, "SGD:S0001"),
            (559292, "SGD:S0001"),
            (559292, "SGD:S0002"),
            (559292, "SGD:S0002"),
            (559292, "SGD:S0003"),
            (559292, "SGD:S0003"),
            (559292, "SGD:S0004"),
        ]

    def test_hash_assignment_is_deterministic_grouped_and_nonempty(self) -> None:
        first = grouped_fold_ids(self.keys, folds=3, seed=731)
        second = grouped_fold_ids(self.keys, folds=3, seed=731)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(first.tolist()), {0, 1, 2})
        for key in set(self.keys):
            rows = [index for index, candidate in enumerate(self.keys) if candidate == key]
            self.assertEqual(np.unique(first[rows]).size, 1)

    def test_ridge_refits_all_records_but_uses_grouped_oof_scale(self) -> None:
        x = np.array([[0.0], [0.0], [1.0], [1.0], [2.0], [2.0]])
        y = np.array([[0.0], [0.0], [1.0], [1.0], [4.0], [4.0]])
        observed = np.ones_like(y, dtype=bool)
        keys = self.keys[:6]

        ordinary = fit_ridge(x, y, observed, alpha=1e-8, scale_floor=1e-6)
        calibrated = fit_grouped_oof_ridge(
            x, y, observed, keys, alpha=1e-8, folds=3, seed=731, scale_floor=1e-6
        )

        np.testing.assert_allclose(calibrated.intercept_, ordinary.intercept_)
        np.testing.assert_allclose(calibrated.coef_, ordinary.coef_)
        self.assertEqual(calibrated.residual_scale_.provenance, "oof-fitting-residuals")
        self.assertGreater(
            calibrated.residual_scale_.values[0], ordinary.residual_scale_.values[0]
        )

    def test_mean_oof_scale_differs_from_in_sample_residual_scale(self) -> None:
        y = np.array([[0.0], [0.0], [2.0], [2.0], [8.0], [8.0]])
        observed = np.ones_like(y, dtype=bool)
        keys = self.keys[:6]

        ordinary = fit_mean(y, observed, scale_floor=1e-6)
        calibrated = fit_grouped_oof_mean(
            y, observed, keys, folds=3, seed=731, scale_floor=1e-6
        )

        np.testing.assert_allclose(calibrated.intercept_, ordinary.intercept_)
        self.assertEqual(calibrated.residual_scale_.provenance, "oof-fitting-residuals")
        self.assertGreater(
            calibrated.residual_scale_.values[0], ordinary.residual_scale_.values[0]
        )

    def test_requires_enough_distinct_genes_for_nonempty_folds(self) -> None:
        with self.assertRaisesRegex(CalibrationError, "distinct action genes"):
            grouped_fold_ids(self.keys[:4], folds=3)

    def test_rejects_non_composite_action_identity(self) -> None:
        with self.assertRaisesRegex(CalibrationError, "composite identity"):
            grouped_fold_ids(["S0001", "S0002", "S0003"], folds=3)

    def test_rejects_readout_without_training_fold_support(self) -> None:
        y = np.array([[1.0], [np.nan], [np.nan]])
        observed = np.array([[1], [0], [0]], dtype=bool)
        keys = [(559292, "SGD:A"), (559292, "SGD:B"), (559292, "SGD:C")]

        with self.assertRaisesRegex(CalibrationError, "support"):
            fit_grouped_oof_mean(y, observed, keys, folds=3)


if __name__ == "__main__":
    unittest.main()
