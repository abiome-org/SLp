"""Arithmetic contracts for the basal-context sensitivity audit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/audit_slp11_minimal_control_context_sensitivity.py"
)
SPEC = importlib.util.spec_from_file_location("context_sensitivity_audit_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_pairwise_common_and_gene_specific_energy_are_orthogonal() -> None:
    left = np.asarray(
        [[0.0, 1.0], [1.0, 0.0], [-1.0, -1.0]], dtype=np.float32
    )
    common = np.asarray([2.0, -1.0], dtype=np.float32)
    residual = np.asarray(
        [[1.0, 0.0], [-0.5, 1.0], [-0.5, -1.0]], dtype=np.float32
    )
    right = left - common - residual
    report = MODULE.pairwise_decomposition(left, right)
    total = report["overallDifferenceRms"] ** 2
    separated = (
        report["commonAcrossGeneDifferenceRms"] ** 2
        + report["geneSpecificResidualDifferenceRms"] ** 2
    )
    np.testing.assert_allclose(total, separated, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        report["commonProfileDifferenceEnergyFraction"]
        + report["geneSpecificDifferenceEnergyFraction"],
        1.0,
    )
    np.testing.assert_allclose(
        report["commonAcrossGeneDifferenceRms"] ** 2,
        np.mean(np.square(common)),
    )


def test_context_summary_gene_centering_is_across_actions() -> None:
    values = np.asarray([[1.0, 4.0], [3.0, 0.0]], dtype=np.float32)
    report = MODULE.context_summary(values)
    centroid = values.mean(0)
    expected = np.mean(np.square(values - centroid))
    np.testing.assert_allclose(report["geneCenteredForecastVariance"], expected)
    np.testing.assert_allclose(
        report["commonProfileEnergyFraction"]
        + report["geneSpecificEnergyFraction"],
        1.0,
    )


def test_unique_action_order_uses_first_roster_occurrence() -> None:
    values = np.asarray(["ENSG2", "ENSG1", "ENSG2", "ENSG3", "ENSG1"])
    assert MODULE.unique_first(values).tolist() == [0, 1, 3]
