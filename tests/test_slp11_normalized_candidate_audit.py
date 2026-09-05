"""Focused numerical tests for the normalized candidate development audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_slp11_normalized_candidate.py"
SPEC = importlib.util.spec_from_file_location("audit_slp11_normalized_candidate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_gene_bootstrap_retains_duplicate_records() -> None:
    target = np.asarray([[1.0, -1.0], [3.0, -3.0], [2.0, -2.0]])
    prediction = np.zeros_like(target)
    observed = np.ones_like(target, dtype=bool)
    genes = np.asarray(["ENSG1", "ENSG1", "ENSG2"])
    summary = AUDIT.gene_summaries(
        prediction, target, observed, genes, np.zeros(2), np.ones_like(target)
    )
    np.testing.assert_array_equal(summary["gene_ids"], ["ENSG1", "ENSG2"])
    np.testing.assert_array_equal(summary["record_counts"], [2, 1])
    # ENSG1 NLL contains both records, rather than treating the records as genes.
    expected = 0.5 * np.log(2.0 * np.pi) + 0.5 * np.mean([1.0, 9.0])
    assert np.isclose(summary["nll"][0], expected)


def test_calibration_bins_use_fixed_boundaries() -> None:
    target = np.asarray([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])
    prediction = np.zeros_like(target)
    observed = np.ones_like(target, dtype=bool)
    result = AUDIT.calibration_moments(
        prediction,
        target,
        observed,
        np.ones_like(target),
        np.asarray([29.0, 30.0, 100.0]),
        np.asarray(["A", "B", "C"]),
    )
    assert [result["bins"][name]["records"] for name in ("lt30", "30to99", "ge100")] == [1, 1, 1]
    assert result["bins"]["lt30"]["zSecondMoment"] == 1.0
    assert result["bins"]["30to99"]["zSecondMoment"] == 4.0
    assert result["bins"]["ge100"]["zSecondMoment"] == 9.0
