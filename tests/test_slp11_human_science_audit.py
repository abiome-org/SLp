"""Numerical tests for the bounded human science audit helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_slp11_human_science import (
    calibration_summary,
    cp10k_log1p,
    pearson,
)


def test_cp10k_transform_is_row_scale_invariant() -> None:
    raw = np.asarray([[1.0, 3.0], [2.0, 6.0]])
    transformed = cp10k_log1p(raw)
    np.testing.assert_allclose(transformed[0], transformed[1])


def test_calibration_detects_scale_multiplier() -> None:
    truth = np.asarray([[2.0, -2.0]])
    result = calibration_summary(np.zeros_like(truth), truth, np.ones_like(truth))
    assert result["standardizedResidualSecondMoment"] == pytest.approx(4.0)
    assert result["validationOptimalGlobalScaleMultiplier"] == pytest.approx(2.0)
    assert result["nllGainAtValidationOptimalGlobalScale"] > 0


def test_pearson_rejects_constant_vector() -> None:
    assert pearson(np.ones(3), np.arange(3)) is None
    assert pearson(np.arange(3), np.arange(3)) == pytest.approx(1.0)
