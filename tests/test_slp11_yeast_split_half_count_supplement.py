from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "supplement_slp11_yeast_split_half_counts.py"
SPEC = importlib.util.spec_from_file_location("slp11_split_count_supplement", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
supplement = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = supplement
SPEC.loader.exec_module(supplement)


def test_subset_support_and_independent_centering_are_exact():
    cells = np.asarray([[2, 5, 10], [2, 4, 9]])
    a = np.asarray([[1.0, 4.0, 2.0], [3.0, 1.0, 7.0], [8.0, 3.0, 5.0]])
    b = a + np.asarray([20.0, -3.0, 6.0])
    means = np.stack([a, b])
    sums = means * cells[:, :, None]
    result = supplement.centered_metrics(sums, cells, np.asarray([True, True, False]))
    assert result["genes"] == 2
    assert result["halfACells"] == 7
    assert result["halfBCells"] == 6
    assert result["halfAMedianCellsPerGene"] == pytest.approx(3.5)
    assert result["metrics"]["equalGeneMeanMse"] == pytest.approx(0)
    assert result["metrics"]["meanGeneProfilePearson"] == pytest.approx(1)


def test_empty_fixed_bin_remains_explicit_and_finite_safe():
    result = supplement.centered_metrics(
        np.zeros((2, 3, 4)), np.ones((2, 3), dtype=np.int64), np.zeros(3, bool)
    )
    assert result == {
        "genes": 0,
        "halfACells": 0,
        "halfBCells": 0,
        "halfAMedianCellsPerGene": None,
        "halfBMedianCellsPerGene": None,
        "metrics": None,
    }
