"""Focused checks for selection of static nuisance covariates."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/run_slp11_degree_controls.py"
SPEC = importlib.util.spec_from_file_location("degree_controls_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_covariates_use_only_degree_coverage_and_presence():
    values = np.zeros((3, 1156), dtype=np.float32)
    values[:, -2] = [0, np.log(2), np.log(4)]
    values[:, -1] = [0, 1, 1]
    values[:, 320] = [1, 0, 1]
    result = MODULE.degree_covariates(values)
    values[:, 4:300] = 90
    changed = MODULE.degree_covariates(values)
    for key in result:
        np.testing.assert_array_equal(result[key], changed[key])
    np.testing.assert_array_equal(result["log_physical_degree"].ravel(), values[:, -2])


def test_invalid_coverage_is_rejected():
    values = np.zeros((2, 1156), dtype=np.float32)
    values[:, -1] = 1
    with pytest.raises(ValueError, match="coverage"):
        MODULE.degree_covariates(values)
