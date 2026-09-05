import importlib.util
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "scripts/audit_slp11_yeast_wildtype_batches.py"
SPEC = importlib.util.spec_from_file_location("wt_batches", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_two_batch_sampling_correction():
    result = MODULE.dispersion(np.array([[1.0], [3.0]]), np.array([[4.0], [9.0]]), np.array([2, 3]))
    assert np.isclose(result["weighted_between_batch_mean_squared_dispersion"], 0.96)
    # Var(weighted centered sample means): w1*w2*(v1/n1+v2/n2).
    assert np.isclose(result["estimated_independent_cell_sampling_contribution"], .4 * .6 * (4 / 2 + 9 / 3))
