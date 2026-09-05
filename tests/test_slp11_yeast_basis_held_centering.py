import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "held_centering", Path(__file__).resolve().parents[1] / "scripts/score_slp11_yeast_basis_held_centering.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_held_centering_removes_arbitrary_shared_profile():
    rng = np.random.default_rng(731)
    signal = rng.normal(size=(150, 23))
    shared = rng.normal(size=23) * 1000
    np.testing.assert_allclose(MODULE.center(signal + shared), MODULE.center(signal), atol=1e-12)
    np.testing.assert_array_equal(MODULE.center(np.tile(shared, (150, 1))), np.zeros_like(signal))
