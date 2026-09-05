from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/consolidate_replogle_k562_fitting_moments.py"
SPEC = importlib.util.spec_from_file_location("replogle_fitting_moments", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_action_moments_use_full_retained_library_and_exact_groups() -> None:
    raw = np.asarray([[1, 1], [2, 0], [0, 4]], dtype=np.float32)
    genes, counts, sums = MOD.action_moments(raw, np.asarray([2, 2, 4]), np.asarray(["ENSG1", "ENSG1", "ENSG2"]))
    assert genes.tolist() == ["ENSG1", "ENSG2"]
    assert counts.tolist() == [2, 1]
    np.testing.assert_allclose(sums.toarray(), [[15000, 5000], [0, 10000]])


def test_action_moments_reject_denominator_or_fractional_drift() -> None:
    for raw, library in [
        (np.asarray([[1, 1]], np.float32), np.asarray([3])),
        (np.asarray([[1.5, .5]], np.float32), np.asarray([2])),
    ]:
        try:
            MOD.action_moments(raw, library, np.asarray(["ENSG1"]))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid raw moment input accepted")
