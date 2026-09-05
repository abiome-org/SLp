import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTROL = load(ROOT / "scripts/build_slp11_rpe1_count_control_reference.py", "rpe_control_test")
RUNNER = load(ROOT / "scripts/run_slp11_rpe1_count_anchored_ridge.py", "rpe_ridge_test")


def test_rpe_control_smoothing_is_positive_and_exact_mass():
    raw = np.asarray([[0, 2, 3], [4, 0, 2]], dtype=np.int64)
    library = raw.sum(1)
    rate, audit = CONTROL.positive_control_rate(raw, library, np.asarray([2, 3]))
    assert rate.dtype == np.float32
    assert np.all(rate > 0)
    expected = 10000 * (raw.astype(np.float64) + 0.5) / (library[:, None] + 1.5)
    np.testing.assert_array_equal(rate, expected.astype(np.float32))
    assert audit["maximumFloat64MassError"] < 1e-9


def test_rpe_control_rejects_library_not_equal_full_panel_sum():
    raw = np.asarray([[1, 2]], dtype=np.int64)
    with pytest.raises(CONTROL.RpeControlError):
        CONTROL.positive_control_rate(raw, np.asarray([4]), np.asarray([1]))


def test_prepared_rpe_protocol_excludes_held_and_development():
    protocol = RUNNER.frozen_protocol()
    assert protocol["outcomeBoundary"] == {
        "fittingMoments": True,
        "reconstructionHeld": False,
        "developmentValidation": False,
        "test": False,
        "syntheticLethality": False,
    }
    assert protocol["selection"]["candidates"][-1] == "mean-limit"
    assert protocol["selection"]["featureNormalization"].startswith("fold-local float64")
