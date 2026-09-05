"""Focused isolation tests for three-seed OOF mean ensembling."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_slp11_three_seed_ensemble.py"
SPEC = importlib.util.spec_from_file_location("run_slp11_three_seed_ensemble", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ENSEMBLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENSEMBLE)


def test_oof_average_preserves_outer_validation_as_unpredicted() -> None:
    train = np.asarray([0, 1, 2])
    validation = np.asarray([3])
    members = []
    for value in (1.0, 2.0, 3.0):
        prediction = np.full((4, 2), np.nan, dtype=np.float32)
        prediction[train] = value
        members.append(prediction)
    result = ENSEMBLE.average_member_oof(members, train, validation)
    np.testing.assert_allclose(result[train], 2.0)
    assert np.isnan(result[validation]).all()


def test_oof_average_rejects_member_validation_prediction() -> None:
    train = np.asarray([0, 1])
    validation = np.asarray([2])
    valid = np.asarray([[1.0], [1.0], [np.nan]], dtype=np.float32)
    leaked = np.asarray([[2.0], [2.0], [0.0]], dtype=np.float32)
    with pytest.raises(ENSEMBLE.EnsembleExperimentError, match="outer validation"):
        ENSEMBLE.average_member_oof([valid, leaked], train, validation)
