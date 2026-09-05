"""Focused contracts for the transition-world mean ensemble wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MODULE = Path(__file__).parents[1] / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from ensemble_inference import combine_member_outputs


def test_ensemble_averages_means_and_keeps_member_states_separate() -> None:
    outputs = [
        {"mean": np.full((2, 3), value), "state": np.full((2, 4), value * 10)}
        for value in (1.0, 2.0, 3.0)
    ]
    result = combine_member_outputs(outputs, np.full((2, 3), 0.5))
    np.testing.assert_allclose(result["mean"], 2.0)
    np.testing.assert_allclose(result["scale"], 0.5)
    assert result["member_means"].shape == (3, 2, 3)
    assert result["member_states"].shape == (3, 2, 4)
    np.testing.assert_array_equal(result["member_states"][:, 0, 0], [10.0, 20.0, 30.0])
