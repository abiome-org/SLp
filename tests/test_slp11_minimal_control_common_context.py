"""Synthetic contract checks for the minimal common-context launcher."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "scripts/run_slp11_minimal_control_common_context.py"
SPEC = importlib.util.spec_from_file_location("minimal_control_common_test", PATH)
PILOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PILOT
SPEC.loader.exec_module(PILOT)


def test_pooled_amplitude_is_one_query_vector_not_context_indexed() -> None:
    targets = np.asarray([[1.0, 2.0], [3.0, 6.0], [5.0, 4.0]])
    oof = np.asarray([[0.0, 2.0], [2.0, 2.0], [4.0, 4.0]])
    amplitude = PILOT.pooled_delta_amplitude(
        targets, oof, np.ones_like(targets, dtype=np.bool_)
    )
    np.testing.assert_allclose(amplitude, [1.0, np.sqrt(16.0 / 3.0)])
    assert amplitude.shape == (2,)
    assert amplitude.dtype == np.float32


def test_advancement_requires_no_original_model_regression() -> None:
    def metrics(nll: float, correlation: float) -> dict[str, float]:
        return {
            "gene_macro_nll": nll,
            "gene_macro_profile_centroid_adjusted_pearson_mean": correlation,
        }

    current = {
        "ctx": {
            "world": metrics(-0.6, 0.2),
            "world_delta_vs_mean": 0.03,
            "world_delta_vs_ridge": 0.03,
        }
    }
    original = {"ctx": {"world": metrics(-0.59, 0.21)}}
    decision = PILOT.advancement_decision(current, original, ("ctx",))
    assert not decision["passed"]
    assert not decision["contexts"]["ctx"]["checks"][
        "noAdjustedPearsonRegressionVsOriginal"
    ]
