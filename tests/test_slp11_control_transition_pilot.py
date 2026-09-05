"""Synthetic launcher checks for the fixed control-anchored pilot."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

PATH = Path(__file__).resolve().parents[1] / "scripts/run_slp11_control_transition_pilot.py"
SPEC = importlib.util.spec_from_file_location("control_transition_pilot_test", PATH)
PILOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PILOT
SPEC.loader.exec_module(PILOT)


def test_singleton_interaction_projection_is_zero_and_frozen() -> None:
    model = PILOT.MODEL.ControlTransition(
        PILOT.MODEL.Config(5, 6, hidden_dim=12, state_dim=7, dropout=0.0)
    )
    PILOT.freeze_singleton_interaction(model)
    assert not model.interaction_projection.weight.requires_grad
    assert torch.count_nonzero(model.interaction_projection.weight) == 0


def test_advancement_requires_every_context_and_original_nonregression() -> None:
    def metrics(nll: float, correlation: float) -> dict[str, float]:
        return {
            "gene_macro_nll": nll,
            "gene_macro_profile_centroid_adjusted_pearson_mean": correlation,
        }

    contexts = ("a", "b", "c")
    current = {
        name: {
            "world": metrics(-0.6, 0.2),
            "world_delta_vs_mean": 0.03,
            "world_delta_vs_ridge": 0.02,
        }
        for name in contexts
    }
    original = {name: {"world": metrics(-0.59, 0.19)} for name in contexts}
    assert PILOT.advancement_decision(current, original, contexts)["passed"]
    current["b"]["world"] = metrics(-0.58, 0.2)
    assert not PILOT.advancement_decision(current, original, contexts)["passed"]
