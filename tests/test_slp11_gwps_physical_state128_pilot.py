"""Focused contracts for the physical state-128 decoder pilot."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/run_slp11_gwps_physical_state128_pilot.py"
SPEC = importlib.util.spec_from_file_location("run_slp11_gwps_physical_state128_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


def test_frozen_command_changes_only_state_capacity(tmp_path: Path) -> None:
    command = PILOT.training_command(tmp_path / "data", tmp_path / "features", tmp_path / "out", "cuda")
    state = command.index("--state-dim")
    hidden = command.index("--hidden")
    assert command[state + 1] == "128"
    assert command[hidden + 1] == "128"
    assert command[command.index("--query-basis-rank") + 1] == "32"
    assert command[command.index("--seed") + 1] == "731"


def test_rule_requires_no_regression_in_every_context() -> None:
    results = {
        "a": {
            "development_rule_passed": True,
            "world": {
                "gene_macro_nll": -1.1,
                "gene_macro_profile_centroid_adjusted_pearson_mean": 0.2,
            },
        }
    }
    state64 = {
        "a": {
            "world": {
                "gene_macro_nll": -1.0,
                "gene_macro_profile_centroid_adjusted_pearson_mean": 0.21,
            }
        }
    }
    decision = PILOT.evaluate_rules(results, state64)
    assert decision["primaryRulePassedAllContexts"] is True
    assert decision["noRegressionPassedAllContexts"] is False
    assert decision["hypothesisPassed"] is False
