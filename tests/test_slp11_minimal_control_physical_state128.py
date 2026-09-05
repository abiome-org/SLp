"""Focused contracts for the physical/state-128 minimal-control synthesis."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/run_slp11_minimal_control_physical_state128.py"
SPEC = importlib.util.spec_from_file_location("minimal_control_physical_state128", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
WRAPPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WRAPPER)


def test_command_explicitly_pins_two_synthesis_changes(tmp_path: Path) -> None:
    args = argparse.Namespace(
        data=tmp_path / "data.npz",
        features=tmp_path / "features.npz",
        hepg2_control=tmp_path / "hepg2.npz",
        state64_report=tmp_path / "state64.json",
        device="cuda",
    )
    command = WRAPPER.training_command(args, tmp_path / "out")
    assert command[command.index("--feature-sha256") + 1] == WRAPPER.PHYSICAL_FEATURE_SHA256
    assert command[command.index("--state-dim") + 1] == "128"
    assert command[command.index("--hidden") + 1] == "128"
    assert command[command.index("--query-basis-rank") + 1] == "32"
    assert command[command.index("--seed") + 1] == "731"
