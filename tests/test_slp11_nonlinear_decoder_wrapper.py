from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_slp11_nonlinear_decoder.py"
SPEC = importlib.util.spec_from_file_location("slp11_nonlinear_decoder_wrapper_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
wrapper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = wrapper
SPEC.loader.exec_module(wrapper)


def test_command_freezes_single_intended_architecture_run(tmp_path: Path) -> None:
    command = wrapper.command(tmp_path, 64)
    joined = " ".join(command)
    assert "--training-objective uniform-row-v1" in joined
    assert "--state-dim 128" in joined
    assert "--batch-size 64" in joined
    assert "--epochs 180" in joined
    assert "--patience 30" in joined
    assert "--max-seconds 1800" in joined
    assert str(wrapper.MODEL) in command


def test_independently_centered_gate_is_nonregression() -> None:
    assert wrapper.centered_nonregression(0.3, 0.3)
    assert wrapper.centered_nonregression(0.31, 0.3)
    assert not wrapper.centered_nonregression(0.299, 0.3)
    with pytest.raises(ValueError, match="finite"):
        wrapper.centered_nonregression(float("nan"), 0.3)
