"""Frozen launcher contract for the isolated v3 decoder experiment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_slp11_state_difference_physical.py"
SPEC = importlib.util.spec_from_file_location("state_difference_physical_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_command_isolates_v3_decoder_change() -> None:
    command = MODULE.command(Path("out"))
    assert argument(command, "--model-sha256") == MODULE.HASHES["model"]
    assert argument(command, "--feature-sha256") == MODULE.HASHES["features"]
    assert argument(command, "--training-objective") == "uniform-row-v1"
    assert argument(command, "--hidden") == "128"
    assert argument(command, "--state-dim") == "128"
    assert argument(command, "--query-basis-rank") == "32"
    assert argument(command, "--context-tokens") == "64"
    assert argument(command, "--seed") == "731"
    assert "650m" not in " ".join(command).lower()
    assert "6517" not in " ".join(command)


def test_primary_comparator_is_v2_physical_state128() -> None:
    assert MODULE.HASHES["comparator"] == (
        "49333ade99f04d96e9d4c4ccc2fc01c002170b38f02d10f88fdc8559d274203d"
    )
    assert MODULE.HASHES["features"] == (
        "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7"
    )
    source = PATH.read_text(encoding="utf-8")
    assert "at least 0.02 nats" in source
    assert "adjusted Pearson at least 0.10" in source
    assert "no NLL or adjusted-Pearson regression" in source
