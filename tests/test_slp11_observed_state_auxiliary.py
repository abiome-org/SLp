"""Frozen-command checks for the observed-state auxiliary pilot."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_slp11_observed_state_auxiliary.py"
SPEC = importlib.util.spec_from_file_location("slp11_observed_aux_runner_test", PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_frozen_command_is_single_uniform_row_auxiliary_change(tmp_path: Path) -> None:
    command = RUNNER.command(tmp_path)
    assert option(command, "--training-objective") == RUNNER.OBJECTIVE
    assert option(command, "--model-source").endswith(
        "slp-1-1-observed-state-transition-v1\\transition_model.py"
    )
    assert option(command, "--feature-sha256") == RUNNER.HASHES["features"]
    assert option(command, "--state-dim") == "128"
    assert option(command, "--hidden") == "128"
    assert option(command, "--query-basis-rank") == "32"
    assert option(command, "--seed") == "731"
    assert option(command, "--epochs") == "180"
    assert option(command, "--patience") == "30"
    assert option(command, "--max-seconds") == "1800"
    assert option(command, "--original-report") == str(RUNNER.COMPARATOR_V2)
    assert "650m" not in " ".join(command).lower()
    assert "6517" not in " ".join(command)


def test_all_frozen_inputs_have_expected_hashes() -> None:
    paths = {
        "launcher": RUNNER.LAUNCHER,
        "model": RUNNER.MODEL,
        "data": RUNNER.DATA,
        "features": RUNNER.FEATURES,
        "hepg2": RUNNER.HEPG2,
        "comparator_v2": RUNNER.COMPARATOR_V2,
        "comparator_v3": RUNNER.COMPARATOR_V3,
    }
    for label, path in paths.items():
        assert RUNNER.sha256_file(path) == RUNNER.HASHES[label]


def test_launcher_explicitly_accepts_auxiliary_objective() -> None:
    source = RUNNER.LAUNCHER.read_text(encoding="utf-8")
    assert 'OBSERVED_STATE_AUX_V1 = "uniform-row-observed-state-aux-v1"' in source
    assert "model.training_loss(" in source
    assert '"trainingLossComponents"' in source
    assert 'score = float(np.mean([report["gene_macro_nll"]' in source
