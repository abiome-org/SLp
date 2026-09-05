from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extended_initialization_preserves_v1_and_zeros_only_new_columns() -> None:
    runner = load(ROOT / "scripts/run_slp11_frangieh_cell_state_transcript_query.py", "transcript_runner_test")
    core = load(ROOT / "modules/slp-1-1-cell-state-v1/cell_state.py", "transcript_core_test")
    model, old = runner.initialize_extended(core, torch.device("cpu"))

    old_state = old.state_dict(); new_state = model.state_dict()
    for name, old_value in old_state.items():
        new_value = new_state[name]
        if name in ("keys.rna.0.weight", "observation.rna.0.weight"):
            assert torch.equal(new_value[:, :1156], old_value)
            assert torch.count_nonzero(new_value[:, 1156:]) == 0
        else:
            assert torch.equal(new_value, old_value)


def test_initial_forward_parity_is_exact() -> None:
    runner = load(ROOT / "scripts/run_slp11_frangieh_cell_state_transcript_query.py", "transcript_runner_parity")
    core = load(ROOT / "modules/slp-1-1-cell-state-v1/cell_state.py", "transcript_core_parity")
    model, old = runner.initialize_extended(core, torch.device("cpu"))
    rng = np.random.default_rng(2)
    reference = {
        "rna_query_features": rng.normal(size=(7, 1415)).astype(np.float32),
        "protein_query_features": rng.normal(size=(3, 20)).astype(np.float32),
    }

    assert runner.initial_parity(model, old, reference, torch.device("cpu")) <= 1e-6
