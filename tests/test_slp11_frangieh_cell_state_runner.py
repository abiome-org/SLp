from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / "scripts/run_slp11_frangieh_cell_state.py"
    spec = importlib.util.spec_from_file_location("frangieh_cell_state_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_guide_aggregation_is_equal_cell_then_equal_guide() -> None:
    runner = load_runner()
    action = np.asarray(["G", "G", "G", "", ""])
    context = np.asarray(["C", "C", "C", "C", "C"])
    guide = np.asarray(["a", "a", "b", "NT1", "NT2"])
    state = np.asarray([[0.0], [2.0], [9.0], [3.0], [7.0]], dtype=np.float32)

    genes, controls = runner.aggregate_guide_states(action, context, guide, state)

    # guide a mean=1 and guide b mean=9; guides receive equal mass.
    np.testing.assert_allclose(genes[("G", "C")], [5.0])
    # Verified controls are a single no-intervention population and remain cell weighted.
    np.testing.assert_allclose(controls["C"], [5.0])


def test_denoising_mask_is_deterministic_and_channel_level() -> None:
    runner = load_runner()
    first = runner.denoising_mask((64, 100), 17, torch.device("cpu"))
    second = runner.denoising_mask((64, 100), 17, torch.device("cpu"))
    different = runner.denoising_mask((64, 100), 18, torch.device("cpu"))

    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert 0.75 < float(first.float().mean()) < 0.85
    assert torch.any(first[0] != first[1])


def test_shards_exclude_original_held_gene_cells_before_matrix_access() -> None:
    runner = load_runner()
    manifest = __import__("json").loads((runner.SHARDS / "manifest.json").read_text())

    assert manifest["counts"]["excluded_validation_cells"] == 19_606
    assert manifest["counts"]["cells"] == 103_862
    assert manifest["counts"]["source_train_cells"] == 64_515
    assert manifest["counts"]["verified_control_cells"] == 39_347
