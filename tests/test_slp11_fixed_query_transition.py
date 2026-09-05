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


def test_shared_seed_initialization_is_bit_exact() -> None:
    runner = load(ROOT / "scripts/run_slp11_source3_fixed_response_basis.py", "fixed_runner_test")
    fixed_module = load(runner.CORE, "fixed_core_init_test")
    old_module = load(runner.OLD_CORE, "old_core_init_test")
    helper = load(runner.BP_HELPER, "fixed_helper_init_test")
    fixed, old = runner.initialize_fixed(fixed_module, old_module, helper)
    assert all(
        torch.equal(value, old.state_dict()[name]) for name, value in fixed.state_dict().items()
    )
    assert not any("query_encoder" in name for name, _ in fixed.named_parameters())
    assert torch.count_nonzero(old.state_dict()["action_encoder.0.weight"][:, 1156:]) == 0


def test_fixed_coordinates_permute_outputs_and_empty_action_is_exact() -> None:
    module = load(
        ROOT / "modules/slp-1-1-fixed-query-transition-v1/transition_model.py",
        "fixed_core_contract_test",
    )
    torch.manual_seed(4)
    model = module.FixedQueryTransition(module.Config(3, 5, state_dim=4, hidden_dim=6, dropout=0.0))
    coordinates = torch.randn(7, 4)
    control = torch.randn(2, 7)
    common = {
        "actions": torch.randn(2, 1, 3),
        "control_mean": control,
        "delta_amplitude": torch.rand(7) + 0.1,
        "observation_scale": torch.ones(2, 7),
        "basal_features": torch.randn(2, 3, 5),
        "basal_values": torch.randn(2, 3),
        "basal_mask": torch.ones(2, 3, dtype=torch.bool),
    }
    direct = model(query_coordinates=coordinates, **common)["mean"]
    order = torch.tensor([4, 1, 6, 0, 5, 3, 2])
    permuted = dict(common)
    permuted["control_mean"] = control[:, order]
    permuted["delta_amplitude"] = common["delta_amplitude"][order]
    permuted["observation_scale"] = common["observation_scale"][:, order]
    moved = model(query_coordinates=coordinates[order], **permuted)["mean"]
    torch.testing.assert_close(moved, direct[:, order])
    empty = np.empty((2, 0, 3), dtype=np.float32)
    result = model(
        torch.as_tensor(empty),
        coordinates,
        control,
        common["delta_amplitude"],
        common["observation_scale"],
        common["basal_features"],
        common["basal_values"],
        common["basal_mask"],
    )
    assert torch.equal(result["mean"], control)
    assert not torch.count_nonzero(result["delta"])
