from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_core():
    path = ROOT / "modules/slp-1-1-action-state-transition-v1/transition.py"
    spec = importlib.util.spec_from_file_location("residual_transition_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_zero_initialization_exactly_recovers_base_and_empty_identity() -> None:
    core = load_core()
    model = core.ResidualStateTransition(core.Config(5, 3, hidden_dim=7, dropout=0.2))
    action = torch.randn(4, 5)
    control = torch.randn(4, 3)
    base = torch.randn(4, 3)
    present = torch.tensor([True, False, True, False])

    output = model(action, control, base, present)

    assert torch.equal(output[present], base[present])
    assert torch.count_nonzero(output[~present]) == 0


def test_final_layer_receives_gradient_at_zero_initialization() -> None:
    core = load_core()
    model = core.ResidualStateTransition(core.Config(5, 3, hidden_dim=7, dropout=0.0))
    output = model(torch.randn(4, 5), torch.randn(4, 3), torch.zeros(4, 3), torch.ones(4, dtype=torch.bool))
    output.square().add(output).mean().backward()

    assert model.residual.weight.grad is not None
    assert torch.count_nonzero(model.residual.weight.grad) > 0


def test_control_state_is_a_real_input_after_residual_learns() -> None:
    core = load_core()
    model = core.ResidualStateTransition(core.Config(2, 2, hidden_dim=4, dropout=0.0))
    with torch.no_grad():
        model.residual.weight.fill_(0.2)
    action = torch.zeros(1, 2)
    base = torch.zeros(1, 2)
    present = torch.ones(1, dtype=torch.bool)

    first = model(action, torch.zeros(1, 2), base, present)
    second = model(action, torch.tensor([[1.0, -0.5]]), base, present)

    assert not torch.equal(first, second)
