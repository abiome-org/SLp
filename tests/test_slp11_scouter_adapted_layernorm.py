from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "modules/slp-1-1-scouter-adapted-baseline-v2/scouter_model.py"
SPEC = importlib.util.spec_from_file_location("slp11_scouter_layernorm_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODEL
SPEC.loader.exec_module(MODEL)


def test_layernorm_v2_contains_no_batchnorm_and_preserves_widths() -> None:
    model = MODEL.ScouterAdaptedBaseline(MODEL.Config(7, 5))
    assert not any(isinstance(item, nn.BatchNorm1d) for item in model.modules())
    assert sum(isinstance(item, nn.LayerNorm) for item in model.modules()) == 3
    assert model.config.control_hidden == (2048, 512)
    assert model.config.control_state_dim == 64
    assert model.config.generator_hidden == (2048,)
    assert model.config.dropout == 0.0


def test_constant_control_train_and_eval_outputs_are_identical() -> None:
    torch.manual_seed(731)
    model = MODEL.ScouterAdaptedBaseline(
        MODEL.Config(
            query_dim=7,
            action_feature_dim=5,
            control_hidden=(11, 9),
            control_state_dim=3,
            generator_hidden=(13,),
        )
    )
    action = torch.randn(4, 1, 5)
    control = torch.randn(1, 7).expand(4, -1)
    model.train()
    training = model(action, control)
    model.eval()
    evaluation = model(action, control)
    assert torch.equal(training, evaluation)


def test_layernorm_action_order_and_mask_contract() -> None:
    torch.manual_seed(4)
    model = MODEL.ScouterAdaptedBaseline(
        MODEL.Config(7, 5, control_hidden=(11,), control_state_dim=3, generator_hidden=(13,))
    ).eval()
    action = torch.randn(3, 2, 5)
    mask = torch.tensor([[True, False], [True, True], [False, True]])
    action[~mask] = torch.nan
    control = torch.randn(3, 7)
    first = model(action, control, mask)
    second = model(action[:, [1, 0]], control, mask[:, [1, 0]])
    assert torch.equal(first, second)
