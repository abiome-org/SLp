from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(version: str):
    name = f"state_difference_test_{version}"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / f"modules/slp-1-1-control-transition-{version}/transition_model.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V2, V3 = load("v2"), load("v3")


def inputs():
    return (torch.randn(2, 1, 3), torch.randn(7, 4), torch.full((2, 7), .25),
            torch.ones(7), torch.ones(2, 7), torch.randn(5, 4),
            torch.randn(2, 5), torch.ones(2, 5, dtype=torch.bool))


def test_zero_learned_latent_effect_must_not_change_molecular_state() -> None:
    torch.manual_seed(731)
    model = V3.MinimalControlTransition(V3.Config(3, 4, hidden_dim=8, state_dim=6, dropout=0)).eval()
    with torch.no_grad():
        for layer in (model.action_encoder, model.transition):
            for parameter in layer.parameters():
                parameter.zero_()
    historical = V2.MinimalControlTransition(V2.Config(3, 4, hidden_dim=8, state_dim=6, dropout=0)).eval()
    historical.load_state_dict(model.state_dict(), strict=True)
    args = inputs()
    with torch.no_grad():
        active = model(*args)
        empty = model(*args, action_mask=torch.zeros(2, 1, dtype=torch.bool))
        old_active = historical(*args)
        old_empty = historical(*args, action_mask=torch.zeros(2, 1, dtype=torch.bool))
    assert torch.equal(active["state"], empty["state"])
    assert torch.equal(active["mean"], args[2])
    assert torch.equal(active["mean"], empty["mean"])
    # The old state alias is a reproducible numerical counterexample.
    assert torch.equal(old_active["state"], old_empty["state"])
    assert not torch.equal(old_active["mean"], old_empty["mean"])


def test_direct_delta_decode_equals_linear_difference_and_empty_is_exact() -> None:
    torch.manual_seed(732)
    model = V3.MinimalControlTransition(V3.Config(3, 4, hidden_dim=8, state_dim=6, dropout=0)).eval()
    args = inputs()
    with torch.no_grad():
        prediction = model(*args)
        queries = model.query_encoder(args[1])
        expected = (model.mean_state(prediction["state"]) -
                    model.mean_state(prediction["basal_state"])) @ queries.T / (6 ** .5)
        empty = model(*args, action_mask=torch.zeros(2, 1, dtype=torch.bool))
    torch.testing.assert_close(prediction["delta"], expected, rtol=2e-5, atol=1e-8)
    assert torch.equal(empty["state"], empty["basal_state"])
    assert torch.count_nonzero(empty["delta"]) == 0
    assert torch.equal(empty["mean"], args[2])
