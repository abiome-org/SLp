import importlib.util
from pathlib import Path
import sys

import pytest
import torch


PATH = Path(__file__).parents[1] / 'modules/slp-1-1-joint-world-v4/world_model.py'
SPEC = importlib.util.spec_from_file_location('joint_world_v4_chain_test', PATH)
WORLD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WORLD
SPEC.loader.exec_module(WORLD)


def example():
    torch.manual_seed(731)
    model = WORLD.SharedWorldModel(WORLD.Config(feature_dim=7, width=16, state_slots=3, heads=4))
    torch.nn.init.normal_(model.transition_output.weight, std=.03)
    state = torch.randn(2, 3, 16)
    actions = torch.randn(2, 2, 7, requires_grad=True)
    mode = torch.tensor([0, 1])
    assay = torch.tensor([0, 1])
    order = torch.tensor([[0, 1], [1, 0]])
    return model, state, actions, mode, assay, order


def test_single_active_action_matches_direct_for_either_slot_order():
    model, state, actions, mode, assay, order = example()
    mask = torch.tensor([[True, False], [False, True]])
    chain = model.transition_chain(state, actions, mask, order, mode, assay)
    direct = model.transition(state, actions, mask, mode, assay)
    torch.testing.assert_close(chain, direct, rtol=0, atol=0)


def test_endpoint_gradient_reaches_both_interventions():
    model, state, actions, mode, assay, order = example()
    final = model.transition_chain(state, actions, torch.ones(2, 2, dtype=torch.bool), order, mode, assay)
    final.square().mean().backward()
    assert torch.isfinite(actions.grad).all()
    assert (actions.grad.abs().sum(-1) > 0).all()


def test_empty_chain_preserves_state_and_order_cannot_duplicate_actions():
    model, state, actions, mode, assay, order = example()
    empty = torch.zeros(2, 2, dtype=torch.bool)
    result = model.transition_chain(state, torch.full_like(actions, float('nan')), empty, order, mode, assay)
    assert torch.equal(result, state)
    with pytest.raises(ValueError, match='permute'):
        model.transition_chain(state, actions, empty, torch.zeros_like(order), mode, assay)
