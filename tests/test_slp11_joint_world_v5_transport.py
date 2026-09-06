import importlib.util
from pathlib import Path
import sys

import pytest
import torch


PATH = Path(__file__).parents[1] / 'modules/slp-1-1-joint-world-v5/world_model.py'
SPEC = importlib.util.spec_from_file_location('joint_world_v5_transport_test', PATH)
WORLD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WORLD
SPEC.loader.exec_module(WORLD)


def example():
    torch.manual_seed(731)
    model = WORLD.SharedWorldModel(WORLD.Config(feature_dim=7, width=16, state_slots=3, heads=4))
    torch.nn.init.normal_(model.transition_output.weight, std=.03)
    anchors = [torch.randn(2, 3, 16, requires_grad=True) for _ in range(3)]
    actions = torch.randn(2, 2, 7, requires_grad=True)
    mode = torch.tensor([0, 1])
    assay = torch.tensor([0, 1])
    return model, anchors, actions, mode, assay


def test_transport_carries_existing_residual_and_new_transition():
    model, (state, prior, next_prior), actions, mode, assay = example()
    mask = torch.tensor([[True, True], [False, False]])
    expected = next_prior + (state-prior) + (model.transition(state, actions, mask, mode, assay)-state)
    actual = model.transport_action(state, prior, next_prior, actions, mask, mode, assay)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert torch.equal(actual[1], state[1])


def test_two_action_endpoint_reaches_both_actions_and_prior_anchors():
    model, (initial, prior_a, prior_ab), actions, mode, assay = example()
    first = torch.tensor([[True, False], [False, True]])
    second = ~first
    za = model.transport_action(initial, initial, prior_a, actions, first, mode, assay)
    zab = model.transport_action(za, prior_a, prior_ab, actions, second, mode, assay)
    queries = torch.randn(5, 7)
    residual = model.decode(zab, queries, assay)-model.decode(prior_ab, queries, assay)
    residual.square().mean().backward()
    for value in (initial, prior_a, prior_ab, actions):
        assert torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0
    assert (actions.grad.abs().sum(-1) > 0).all()


def test_zero_transition_reduces_to_prior_and_rejects_misaligned_anchors():
    model, (initial, prior_a, prior_ab), actions, mode, assay = example()
    torch.nn.init.zeros_(model.transition_output.weight)
    torch.nn.init.zeros_(model.transition_output.bias)
    first = torch.tensor([[True, False], [False, True]])
    za = model.transport_action(initial, initial, prior_a, actions, first, mode, assay)
    zab = model.transport_action(za, prior_a, prior_ab, actions, ~first, mode, assay)
    assert torch.equal(za, prior_a)
    assert torch.equal(zab, prior_ab)
    with pytest.raises(ValueError, match='identical shapes'):
        model.transport_action(initial, prior_a[:, :1], prior_ab, actions, first, mode, assay)
