import importlib.util
from pathlib import Path
import sys

import pytest


torch = pytest.importorskip("torch")
MODULE = Path(__file__).parents[1] / "modules" / "slp-1-1-compositional-state-v1" / "operator.py"
SPEC = importlib.util.spec_from_file_location("slp11_compositional_operator", MODULE)
operator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = operator
SPEC.loader.exec_module(operator)


def inputs(batch=4):
    generator = torch.Generator().manual_seed(91)
    return (torch.randn(batch, 32, generator=generator),
            torch.randn(batch, 2, 577, generator=generator))


def test_default_zero_initialization_and_empty_action_are_exact_identity():
    state, actions = inputs()
    model = operator.CompositionalStateOperator().eval()
    assert torch.equal(model(state, actions, torch.ones(4, 2, dtype=torch.bool)), state)
    with torch.no_grad():
        model.delta_head[-1].weight.normal_()
        model.delta_head[-1].bias.normal_()
    assert torch.equal(model(state, actions, torch.zeros(4, 2, dtype=torch.bool)), state)


def test_action_permutation_and_masked_values_have_no_effect():
    state, actions = inputs()
    model = operator.CompositionalStateOperator(operator.Config(zero_init_delta=False)).eval()
    both = torch.ones(4, 2, dtype=torch.bool)
    assert torch.allclose(model(state, actions, both), model(state, actions.flip(1), both), atol=1e-6)
    one = torch.tensor([[True, False]]).expand(4, -1)
    changed = actions.clone()
    changed[:, 1] = torch.randn_like(changed[:, 1]) * 1000
    assert torch.equal(model(state, actions, one), model(state, changed, one))


def test_nonzero_operator_uses_current_molecular_state():
    state, actions = inputs()
    model = operator.CompositionalStateOperator(operator.Config(zero_init_delta=False)).eval()
    mask = torch.ones(4, 2, dtype=torch.bool)
    first = model(state, actions, mask)
    shifted = model(state + 0.25, actions, mask)
    assert not torch.allclose(first - state, shifted - (state + 0.25))


def test_state_dict_round_trip_is_deterministic(tmp_path):
    state, actions = inputs()
    config = operator.Config(zero_init_delta=False)
    original = operator.CompositionalStateOperator(config).eval()
    expected = original(state, actions, torch.ones(4, 2, dtype=torch.bool))
    path = tmp_path / "operator.pt"
    torch.save(original.state_dict(), path)
    restored = operator.CompositionalStateOperator(config).eval()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    assert torch.equal(expected, restored(state, actions, torch.ones(4, 2, dtype=torch.bool)))


def test_difference_readout_preserves_additive_for_state_independent_action_offset():
    """The diagnostic isolates how the action increment changes by background.

    Even a biased control-to-single action model must contribute zero inferred
    interaction when T(state, action) = state + g(action).
    """
    generator = torch.Generator().manual_seed(117)
    observed_a = torch.randn(5, 32, generator=generator)
    observed_b = torch.randn(5, 32, generator=generator)
    effect_a = torch.randn(5, 32, generator=generator) + 3.0
    effect_b = torch.randn(5, 32, generator=generator) - 2.0
    predicted_a = effect_a  # T(0, A)
    predicted_b = effect_b  # T(0, B)

    ab = observed_a + effect_b
    ba = observed_b + effect_a
    interaction = 0.5 * (
        ab - observed_a - predicted_b + ba - observed_b - predicted_a
    )
    additive = observed_a + observed_b
    torch.testing.assert_close(interaction, torch.zeros_like(interaction), atol=1e-6, rtol=0)
    torch.testing.assert_close(additive + interaction, additive, atol=1e-6, rtol=0)

    swapped_a = observed_a.roll(1, 0)
    swapped_b = observed_b.roll(1, 0)
    swapped_interaction = 0.5 * (
        (swapped_a + effect_b) - swapped_a - predicted_b
        + (swapped_b + effect_a) - swapped_b - predicted_a
    )
    torch.testing.assert_close(swapped_interaction, torch.zeros_like(swapped_interaction), atol=1e-6, rtol=0)
    torch.testing.assert_close(additive + swapped_interaction, additive, atol=1e-6, rtol=0)
