import importlib.util
import sys
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).parents[1] / "modules/slp-1-1-paired-state-v1/paired_model.py"
SPEC = importlib.util.spec_from_file_location("paired_state_test", PATH)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def fixture():
    torch.manual_seed(731)
    model = MOD.PairedStateModel(
        MOD.Config(5, 5, 3, hidden=8, state=4, decoder_hidden=6)
    ).eval()
    controls = {
        "rna": {
            "features": torch.randn(7, 5),
            "values": torch.randn(2, 7),
            "observed": torch.ones(2, 7, dtype=torch.bool),
        },
        "protein": {
            "features": torch.eye(3),
            "values": torch.randn(2, 3),
            "observed": torch.ones(2, 3, dtype=torch.bool),
        },
    }
    return model, controls, torch.randn(2, 2, 5)


@pytest.mark.parametrize("empty_shape", [True, False])
def test_empty_intervention_is_exact_control_in_both_modalities(empty_shape):
    model, controls, action = fixture()
    if empty_shape:
        action = action[:, :0]
    encoded = model.encode(
        action, torch.zeros(action.shape[:2], dtype=torch.bool), controls
    )
    for name, control in controls.items():
        out = model.observe(
            encoded,
            name,
            control["features"],
            control["values"],
            torch.ones(control["features"].shape[0]),
        )
        assert torch.equal(out["mean"], control["values"])
        assert torch.count_nonzero(out["delta"]) == 0


def test_actions_and_control_tokens_are_permutation_invariant():
    model, controls, action = fixture()
    mask = torch.ones(action.shape[:2], dtype=torch.bool)
    expected = model.encode(action, mask, controls)
    shuffled = {}
    for name, control in controls.items():
        order = torch.arange(len(control["features"]) - 1, -1, -1)
        shuffled[name] = {
            "features": control["features"][order],
            "values": control["values"][:, order],
            "observed": control["observed"][:, order],
        }
    actual = model.encode(action.flip(1), mask, shuffled)
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key])


def test_each_head_is_query_chunk_invariant_and_shared_state_gets_both_gradients():
    model, controls, action = fixture()
    for name, control in controls.items():
        model.zero_grad()
        encoded = model.encode(
            action, torch.ones(action.shape[:2], dtype=torch.bool), controls
        )
        q, mean = control["features"], control["values"]
        expected = model.observe(encoded, name, q, mean, torch.ones(len(q)))["mean"]
        chunks = [
            model.observe(
                encoded, name, q[i : i + 1], mean[:, i : i + 1], torch.ones(1)
            )["mean"]
            for i in range(len(q))
        ]
        torch.testing.assert_close(expected, torch.cat(chunks, 1))
        expected.square().mean().backward()
        assert model.action_encoder[0].weight.grad.abs().sum() > 0
        assert model.transition[0].weight.grad.abs().sum() > 0


def test_masked_nonfinite_controls_and_actions_are_inert():
    model, controls, action = fixture()
    controls["protein"]["observed"][:] = False
    controls["protein"]["values"][:] = float("nan")
    controls["protein"]["features"][:] = float("nan")
    mask = torch.tensor([[True, False], [True, False]])
    expected = model.encode(action, mask, controls)
    action[:, 1] = float("nan")
    actual = model.encode(action, mask, controls)
    torch.testing.assert_close(actual["state"], expected["state"])
    controls["rna"]["observed"][:] = False
    with pytest.raises(ValueError, match="at least one"):
        model.encode(action, mask, controls)


def test_masked_loss_has_zero_gradient_and_observed_targets_must_be_finite():
    prediction = torch.zeros(2, 3, requires_grad=True)
    truth = torch.tensor([[1.0, float("nan"), 2.0], [2.0, 3.0, 4.0]])
    mask = torch.isfinite(truth)
    loss = MOD.scaled_mse(prediction, truth, mask, torch.ones_like(truth))
    loss.backward()
    assert prediction.grad[0, 1] == 0
    with pytest.raises(ValueError, match="unmasked"):
        MOD.scaled_mse(prediction, truth, torch.ones_like(mask), torch.ones_like(truth))
