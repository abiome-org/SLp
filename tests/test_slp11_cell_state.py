import importlib.util
import sys
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "modules/slp-1-1-cell-state-v1/cell_state.py"
SPEC = importlib.util.spec_from_file_location("cell_state_under_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def fixture():
    torch.manual_seed(731)
    model = MODULE.CellState(MODULE.Config(5, 3, key_dim=4, state_dim=8, hidden_dim=16, dropout=0))
    rna_features, protein_features = torch.randn(7, 5), torch.randn(3, 3)
    rna_values, protein_values = torch.randn(4, 7), torch.randn(4, 3)
    rna_mask, protein_mask = torch.ones(4, 7, dtype=torch.bool), torch.ones(4, 3, dtype=torch.bool)
    return model, [rna_features, rna_values, rna_mask, protein_features, protein_values, protein_mask]


def test_query_permutation_and_chunking():
    model, inputs = fixture()
    state = model.encode(*inputs)
    permutation = torch.tensor([6, 2, 1, 5, 0, 4, 3])
    reordered = [inputs[0][permutation], inputs[1][:, permutation], inputs[2][:, permutation], *inputs[3:]]
    torch.testing.assert_close(model.encode(*reordered), state)
    whole = model.observe(state, inputs[0], "rna")
    chunked = torch.cat([model.observe(state, inputs[0][:3], "rna"), model.observe(state, inputs[0][3:], "rna")], 1)
    torch.testing.assert_close(whole, chunked)


def test_masked_values_cannot_enter_state():
    model, inputs = fixture()
    inputs[2][:, :2] = False
    original = model.encode(*inputs)
    inputs[1][:, :2] = float("nan")
    torch.testing.assert_close(model.encode(*inputs), original)
    inputs[2][:] = False
    assert torch.isfinite(model.encode(*inputs)).all()
    inputs[5][:] = False
    with pytest.raises(ValueError, match="one observed modality"):
        model.encode(*inputs)


def test_linear_observation_averaging_and_control_identity():
    model, inputs = fixture()
    state = model.encode(*inputs)
    expected = model.observe(state, inputs[0], "rna").mean(0, keepdim=True)
    actual = model.observe(state.mean(0, keepdim=True), inputs[0], "rna")
    torch.testing.assert_close(actual, expected)
    control = torch.randn(4, 7)
    scale = torch.rand(7) + 0.05
    identity = model.observe_delta(torch.zeros_like(state), inputs[0], "rna", control, scale)
    assert torch.equal(identity, control)


def test_balanced_loss_and_encoder_receive_gradients():
    model, inputs = fixture()
    state = model.encode(*inputs)
    rna = model.observe(state, inputs[0], "rna")
    protein = model.observe(state, inputs[3], "protein")
    loss = MODULE.balanced_reconstruction_loss(rna, inputs[1], inputs[2], protein, inputs[4], inputs[5])
    expected = 0.5 * ((rna - inputs[1]).square().mean() + (protein - inputs[4]).square().mean())
    torch.testing.assert_close(loss, expected)
    loss.backward()
    for parameters in (model.keys["rna"].parameters(), model.keys["protein"].parameters(), model.encoder.parameters()):
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in parameters)


def test_query_panel_size_is_data_not_parameter_shape():
    model, inputs = fixture()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    state = model.encode(*inputs)
    assert model.observe(state, torch.randn(11, 5), "rna").shape == (4, 11)
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count
