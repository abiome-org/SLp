"""Contracts for the training-only observed-response encoder."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "modules/slp-1-1-observed-state-transition-v1/transition_model.py"
SPEC = importlib.util.spec_from_file_location("slp11_observed_state_test", PATH)
assert SPEC is not None and SPEC.loader is not None
MODEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODEL
SPEC.loader.exec_module(MODEL)

V3_SPEC = importlib.util.spec_from_file_location(
    "slp11_observed_state_v3_reference",
    ROOT / "modules/slp-1-1-control-transition-v3/transition_model.py",
)
assert V3_SPEC is not None and V3_SPEC.loader is not None
V3 = importlib.util.module_from_spec(V3_SPEC)
sys.modules[V3_SPEC.name] = V3
V3_SPEC.loader.exec_module(V3)


def fixture():
    torch.manual_seed(731)
    model = MODEL.MinimalControlTransition(
        MODEL.Config(3, 4, hidden_dim=8, state_dim=6, dropout=0.0)
    )
    actions = torch.randn(2, 1, 3)
    queries = torch.randn(7, 4)
    control = torch.randn(2, 7)
    amplitude = torch.rand(7) + 0.5
    scale = torch.rand(2, 7) + 0.5
    basal_features = torch.randn(5, 4)
    basal_values = torch.randn(2, 5)
    basal_mask = torch.ones(2, 5, dtype=torch.bool)
    inputs = (
        actions,
        queries,
        control,
        amplitude,
        scale,
        basal_features,
        basal_values,
        basal_mask,
    )
    return model, inputs


def test_forecast_never_invokes_training_only_response_encoder() -> None:
    model, inputs = fixture()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("training-only response encoder entered inference")

    model.response_keys.forward = forbidden
    model.response_state.forward = forbidden
    prediction = model(*inputs)
    assert prediction["mean"].shape == (2, 7)


def test_initial_forecast_parameters_and_outputs_match_v3_exactly() -> None:
    config = {
        "action_feature_dim": 3,
        "query_feature_dim": 4,
        "hidden_dim": 8,
        "state_dim": 6,
        "dropout": 0.0,
    }
    torch.manual_seed(731)
    observed_model = MODEL.MinimalControlTransition(MODEL.Config(**config)).eval()
    torch.manual_seed(731)
    v3_model = V3.MinimalControlTransition(V3.Config(**config)).eval()
    shared = v3_model.state_dict()
    for name, value in shared.items():
        assert torch.equal(observed_model.state_dict()[name], value)
    _, inputs = fixture()
    with torch.no_grad():
        observed_prediction = observed_model(*inputs)
        v3_prediction = v3_model(*inputs)
    for name in (
        "mean",
        "scale",
        "delta",
        "state",
        "basal_state",
        "intervention_delta",
    ):
        assert torch.equal(observed_prediction[name], v3_prediction[name])


def test_masked_nonfinite_target_is_inert() -> None:
    model, inputs = fixture()
    model.eval()
    prediction = model(*inputs)
    target = inputs[2] + torch.randn(2, 7)
    observed = torch.ones(2, 7, dtype=torch.bool)
    observed[0, 3] = False
    masked_nan = target.clone()
    masked_nan[0, 3] = torch.nan
    masked_finite = target.clone()
    masked_finite[0, 3] = 1_000_000.0
    loss_nan = model.training_loss(
        prediction, inputs[1], masked_nan, observed, inputs[2], inputs[3]
    )
    loss_finite = model.training_loss(
        prediction, inputs[1], masked_finite, observed, inputs[2], inputs[3]
    )
    for name in loss_nan:
        torch.testing.assert_close(loss_nan[name], loss_finite[name], rtol=0, atol=0)


def test_zero_observed_response_has_exact_zero_posterior_delta() -> None:
    model, inputs = fixture()
    observed = torch.ones(2, 7, dtype=torch.bool)
    observed[0, 2] = False
    target = inputs[2].clone()
    target[0, 2] = torch.nan
    posterior = model.encode_observed_response(
        inputs[1], target, observed, inputs[2], inputs[3]
    )
    assert torch.count_nonzero(posterior) == 0


def test_auxiliary_training_gradients_reach_response_encoder_and_transition() -> None:
    model, inputs = fixture()
    model.train()
    prediction = model(*inputs)
    target = inputs[2] + 0.4 * torch.randn(2, 7)
    observed = torch.ones(2, 7, dtype=torch.bool)
    losses = model.training_loss(
        prediction, inputs[1], target, observed, inputs[2], inputs[3]
    )
    losses["total"].backward()
    response_parameters = [
        *model.response_keys.parameters(),
        *model.response_state.parameters(),
    ]
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in response_parameters
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.transition.parameters()
    )


def test_query_permutation_is_consistent_and_empty_identity_remains_exact() -> None:
    model, inputs = fixture()
    model.eval()
    permutation = torch.tensor([4, 1, 6, 0, 3, 5, 2])
    with torch.no_grad():
        original = model(*inputs)
        permuted = model(
            inputs[0],
            inputs[1][permutation],
            inputs[2][:, permutation],
            inputs[3][permutation],
            inputs[4][:, permutation],
            inputs[5],
            inputs[6],
            inputs[7],
        )
        empty = model(
            torch.empty(2, 0, 3),
            *inputs[1:],
            action_mask=torch.empty(2, 0, dtype=torch.bool),
        )
    torch.testing.assert_close(permuted["mean"], original["mean"][:, permutation])
    torch.testing.assert_close(permuted["delta"], original["delta"][:, permutation])
    assert torch.equal(empty["mean"], inputs[2])
    assert torch.equal(empty["state"], empty["basal_state"])
    assert torch.count_nonzero(empty["intervention_delta"]) == 0


def test_loss_formula_and_stop_gradient_contract() -> None:
    model, inputs = fixture()
    prediction = model(*inputs)
    target = inputs[2] + torch.randn(2, 7)
    observed = torch.ones(2, 7, dtype=torch.bool)
    losses = model.training_loss(
        prediction, inputs[1], target, observed, inputs[2], inputs[3]
    )
    torch.testing.assert_close(
        losses["total"],
        losses["forecast_nll"]
        + 0.1 * losses["reconstruction_nll"]
        + 0.1 * losses["latent_match"],
    )
    assert set(losses) == {
        "total",
        "forecast_nll",
        "reconstruction_nll",
        "latent_match",
    }
