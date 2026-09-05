"""Numerical tests for the minimal control-anchored transition revision."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

PATH = (
    Path(__file__).resolve().parents[1]
    / "modules/slp-1-1-control-transition-v2/transition_model.py"
)
SPEC = importlib.util.spec_from_file_location("minimal_control_transition_test", PATH)
MODEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODEL
SPEC.loader.exec_module(MODEL)

ORIGINAL_PATH = (
    Path(__file__).resolve().parents[1]
    / "modules/slp-1-1-world-transition-v1/transition_model.py"
)
ORIGINAL_SPEC = importlib.util.spec_from_file_location(
    "minimal_control_original_topology_test", ORIGINAL_PATH
)
ORIGINAL = importlib.util.module_from_spec(ORIGINAL_SPEC)
sys.modules[ORIGINAL_SPEC.name] = ORIGINAL
ORIGINAL_SPEC.loader.exec_module(ORIGINAL)


def fixture() -> tuple:
    torch.manual_seed(731)
    model = MODEL.MinimalControlTransition(
        MODEL.Config(5, 7, hidden_dim=16, state_dim=8, dropout=0.0)
    )
    model.eval()
    actions = torch.randn(3, 3, 5)
    queries = torch.randn(13, 7)
    control = torch.randn(3, 13)
    amplitude = torch.rand(13) + 0.5
    scale = torch.rand(3, 13) + 0.5
    basal_features = torch.randn(3, 4, 7)
    basal_values = torch.randn(3, 4)
    basal_mask = torch.tensor(
        [[1, 1, 0, 1], [1, 0, 1, 1], [1, 1, 0, 0]], dtype=torch.bool
    )
    return model, actions, queries, control, amplitude, scale, basal_features, basal_values, basal_mask


def call(args: tuple, action_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    model, actions, queries, control, amplitude, scale, basal_f, basal_v, basal_m = args
    return model(
        actions, queries, control, amplitude, scale, basal_f, basal_v, basal_m,
        action_mask=action_mask,
    )


def test_encoder_transition_and_mean_topology_matches_original() -> None:
    minimal = MODEL.MinimalControlTransition(
        MODEL.Config(5, 7, hidden_dim=16, state_dim=8, dropout=0.2)
    )
    original = ORIGINAL.TransitionWorld(
        ORIGINAL.Config(
            5, hidden=16, state_dim=8, dropout=0.2, query_feature_dim=7
        )
    )
    for name in (
        "action_encoder",
        "context_encoder",
        "transition",
        "query_encoder",
        "mean_state",
    ):
        left = getattr(minimal, name)
        right = getattr(original, name)
        assert [(key, value.shape) for key, value in left.state_dict().items()] == [
            (key, value.shape) for key, value in right.state_dict().items()
        ]


def test_empty_action_identity_and_shared_amplitude_contract() -> None:
    args = fixture()
    model, actions, queries, control, amplitude, scale, basal_f, basal_v, basal_m = args
    actions[:] = float("nan")
    result = call(args, torch.zeros(3, 3, dtype=torch.bool))
    assert torch.equal(result["mean"], control)
    assert torch.equal(result["scale"], scale)
    assert torch.count_nonzero(result["delta"]) == 0
    assert torch.count_nonzero(result["intervention_delta"]) == 0
    assert torch.equal(result["state"], result["basal_state"])
    try:
        model(
            actions, queries, control, amplitude.expand(3, -1), scale,
            basal_f, basal_v, basal_m,
            action_mask=torch.zeros(3, 3, dtype=torch.bool),
        )
    except ValueError as error:
        assert "shared [Q]" in str(error)
    else:
        raise AssertionError("context-indexed decoder amplitude was accepted")


def test_masked_nonfinite_padding_has_zero_gradient() -> None:
    args = fixture()
    _, actions, _, _, _, _, basal_f, basal_v, basal_m = args
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.bool)
    actions.requires_grad_()
    with torch.no_grad():
        actions[~mask] = float("nan")
        basal_f[~basal_m] = float("nan")
        basal_v[~basal_m] = float("nan")
    prediction = call(args, mask)
    target = torch.randn(3, 13)
    observed = torch.ones(3, 13, dtype=torch.bool)
    target[:, -1] = float("nan")
    observed[:, -1] = False
    loss = MODEL.gaussian_loss(prediction, target, observed)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(actions.grad).all()
    assert torch.count_nonzero(actions.grad[mask]) > 0
    assert torch.count_nonzero(actions.grad[~mask]) == 0


def test_action_order_is_invariant() -> None:
    args = fixture()
    mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    args[1][~mask] = float("nan")
    first = call(args, mask)
    order = torch.tensor([2, 0, 1])
    reordered = list(args)
    reordered[1] = args[1][:, order]
    second = call(tuple(reordered), mask[:, order])
    for key in ("mean", "delta", "state", "intervention_delta"):
        torch.testing.assert_close(first[key], second[key], rtol=1e-6, atol=1e-7)


def test_query_order_and_chunks_are_independent() -> None:
    args = fixture()
    mask = torch.ones(3, 3, dtype=torch.bool)
    full = call(args, mask)
    model, actions, queries, control, amplitude, scale, basal_f, basal_v, basal_m = args
    order = torch.tensor([9, 1, 12, 0, 5])
    ordered = model(
        actions, queries[order], control[:, order], amplitude[order], scale[:, order],
        basal_f, basal_v, basal_m, action_mask=mask,
    )
    torch.testing.assert_close(full["mean"][:, order], ordered["mean"])
    pieces = []
    for start, stop in ((0, 4), (4, 10), (10, 13)):
        pieces.append(
            model(
                actions, queries[start:stop], control[:, start:stop],
                amplitude[start:stop], scale[:, start:stop], basal_f, basal_v,
                basal_m, action_mask=mask,
            )["mean"]
        )
    torch.testing.assert_close(full["mean"], torch.cat(pieces, dim=1))
