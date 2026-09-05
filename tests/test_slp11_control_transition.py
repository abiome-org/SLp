"""Numerical contracts for the control-anchored transition candidate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

PATH = (
    Path(__file__).resolve().parents[1]
    / "modules/slp-1-1-control-transition-v1/transition_model.py"
)
SPEC = importlib.util.spec_from_file_location("control_transition_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def inputs() -> tuple:
    torch.manual_seed(731)
    config = MODULE.Config(
        action_feature_dim=5,
        query_feature_dim=6,
        assay_feature_dim=3,
        hidden_dim=16,
        state_dim=8,
        dropout=0.0,
    )
    model = MODULE.ControlTransition(config)
    model.eval()
    actions = torch.randn(3, 3, 5)
    queries = torch.randn(11, 6)
    control_mean = torch.randn(3, 11)
    control_scale = torch.rand(3, 11) + 0.5
    basal_features = torch.randn(3, 4, 6)
    basal_values = torch.randn(3, 4)
    basal_mask = torch.tensor(
        [[1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0]], dtype=torch.bool
    )
    assay_features = torch.randn(3, 3)
    return (
        model,
        actions,
        queries,
        control_mean,
        control_scale,
        basal_features,
        basal_values,
        basal_mask,
        assay_features,
    )


def call(model, actions, queries, control_mean, control_scale, basal_features,
         basal_values, basal_mask, assay_features, action_mask):
    return model(
        actions,
        queries,
        control_mean,
        control_scale,
        basal_features,
        basal_values,
        basal_mask,
        action_mask=action_mask,
        assay_features=assay_features,
    )


def test_empty_and_fully_masked_actions_are_exact_control_identity() -> None:
    args = inputs()
    model, actions, queries, mean, scale, basal_f, basal_v, basal_m, assay = args
    actions[:] = float("nan")
    masked = call(
        model, actions, queries, mean, scale, basal_f, basal_v, basal_m, assay,
        torch.zeros(3, 3, dtype=torch.bool),
    )
    assert torch.equal(masked["mean"], mean)
    assert torch.equal(masked["scale"], scale)
    assert torch.count_nonzero(masked["delta"]) == 0
    assert torch.count_nonzero(masked["intervention_delta"]) == 0
    assert torch.equal(masked["state"], masked["basal_state"])

    empty = call(
        model, torch.empty(3, 0, 5), queries, mean, scale, basal_f, basal_v,
        basal_m, assay, torch.empty(3, 0, dtype=torch.bool),
    )
    assert torch.equal(empty["mean"], mean)
    assert torch.equal(empty["scale"], scale)
    assert torch.equal(empty["delta"], torch.zeros_like(empty["delta"]))


def test_valid_action_gradients_flow_and_masked_nonfinite_padding_is_inert() -> None:
    args = inputs()
    model, actions, queries, mean, scale, basal_f, basal_v, basal_m, assay = args
    actions.requires_grad_()
    action_mask = torch.tensor(
        [[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.bool
    )
    with torch.no_grad():
        actions[~action_mask] = float("nan")
        basal_f[~basal_m] = float("nan")
        basal_v[~basal_m] = float("nan")
    prediction = call(
        model, actions, queries, mean, scale, basal_f, basal_v, basal_m, assay,
        action_mask,
    )
    target = torch.randn(3, 11)
    observed = torch.ones(3, 11, dtype=torch.bool)
    target[:, -1] = float("nan")
    observed[:, -1] = False
    loss = MODULE.gaussian_loss(prediction, target, observed)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(actions.grad).all()
    assert torch.count_nonzero(actions.grad[action_mask]) > 0
    assert torch.count_nonzero(actions.grad[~action_mask]) == 0
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.parameters()
    )


def test_action_order_and_masked_padding_do_not_change_predictions() -> None:
    args = inputs()
    model, actions, queries, mean, scale, basal_f, basal_v, basal_m, assay = args
    mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    actions[~mask] = float("nan")
    first = call(
        model, actions, queries, mean, scale, basal_f, basal_v, basal_m, assay,
        mask,
    )
    order = torch.tensor([2, 0, 1])
    second = call(
        model, actions[:, order], queries, mean, scale, basal_f, basal_v, basal_m,
        assay, mask[:, order],
    )
    for key in ("mean", "scale", "delta", "state", "intervention_delta"):
        torch.testing.assert_close(first[key], second[key], rtol=1e-6, atol=1e-7)


def test_queries_are_order_independent_and_chunk_invariant() -> None:
    args = inputs()
    model, actions, queries, mean, scale, basal_f, basal_v, basal_m, assay = args
    mask = torch.ones(3, 3, dtype=torch.bool)
    full = call(
        model, actions, queries, mean, scale, basal_f, basal_v, basal_m, assay,
        mask,
    )
    order = torch.tensor([7, 1, 10, 0, 4])
    ordered = call(
        model, actions, queries[order], mean[:, order], scale[:, order], basal_f,
        basal_v, basal_m, assay, mask,
    )
    torch.testing.assert_close(full["mean"][:, order], ordered["mean"])
    torch.testing.assert_close(full["scale"][:, order], ordered["scale"])
    torch.testing.assert_close(full["delta"][:, order], ordered["delta"])

    chunks = []
    scale_chunks = []
    for start, stop in ((0, 4), (4, 9), (9, 11)):
        part = call(
            model, actions, queries[start:stop], mean[:, start:stop],
            scale[:, start:stop], basal_f, basal_v, basal_m, assay, mask,
        )
        chunks.append(part["mean"])
        scale_chunks.append(part["scale"])
    torch.testing.assert_close(full["mean"], torch.cat(chunks, dim=1))
    torch.testing.assert_close(full["scale"], torch.cat(scale_chunks, dim=1))
