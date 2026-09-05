from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
V3_PATH = ROOT / "modules/slp-1-1-control-transition-v3/transition_model.py"
V4_PATH = ROOT / "modules/slp-1-1-control-transition-v4/transition_model.py"


def load(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


v3 = load(V3_PATH, "slp11_control_transition_v3_test")
v4 = load(V4_PATH, "slp11_control_transition_v4_test")


def fixture(module=v4, *, state_dim: int = 5):
    torch.manual_seed(731)
    config = module.Config(
        action_feature_dim=4,
        query_feature_dim=6,
        hidden_dim=7,
        state_dim=state_dim,
        dropout=0.2,
    )
    model = module.MinimalControlTransition(config).eval()
    batch, queries, tokens = 3, 11, 4
    inputs = {
        "actions": torch.randn(batch, 4),
        "queries": torch.randn(queries, 6),
        "control_mean": torch.randn(batch, queries),
        "delta_amplitude": torch.rand(queries) + 0.1,
        "observation_scale": torch.rand(batch, queries) + 0.1,
        "basal_features": torch.randn(tokens, 6),
        "basal_values": torch.randn(batch, tokens),
        "basal_mask": torch.ones(batch, tokens, dtype=torch.bool),
    }
    return model, inputs


def call(model, values):
    return model(
        values["actions"],
        values["queries"],
        values["control_mean"],
        values["delta_amplitude"],
        values["observation_scale"],
        values["basal_features"],
        values["basal_values"],
        values["basal_mask"],
    )


def test_v3_encoder_and_transition_topologies_are_unchanged() -> None:
    torch.manual_seed(99)
    old = v3.MinimalControlTransition(v3.Config(4, 6, 7, 5, 0.2))
    torch.manual_seed(99)
    new = v4.MinimalControlTransition(v4.Config(4, 6, 7, 5, 0.2))

    for name in ("action_encoder", "context_encoder", "transition", "query_encoder"):
        old_state = getattr(old, name).state_dict()
        new_state = getattr(new, name).state_dict()
        assert old_state.keys() == new_state.keys()
        for key in old_state:
            torch.testing.assert_close(old_state[key], new_state[key], rtol=0, atol=0)


def test_empty_and_fully_masked_nonempty_actions_have_exact_zero_change() -> None:
    model, values = fixture()
    empty = dict(values)
    empty["actions"] = torch.empty(3, 0, 4)
    with torch.no_grad():
        empty_result = model(
            empty["actions"],
            empty["queries"],
            empty["control_mean"],
            empty["delta_amplitude"],
            empty["observation_scale"],
            empty["basal_features"],
            empty["basal_values"],
            empty["basal_mask"],
            action_mask=torch.empty(3, 0, dtype=torch.bool),
        )
        masked_result = model(
            values["actions"],
            values["queries"],
            values["control_mean"],
            values["delta_amplitude"],
            values["observation_scale"],
            values["basal_features"],
            values["basal_values"],
            values["basal_mask"],
            action_mask=torch.zeros(3, 1, dtype=torch.bool),
        )
    for result in (empty_result, masked_result):
        assert torch.equal(result["mean"], values["control_mean"])
        assert torch.count_nonzero(result["delta"]) == 0
        assert torch.count_nonzero(result["intervention_delta"]) == 0
        assert torch.equal(result["state"], result["basal_state"])


def test_query_permutation_and_chunks_are_consistent() -> None:
    model, values = fixture()
    permutation = torch.tensor([7, 2, 10, 0, 5, 1, 3, 9, 8, 6, 4])
    with torch.no_grad():
        full = call(model, values)
        permuted_values = dict(values)
        for name in ("queries", "delta_amplitude"):
            permuted_values[name] = values[name][permutation]
        for name in ("control_mean", "observation_scale"):
            permuted_values[name] = values[name][:, permutation]
        permuted = call(model, permuted_values)
        chunks = []
        for selection in (slice(0, 4), slice(4, 8), slice(8, 11)):
            chunk = dict(values)
            chunk["queries"] = values["queries"][selection]
            chunk["delta_amplitude"] = values["delta_amplitude"][selection]
            chunk["control_mean"] = values["control_mean"][:, selection]
            chunk["observation_scale"] = values["observation_scale"][:, selection]
            chunks.append(call(model, chunk)["mean"])
    torch.testing.assert_close(permuted["mean"], full["mean"][:, permutation])
    torch.testing.assert_close(torch.cat(chunks, dim=1), full["mean"], rtol=0, atol=2e-7)


def test_decoder_difference_depends_on_basal_state_for_identical_delta() -> None:
    model, _ = fixture(state_dim=2)
    query_state = torch.tensor([[0.2, -0.4], [1.0, 0.3], [-0.5, 0.7]])
    basal = torch.tensor([[0.1, 0.2], [1.4, -0.8]])
    delta = torch.tensor([[0.6, -0.2], [0.6, -0.2]])
    with torch.no_grad():
        difference = model.decode_state_queries(
            basal + delta, query_state
        ) - model.decode_state_queries(basal, query_state)
    assert not torch.allclose(difference[0], difference[1])


def test_nonlinear_decoder_can_exceed_latent_matrix_rank() -> None:
    torch.manual_seed(17)
    model, _ = fixture(state_dim=2)
    basal = torch.zeros(4, 2)
    states = torch.tensor([[0.1, 0.9], [0.7, -0.2], [-0.4, 0.5], [1.2, 0.3]])
    queries = torch.tensor([[0.2, -0.3], [0.8, 0.4], [-0.6, 1.0], [1.1, -0.7]])
    with torch.no_grad():
        response = model.decode_state_queries(states, queries) - model.decode_state_queries(
            basal, queries
        )
    assert int(torch.linalg.matrix_rank(response, atol=1e-7, rtol=1e-6)) > 2


def test_full_source_reload_preserves_predictions() -> None:
    model, values = fixture()
    with torch.no_grad():
        expected = call(model, values)["mean"]
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    reloaded_module = load(V4_PATH, "slp11_control_transition_v4_reload_test")
    clone = reloaded_module.MinimalControlTransition(
        reloaded_module.Config(4, 6, 7, 5, 0.2)
    ).eval()
    buffer.seek(0)
    clone.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        actual = call(clone, values)["mean"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_gaussian_loss_remains_uniform_over_records() -> None:
    prediction = {
        "mean": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        "scale": torch.ones(2, 2),
    }
    target = torch.tensor([[1.0, float("nan")], [2.0, 2.0]])
    observed = torch.tensor([[True, False], [True, True]])
    loss = v4.gaussian_loss(prediction, target, observed)
    expected = np.mean(
        [0.5 * (np.log(2 * np.pi) + 1.0), 0.5 * (np.log(2 * np.pi) + 4.0)]
    )
    assert float(loss) == pytest.approx(expected)
