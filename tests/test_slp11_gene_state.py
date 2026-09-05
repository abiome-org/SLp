import importlib.util
import sys
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).parents[1] / "modules/slp-1-1-gene-state-v1/gene_state.py"
SPEC = importlib.util.spec_from_file_location("slp11_gene_state_test", PATH)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def ring_adjacency(nodes):
    rows = torch.arange(nodes)
    columns = (rows - 1) % nodes
    with torch.sparse.check_sparse_tensor_invariants():
        return torch.sparse_coo_tensor(
            torch.stack((rows, columns)), torch.ones(nodes), (nodes, nodes),
        ).coalesce()


def fixture(batch=2, nodes=6, features=4):
    torch.manual_seed(731)
    model = MOD.GeneStateCore(
        MOD.Config(features, state=3, transition_hidden=7, decoder_hidden=5),
    ).eval()
    static = torch.randn(nodes, features)
    basal = torch.randn(batch, nodes)
    observed = torch.ones(batch, nodes, dtype=torch.bool)
    strength = torch.zeros(batch, nodes)
    strength[0, 0] = 0.5
    strength[1, 3] = -1.25
    return model, static, basal, observed, strength, ring_adjacency(nodes)


def test_node_permutation_equivariance_including_graph_actions_and_queries():
    model, static, basal, observed, strength, adjacency = fixture()
    expected = model.encode(static, basal, observed, strength, adjacency)
    query = torch.tensor([0, 3, 5])
    control = torch.randn(2, 3)
    amplitude = torch.tensor([0.5, 1.0, 2.0])
    expected_mean = model.observe(expected, query, control, amplitude)["mean"]

    permutation = torch.tensor([2, 5, 0, 4, 1, 3])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(len(permutation))
    dense_adjacency = adjacency.to_dense()
    permuted_adjacency = dense_adjacency[permutation][:, permutation].to_sparse().coalesce()
    actual = model.encode(
        static[permutation], basal[:, permutation], observed[:, permutation],
        strength[:, permutation], permuted_adjacency,
    )
    for name in ("static_state",):
        torch.testing.assert_close(actual[name], expected[name][permutation])
    for name in ("basal_node_state", "initial_local_delta", "local_delta", "local_state"):
        torch.testing.assert_close(actual[name], expected[name][:, permutation])
    for name in ("global_basal_state", "global_action", "global_delta", "global_state"):
        torch.testing.assert_close(actual[name], expected[name])
    actual_mean = model.observe(actual, inverse[query], control, amplitude)["mean"]
    torch.testing.assert_close(actual_mean, expected_mean)


def test_no_action_is_exact_control_identity_and_static_projection_runs_once():
    model, static, basal, observed, strength, adjacency = fixture()
    calls = []
    handle = model.static_linear.register_forward_hook(lambda *args: calls.append(1))
    encoded = model.encode(static, basal, observed, torch.zeros_like(strength), adjacency)
    handle.remove()
    assert len(calls) == 1
    assert torch.count_nonzero(encoded["global_delta"]) == 0
    assert torch.count_nonzero(encoded["local_delta"]) == 0
    query = torch.tensor([0, 2, 5])
    control = torch.randn(2, 3)
    result = model.observe(encoded, query, control, torch.ones(3))
    assert torch.count_nonzero(result["delta"]) == 0
    assert torch.equal(result["mean"], control)


def test_two_message_steps_limit_local_reachability_to_two_edges():
    torch.manual_seed(4)
    model = MOD.GeneStateCore(MOD.Config(2, state=3, transition_hidden=6, decoder_hidden=4)).eval()
    nodes = 5
    # Messages follow 0 -> 1 -> 2 -> 3; node 4 is disconnected.
    with torch.sparse.check_sparse_tensor_invariants():
        adjacency = torch.sparse_coo_tensor(
            torch.tensor([[1, 2, 3], [0, 1, 2]]), torch.ones(3), (nodes, nodes),
        ).coalesce()
    static = torch.randn(nodes, 2)
    basal = torch.randn(1, nodes)
    observed = torch.ones(1, nodes, dtype=torch.bool)
    strength = torch.zeros(1, nodes)
    strength[0, 0] = 1.0
    encoded = model.encode(static, basal, observed, strength, adjacency)
    assert torch.count_nonzero(encoded["initial_local_delta"][0, 1:]) == 0
    assert torch.count_nonzero(encoded["local_delta"][0, 3:]) == 0
    assert torch.count_nonzero(encoded["local_delta"][0, :3]) > 0
    # The separate global route is intentionally broad and not graph-local.
    assert torch.count_nonzero(encoded["global_delta"]) > 0


def test_query_order_and_chunking_do_not_change_predictions():
    model, static, basal, observed, strength, adjacency = fixture()
    encoded = model.encode(static, basal, observed, strength, adjacency)
    query = torch.tensor([5, 1, 4, 0])
    control = torch.randn(2, 4)
    amplitude = torch.tensor([0.5, 0.7, 0.9, 1.1])
    expected = model.observe(encoded, query, control, amplitude)["mean"]
    chunks = [
        model.observe(encoded, query[i:i + 2], control[:, i:i + 2], amplitude[i:i + 2])["mean"]
        for i in range(0, len(query), 2)
    ]
    torch.testing.assert_close(torch.cat(chunks, dim=1), expected)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = model.observe(
        encoded, query[permutation], control[:, permutation], amplitude[permutation],
    )["mean"]
    torch.testing.assert_close(permuted, expected[:, permutation])


def test_masked_nonfinite_basal_values_are_inert_and_observed_values_are_checked():
    model, static, basal, observed, strength, adjacency = fixture()
    observed[:, 2] = False
    finite = basal.clone()
    finite[:, 2] = 999.0
    nonfinite = basal.clone()
    nonfinite[:, 2] = float("nan")
    expected = model.encode(static, finite, observed, strength, adjacency)
    actual = model.encode(static, nonfinite, observed, strength, adjacency)
    for name in ("basal_node_state", "global_basal_state", "global_delta", "local_delta"):
        torch.testing.assert_close(actual[name], expected[name])
    observed[:, 1] = False
    broken = nonfinite.clone()
    broken[:, 0] = float("inf")
    with pytest.raises(ValueError, match="observed basal"):
        model.encode(static, broken, observed, strength, adjacency)
    with pytest.raises(ValueError, match="at least one observed"):
        model.encode(static, basal, torch.zeros_like(observed), strength, adjacency)


def test_action_strength_is_a_dose_on_external_nodes_without_identity_embeddings():
    model, static, basal, observed, strength, adjacency = fixture(batch=2)
    assert not any(isinstance(layer, torch.nn.Embedding) for layer in model.modules())
    encoded = model.encode(static, basal, observed, strength, adjacency)
    doubled = model.encode(static, basal, observed, strength * 2, adjacency)
    torch.testing.assert_close(doubled["global_action"], encoded["global_action"] * 2)
    torch.testing.assert_close(doubled["initial_local_delta"], encoded["initial_local_delta"] * 2)


def test_gradients_reach_static_basal_global_local_message_and_decoder_modules():
    model, static, basal, observed, strength, adjacency = fixture()
    model.train()
    encoded = model.encode(static, basal, observed, strength, adjacency)
    prediction = model.observe(
        encoded, torch.arange(len(static)), torch.zeros(2, len(static)), torch.ones(len(static)),
    )["mean"]
    prediction.square().mean().backward()
    groups = (
        model.static_linear,
        model.value_linear,
        model.observed_flag_linear,
        model.global_delta_mlp,
        model.local_initial_mlp,
        model.message_update_mlp,
        model.decoder,
    )
    for group in groups:
        gradients = [parameter.grad for parameter in group.parameters()]
        assert gradients and all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0


def test_sparse_row_normalization_and_positive_amplitude_are_guarded():
    model, static, basal, observed, strength, adjacency = fixture()
    with torch.sparse.check_sparse_tensor_invariants():
        bad = torch.sparse_coo_tensor(
            adjacency.indices(), adjacency.values() * 2, adjacency.shape,
        ).coalesce()
    with pytest.raises(ValueError, match="sum to one"):
        model.encode(static, basal, observed, strength, bad)
    encoded = model.encode(static, basal, observed, strength, adjacency)
    with pytest.raises(ValueError, match="positive"):
        model.observe(encoded, torch.tensor([0]), torch.zeros(2, 1), torch.zeros(1))
