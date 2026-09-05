import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PATH = Path(__file__).parents[1] / "scripts/run_slp11_gene_state_source3.py"
SPEC = importlib.util.spec_from_file_location("slp11_gene_state_runner_test", PATH)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_control_values_map_only_to_query_nodes_and_use_control_only_statistics():
    values = np.array([[1.0, 3.0, 100.0], [2.0, 6.0, -50.0]])
    observed = np.array([[True, True, False], [True, True, False]])
    node_values, node_observed, stats = MOD.basal_node_inputs(
        values, observed, np.array([4, 1, 3]), nodes=6,
    )
    expected = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(node_values[:, [4, 1]], expected)
    np.testing.assert_array_equal(node_values[:, [0, 2, 3, 5]], 0.0)
    np.testing.assert_array_equal(node_observed[:, [4, 1]], True)
    np.testing.assert_array_equal(node_observed[:, [0, 2, 3, 5]], False)
    np.testing.assert_allclose(stats, [[2.0, 1.0], [4.0, 2.0]])


def test_exposure_scale_uses_counts_only_in_sampling_variance():
    biological = np.array([[4.0, 9.0], [1.0, 16.0]])
    sampling = np.array([[8.0, 4.0], [3.0, 12.0]])
    actual = MOD.fixed_exposure_scales(
        biological, sampling, np.array([0, 1]), np.array([2.0, 3.0]),
    )
    np.testing.assert_allclose(actual, np.sqrt([[8.0, 11.0], [2.0, 20.0]]))
    # The helper returns scales only; it has no mean/state argument or output.
    assert actual.shape == (2, 2)


def test_action_presence_maps_external_indices_without_identity_parameters():
    index = torch.tensor([2, 0, 2])
    strength = MOD.dense_action_strength(index, nodes=4)
    torch.testing.assert_close(
        strength,
        torch.tensor([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
    )
    assert torch.equal(strength.sum(1), torch.ones(3))


def test_gaussian_loss_is_uniform_per_record_and_masks_missing_targets():
    prediction = torch.zeros(2, 3, requires_grad=True)
    target = torch.tensor([[1.0, float("nan"), 3.0], [2.0, 0.0, 0.0]])
    observed = torch.tensor([[True, False, True], [True, False, False]])
    scale = torch.ones(2, 3)
    loss = MOD.gaussian_loss(prediction, target, observed, scale)
    expected = 0.5 * ((0.5 * (1.0 + 9.0) / 2) + (0.5 * 4.0)) + 0.5 * np.log(2 * np.pi)
    assert float(loss.detach()) == pytest.approx(expected)
    loss.backward()
    assert prediction.grad[0, 1] == 0
    assert prediction.grad[1, 1] == 0


def test_profile_choice_requires_b64_memory_and_throughput_rule():
    batch32 = {
        "batch": 32, "nodes": 24019, "queries": 7036, "edges": 83264,
        "staticFeatures": 610,
        "peakReservedBytes": 3_347_054_592,
        "trainingExamplesPerSecond": 233.4,
        "meanForwardBackwardSeconds": 0.137,
        "meanForwardSeconds": 0.026,
    }
    batch64 = {
        "batch": 64, "nodes": 24019, "queries": 7036, "edges": 83264,
        "staticFeatures": 610,
        "peakReservedBytes": 6_492_782_592,
        "trainingExamplesPerSecond": 316.5,
        "meanForwardBackwardSeconds": 0.202,
        "meanForwardSeconds": 0.0515,
    }
    choice = MOD.validate_profile_choice(batch64, batch32, 10719, 2339)
    assert choice["selectedBatch"] == 64
    assert choice["throughputRatioVsB32"] > 1.15
    assert choice["reservedGiB"] < 9
    assert choice["projectedTotalSeconds"] < 1800
    rejected = dict(batch64, trainingExamplesPerSecond=250.0)
    with pytest.raises(ValueError, match="batch-selection"):
        MOD.validate_profile_choice(rejected, batch32, 10719, 2339)


def test_selection_nll_weights_contexts_and_genes_before_records():
    prediction = np.array([[0.0], [0.0], [0.0], [0.0]])
    target = np.array([[1.0], [1.0], [3.0], [2.0]])
    observed = np.ones_like(target, dtype=bool)
    scale = np.ones_like(target)
    # Context0 has two records for A and one for B; context1 has one for C.
    score = MOD.selection_gene_macro_nll(
        prediction, target, observed, scale,
        np.array(["A", "A", "B", "C"]), np.array([0, 0, 0, 1]),
    )
    constant = 0.5 * np.log(2 * np.pi)
    context0 = ((0.5 * 1.0) + (0.5 * 9.0)) / 2 + constant
    context1 = 0.5 * 4.0 + constant
    assert score == pytest.approx((context0 + context1) / 2)


def test_constant_world_forecast_preserves_undefined_correlation_as_none():
    baselines = MOD.load_source(MOD.BASELINES, "gene_state_constant_baselines_test")
    data = {
        "action_ids": np.array(["A", "B"]),
        "targets": np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]]),
        "observed": np.ones((2, 3), dtype=bool),
        "target_value_space": np.asarray("synthetic-z"),
    }
    report = MOD.gene_metrics(
        baselines,
        np.zeros((2, 3)),
        np.ones((2, 3)),
        data,
        np.array([0, 1]),
        np.zeros(3),
    )
    assert report["gene_macro_profile_centroid_adjusted_pearson_mean"] is None
    assert report["profile_centroid_adjusted_pearson_undefined"] == 2


def test_response32_feature_contract_keeps_static_and_explicit_missingness():
    static = np.zeros((3, 577), dtype=np.float32)
    static[0, 0] = 2.0
    combined = np.zeros((3, 610), dtype=np.float32)
    combined[:, :577] = static
    combined[0, 577:609] = 0.5
    combined[0, 609] = 1.0
    graph = {
        "node_ids": np.array(["a", "b", "c"]),
        "static_features": static,
        "static_feature_observed": np.array([True, False, False]),
        "node_features": combined,
        "response_query_feature_observed": np.array([True, False, False]),
        "query_node_index": np.array([0]),
    }
    np.testing.assert_array_equal(MOD.graph_node_features(graph, "static577"), static)
    np.testing.assert_array_equal(MOD.graph_node_features(graph, "response32"), combined)
    broken = {**graph, "node_features": combined.copy()}
    broken["node_features"][1, 580] = 1.0
    with pytest.raises(ValueError, match="response32"):
        MOD.graph_node_features(broken, "response32")
