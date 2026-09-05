from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_slp11_k562_count_latent_state.py"
SPEC = importlib.util.spec_from_file_location("k562_count_latent_runner", PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def routing():
    return {
        "action_ids": np.asarray(["", "", "", "g1", "g1", "g1", "g2", "g2", "held"]),
        "population_ids": np.asarray(["c", "c", "c", "p1", "p1", "p2", "p3", "p3", "p4"]),
        "gem_group": np.asarray([1, 1, 2, 1, 1, 2, 1, 2, 1]),
        "intervention_role": np.asarray(["control"] * 3 + ["train"] * 5 + ["validation"]),
        "reconstruction_role": np.asarray(["train", "train", "train", "train", "validation", "train", "train", "train", "none"]),
        "is_control": np.asarray([True, True, True, False, False, False, False, False, False]),
    }


def test_balanced_sampling_is_deterministic_and_hierarchical():
    index = RUNNER.build_balanced_sampling_index(**routing())
    assert index.control_gems == (1, 2)
    assert index.target_genes == ("g1", "g2")
    assert index.target_populations["g1"] == ("p1", "p2")
    first = RUNNER.draw_balanced_rows(index, np.random.default_rng(731), 2000, 2000)
    second = RUNNER.draw_balanced_rows(index, np.random.default_rng(731), 2000, 2000)
    np.testing.assert_array_equal(first, second)
    assert set(first[:2000]) == {0, 1, 2}
    assert set(first[2000:]) == {3, 5, 6, 7}
    # GEMs and genes, rather than their number of cells, receive similar mass.
    control_gem = routing()["gem_group"][first[:2000]]
    target_gene = routing()["action_ids"][first[2000:]]
    assert 0.45 < np.mean(control_gem == 1) < 0.55
    assert 0.45 < np.mean(target_gene == "g1") < 0.55


def test_static_scaler_uses_only_unique_fitting_actions():
    values = np.asarray([[1, 10], [3, 14], [1000, -1000]], dtype=np.float64)
    normalized, mean, scale = RUNNER.normalize_static_features(values, np.asarray([0, 1]))
    np.testing.assert_allclose(mean, [2, 12])
    np.testing.assert_allclose(scale, [1, 2])
    np.testing.assert_allclose(normalized[:2], [[-1, -1], [1, 1]])
    np.testing.assert_allclose(normalized[2], [8, -8])


def test_control_anchor_uses_aggregate_smoothed_raw_counts():
    counts = np.asarray([[2, 3, 5], [4, 0, 6]], dtype=np.float64)
    result = RUNNER.basal_rates_from_control_sums(counts, counts.sum(1), 3)
    expected = 10_000 * (counts + 0.5) / (counts.sum(1, keepdims=True) + 1.5)
    np.testing.assert_allclose(result, expected)
    assert np.all(result > 0)
    with pytest.raises(ValueError, match="sufficient"):
        RUNNER.basal_rates_from_control_sums(counts, np.asarray([9, 10]), 3)


def test_registered_inputs_require_exact_query_identity():
    queries = np.asarray([f"q{i}" for i in range(8563)])
    rng = np.random.default_rng(731)
    raw = rng.normal(size=(8563, 577)).astype(np.float32)
    static = {
        "entity_id": queries,
        "feature_values": raw,
        "normalized_feature_values": raw,
        "feature_mean": np.zeros(577, dtype=np.float32),
        "feature_scale": np.ones(577, dtype=np.float32),
    }
    roster = {
        "query_ids": queries,
        "query_entity_index": np.arange(8563, dtype=np.int64),
    }
    counts = np.ones((48, 8563), dtype=np.int64)
    control = {
        "query_ids": queries,
        "raw_count_sum": counts,
        "library_count_sum": counts.sum(1),
        "gem_group": np.arange(1, 49),
    }
    result = RUNNER.registered_model_inputs(static, roster, control)
    assert result["query_features"].shape == (8563, 577)
    assert result["basal_rate"].shape == (48, 8563)
    changed = dict(control)
    changed["query_ids"] = queries[::-1]
    with pytest.raises(ValueError, match="identity"):
        RUNNER.registered_model_inputs(static, roster, changed)


def test_gene_gem_weights_and_count_target_use_only_metadata_and_raw_counts():
    action = np.asarray(["a", "a", "b", "b", "b"])
    gem = np.asarray([1, 2, 1, 1, 2])
    weights = RUNNER.gene_gem_weights(action, gem, np.asarray(["a", "b"]), np.asarray([1, 2]))
    np.testing.assert_allclose(weights, [[0.5, 0.5], [2 / 3, 1 / 3]])
    counts = np.asarray([[1, 3], [2, 2], [0, 4], [4, 0], [2, 6]])
    expected_a = np.log1p(np.mean(counts[:2] * 2500, axis=0))
    expected_b = np.log1p(np.mean(counts[2:] * (10_000 / counts[2:].sum(1)[:, None]), axis=0))
    actual = RUNNER.aggregate_gene_cp10k(counts, counts.sum(1), action, np.asarray(["a", "b"]))
    np.testing.assert_allclose(actual, np.stack((expected_a, expected_b)))


def test_anchored_centered_metric_rejects_common_profile_and_is_translation_stable():
    common = np.asarray([[1.0, 4.0, -2.0, 3.0]])
    truth = np.asarray([[0.0, 1.0, 0.0, -1.0], [1.0, 0.0, 2.0, 0.0], [-1.0, 2.0, 0.0, 1.0]])
    anchor = np.asarray([[2.0, 2.0, 2.0, 2.0]] * 3)
    constant_prediction = anchor + np.repeat(common, 3, axis=0)
    undefined = RUNNER.profile_metrics(constant_prediction, truth, anchor)
    assert undefined["independentlyQueryCenteredPearson"] is None
    assert undefined["independentlyQueryCenteredUndefinedGenes"] == 3
    prediction = truth + np.asarray([[0.1, -0.2, 0.3, -0.1]])
    initial = RUNNER.profile_metrics(prediction, truth, anchor)
    shift = np.asarray([[1e8, -2e8, 3e8, -4e8]])
    translated = RUNNER.profile_metrics(prediction + shift, truth + shift, anchor + shift)
    assert initial["independentlyQueryCenteredPearson"] == pytest.approx(
        translated["independentlyQueryCenteredPearson"], abs=1e-8
    )


def test_parameter_norms_cover_every_core_component():
    module_path = ROOT / "modules/slp-1-1-count-latent-state-v1/count_latent_state.py"
    spec = importlib.util.spec_from_file_location("runner_count_core", module_path)
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    model = core.CountLatentState(core.Config(3, hidden_dim=4, state_dim=2, key_dim=2))
    norms = RUNNER.parameter_group_norms(model)
    assert set(norms) == {
        "action_encoder", "cell_keys", "context_encoder", "control_prior",
        "intervention_prior", "posterior", "query_dispersion", "query_loading",
    }
    assert all(np.isfinite(list(norms.values())))


def test_antithetic_noise_and_latent_diagnostics_are_exact():
    noise = RUNNER.deterministic_antithetic_noise(5, 3, draws=4)
    np.testing.assert_array_equal(noise[:2], -noise[2:])
    prior = {
        "mean": torch.zeros(5, 3),
        "logvar": torch.zeros(5, 3),
    }
    posterior = {
        "mean": torch.ones(5, 3) * 0.2,
        "logvar": torch.zeros(5, 3),
    }
    result = RUNNER.latent_diagnostics(None, posterior, prior)
    assert result["activeLatentUnitsKlAbovePoint01"] == 3
    assert result["totalKlPerCell"] == pytest.approx(0.06)
    assert result["posteriorLogvarLowerClampFraction"] == 0
