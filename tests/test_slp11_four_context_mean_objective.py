import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[1]
PATH = ROOT / "modules/slp-1-1-world-transition-v1/mean_objective.py"
SPEC = importlib.util.spec_from_file_location("slp11_mean_objective_test", PATH)
OBJECTIVE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(OBJECTIVE)

RUNNER_PATH = ROOT / "scripts/run_slp11_four_context_mean_objective_pair.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("slp11_mean_pair_test", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
sys.modules[RUNNER_SPEC.name] = RUNNER
RUNNER_SPEC.loader.exec_module(RUNNER)


def test_context_query_sd_uses_only_fitting_observed_rows_and_floor():
    targets = np.array(
        [[0.0, 2.0], [2.0, 999.0], [100.0, 100.0], [4.0, 8.0], [8.0, 8.0]],
        dtype=np.float32,
    )
    observed = np.ones_like(targets, dtype=bool)
    observed[1, 1] = False
    scales = OBJECTIVE.context_query_sd(
        targets,
        observed,
        np.array([0, 0, 0, 1, 1]),
        np.array([0, 1, 3, 4]),
        2,
        floor=0.05,
    )
    np.testing.assert_allclose(scales[0], [1.0, 0.05])
    np.testing.assert_allclose(scales[1], [2.0, 0.05])


def test_masked_standardized_mse_uses_row_means_and_fixed_global_weights():
    prediction = torch.tensor([[2.0, 50.0], [3.0, 4.0]])
    target = torch.zeros_like(prediction)
    observed = torch.tensor([[True, False], [True, True]])
    scale = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    weights = torch.tensor([0.5, 1.5])
    loss = OBJECTIVE.masked_standardized_mse(
        prediction, target, observed, scale, weights
    )
    # Row losses are 1 and mean([9,4])=6.5; weights are not batch-renormalized.
    assert torch.isclose(loss, torch.tensor((0.5 + 1.5 * 6.5) / 2))


def test_masked_standardized_mse_rejects_empty_observed_row():
    with pytest.raises(OBJECTIVE.MeanObjectiveError):
        OBJECTIVE.masked_standardized_mse(
            torch.zeros(2, 2),
            torch.zeros(2, 2),
            torch.tensor([[True, True], [False, False]]),
            torch.ones(2, 2),
            torch.ones(2),
        )


def test_shuffled_batches_are_deterministic_full_and_fitting_only():
    fitting = np.arange(101, 151, dtype=np.int64)
    first = list(
        OBJECTIVE.deterministic_shuffled_batches(
            fitting, batch_size=8, steps=17, seed=731
        )
    )
    second = list(
        OBJECTIVE.deterministic_shuffled_batches(
            fitting, batch_size=8, steps=17, seed=731
        )
    )
    assert len(first) == 17
    assert all(len(batch) == 8 for batch in first)
    assert all(set(batch).issubset(set(fitting)) for batch in first)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))


def test_default_seed_is_explicit_731_and_initialization_is_identical():
    module = RUNNER.load_model(RUNNER.MODEL)
    default = RUNNER.initialize_model(module)
    explicit = RUNNER.initialize_model(module, 731)
    assert RUNNER.SEED == 731
    assert all(
        torch.equal(default.state_dict()[name], explicit.state_dict()[name])
        for name in default.state_dict()
    )
    default_batches = list(
        OBJECTIVE.deterministic_shuffled_batches(
            np.arange(80), batch_size=8, steps=4, seed=RUNNER.SEED
        )
    )
    explicit_batches = list(
        OBJECTIVE.deterministic_shuffled_batches(
            np.arange(80), batch_size=8, steps=4, seed=731
        )
    )
    assert all(
        np.array_equal(left, right)
        for left, right in zip(default_batches, explicit_batches)
    )


def test_frozen_source3_inference_inputs_are_bitwise_preserved():
    data_path = ROOT / "data/derived/slp11-human-four-context-v2/development.npz"
    reference_path = ROOT / (
        "results/slp11-transition/"
        "human-gwps-fixed-context-minimal-control-physical-state128-response32-seed731-v1/"
        "model/reference.npz"
    )
    with np.load(data_path, allow_pickle=False) as data, np.load(
        reference_path, allow_pickle=False
    ) as reference:
        assert np.array_equal(reference["query_ids"], data["query_ids"])
        assert np.array_equal(reference["control_mean"], data["basal_control"][:3])
        selected = reference["context_query_indices"]
        assert np.array_equal(
            reference["context_features"], reference["query_features"][selected]
        )
        common = data["context_basal_observed"].all(axis=0)
        basal = data["context_basal_expression"]
        mean = np.asarray([basal[index, common].mean() for index in range(3)])[:, None]
        std = np.maximum(
            np.asarray([basal[index, common].std() for index in range(3)])[:, None],
            1e-5,
        )
        normalized = ((basal[:3] - mean) / std).astype(np.float32)
        assert np.array_equal(reference["context_values"], normalized[:, selected])


def test_undefined_correlations_fail_every_required_gate():
    metrics = {
        name: {
            "gene_profile_raw_mse": 0.5,
            "independently_query_centered_profile_pearson": None,
        }
        for name in RUNNER.CONTEXTS
    }
    reports = {
        "source3": {"validationMetrics": metrics},
        "source4": {"validationMetrics": metrics},
    }
    baseline = {
        "contexts": {
            name: {
                "mean": {"gene_profile_raw_mse": 1.0},
                "ridge": {
                    "gene_profile_raw_mse": 0.75,
                    "independently_query_centered_profile_pearson": 0.2,
                },
            }
            for name in RUNNER.CONTEXTS
        }
    }
    decision = RUNNER.decide(reports, baseline)
    assert not decision["adaptiveComparison"]["passed"]
    assert not decision["standaloneWorld"]["passed"]
    assert all(
        not context["passed"]
        for context in decision["adaptiveComparison"]["contexts"].values()
    )
