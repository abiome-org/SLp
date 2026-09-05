import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).parents[1] / "scripts/score_slp11_frangieh_paired_state_vs_static.py"
SPEC = importlib.util.spec_from_file_location("paired_scoring", PATH)
SCORING = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SCORING)


def test_fixed_gates_require_all_mse_and_defined_world_correlation():
    baselines = {
        "mean": {"raw_mse": 2.0, "query_centroid_adjusted_profile_pearson": float("nan")},
        "base577": {"raw_mse": 1.5, "query_centroid_adjusted_profile_pearson": 0.1},
        "physical1156": {"raw_mse": 1.4, "query_centroid_adjusted_profile_pearson": 0.11},
    }
    passing = SCORING.evaluate_gates(
        {"raw_mse": 1.0, "query_centroid_adjusted_profile_pearson": 0.12}, baselines
    )
    assert passing["pass"]
    undefined = SCORING.evaluate_gates(
        {"raw_mse": 1.0, "query_centroid_adjusted_profile_pearson": float("nan")}, baselines
    )
    assert not undefined["pass"]
    weak_mse = SCORING.evaluate_gates(
        {"raw_mse": 1.395, "query_centroid_adjusted_profile_pearson": 0.12}, baselines
    )
    assert not weak_mse["pass"]


def test_bootstrap_is_paired_and_deterministic():
    truth = np.zeros((3, 2))
    world = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    baseline = world + 1.0
    samples = np.random.default_rng(731).integers(0, 3, size=(1000, 3))
    first = SCORING.paired_mse_bootstrap(world, baseline, truth, samples)
    second = SCORING.paired_mse_bootstrap(world, baseline, truth, samples)
    assert first == second
    assert first["raw_mse_difference_baseline_minus_world_ci025"] > 0


def test_alignment_and_nonfinite_values_fail_closed():
    with pytest.raises(ValueError, match="mismatch"):
        SCORING._require_equal("query", np.array(["a"]), np.array(["b"]))
    with pytest.raises(ValueError, match="finite"):
        SCORING.score_metrics(np.array([[np.nan, 0.0]]), np.zeros((1, 2)))
