from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from context_transfer_baselines import (
    ContextTransferError,
    control_context_distances,
    equal_source_fitting_centroid,
    gene_macro_transfer_metrics,
    population_roster,
    same_gene_source_response_forecast,
)


def test_constant_baseline_has_undefined_independently_centered_correlation() -> None:
    truth = np.array([[1.0, 3.0, -2.0], [2.0, -1.0, 4.0]])
    prediction = np.broadcast_to(np.array([0.2, -0.5, 1.0]), truth.shape).copy()
    result = gene_macro_transfer_metrics(
        prediction, truth, np.ones_like(truth, dtype=bool), np.array(["a", "b"]),
        training_centroid=np.zeros(3),
    )
    assert result["independentlyGeneCenteredProfilePearsonMean"] is None
    assert result["independentlyGeneCenteredUndefinedGenes"] == 2


def test_gene_weighting_is_insensitive_to_duplicated_identical_records() -> None:
    prediction = np.array([[1.0, 2.0, 4.0], [2.0, 0.0, -1.0]])
    truth = np.array([[1.2, 1.5, 3.8], [1.0, -1.0, 0.0]])
    mask = np.ones_like(truth, dtype=bool)
    first = gene_macro_transfer_metrics(
        prediction, truth, mask, np.array(["a", "b"]), training_centroid=np.zeros(3)
    )
    second = gene_macro_transfer_metrics(
        np.vstack((prediction, prediction[0])), np.vstack((truth, truth[0])),
        np.vstack((mask, mask[0])), np.array(["a", "b", "a"]),
        training_centroid=np.zeros(3),
    )
    for key in ("geneMacroMse", "independentlyGeneCenteredProfilePearsonMean"):
        assert second[key] == pytest.approx(first[key], abs=1e-15)


def test_missing_and_zero_variance_support_are_reported_without_imputation() -> None:
    prediction = np.array([[1.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    truth = np.array([[1.5, 0.0, 0.0], [3.0, 3.0, 3.0]])
    observed = np.array([[True, False, False], [True, True, True]])
    result = gene_macro_transfer_metrics(
        prediction, truth, observed, np.array(["a", "b"]),
        training_centroid=np.zeros(3),
    )
    assert result["queriesWithAnySupport"] == 3
    assert result["independentlyGeneCenteredUndefinedGenes"] >= 1


def test_forecast_helpers_have_no_target_context_outcome_argument() -> None:
    parameters = inspect.signature(same_gene_source_response_forecast).parameters
    assert "target_context_truth" not in parameters
    assert "target_context_observed" not in parameters
    fitting_targets = np.array([[1.0, 3.0], [2.0, 4.0], [5.0, 7.0]])
    fitting_mask = np.ones_like(fitting_targets, dtype=bool)
    contexts = np.array([0, 0, 1])
    actions = np.array(["a", "b", "a"])
    centroid = equal_source_fitting_centroid(
        fitting_targets, fitting_mask, contexts, actions
    )
    forecast, seen = same_gene_source_response_forecast(
        fitting_targets, fitting_mask, contexts, actions, np.array(["a", "c"]), centroid
    )
    np.testing.assert_allclose(forecast[0], [3.0, 5.0])
    np.testing.assert_allclose(forecast[1], centroid)
    np.testing.assert_array_equal(seen, [True, False])


def test_control_distance_standardizes_within_each_context_and_roster_is_exact() -> None:
    fitting = np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0], [2.0, 1.0, 0.0]])
    target = np.array([5.0, 7.0, 9.0])
    distances = control_context_distances(fitting, target, np.ones(3, dtype=bool))
    assert distances[0] == pytest.approx(0.0, abs=1e-14)
    assert distances[1] == pytest.approx(0.0, abs=1e-14)
    assert distances[2] > 0.0
    roster = population_roster(
        np.array(["a", "a", "b"]), np.array(["p1", "p1", "p2"]),
        np.array(["c1", "c1", "c2"]), np.array(["t1", "t1", "t2"]),
    )
    np.testing.assert_array_equal(roster.population_ids, ["p1", "p2"])
    with pytest.raises(ContextTransferError, match="multiple construct"):
        population_roster(
            np.array(["a", "a"]), np.array(["p", "p"]),
            np.array(["c1", "c2"]), np.array(["t", "t"]),
        )
