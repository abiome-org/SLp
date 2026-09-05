from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from context_transfer_scoring import (
    ContextTransferScoringError,
    bootstrap_gene_profiles,
    collapse_gene_profiles,
    score_gene_profiles,
)


def score(prediction, truth, observed, actions, records):
    profiles = collapse_gene_profiles(prediction, truth, observed, actions, records)
    return score_gene_profiles(profiles, np.zeros(prediction.shape[1]))


def test_constant_prediction_primary_correlation_is_undefined_with_tolerance() -> None:
    prediction = np.tile(np.array([1.0, -2.0, 3.0, 0.5]), (3, 1))
    prediction[1] += np.finfo(np.float64).eps
    truth = np.array([[1.0, 2.0, 4.0, 8.0], [3.0, -1.0, 2.0, 0.0], [2.0, 5.0, 1.0, -2.0]])
    report = score(
        prediction, truth, np.ones_like(truth, dtype=bool),
        np.array(["a", "b", "c"]), np.array(["r1", "r2", "r3"]),
    )
    assert report["primaryIndependentlyCenteredGeneMacroProfilePearson"] is None
    assert report["primaryPearsonUndefinedGenes"] == 3


def test_duplicate_all_constructs_of_one_gene_does_not_change_primary_metrics() -> None:
    prediction = np.array([
        [1.0, 2.0, 4.0], [3.0, 0.0, 1.0], [2.0, 5.0, -1.0],
    ])
    truth = np.array([
        [1.5, 1.0, 4.0], [2.0, 1.0, 0.0], [1.0, 4.0, 0.0],
    ])
    observed = np.ones_like(truth, dtype=bool)
    first = score(
        prediction, truth, observed, np.array(["a", "a", "b"]),
        np.array(["a1", "a2", "b1"]),
    )
    second = score(
        np.vstack((prediction, prediction[:2])), np.vstack((truth, truth[:2])),
        np.vstack((observed, observed[:2])), np.array(["a", "a", "b", "a", "a"]),
        np.array(["a1", "a2", "b1", "a1-copy", "a2-copy"]),
    )
    for key in (
        "primaryGeneAveragedProfileMse", "secondaryEqualGeneMeanConstructMse",
        "primaryIndependentlyCenteredGeneMacroProfilePearson",
    ):
        assert second[key] == pytest.approx(first[key], abs=1e-15)


def test_duplicate_exact_record_id_is_rejected() -> None:
    with pytest.raises(ContextTransferScoringError, match="duplicate exact record"):
        collapse_gene_profiles(
            np.ones((2, 2)), np.zeros((2, 2)), np.ones((2, 2), dtype=bool),
            np.array(["a", "b"]), np.array(["same", "same"]),
        )


def test_query_specific_count_mask_controls_missing_support_without_imputation() -> None:
    query_num_cells = np.array([[3, 0, 0], [0, 4, 0]], dtype=np.uint32)
    observed = query_num_cells > 0
    prediction = np.array([[1.0, 999.0, 2.0], [3.0, 4.0, 999.0]])
    truth = np.array([[2.0, -999.0, 2.0], [1.0, 4.0, -999.0]])
    profiles = collapse_gene_profiles(
        prediction, truth, observed, np.array(["a", "b"]), np.array(["r1", "r2"])
    )
    np.testing.assert_array_equal(profiles.observed, observed)
    np.testing.assert_allclose(profiles.gene_profile_mse, [1.0, 0.0])
    report = score_gene_profiles(profiles, np.zeros(3))
    assert report["primaryPearsonUndefinedGenes"] == 2


def test_primary_mse_uses_gene_averaged_profiles_and_construct_mse_is_secondary() -> None:
    prediction = np.zeros((3, 2))
    truth = np.array([[2.0, 2.0], [-2.0, -2.0], [1.0, 1.0]])
    profiles = collapse_gene_profiles(
        prediction, truth, np.ones_like(truth, dtype=bool),
        np.array(["a", "a", "b"]), np.array(["a1", "a2", "b1"]),
    )
    report = score_gene_profiles(profiles, np.zeros(2))
    assert report["primaryGeneAveragedProfileMse"] == pytest.approx(0.5)
    assert report["secondaryEqualGeneMeanConstructMse"] == pytest.approx(2.5)


def test_bootstrap_is_seeded_and_recomputes_centroids_for_each_gene_draw() -> None:
    prediction = np.array([
        [1.0, 2.0, 3.0, 5.0], [2.0, 0.0, 4.0, 1.0],
        [0.0, 3.0, 1.0, 2.0], [4.0, 2.0, 0.0, 1.0],
    ])
    truth = np.array([
        [1.0, 1.0, 4.0, 4.0], [3.0, 0.0, 2.0, 1.0],
        [1.0, 2.0, 0.0, 3.0], [3.0, 4.0, 1.0, 0.0],
    ])
    profiles = collapse_gene_profiles(
        prediction, truth, np.ones_like(truth, dtype=bool),
        np.array(["a", "b", "c", "d"]), np.array(["1", "2", "3", "4"]),
    )
    first = bootstrap_gene_profiles(profiles, samples=9, seed=731, query_chunk=2)
    second = bootstrap_gene_profiles(profiles, samples=9, seed=731, query_chunk=3)
    assert first == second
    assert first["centroidsRecomputedForEveryDraw"] is True
    assert first["decisionUse"] == "descriptive-only"
