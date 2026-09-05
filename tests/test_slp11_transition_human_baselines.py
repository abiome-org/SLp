"""Focused tests for training-only human context null baselines."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).parents[1] / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from human_baselines import (
    HumanBaselineError,
    action_oof_fold,
    duplicate_guide_consistency,
    evaluate_context_baselines,
    fit_context_references,
    randomized_pca_explained_variance,
)


def fixture() -> dict[str, np.ndarray]:
    actions = np.asarray(["A", "B", "C", "D", "A", "B", "C", "D"])
    contexts = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    targets = np.asarray(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
            [11.0, 12.0],
            [12.0, 14.0],
            [13.0, 16.0],
            [14.0, 18.0],
        ]
    )
    return {
        "targets": targets,
        "observed": np.ones_like(targets, dtype=np.bool_),
        "context_index": contexts,
        "action_ids": actions,
        "basal_control": np.asarray([[0.0, 0.0], [10.0, 10.0]]),
        "context_ids": np.asarray(["K562", "RPE1"]),
    }


def test_action_oof_fold_is_shared_across_context_occurrences() -> None:
    assert action_oof_fold("ENSG000001", folds=7) == action_oof_fold(
        "ENSG000001", folds=7
    )


def test_fit_context_references_uses_only_selected_training_rows() -> None:
    data = fixture()
    train = np.asarray([0, 2, 4, 6], dtype=np.int64)
    model = fit_context_references(**data, train_indices=train, folds=2)
    np.testing.assert_allclose(model.perturbation_mean, [[2.0, 4.0], [12.0, 14.0]])
    np.testing.assert_allclose(model.basal_mean, data["basal_control"])
    assert model.perturbation_scale.shape == (2, 2)
    assert np.all(model.perturbation_scale > 0)
    assert model.scale_provenance == "action-grouped-oof-training-residuals"

    changed = data["targets"].copy()
    changed[[1, 3, 5, 7]] = 1_000_000.0
    refit = fit_context_references(
        changed,
        data["observed"],
        data["context_index"],
        data["action_ids"],
        train,
        data["basal_control"],
        data["context_ids"],
        folds=2,
    )
    np.testing.assert_array_equal(refit.perturbation_mean, model.perturbation_mean)
    np.testing.assert_array_equal(refit.perturbation_scale, model.perturbation_scale)


def test_duplicate_action_rows_are_always_in_one_fold() -> None:
    data = fixture()
    folds = [action_oof_fold(str(action), folds=11) for action in data["action_ids"]]
    assert folds[:4] == folds[4:]


def test_evaluation_is_per_context_and_constant_gene_predictions_are_undefined() -> None:
    data = fixture()
    train = np.asarray([0, 2, 4, 6], dtype=np.int64)
    validation = np.asarray([1, 3, 5, 7], dtype=np.int64)
    model = fit_context_references(**data, train_indices=train, folds=2)
    result = evaluate_context_baselines(
        model,
        **{key: value for key, value in data.items() if key != "context_ids"},
        validation_indices=validation,
    )
    assert set(result) == {"K562", "RPE1"}
    metric = result["K562"]["basal_control"]
    assert metric["records"] == 2
    assert metric["gaussianNllNatsPerObservedValue"] > 0
    assert metric["perGeneCenteredPearsonMean"] is None
    assert metric["perGeneCenteredPearsonUndefinedGenes"] == 2


def test_randomized_pca_is_deterministic_and_bounded() -> None:
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(30, 12))
    first = randomized_pca_explained_variance(matrix, components=10)
    second = randomized_pca_explained_variance(matrix, components=10)
    assert first == second
    assert len(first["explainedVarianceRatio"]) == 10
    assert 0 < first["topComponentsCumulativeRatio"] <= 1.000000000001


def test_machine_scale_variation_in_constant_predictions_is_undefined() -> None:
    data = fixture()
    train = np.asarray([0, 2, 4, 6], dtype=np.int64)
    validation = np.asarray([1, 3, 5, 7], dtype=np.int64)
    model = fit_context_references(**data, train_indices=train, folds=2)
    model.perturbation_mean[0, 0] = np.nextafter(
        model.perturbation_mean[0, 0], np.inf
    )
    result = evaluate_context_baselines(
        model,
        **{key: value for key, value in data.items() if key != "context_ids"},
        validation_indices=validation,
    )
    metric = result["K562"]["training_perturbation_mean"]
    assert metric["perGeneCenteredPearsonDefinedGenes"] == 0


def test_duplicate_guide_consistency_labels_summary_unit() -> None:
    targets = np.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 1.0, 2.0]])
    result = duplicate_guide_consistency(
        targets, np.asarray(["A", "A", "B"]), np.zeros(3), bootstrap_samples=20
    )
    assert result["duplicateActionGroups"] == 1
    assert result["guideSummaryPairs"] == 1
    assert result["meanPairPearson"] == pytest.approx(1.0)
    assert result["isBiologicalReplicateNoiseCeiling"] is False
    assert "guide/cell-mean" in result["unit"]


def test_rejects_validation_overlap_or_bad_mask_contract() -> None:
    data = fixture()
    bad = data["observed"].astype(np.int8)
    with pytest.raises(HumanBaselineError, match="boolean"):
        fit_context_references(
            data["targets"],
            bad,
            data["context_index"],
            data["action_ids"],
            np.arange(8),
            data["basal_control"],
            data["context_ids"],
        )
