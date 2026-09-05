from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

import response_basis
from response_basis import (
    ResponseBasisError,
    fit_grouped_oof_response_basis,
    fit_grouped_oof_response_basis_grid,
)


def _fixture():
    rng = np.random.default_rng(19)
    genes = [f"ENSG{index:011d}" for index in range(1, 13)]
    features_by_gene = rng.normal(size=(len(genes), 4))
    basis = rng.normal(size=(2, 8))
    context_mean = np.stack((np.linspace(0, 1, 8), np.linspace(1, 0, 8)))
    features = np.repeat(features_by_gene, 2, axis=0)
    contexts = np.tile([0, 1], len(genes))
    actions = tuple(action for action in genes for _ in range(2))
    coefficients = features @ rng.normal(size=(4, 2))
    targets = context_mean[contexts] + coefficients @ basis
    observed = np.ones_like(targets, dtype=np.bool_)
    return features, targets, observed, contexts, actions


def test_response_basis_predicts_shape_and_uses_oof_context_scales() -> None:
    features, targets, observed, contexts, actions = _fixture()
    model = fit_grouped_oof_response_basis(
        features, targets, observed, contexts, actions, rank=2, alpha=1e-8, folds=3
    )

    prediction = model.predict(features, contexts)
    np.testing.assert_allclose(prediction, targets, atol=1e-6)
    assert model.scales(contexts).shape == targets.shape
    assert model.residual_scale_.values.shape == (2, targets.shape[1])
    assert model.residual_scale_.provenance.startswith("action-grouped-oof")
    assert model.baseline_family == "training-response-basis-plus-feature-linear-ridge"


def test_grid_uses_nested_response_bases_and_keeps_actions_out_of_each_fold() -> None:
    features, targets, observed, contexts, actions = _fixture()
    models = fit_grouped_oof_response_basis_grid(
        features,
        targets,
        observed,
        contexts,
        actions,
        ranks=(1, 2),
        alphas=(0.1, 10.0),
        folds=3,
    )

    assert set(models) == {(1, 0.1), (1, 10.0), (2, 0.1), (2, 10.0)}
    np.testing.assert_array_equal(models[(1, 0.1)].components_, models[(2, 0.1)].components_[:1])
    for audit in models[(2, 10.0)].fold_audit_:
        assert audit["actionOverlap"] == 0
        assert audit["fittingActions"] + audit["heldActions"] == 12
        assert audit["basisFloat64Sha256"]


def test_oof_calibration_refits_basis_without_held_action_rows(monkeypatch) -> None:
    features, targets, observed, contexts, actions = _fixture()
    fitting_row_counts: list[int] = []
    original = response_basis._response_components

    def traced_components(residuals, rank, seed):
        fitting_row_counts.append(residuals.shape[0])
        return original(residuals, rank, seed)

    monkeypatch.setattr(response_basis, "_response_components", traced_components)
    fit_grouped_oof_response_basis(
        features, targets, observed, contexts, actions, rank=2, alpha=1.0, folds=3
    )

    assert len(fitting_row_counts) == 4
    assert all(0 < count < len(actions) for count in fitting_row_counts[:-1])
    assert fitting_row_counts[-1] == len(actions)


def test_context_mean_is_fit_from_supplied_training_rows() -> None:
    features, targets, observed, contexts, actions = _fixture()
    model = fit_grouped_oof_response_basis(
        features, targets, observed, contexts, actions, rank=2, alpha=1.0, folds=3
    )
    for context in (0, 1):
        np.testing.assert_allclose(
            model.context_means_[context], targets[contexts == context].mean(axis=0)
        )


def test_missing_outcome_is_rejected_instead_of_imputed() -> None:
    features, targets, observed, contexts, actions = _fixture()
    observed[0, 0] = False
    with pytest.raises(ResponseBasisError, match="no outcome imputation"):
        fit_grouped_oof_response_basis(
            features, targets, observed, contexts, actions, rank=2, alpha=1.0
        )


def test_repeated_action_static_features_must_agree_across_contexts() -> None:
    features, targets, observed, contexts, actions = _fixture()
    features[1, 0] += 1.0
    with pytest.raises(ResponseBasisError, match="inconsistent static features"):
        fit_grouped_oof_response_basis(
            features, targets, observed, contexts, actions, rank=2, alpha=1.0
        )
