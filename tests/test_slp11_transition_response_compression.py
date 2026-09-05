from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from response_compression import (
    ResponseCompressionError,
    fit_response_compression,
    gene_macro_point_metrics,
)
from transition_baselines import fit_ridge


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(731)
    features = np.repeat(rng.normal(size=(12, 4)), 2, axis=0)
    contexts = np.tile(np.arange(2), 12)
    context_means = np.stack((np.linspace(-1, 1, 10), np.linspace(1, -1, 10)))
    basis = rng.normal(size=(2, 10))
    scores = features[:, :2] @ np.array([[1.2, -0.4], [0.3, 0.9]])
    targets = context_means[contexts] + scores @ basis
    return features, targets, np.ones_like(targets, dtype=np.bool_), contexts


def test_oracle_and_feature_forecast_recover_low_rank_fixture() -> None:
    features, targets, observed, contexts = _fixture()
    model = fit_response_compression(
        features, targets, observed, contexts, maximum_rank=2, alpha=1e-8, scale_floor=1e-8
    )

    np.testing.assert_allclose(model.oracle_reconstruct(targets, observed, contexts, 2), targets, atol=2e-5)
    np.testing.assert_allclose(model.predict(features, contexts, 2), targets, atol=2e-5)
    assert 0.9999 <= model.retained_training_variance(2) <= 1.0001


def test_fit_statistics_do_not_consume_later_diagnostic_outcomes() -> None:
    features, targets, observed, contexts = _fixture()
    model = fit_response_compression(features[:16], targets[:16], observed[:16], contexts[:16], maximum_rank=2)
    altered = targets[16:].copy()
    altered[:, 0] += 1_000.0

    original_prediction = model.predict(features[16:], contexts[16:], 2)
    altered_prediction = model.predict(features[16:], contexts[16:], 2)
    np.testing.assert_array_equal(original_prediction, altered_prediction)
    assert not np.allclose(
        model.oracle_reconstruct(targets[16:], observed[16:], contexts[16:], 2),
        model.oracle_reconstruct(altered, observed[16:], contexts[16:], 2),
    )


def test_complete_panel_and_rank_are_enforced() -> None:
    features, targets, observed, contexts = _fixture()
    observed[0, 0] = False
    with pytest.raises(ResponseCompressionError, match="complete observed panel"):
        fit_response_compression(features, targets, observed, contexts, maximum_rank=2)
    with pytest.raises(ResponseCompressionError, match="rank must"):
        fit_response_compression(
            features, targets, np.ones_like(targets, dtype=np.bool_), contexts, maximum_rank=2
        ).predict(features, contexts, 3)


def test_gene_macro_metrics_weight_genes_equally_and_label_centroid_adjustment() -> None:
    truth = np.array([[1.0, 2.0, 4.0], [1.0, 2.0, 4.0], [4.0, 2.0, 1.0]])
    prediction = np.array([[1.1, 2.1, 3.9], [1.1, 2.1, 3.9], [3.8, 2.0, 1.2]])
    observed = np.ones_like(truth, dtype=np.bool_)
    metrics = gene_macro_point_metrics(
        prediction, truth, observed, ["gene-a", "gene-a", "gene-b"], np.zeros(3)
    )

    mse_a = np.mean(np.square(prediction[0] - truth[0]))
    mse_b = np.mean(np.square(prediction[2] - truth[2]))
    assert metrics["gene_macro_mse"] == pytest.approx((mse_a + mse_b) / 2)
    assert metrics["gene_macro_source_centroid_adjusted_mse"] == metrics["gene_macro_mse"]
    assert metrics["intervention_genes"] == 2


def test_projected_full_ridge_equals_context_local_latent_ridge() -> None:
    features, targets, observed, contexts = _fixture()
    training = np.arange(16)
    scoring = np.arange(16, 24)
    alpha = 3.0
    basis = fit_response_compression(
        features[training],
        targets[training],
        observed[training],
        contexts[training],
        maximum_rank=2,
        alpha=alpha,
    )
    full_prediction = np.empty_like(targets[scoring])
    explicit_prediction = np.empty_like(targets[scoring])
    standardized_training = (
        targets[training] - basis.context_means_[contexts[training]]
    ) / basis.query_scales_
    scores = standardized_training @ basis.components_.T
    for context in range(2):
        fit_rows = training[contexts[training] == context]
        score_rows = scoring[contexts[scoring] == context]
        full_model = fit_ridge(
            features[fit_rows], targets[fit_rows], observed[fit_rows], alpha
        )
        full_prediction[contexts[scoring] == context] = full_model.predict(features[score_rows])
        latent_model = fit_ridge(
            features[fit_rows],
            scores[contexts[training] == context],
            np.ones((fit_rows.size, 2), dtype=np.bool_),
            alpha,
        )
        explicit_prediction[contexts[scoring] == context] = (
            basis.context_means_[context]
            + latent_model.predict(features[score_rows]) @ basis.components_ * basis.query_scales_
        )

    projected = basis.project_forecast(full_prediction, contexts[scoring], 2)
    np.testing.assert_allclose(projected, explicit_prediction, atol=1e-10)
