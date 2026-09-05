from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / "scripts/run_slp11_yeast_batch_ridge.py"
    spec = importlib.util.spec_from_file_location("yeast_batch_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_population_weights_equalize_genes_and_use_cells_within_gene() -> None:
    runner = load_runner()
    actions = np.asarray(["a", "a", "b"])
    weights = runner.population_weights(actions, np.asarray([2, 6, 5]))
    np.testing.assert_allclose(weights, [0.25, 0.75, 1.0])
    assert weights[actions == "a"].sum() == 1.0
    assert weights[actions == "b"].sum() == 1.0


def test_pooling_and_batch_residual_subtraction_are_cell_weighted() -> None:
    runner = load_runner()
    actions = np.asarray(["a", "a", "b"])
    cells = np.asarray([1, 3, 2])
    truth = np.asarray([[10.0], [22.0], [7.0]])
    batch_mean = np.asarray([[4.0], [10.0], [4.0]])
    genes, raw, residual = runner.pool_genes(actions, cells, truth, truth - batch_mean)
    assert genes.tolist() == ["a", "b"]
    np.testing.assert_allclose(raw[:, 0], [19.0, 7.0])
    np.testing.assert_allclose(residual[:, 0], [10.5, 3.0])


def test_feature_normalizer_uses_unique_genes_not_population_count() -> None:
    runner = load_runner()
    actions = np.asarray(["a", "a", "b"])
    features = np.asarray([[0.0, 1.0], [0.0, 1.0], [4.0, 1.0]])
    mean, scale = runner.fit_feature_normalizer(features, actions)
    np.testing.assert_allclose(mean, [2.0, 1.0])
    np.testing.assert_allclose(scale, [2.0, 1.0])


def test_constant_profile_correlation_is_undefined_despite_roundoff() -> None:
    runner = load_runner()
    truth = np.asarray([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]])
    prediction = np.full_like(truth, 1.0e12)
    prediction[:, 0] = np.nextafter(prediction[:, 0], np.inf)
    result = runner.metrics(truth, prediction)
    assert result["ordinaryUndefinedGenes"] == 2
    assert result["independentlyQueryCenteredUndefinedGenes"] == 2
    assert result["ordinaryGeneProfilePearson"] is None


def test_batch_mean_prediction_residual_is_exact_zero() -> None:
    runner = load_runner()
    stats_type = runner._load_python(runner.CORE_PATH, "batch_mean_zero_test").BatchRidgeStatistics
    stats = stats_type(577, 2)
    stats.update("B1", np.zeros((1, 577)), np.asarray([[2.0, 5.0]]), np.asarray([1.0]))
    model = runner.batch_only_model(stats)
    static = runner.StaticFeatures(
        np.asarray(["g"]), np.asarray([[0.0] + [0.0] * 576]), {"g": 0},
        np.asarray([True]), np.asarray([False]),
    )
    raw, residual = runner.predict_pooled_genes(
        model, model, static, np.asarray(["g"]), {"g": {"B1": 1.0}},
        np.zeros(577), np.ones(577),
    )
    np.testing.assert_array_equal(raw, [[2.0, 5.0]])
    np.testing.assert_array_equal(residual, [[0.0, 0.0]])


def test_common_expression_profile_is_not_perturbation_signal():
    runner = load_runner()
    rng = np.random.default_rng(731)
    common = np.linspace(0.01, 3.0, 107)
    prediction = np.broadcast_to(common, (346, 107)).copy()
    prediction[::2] *= np.nextafter(1.0, 2.0)
    truth = common + rng.normal(0, 0.1, prediction.shape)
    result = runner.metrics(truth, prediction)
    assert result["ordinaryUndefinedGenes"] == 0
    assert result["independentlyQueryCenteredUndefinedGenes"] == 346
    assert result["independentlyQueryCenteredPearson"] is None
