from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_module():
    path = Path(__file__).parents[1] / "scripts/audit_slp11_fitting_convergence.py"
    spec = importlib.util.spec_from_file_location(
        "fitting_convergence_audit_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sample_is_deterministic_and_value_independent() -> None:
    module = load_module()
    genes = np.asarray(["g3", "g1", "g2", "g4", "g1"])
    first = module.deterministic_gene_sample(genes, count=3, seed=731, label="x")
    second = module.deterministic_gene_sample(genes[::-1], count=3, seed=731, label="x")
    assert np.array_equal(first, second)
    assert len(first) == 3


def test_gene_collapse_respects_row_mass_and_centered_metrics() -> None:
    module = load_module()
    genes = np.asarray(["a", "a", "b"])
    target = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 2.0]])
    prediction = target.copy()
    observed = np.ones_like(target, dtype=np.bool_)
    pred, truth, mask, _ = module.collapse_profiles(
        genes, prediction, target, observed, np.asarray([1.0, 3.0, 2.0])
    )
    assert np.allclose(truth[0], [1.5, 2.5])
    score = module.profile_metrics(pred, truth, mask)
    assert score["geneProfileMse"] == 0.0
    assert np.isclose(score["independentlyQueryCenteredPearson"], 1.0)


def test_sample_objective_keeps_frozen_weights() -> None:
    module = load_module()
    target = np.zeros((2, 2))
    prediction = np.asarray([[1.0, 1.0], [2.0, 2.0]])
    observed = np.ones_like(target, dtype=np.bool_)
    score = module.sampled_objective(
        prediction, target, observed, np.ones_like(target), np.asarray([2.0, 0.5])
    )
    assert np.isclose(score["exactFrozenWeightMean"], 2.0)
    assert np.isclose(score["sampleWeightNormalized"], 1.6)


def test_centered_profile_is_translation_invariant() -> None:
    module = load_module()
    prediction = np.asarray([[1.0, 2.0, 4.0], [2.0, 1.0, 5.0], [4.0, 3.0, 2.0]])
    truth = np.asarray([[2.0, 3.0, 3.0], [3.0, 1.0, 4.0], [5.0, 2.0, 1.0]])
    mask = np.ones_like(prediction, dtype=np.bool_)
    first = module.profile_metrics(prediction, truth, mask)
    common_prediction = np.asarray([1000.0, -2000.0, 500.0])
    common_truth = np.asarray([-800.0, 300.0, 1200.0])
    second = module.profile_metrics(
        prediction + common_prediction, truth + common_truth, mask
    )
    assert np.isclose(
        first["independentlyQueryCenteredPearson"],
        second["independentlyQueryCenteredPearson"],
    )


def test_constant_across_gene_prediction_has_undefined_centered_correlation() -> None:
    module = load_module()
    prediction = np.broadcast_to(np.asarray([1.0, 2.0, 3.0]), (3, 3)).copy()
    truth = np.asarray([[1.0, 3.0, 2.0], [2.0, 1.0, 4.0], [4.0, 2.0, 1.0]])
    score = module.profile_metrics(
        prediction, truth, np.ones_like(prediction, dtype=np.bool_)
    )
    assert score["independentlyQueryCenteredPearson"] is None
    assert score["undefinedGenes"] == 3
