from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load(
    ROOT / "scripts/run_slp11_guide_composition_ridge_cv.py",
    "test_guide_composition_cv_runner",
)
RIDGE = load(
    ROOT / "modules/slp-1-1-count-static-ridge-v1/count_static_ridge.py",
    "test_guide_composition_cv_ridge",
)


def test_oof_predictions_use_only_other_global_gene_folds():
    rng = np.random.default_rng(731)
    genes = np.asarray([f"ENSG{i:011d}" for i in range(36)])
    features = rng.normal(size=(len(genes), 5)).astype(np.float32)
    coefficient = rng.normal(size=(5, 4))
    target = features @ coefficient + rng.normal(scale=0.05, size=(len(genes), 4))
    selected, scores, reports, prediction, mean_prediction = (
        RUNNER.cross_validated_predictions(RIDGE, genes, features, target)
    )
    assert selected in RIDGE.ALPHAS
    assert set(scores) == set(RIDGE.ALPHAS)
    assert {item["fold"] for item in reports} == {0, 1, 2}
    folds = np.asarray([RIDGE.global_gene_fold(gene, 731) for gene in genes])
    for fold in range(3):
        fit, held = folds != fold, folds == fold
        state = RIDGE.fit_state(features[fit], target[fit])
        np.testing.assert_allclose(
            prediction[held], RIDGE.predict_residual(state, features[held], selected)
        )
        np.testing.assert_allclose(
            mean_prediction[held],
            RIDGE.predict_residual(state, features[held], "mean-limit"),
        )
    observed = np.mean(np.square(prediction - target))
    np.testing.assert_allclose(observed, scores[selected], rtol=1e-11)


def test_extra_signal_can_improve_augmented_oof_without_changing_folds():
    rng = np.random.default_rng(911)
    genes = np.asarray([f"ENSG{i:011d}" for i in range(60)])
    static = rng.normal(size=(len(genes), 3)).astype(np.float32)
    guide = rng.normal(size=(len(genes), 2)).astype(np.float32)
    target = guide @ rng.normal(size=(2, 5))
    _, raw_scores, raw_folds, _, _ = RUNNER.cross_validated_predictions(
        RIDGE, genes, static, target
    )
    _, augmented_scores, augmented_folds, _, _ = RUNNER.cross_validated_predictions(
        RIDGE, genes, np.concatenate((static, guide), axis=1), target
    )
    assert min(augmented_scores.values()) < 0.01 * min(raw_scores.values())
    assert [item["heldGenes"] for item in raw_folds] == [
        item["heldGenes"] for item in augmented_folds
    ]
