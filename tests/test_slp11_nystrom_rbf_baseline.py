from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/run_slp11_nystrom_rbf_baseline.py"
SPEC = importlib.util.spec_from_file_location("slp11_nystrom_rbf_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def test_global_gene_fold_is_context_independent_and_deterministic() -> None:
    genes = [f"ENSG{index:011d}" for index in range(30)]
    first = [RUNNER.global_gene_fold(gene, seed=731) for gene in genes]
    second = [RUNNER.global_gene_fold(gene, seed=731) for gene in reversed(genes)]
    assert first == list(reversed(second))
    assert set(first) == {0, 1, 2}


def test_nystrom_psd_floor_and_fold_local_statistics() -> None:
    generator = np.random.default_rng(8)
    ids = tuple(f"ENSG{index:011d}" for index in range(520))
    values = generator.normal(size=(520, 1156)).astype(np.float32)
    model, report = RUNNER.fit_nystrom(ids, values, landmarks=512, bandwidth_sample=64)
    assert np.all(model.eigenvalues > 1e-6)
    assert report["retainedEigenvalues"] + report["droppedEigenvalues"] == 512
    original_mean = model.feature_mean.copy()
    held_outlier = np.full((1, 1156), 1e6, dtype=np.float32)
    assert np.array_equal(original_mean, RUNNER.fit_nystrom(ids, values, landmarks=512, bandwidth_sample=64)[0].feature_mean)
    assert not np.allclose(original_mean, np.vstack((values, held_outlier)).mean(axis=0))


def test_independent_centroid_removes_shared_prediction_and_truth_profiles() -> None:
    truth = np.asarray([[1.0, 4.0, 2.0], [3.0, 2.0, 8.0], [7.0, 6.0, 5.0]])
    prediction = truth + np.asarray([10.0, -5.0, 3.0])
    score = RUNNER.score_profiles(prediction, truth, truth.mean(axis=0))
    assert score["independentlyCenteredPearson"] == pytest.approx(1.0)


def test_collapsing_constructs_weights_each_gene_equally() -> None:
    prediction = np.asarray([[0.0, 2.0], [2.0, 4.0], [8.0, 10.0]])
    truth = prediction.copy()
    genes, collapsed, _ = RUNNER.collapse_prediction(
        prediction, truth, np.asarray(["a", "a", "b"])
    )
    assert genes == ("a", "b")
    assert np.array_equal(collapsed, np.asarray([[1.0, 3.0], [8.0, 10.0]]))
