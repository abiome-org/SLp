from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/run_slp11_bp_ridge_screen.py"
SPEC = importlib.util.spec_from_file_location("slp11_bp_ridge_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def test_global_gene_fold_is_deterministic_and_shared() -> None:
    genes = [f"ENSG{index:011d}" for index in range(60)]
    first = [RUNNER.global_gene_fold(gene, 731) for gene in genes]
    reverse = [RUNNER.global_gene_fold(gene, 731) for gene in reversed(genes)]
    assert first == list(reversed(reverse))
    assert set(first) == {0, 1, 2}


def test_feature_statistics_are_fitting_only() -> None:
    fitting = np.asarray([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    standardized, mean, scale = RUNNER.standardize_fit(fitting)
    assert np.allclose(mean, [1.0, 3.0])
    assert np.allclose(scale, [1.0, 1.0])
    assert np.allclose(standardized.mean(axis=0), 0.0)
    held_outlier = np.asarray([[1e6, 1e6]], dtype=np.float32)
    assert not np.allclose(mean, np.vstack((fitting, held_outlier)).mean(axis=0))


def test_candidate_order_resolves_exact_ties() -> None:
    scores = {label: 1.0 for label in RUNNER.CANDIDATE_ORDER}
    selected = min(
        RUNNER.CANDIDATE_ORDER,
        key=lambda label: (scores[label], RUNNER.CANDIDATE_ORDER.index(label)),
    )
    assert selected == "0.1"


def test_collapsing_constructs_weights_genes_equally() -> None:
    prediction = np.asarray([[0.0, 2.0], [2.0, 4.0], [8.0, 10.0]])
    genes, collapsed, truth = RUNNER.collapse_prediction(
        prediction, prediction, np.asarray(["a", "a", "b"])
    )
    assert genes == ("a", "b")
    assert np.array_equal(collapsed, np.asarray([[1.0, 3.0], [8.0, 10.0]]))
    assert np.array_equal(collapsed, truth)


def test_independent_query_centering_removes_shared_profile() -> None:
    truth = np.asarray([[1.0, 4.0, 2.0], [3.0, 2.0, 8.0], [7.0, 6.0, 5.0]])
    prediction = truth + np.asarray([10.0, -5.0, 3.0])
    score = RUNNER.score_profiles(prediction, truth)
    assert score["independentlyQueryCenteredPearson"] == pytest.approx(1.0)


def test_v2_identity_contract_does_not_require_unstored_query_ids() -> None:
    archive = {
        "mean": np.zeros((2, 3), dtype=np.float32),
        "record_ids": np.asarray(["r1", "r2"]),
        "action_ids": np.asarray(["a1", "a2"]),
        "context_index": np.asarray([0, 1]),
    }
    assert RUNNER.validate_v2_identity(
        archive,
        np.asarray(["r1", "r2"]),
        np.asarray(["a1", "a2"]),
        np.asarray([0, 1]),
        3,
    )
