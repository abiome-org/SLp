from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load(
    ROOT / "scripts/run_slp11_control_coexpression_ridge_cv.py",
    "test_control_coexpression_cv_runner",
)
RIDGE = load(
    ROOT / "modules/slp-1-1-count-static-ridge-v1/count_static_ridge.py",
    "test_control_coexpression_cv_ridge",
)


def test_stable_action_join_appends_presence_and_keeps_absent_zero():
    action_ids = np.asarray(["g2", "g1", "g3"])
    values = np.zeros((3, 64), dtype=np.float32)
    values[1] = np.arange(64, dtype=np.float32)
    values[2] = -1
    present = np.asarray([False, True, True])
    result = RUNNER.join_action_coexpression(
        np.asarray(["g1", "g2", "g3"]), action_ids, values, present
    )
    np.testing.assert_array_equal(result[0, :64], np.arange(64))
    np.testing.assert_array_equal(result[1, :64], 0)
    np.testing.assert_array_equal(result[:, -1], [1, 0, 1])


def test_join_rejects_missing_roster_and_nonzero_absent_features():
    with pytest.raises(ValueError, match="absent from action roster"):
        RUNNER.join_action_coexpression(
            np.asarray(["missing"]),
            np.asarray(["g1"]),
            np.zeros((1, 64), np.float32),
            np.ones(1, dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="invalid control-coexpression"):
        RUNNER.join_action_coexpression(
            np.asarray(["g1"]),
            np.asarray(["g1"]),
            np.ones((1, 64), np.float32),
            np.zeros(1, dtype=np.bool_),
        )


def test_cv_expanded_per_gene_errors_equal_selected_candidate_score():
    rng = np.random.default_rng(731)
    genes = np.asarray([f"ENSG{i:011d}" for i in range(45)])
    features = rng.normal(size=(len(genes), 7)).astype(np.float32)
    target = features @ rng.normal(size=(7, 5)) + rng.normal(
        scale=0.01, size=(len(genes), 5)
    )
    selected, scores, reports, folds, per_gene = RUNNER.cross_validated_predictions(
        RIDGE, genes, features, target
    )
    assert {item["fold"] for item in reports} == {0, 1, 2}
    np.testing.assert_allclose(per_gene.mean(), scores[selected], rtol=1e-11, atol=1e-14)
    np.testing.assert_array_equal(
        folds, [RIDGE.global_gene_fold(gene, 731) for gene in genes]
    )
