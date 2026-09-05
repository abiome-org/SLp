import importlib.util
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).parents[1] / "modules/slp-1-1-sl-readout-v1/readout.py"
SPEC = importlib.util.spec_from_file_location("sl_readout", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_inner_split_is_deterministic_and_gene_disjoint():
    pairs = np.array([["a", "b"], ["a", "c"], ["d", "e"], ["f", "g"], ["h", "i"]])
    first = M.inner_split(pairs, 42, 0)
    second = M.inner_split(pairs, 42, 0)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]
    for pair in pairs[first[0]]:
        assert all(g not in first[2] for g in pair)
    for pair in pairs[first[1]]:
        assert all(g in first[2] for g in pair)


def test_feature_loader_rejects_reordered_world_ids(tmp_path):
    roster = tmp_path / "roster"
    roster.mkdir()
    ids = np.array([["a", "b"], ["c", "d"]])
    np.savez(roster / "pairs.npz", pair_ids=ids, gene_indices=np.zeros((2, 2)))
    np.save(tmp_path / "base.npy", np.ones((2, 1081), np.float32))
    np.savez(tmp_path / "world.npz", features=np.ones((2, 1054)), pair_ids=ids[::-1],
             source_pair_roster_sha256=np.array(M.sha256(roster / "pairs.npz")),
             source_start=np.array(0), source_stop=np.array(2))
    with pytest.raises(ValueError, match="not exactly aligned"):
        M.load_features(tmp_path / "base.npy", tmp_path / "world.npz", roster)


def test_baseline_directory_concatenates_only_selected_rows(tmp_path):
    widths = M.BASELINE_WIDTHS
    for name, width in zip(M.BASELINE_BLOCKS, widths):
        np.save(tmp_path / name, np.arange(4 * width, dtype=np.float32).reshape(4, width))
    features = M.BaselineFeatures(tmp_path, 4)
    selected = features[np.array([3, 1])]
    assert features.shape == (4, 1081)
    assert selected.shape == (2, 1081)
    assert np.array_equal(selected[:, :1032], np.load(tmp_path / M.BASELINE_BLOCKS[0])[[3, 1]])


def test_baseline_directory_rejects_wrong_block_width(tmp_path):
    for name, width in zip(M.BASELINE_BLOCKS, (1031, 40, 9)):
        np.save(tmp_path / name, np.zeros((2, width), np.float32))
    with pytest.raises(ValueError, match="expected baseline block widths"):
        M.BaselineFeatures(tmp_path, 2)


def test_logloss_prefers_correct_forecast():
    y = np.array([0, 1, 0, 1])
    assert M.logloss(y, [.1, .9, .2, .8]) < M.logloss(y, [.5, .5, .5, .5])


def test_average_precision_is_distinct_from_trapezoidal_pr_auc():
    from sklearn.metrics import average_precision_score
    y = np.array([0, 1, 0, 1, 0])
    prediction = np.array([.95, .8, .6, .4, .1])
    ap = average_precision_score(y, prediction)
    trapezoid = M.trapezoidal_pr_auc(y, prediction)
    assert not np.isclose(ap, trapezoid)
    assert np.isclose(ap, 0.5)
    assert np.isclose(trapezoid, 1 / 3)


def test_outer_split_requires_stable_gene_isolation():
    pairs = np.array([["a", "b"], ["c", "d"], ["b", "e"]])
    assert M.assert_outer_gene_disjoint(pairs, np.array([0]), np.array([1])) == (2, 2)
    with pytest.raises(ValueError, match="not stable-gene-disjoint"):
        M.assert_outer_gene_disjoint(pairs, np.array([0]), np.array([2]))


def test_manifest_receipt_accepts_explicit_path_and_digest():
    receipt = {"inputs": [{"path": "some/roster/manifest.json", "sha256": "abc"}]}
    assert M._manifest_has_receipt(receipt, "manifest.json", "abc")
    assert not M._manifest_has_receipt(receipt, "manifest.json", "def")
