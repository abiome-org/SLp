import importlib.util
import sys
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "modules/slp-1-1-count-panel-data-v1/panel_data.py"
SPEC = importlib.util.spec_from_file_location("panel_data", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_identity_arrays_are_unique_independent_and_readonly():
    original = np.asarray(["Q2", "Q1"])
    result = MOD._readonly_identity(original)
    original[0] = "XX"
    assert result.tolist() == ["Q2", "Q1"]
    assert not result.flags.writeable
    with pytest.raises(ValueError, match="unique"):
        MOD._readonly_identity(np.asarray(["Q1", "Q1"]))


def synthetic_panel(tmp_path):
    counts_path = tmp_path / "counts.uint16"
    counts = np.asarray([[1, 2], [3, 1], [2, 3], [4, 2], [1, 4]], dtype="<u2")
    mmap = np.memmap(counts_path, dtype="<u2", mode="w+", shape=counts.shape)
    mmap[:] = counts
    mmap.flush()
    mmap = np.memmap(counts_path, dtype="<u2", mode="r", shape=counts.shape)
    metadata = {
        "action_ids": np.asarray(["", "", "G1", "G1", "G2"]),
        "population_ids": np.asarray(["control-a", "control-b", "p1", "p2", "p3"]),
        "gem_group": np.asarray([1, 2, 1, 2, 2]),
        "is_control": np.asarray([True, True, False, False, False]),
        "library_size": counts.sum(1).astype(np.int64),
    }
    action_index, context_index, control_rows, target_rows = MOD._validate_cell_metadata(
        metadata, np.asarray(["G1", "G2"]), np.asarray([1, 2])
    )
    return MOD.PanelData(
        source_id="synthetic",
        query_features=np.zeros((2, 3), np.float32),
        gene_action_features=np.asarray([[1, 2, 3], [4, 5, 6]], np.float32),
        basal_rate=np.ones((2, 2), np.float32),
        gene_ids=np.asarray(["G1", "G2"]),
        query_ids=np.asarray(["Q2", "Q1"]),
        context_ids=np.asarray(["source::gem:1", "source::gem:2"]),
        population_targets=np.ones((2, 2), np.float32),
        population_context_weights=np.asarray([[0.5, 0.5], [0, 1]], np.float32),
        fitting_mean_scale=1.0,
        cell_metadata=MappingProxyType(metadata),
        counts=mmap,
        _cell_action_index=action_index,
        _cell_context_index=context_index,
        _control_rows_by_context=control_rows,
        _target_rows_by_gene_population=target_rows,
    )


def test_role_and_axis_contracts_fail_closed():
    metadata = {
        "action_ids": np.asarray(["", "BAD"]),
        "population_ids": np.asarray(["control", "p"]),
        "gem_group": np.asarray([1, 1]),
        "is_control": np.asarray([True, False]),
        "library_size": np.asarray([1, 1]),
    }
    with pytest.raises(ValueError, match="roles"):
        MOD._validate_cell_metadata(metadata, np.asarray(["G1"]), np.asarray([1]))
    metadata["action_ids"] = np.asarray(["", "G1"])
    metadata["gem_group"] = np.asarray([1, 2])
    with pytest.raises(ValueError, match="unknown action or GEM"):
        MOD._validate_cell_metadata(metadata, np.asarray(["G1"]), np.asarray([1]))


def test_basal_and_population_contract_matches_manual_formula():
    control = {
        "raw_count_sum": np.asarray([[4, 6], [8, 2]]),
        "library_count_sum": np.asarray([10, 10]),
        "num_cells": np.asarray([2, 2]),
    }
    basal = MOD._basal_rate(control, 2)
    expected = (10_000 * (control["raw_count_sum"] + 0.5) / 11).astype(np.float32)
    assert np.array_equal(basal, expected)
    moments = {
        "cp10k_sum": np.asarray([[2.0, 4.0], [6.0, 2.0]]),
        "cell_count": np.asarray([2, 2]),
        "gem_cell_count": np.asarray([[1, 1], [0, 2]]),
    }
    target, weights, scale = MOD._population_contract(moments, basal)
    assert np.allclose(target, np.log1p([[1, 2], [3, 1]]))
    assert np.array_equal(weights, np.asarray([[0.5, 0.5], [0, 1]], np.float32))
    assert np.isfinite(scale) and scale > 0


def test_cell_sampling_balances_roles_and_preserves_library(tmp_path):
    panel = synthetic_panel(tmp_path)
    batch = panel.sample_cells(np.random.default_rng(731), n_controls=8, n_targets=8)
    assert batch["actions"].shape == (16, 1, 3)
    assert not batch["action_mask"][:8].any()
    assert batch["action_mask"][8:].all()
    assert np.all(batch["actions"][:8] == 0)
    assert np.array_equal(batch["counts"].sum(1), batch["library"])
    assert batch["observed"].all()
    assert set(batch["context_index"][:8]) == {0, 1}


def test_population_sampling_is_unique_and_feature_replacement_is_explicit(tmp_path):
    panel = synthetic_panel(tmp_path)
    batch = panel.sample_populations(np.random.default_rng(4), n=2)
    assert len(np.unique(batch["gene_index"])) == 2
    assert batch["actions"].shape == (2, 1, 3)
    assert np.allclose(batch["context_weights"].sum(1), 1)
    replaced = panel.replace_features(
        np.ones((2, 5), np.float64), np.ones((2, 5), np.float64)
    )
    assert replaced.query_features.shape == (2, 5)
    assert replaced.gene_action_features.shape == (2, 5)
    assert replaced.query_features.dtype == np.float32
    assert replaced.query_features.flags.writeable is False
    assert panel.query_features.shape == (2, 3)
    with pytest.raises(ValueError, match="gene_action_features"):
        panel.replace_features(np.ones((2, 4)), np.ones((2, 3)))
