import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PATH = Path(__file__).parents[1] / "scripts/run_slp11_paired_state.py"
SPEC = importlib.util.spec_from_file_location("paired_state_runner_test", PATH)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_inference_device_materialization_does_not_require_outcomes(tmp_path, monkeypatch):
    reference, features, _ = write_inference_fixture(tmp_path)
    prepared = MOD.load_inference_inputs(
        reference, features, np.array(["a1", "a2"]), np.array([0, 1])
    )
    original = torch.as_tensor
    monkeypatch.setattr(MOD.torch, "as_tensor", lambda values, device=None: original(values))
    tensors = MOD.to_device(prepared)
    assert "targets" not in tensors
    assert tensors["basal_features"]["rna"].shape == (1, 2)


def write_inference_fixture(tmp_path):
    feature_ids = np.array(["a1", "a2", "q1", "q2"])
    values = np.array(
        [[1.0, 2.0], [5.0, -2.0], [3.0, 6.0], [-3.0, 10.0]], dtype=np.float32
    )
    features = tmp_path / "features.npz"
    np.savez(features, entity_id=feature_ids, feature_values=values)
    mean = np.array([1.0, 2.0])
    scale = np.array([2.0, 4.0])
    standardized_queries = np.clip((values[2:] - mean) / scale, -2.0, 2.0).astype(
        np.float32
    )
    reference = tmp_path / "reference.npz"
    np.savez(
        reference,
        context_names=np.array(["c0", "c1"]),
        feature_mean=mean,
        feature_scale=scale,
        feature_clip=np.array(2.0),
        rna_query_ids=np.array(["q1", "q2"]),
        protein_query_ids=np.array(["p1", "p2"]),
        rna_query_features=standardized_queries,
        protein_query_features=np.eye(2, dtype=np.float32),
        rna_controls=np.ones((2, 2), dtype=np.float32),
        protein_controls=np.full((2, 2), 2.0, dtype=np.float32),
        rna_amplitude=np.array([0.5, 0.6], dtype=np.float32),
        protein_amplitude=np.array([0.7, 0.8], dtype=np.float32),
        rna_basal_indices=np.array([0]),
        protein_basal_indices=np.array([0, 1]),
        rna_basal_values=np.array([[0.0], [1.0]], dtype=np.float32),
        protein_basal_values=np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
    )
    return reference, features, values


def test_reload_rebuilds_actions_from_raw_features_and_uses_frozen_queries(tmp_path):
    reference, features, values = write_inference_fixture(tmp_path)
    actual = MOD.load_inference_inputs(
        reference, features, ["a2", "a1"], np.array([1, 0])
    )
    expected = np.clip((values[[1, 0]] - [1.0, 2.0]) / [2.0, 4.0], -2.0, 2.0)
    np.testing.assert_array_equal(actual["actions"], expected.astype(np.float32))
    np.testing.assert_array_equal(actual["context_index"], [1, 0])
    assert "targets" not in actual
    assert "scales" not in actual


def test_reload_rejects_frozen_query_features_that_disagree_with_raw_pack(tmp_path):
    reference, features, _ = write_inference_fixture(tmp_path)
    with np.load(reference, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    arrays["rna_query_features"] = arrays["rna_query_features"].copy()
    arrays["rna_query_features"][0, 0] += 0.25
    np.savez(reference, **arrays)
    with pytest.raises(ValueError, match="RNA query features disagree"):
        MOD.load_inference_inputs(reference, features, ["a1"], np.array([0]))


def test_runtime_contract_reserves_a_complete_validation_block():
    projection = MOD.runtime_projection(0.1, 2.0, fitting_rows=64, validation_rows=16)
    assert projection["seconds_per_epoch"] == pytest.approx(0.2)
    assert projection["first_complete_evaluation_projection_seconds"] == pytest.approx(
        3.0
    )
    assert MOD.projected_to_next_evaluation(1, 0.2, 2.0) == pytest.approx(3.0)
    assert MOD.projected_to_next_evaluation(5, 0.2, 2.0) == pytest.approx(2.2)
    assert MOD.projected_to_next_evaluation(6, 0.2, 2.0) == pytest.approx(3.0)
    assert projection["bounded_total_projection_seconds"] < 3600
