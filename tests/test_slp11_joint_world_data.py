from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "prepare_slp11_joint_world_data.py"


def _load():
    spec = importlib.util.spec_from_file_location("prepare_slp11_joint_world_data", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREPARE = _load()


def test_real_fitting_sources_form_native_population_contracts() -> None:
    k562, k_features = PREPARE.load_crispri_source("k562")
    rpe1, r_features = PREPARE.load_crispri_source("rpe1")
    norman, n_features = PREPARE.load_norman_source()

    assert k562["target_cp10k_mean"].shape == (1443, 8563)
    assert rpe1["target_cp10k_mean"].shape == (1666, 8749)
    assert k562["action_matched_basal_ln1p_mean"].shape == (1443, 8563)
    assert rpe1["action_matched_basal_ln1p_mean"].shape == (1666, 8749)
    assert norman["target_control_z_mean"].shape == (130, 7226)
    assert norman["control_target_control_z"].shape == (20, 7226)
    assert norman["combination_single_rows"].shape == (59, 2)
    assert k562["action_features"].shape == (1443, 2, 577)
    assert rpe1["action_features"].shape == (1666, 2, 577)
    assert norman["action_features"].shape == (130, 2, 577)
    assert k562["targets"].shape == k562["basal"].shape == k562["observed"].shape
    assert rpe1["targets"].shape == rpe1["basal"].shape == rpe1["observed"].shape
    assert norman["targets"].shape == norman["basal"].shape == norman["observed"].shape
    assert np.bincount(norman["combination_fold"], minlength=3).tolist() == [23, 19, 17]
    assert not bool(k562["uncertainty_available"])
    assert not bool(rpe1["uncertainty_available"])
    assert not bool(norman["uncertainty_available"])
    roster, mean, scale = PREPARE.global_feature_normalization(
        {"k562": k_features, "rpe1": r_features, "norman": n_features}
    )
    assert len(roster) == len(k_features) + len(r_features) + len(n_features)
    assert mean.shape == scale.shape == (577,)
    assert np.all(scale > 0)


def test_normalization_source_qualifies_feature_revisions() -> None:
    left = {"A": np.asarray([1.0, 2.0]), "B": np.asarray([3.0, 4.0])}
    right = {"A": np.asarray([1.0, 2.0]), "C": np.asarray([5.0, 6.0])}
    roster, mean, scale = PREPARE.global_feature_normalization({"left": left, "right": right})
    assert roster.tolist() == ["left|A", "left|B", "right|A", "right|C"]
    np.testing.assert_allclose(mean, [2.5, 3.5])
    assert np.all(scale > 0)


def test_deterministic_npz_bytes() -> None:
    arrays = {"z": np.arange(4, dtype=np.int64), "a": np.asarray("value")}
    assert PREPARE.deterministic_npz(arrays) == PREPARE.deterministic_npz(arrays)
