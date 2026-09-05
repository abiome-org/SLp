import importlib.util
import sys
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "scripts/prepare_slp11_joint_world_omf_training.py"
SPEC = importlib.util.spec_from_file_location("slp11_joint_world_omf_training", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_fold_zero_outcomes_are_physically_removed_and_routes_remapped():
    source = Path(__file__).resolve().parents[1] / "data/derived/slp11-joint-world-populations-string-v1/norman.npz"
    with np.load(source, allow_pickle=False) as archive:
        original = {name: np.asarray(archive[name]) for name in archive.files}
    result = MODULE.filter_norman(original, 0)
    assert result["targets"].shape == (107, 7226)
    assert result["action_features"].shape == (107, 2, 642)
    assert result["single_rows"].tolist() == list(range(71))
    assert result["combination_rows"].tolist() == list(range(71, 107))
    assert len(result["combination_single_rows"]) == 36
    assert set(result["combination_fold"].tolist()) == {1, 2}
    assert np.all(result["combination_single_rows"] < 71)
    assert len(result["action_offsets"]) == 108
    assert result["action_offsets"][-1] == len(result["action_ids"])
    np.testing.assert_array_equal(result["query_ids"], original["query_ids"])
    np.testing.assert_array_equal(result["feature_mean"], original["feature_mean"])
