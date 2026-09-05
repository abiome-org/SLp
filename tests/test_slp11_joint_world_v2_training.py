import importlib.util
from pathlib import Path

import numpy as np
import sys


PATH = Path(__file__).parents[1] / "scripts/prepare_slp11_joint_world_v2_training.py"
SPEC = importlib.util.spec_from_file_location("v2_training", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

MODULE_DIR = Path(__file__).parents[1] / "modules/slp-1-1-joint-world-v2"
sys.path.insert(0, str(MODULE_DIR))
TRAIN_SPEC = importlib.util.spec_from_file_location("joint_v2_train", MODULE_DIR / "train.py")
TRAIN = importlib.util.module_from_spec(TRAIN_SPEC)
TRAIN_SPEC.loader.exec_module(TRAIN)


def test_pair_fold_is_order_invariant_and_seeded():
    left = MODULE.canonical_pair_fold(["ENSG2", "ENSG1"], 731)
    assert left == MODULE.canonical_pair_fold(["ENSG1", "ENSG2"], 731)
    assert left in range(3)
    assert MODULE.canonical_pair_fold(["ENSG2", "ENSG1"], 732) in range(3)


def test_fold_metadata_aligns_only_combination_rows():
    payload = {
        "single_rows": np.asarray([0, 1]),
        "combination_rows": np.asarray([2]),
        "combination_single_rows": np.asarray([[0, 1]]),
        "action_ids": np.asarray([["A", ""], ["B", ""], ["B", "A"]]),
    }
    result = MODULE.add_shared_folds(payload)
    assert result["combination_fold"].shape == (1,)
    assert np.array_equal(result["single_rows"], [0, 1])


def test_manifest_drives_source_discovery_and_normalized_weights():
    sources, weights = TRAIN.resolve_source_configuration({
        "trainingSources": ["a", "b"], "sourceWeights": {"a": 3, "b": 1}})
    assert sources == ("a", "b")
    assert weights == {"a": .75, "b": .25}
