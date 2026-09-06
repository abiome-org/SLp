import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "modules/slp-1-1-joint-world-v5"
sys.path.insert(0, str(DIRECTORY))
try:
    spec = importlib.util.spec_from_file_location("joint_world_v5_train_contract", DIRECTORY / "train.py")
    TRAIN = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = TRAIN
    spec.loader.exec_module(TRAIN)
finally:
    sys.path.pop(0)


def test_compose_action_prior_is_zero_for_empty_and_unchanged_for_active():
    per_action = torch.tensor([[[2., 3.], [0., 0.]], [[0., 0.], [0., 0.]],
                               [[2., 3.], [5., 7.]]])
    mask = torch.tensor([[True, False], [False, False], [True, True]])
    template = torch.tensor([1., 2.])
    result = TRAIN.compose_action_prior(per_action, mask, .25, template)
    assert torch.equal(result[0], torch.tensor([2., 3.]))
    assert torch.equal(result[1], torch.zeros(2))
    assert torch.equal(result[2], torch.tensor([6.75, 9.5]))


def valid_data():
    action_features = np.zeros((3, 2, 4), np.float32)
    action_features[0, 0] = [1, 2, 3, 4]
    action_features[1, 0] = [5, 6, 7, 8]
    action_features[2, 0] = action_features[0, 0]
    action_features[2, 1] = action_features[1, 0]
    return dict(targets=np.zeros((3, 5)), basal=np.zeros((3, 5)),
                action_features=action_features,
                action_mask=np.array([[1, 0], [1, 0], [1, 1]], bool),
                single_rows=np.array([0, 1]), combination_rows=np.array([2]),
                combination_single_rows=np.array([[0, 1]]))


def test_combination_contract_accepts_aligned_pair_and_parents():
    TRAIN.validate_combinations(valid_data(), "fixture")


@pytest.mark.parametrize("mutation,match", [
    ("parent_descriptor", "does not match"),
    ("parent_basal", "basal states do not align"),
    ("parent_not_single", "declared single"),
    ("pair_cardinality", "exactly two"),
])
def test_combination_contract_rejects_malformed_parents(mutation, match):
    data = valid_data()
    if mutation == "parent_descriptor": data["action_features"][2, 1, 0] += 1
    if mutation == "parent_basal": data["basal"][1, 0] = 1
    if mutation == "parent_not_single": data["single_rows"] = np.array([0])
    if mutation == "pair_cardinality": data["action_mask"][2, 1] = False
    with pytest.raises(ValueError, match=match):
        TRAIN.validate_combinations(data, "fixture")
