"""A missing assay is not a measured zero or a validation-selected feature."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1]/"modules/slp-1-1-world-transition-v1/complete_panel.py"
SPEC = importlib.util.spec_from_file_location("complete_panel_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_selection_uses_training_availability_and_aligns_controls():
    data = {"split_train":np.array([0, 1]), "split_validation":np.array([2]),
            "split_test":np.array([], dtype=int), "query_ids":np.array(["A", "B", "C"]),
            "observed":np.array([[1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=bool),
            "targets":np.arange(9).reshape(3, 3), "control_targets":np.arange(6).reshape(2, 3),
            "action_ids":np.array(["X", "Y", "Z"])}
    result, indices = MODULE.select_complete_panel(data)
    np.testing.assert_array_equal(indices, [0, 2])
    np.testing.assert_array_equal(result["control_targets"], [[0, 2], [3, 5]])
    assert not result["observed"][2, 0]
    np.testing.assert_array_equal(result["action_ids"], data["action_ids"])
    data["targets"] = np.full((3, 3), -1e8)
    np.testing.assert_array_equal(MODULE.select_complete_panel(data)[1], indices)


def test_refuses_test_snapshot():
    with pytest.raises(ValueError, match="development"):
        MODULE.select_complete_panel({"split_test":np.array([0])})
