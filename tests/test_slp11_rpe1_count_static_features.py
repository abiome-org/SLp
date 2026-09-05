import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/build_slp11_rpe1_count_static_features.py"
SPEC = importlib.util.spec_from_file_location("rpe1_count_static_test", PATH)
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def test_action_roster_excludes_controls_and_unresolved_and_preserves_global_role():
    routing = {
        "action_ids": np.asarray(["", "nan", "ENSG00000000001", "ENSG00000000002", "ENSG00000000001"]),
        "intervention_role": np.asarray(["control", "unresolved-excluded", "train", "validation", "train"]),
        "is_control": np.asarray([True, False, False, False, False]),
        "unresolved_action": np.asarray([False, True, False, False, False]),
    }
    actions, roles = BUILD.action_rosters(routing)
    np.testing.assert_array_equal(actions, ["ENSG00000000001", "ENSG00000000002"])
    assert roles == {"ENSG00000000001": "train", "ENSG00000000002": "validation"}


def test_float64_normalizer_reproduces_exact_transform_without_rounded_stats():
    values = np.asarray([[0, 1, 2], [3, 2, 1], [6, 3, 0]], dtype=np.float32)
    mean, sd, scale = BUILD.normalizer(values, np.asarray([0, 2]))
    assert mean.dtype == sd.dtype == scale.dtype == np.float64
    first = ((values.astype(np.float64) - mean) / scale).astype(np.float32)
    second = ((values.astype(np.float64) - mean) / scale).astype(np.float32)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(mean, [3, 2, 1])


def test_near_constant_normalizer_column_uses_unit_scale():
    values = np.asarray([[1, 2], [1, 4], [1, 6]], dtype=np.float32)
    _, sd, scale = BUILD.normalizer(values, np.arange(3))
    assert sd[0] == 0
    assert scale[0] == 1
    assert scale[1] == sd[1]
