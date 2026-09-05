from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    path = ROOT / "scripts/build_slp11_action_aligned_basal.py"
    spec = importlib.util.spec_from_file_location("action_aligned_basal_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_alignment_keeps_missing_explicit() -> None:
    module = load_builder()
    values = np.asarray([[2.0, 5.0], [3.0, 7.0]])
    observed = np.asarray([[True, False], [True, True]])
    aligned, mask = module.align_action_values(
        np.asarray(["ENSG_A", "ENSG_MISSING", "ENSG_B"]),
        np.asarray(["ENSG_B", "ENSG_A"]), values, observed,
    )
    np.testing.assert_array_equal(mask, [[False, False, True], [True, False, True]])
    np.testing.assert_array_equal(aligned, [[0.0, 0.0, 2.0], [7.0, 0.0, 3.0]])


def test_alignment_rejects_duplicate_stable_ids() -> None:
    module = load_builder()
    with pytest.raises(ValueError, match="alignment"):
        module.align_action_values(
            np.asarray(["g", "g"]), np.asarray(["q"]),
            np.ones((1, 1)), np.ones((1, 1), dtype=bool),
        )


def test_weighted_normalizer_uses_only_observed_positive_weight() -> None:
    module = load_builder()
    mean, scale, count = module.weighted_normalizer(
        np.asarray([0.0, 2.0, 100.0, 4.0]),
        np.asarray([True, True, False, True]),
        np.asarray([0, 0, 0, 1]),
        np.asarray([1.0, 3.0, 9.0, 1.0]),
        2,
    )
    np.testing.assert_allclose(mean, [1.5, 4.0])
    np.testing.assert_allclose(scale, [np.sqrt(0.75), 1.0])
    np.testing.assert_array_equal(count, [2, 1])
