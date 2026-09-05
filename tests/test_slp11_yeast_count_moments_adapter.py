from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "aggregate_slp11_yeast_count_moments.py"
SPEC = importlib.util.spec_from_file_location("slp11_yeast_moments_adapter", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
adapter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adapter
SPEC.loader.exec_module(adapter)


def test_batch_shard_keeps_ids_roles_support_and_only_sufficient_statistics(tmp_path):
    # Source rows 0 and 2 are strict queries; row 1 is denominator-only.
    dense = np.asarray(
        [
            [2, 0, 9, 1, 0],
            [3, 1, 9, 0, 0],
            [0, 4, 9, 0, 0],
        ],
        dtype=np.float64,
    )
    output = tmp_path / "B01.npz"
    result = adapter.aggregate_batch(
        sparse.csc_matrix(dense),
        np.asarray([0, 1, 3, 4]),
        np.asarray(["", "SGD:A", "IGNORED", "SGD:B", "SGD:A"]),
        np.asarray(["control", "train", "train", "validation", "train"]),
        np.asarray([True, False, False, False, False]),
        np.asarray([0, -1, 1]),
        np.ones(3, dtype=np.bool_),
        np.asarray(["SGD:Q0", "SGD:Q1"]),
        context="Control",
        batch_id="B01",
        output_path=output,
        block_cells=1,
    )

    with np.load(output) as shard:
        assert shard["context"].item() == "Control"
        assert shard["batch_id"].item() == "B01"
        np.testing.assert_array_equal(
            shard["group_action_id"], ["CONTROL:WT", "SGD:A", "SGD:B"]
        )
        np.testing.assert_array_equal(
            shard["development_role"], ["control", "train", "validation"]
        )
        np.testing.assert_array_equal(shard["num_cells"], [1, 1, 1])
        np.testing.assert_array_equal(shard["total_cells"], [1, 2, 1])
        np.testing.assert_array_equal(shard["zero_library_cells"], [0, 1, 0])
        np.testing.assert_array_equal(shard["mean_observed"], [True, True, True])
        np.testing.assert_array_equal(shard["variance_observed"], [False] * 3)
        expected = np.asarray(
            [
                [np.log1p(4000), 0],
                [0, np.log1p(8000)],
                [np.log1p(10000), 0],
            ]
        )
        np.testing.assert_allclose(shard["sum"], expected)
        np.testing.assert_allclose(shard["sum_squares"], expected**2)
        assert "mean" not in shard.files
        assert "cell_variance" not in shard.files

    assert result["fittingGroups"] == 1
    assert result["validationGroups"] == 1
    assert result["positiveLibraryCells"] == 3


def test_action_cannot_span_development_roles(tmp_path):
    with pytest.raises(ValueError, match="spans development roles"):
        adapter.aggregate_batch(
            sparse.csc_matrix(np.ones((1, 2))),
            np.arange(2),
            np.asarray(["SGD:A", "SGD:A"]),
            np.asarray(["train", "validation"]),
            np.zeros(2, dtype=np.bool_),
            np.asarray([0]),
            np.ones(1, dtype=np.bool_),
            np.asarray(["SGD:Q"]),
            context="NaCl",
            batch_id="B02",
            output_path=tmp_path / "unused.npz",
        )
