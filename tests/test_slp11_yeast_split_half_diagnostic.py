from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_slp11_yeast_split_half_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("slp11_yeast_split_half", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diagnostic
SPEC.loader.exec_module(diagnostic)


def test_split_is_deterministic_preserves_batch_strata_and_excludes_validation():
    barcodes = np.asarray([f"cell-{index}" for index in range(10)])
    batch = np.asarray(["B1"] * 5 + ["B2"] * 5)
    action = np.asarray(["A", "A", "A", "", "", "A", "B", "B", "B", "B"])
    roles = np.asarray(
        [
            "train",
            "train",
            "train",
            "control",
            "control",
            "validation",
            "train",
            "train",
            "train",
            "train",
        ]
    )
    controls = roles == "control"
    first, stats = diagnostic.stable_half_split(
        barcodes, batch, action, roles, controls
    )
    second, _ = diagnostic.stable_half_split(barcodes, batch, action, roles, controls)
    np.testing.assert_array_equal(first, second)
    assert first[5] == -2
    for indices in ([0, 1, 2], [3, 4], [6, 7, 8, 9]):
        assert set(first[list(indices)]) == {0, 1}
    assert stats["pairedCellsIncluded"] == 9
    assert stats["validationCellsExcluded"] == 1
    assert stats["singletonCellsExcluded"] == 0


def test_singleton_fitting_population_is_explicitly_excluded():
    half, stats = diagnostic.stable_half_split(
        np.asarray(["a", "b", "c"]),
        np.asarray(["B1", "B1", "B1"]),
        np.asarray(["A", "B", "B"]),
        np.asarray(["train", "train", "train"]),
        np.zeros(3, dtype=np.bool_),
    )
    assert half[0] == -1
    assert set(half[1:]) == {0, 1}
    assert stats["singletonCellsExcluded"] == 1


def test_selected_csc_reader_matches_dense_and_never_reads_other_columns():
    dense = np.arange(30, dtype=np.float64).reshape(5, 6)
    dense[(dense % 3) != 0] = 0
    matrix = sparse.csc_matrix(dense)
    result = diagnostic.selected_columns_as_csr(
        matrix.indptr,
        matrix.indices,
        matrix.data,
        np.asarray([1, 4]),
        matrix.shape[0],
    )
    np.testing.assert_array_equal(result.toarray(), dense[:, [1, 4]].T)
    with pytest.raises(ValueError, match="sorted"):
        diagnostic.selected_columns_as_csr(
            matrix.indptr,
            matrix.indices,
            matrix.data,
            np.asarray([4, 1]),
            matrix.shape[0],
        )


def test_query_centered_metric_removes_common_query_profile():
    a = np.asarray([[1.0, 3.0, 2.0], [2.0, 1.0, 5.0]])
    offset = np.asarray([9.0, -2.0, 4.0])
    b = a + offset
    centered_a, centered_b = diagnostic._center_queries(a, b)
    np.testing.assert_allclose(centered_a, centered_b)
    summary = diagnostic._metric_summary(centered_a, centered_b)
    assert summary["equalGeneMeanMse"] == pytest.approx(0)
    assert summary["meanGeneProfilePearson"] == pytest.approx(1)
