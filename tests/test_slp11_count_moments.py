import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

PATH = (
    Path(__file__).resolve().parents[1]
    / "modules/slp-1-1-count-moments-v1/count_moments.py"
)
SPEC = importlib.util.spec_from_file_location("count_moments", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CountMoments = MODULE.CountMoments


def test_dense_oracle_duplicate_queries_and_unmapped_denominator():
    raw = np.array([[2, 3, 5, 7], [0, 0, 4, 0], [0, 0, 0, 0], [4, 0, 0, 0]])
    # First two source rows resolve to the same gene; third stays denominator-only.
    model = CountMoments(np.array([0, 0, -1, 1]), np.ones(4, bool), 2, 3)
    valid = model.update(sparse.csr_matrix(raw), np.array([0, 0, 0, 1]))
    expected = np.log1p(10000 * np.array([[5, 7], [0, 0]]) / np.array([[17], [4]]))
    result = model.summary()
    np.testing.assert_array_equal(valid, [True, True, False, True])
    np.testing.assert_allclose(result["mean"][0], expected.mean(axis=0))
    np.testing.assert_allclose(result["cell_variance"][0], expected.var(axis=0, ddof=1))
    np.testing.assert_array_equal(result["num_cells"], [2, 1, 0])
    np.testing.assert_array_equal(result["zero_library_cells"], [1, 0, 0])
    assert np.isnan(result["mean"][2]).all()
    assert np.isnan(result["cell_variance"][1:]).all()


def test_chunking_order_and_sparse_duplicate_invariance():
    rng = np.random.default_rng(731)
    raw = rng.poisson(1, (50, 5))
    groups = rng.integers(0, 3, 50)
    first = CountMoments(np.arange(5), np.ones(5, bool), 5, 3)
    second = CountMoments(np.arange(5), np.ones(5, bool), 5, 3)
    first.update(sparse.csr_matrix(raw), groups)
    order = rng.permutation(50)
    for indices in np.array_split(order, 7):
        second.update(sparse.csr_matrix(raw[indices]), groups[indices])
    for key, value in first.summary().items():
        np.testing.assert_allclose(value, second.summary()[key], rtol=1e-12, atol=1e-12)
    duplicate = sparse.csr_matrix(([2, 3], [0, 0], [0, 2]), shape=(1, 5))
    third = CountMoments(np.arange(5), np.ones(5, bool), 5, 1)
    third.update(duplicate, np.array([0]))
    assert third.summary()["mean"][0, 0] == np.log1p(10000)


@pytest.mark.parametrize("bad", [-1, 0.5, np.nan, np.inf])
def test_rejects_noncounts_without_updating(bad):
    model = CountMoments(np.array([0]), np.ones(1, bool), 1, 1)
    with pytest.raises(ValueError, match="counts"):
        model.update(sparse.csr_matrix([[bad]]), np.array([0]))
    assert model.total_cells.sum() == 0


def test_query_denominator_mismatch_rejected():
    with pytest.raises(ValueError, match="denominator"):
        CountMoments(np.array([0, -1]), np.array([False, True]), 1, 1)
