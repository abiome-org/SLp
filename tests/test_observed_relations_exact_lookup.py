import numpy as np
import pytest

from modules.training.world import observed_relations


def pack():
    return {
        "pairs": np.array([[0, 2], [1, 3], [2, 4]], dtype=np.int32),
        "relations": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float16),
    }


def test_observed_relations_preserves_valid_exact_lookup():
    actual = observed_relations(pack(), np.array([[2, 4], [0, 2]], dtype=np.int64), 5)
    assert actual.dtype == np.float32
    assert np.array_equal(actual, np.array([[5, 6], [1, 2]], dtype=np.float32))


@pytest.mark.parametrize("pairs", [
    np.array([[0, 3]], dtype=np.int32),  # absent key formerly took its successor
    np.array([[2, 0]], dtype=np.int32),  # reversed orientation is not an exact key
    np.array([[4, 3]], dtype=np.int32),  # valid indices, key beyond the source table
    np.array([[0, 5]], dtype=np.int32),  # out of bounds
    np.array([[-1, 2]], dtype=np.int32),
])
def test_observed_relations_rejects_nonmatching_or_invalid_pairs(pairs):
    with pytest.raises(ValueError):
        observed_relations(pack(), pairs, 5)
