from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import build_slp11_norman_static_features as FEATURES


def test_copy_frozen_rows_preserves_old_values_exactly() -> None:
    old = np.arange(6, dtype=np.float32).reshape(3, 2)
    extended, rows = FEATURES.copy_frozen_rows(
        old, ("B", "D", "F"), ("A", "B", "C", "D", "E", "F")
    )
    assert np.array_equal(extended[[1, 3, 5]], old)
    assert np.array_equal(extended[[0, 2, 4]], np.zeros((3, 2), dtype=np.float32))
    assert rows == {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}


def test_fixed_term_matrix_omits_terms_outside_frozen_basis() -> None:
    matrix, omitted = FEATURES.fixed_term_matrix(
        [frozenset({"GO:1", "GO:3"}), frozenset(), frozenset({"GO:2"})],
        ("GO:1", "GO:2"),
    )
    assert matrix.shape == (3, 2)
    assert np.array_equal(matrix.toarray(), [[1, 0], [0, 0], [0, 1]])
    assert omitted == 1
