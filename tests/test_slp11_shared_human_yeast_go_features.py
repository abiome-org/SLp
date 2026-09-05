from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_slp11_shared_human_yeast_go_features as shared


class SharedGoTests(unittest.TestCase):
    def test_species_weights_and_identical_rows(self) -> None:
        yeast = sparse.csr_matrix(
            np.asarray([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
        )
        human = sparse.csr_matrix(
            np.asarray([[1, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=np.float32)
        )
        y_values, h_values, model = shared.fit_shared_svd(
            yeast, human, components=2, seed=731
        )
        self.assertTrue(np.array_equal(y_values[0], h_values[0]))
        self.assertAlmostEqual(2 * (1 / np.sqrt(2)) ** 2, 1.0)
        self.assertAlmostEqual(3 * (1 / np.sqrt(3)) ** 2, 1.0)
        self.assertEqual(model.components_.shape, (2, 3))
        audit = shared.identical_cross_species_audit(
            yeast, human, y_values, h_values
        )
        self.assertTrue(audit["exactVectorEquality"])
        self.assertGreaterEqual(audit["sharedExactAnnotationPatterns"], 1)

    def test_binary_matrix_uses_exact_ids_and_sorted_terms(self) -> None:
        matrix = shared.binary_matrix(
            ("SGD:S000000001", "SGD:S000000002"),
            {"SGD:S000000001": frozenset({"GO:0000002", "GO:0000001"})},
            ("GO:0000001", "GO:0000002"),
        )
        np.testing.assert_array_equal(matrix.toarray(), [[1, 1], [0, 0]])


if __name__ == "__main__":
    unittest.main()
