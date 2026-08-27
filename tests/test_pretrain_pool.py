"""Leakage guard for relation pretraining data."""

import unittest

import numpy as np

from src.training.sl_predict import relation_pretrain_pool


class RelationPretrainPoolTest(unittest.TestCase):
    def test_excludes_every_benchmark_pair_regardless_of_orientation(self):
        pool = relation_pretrain_pool(
            ppi_pairs=np.array([[0, 1], [2, 3], [4, 5]], dtype="int32"),
            random_pairs=np.array([[1, 0], [3, 2], [6, 7]], dtype="int32"),
            benchmark_pairs=np.array([[1, 0], [3, 2]], dtype="int32"),
        )
        self.assertEqual({tuple(pair) for pair in pool}, {(4, 5), (6, 7)})


if __name__ == "__main__":
    unittest.main()
