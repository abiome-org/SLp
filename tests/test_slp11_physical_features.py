"""Physical feature aggregation does not count duplicated evidence twice."""
import importlib.util
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1]/"modules/slp-1-1-world-transition-v1/physical_features.py"
SPEC = importlib.util.spec_from_file_location("physical_features_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_symmetry_duplicate_max_and_isolated_node():
    x = np.array([[1., 0.], [0., 2.], [3., 4.], [8., 9.]], dtype=np.float32)
    edges = [(0, 1, .8), (1, 0, .7), (0, 2, 1.), (2, 2, 1.)]
    result, report = MODULE.neighborhood_features(x, edges)
    np.testing.assert_array_equal(result[:, :2], x)
    np.testing.assert_allclose(result[0, 2:4], [.0/1.8+3/1.8, (1.6+4)/1.8])
    np.testing.assert_array_equal(result[3, 2:], 0)
    assert report["edges"] == 2
    np.testing.assert_array_equal(result, MODULE.neighborhood_features(x, reversed(edges))[0])


def test_entity_relabeling_is_equivariant():
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    order = np.array([2, 0, 3, 1])
    inverse = np.argsort(order)
    edges = [(0, 1, .8), (1, 2, .9)]
    original = MODULE.neighborhood_features(x, edges)[0]
    changed = MODULE.neighborhood_features(x[order], [(inverse[i], inverse[j], w) for i,j,w in edges])[0]
    np.testing.assert_allclose(changed, original[order])
