import importlib.util
from pathlib import Path

import numpy as np

PATH = Path(__file__).parents[1] / "scripts/audit_slp11_frangieh_guide_reproducibility.py"
SPEC = importlib.util.spec_from_file_location("frangieh_guides", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_sorted_guides_alternate_with_equal_guide_weight():
    genes, a, b, minimum = MOD.alternating_guide_sides(
        np.array(["A", "A", "A", "B"]),
        np.array(["g3", "g1", "g2", "g1"]),
        np.array([100, 2, 5, 4]),
        np.array([[30.0], [10.0], [20.0], [99.0]]),
    )
    np.testing.assert_array_equal(genes, ["A"])
    np.testing.assert_array_equal(a, [[20.0]])
    np.testing.assert_array_equal(b, [[20.0]])
    np.testing.assert_array_equal(minimum, [5])


def test_different_gene_shuffle_is_deterministic_derangement():
    genes = np.array(["A", "B", "C", "D"])
    first = MOD.deterministic_different_gene_order(genes)
    second = MOD.deterministic_different_gene_order(genes)
    np.testing.assert_array_equal(first, second)
    assert not np.any(genes == genes[first])
