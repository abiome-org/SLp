import importlib.util
from pathlib import Path

import numpy as np

PATH = Path(__file__).parents[1] / "scripts/audit_slp11_frangieh_guide_cell_weighting.py"
SPEC = importlib.util.spec_from_file_location("frangieh_guide_weighting", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_cell_weighting_preserves_frozen_alternating_sides():
    genes, side_a, side_b = MOD.cell_weighted_guide_sides(
        np.array(["A", "A", "A", "B"]),
        np.array(["g3", "g1", "g2", "g1"]),
        np.array([9, 1, 3, 1]),
        np.array([[30.0], [10.0], [20.0], [99.0]]),
    )
    np.testing.assert_array_equal(genes, ["A"])
    np.testing.assert_array_equal(side_a, [[28.0]])
    np.testing.assert_array_equal(side_b, [[20.0]])
