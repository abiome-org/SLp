import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).parents[1] / (
    "scripts/run_slp11_four_context_mean_seed_extension.py"
)
SPEC = importlib.util.spec_from_file_location("slp11_mean_seed_extension_test", PATH)
EXTENSION = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = EXTENSION
SPEC.loader.exec_module(EXTENSION)


def test_arithmetic_ensemble_is_equal_weight_with_float64_accumulation():
    members = [
        np.array([[1.0, 2.0]], dtype=np.float32),
        np.array([[2.0, 4.0]], dtype=np.float32),
        np.array([[6.0, 9.0]], dtype=np.float32),
    ]
    result = EXTENSION.arithmetic_ensemble(members)
    np.testing.assert_allclose(result, [[3.0, 5.0]])
    assert result.dtype == np.float32


def test_arithmetic_ensemble_requires_exactly_three_aligned_finite_members():
    value = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        EXTENSION.arithmetic_ensemble([value, value])
    with pytest.raises(ValueError):
        EXTENSION.arithmetic_ensemble([value, value, np.zeros((3, 2))])
    invalid = value.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError):
        EXTENSION.arithmetic_ensemble([value, value, invalid])


def test_only_seed_is_varied_in_new_arm_contract():
    assert EXTENSION.NEW_SEEDS == (732, 733)
    assert EXTENSION.ARMS == {"source3": 3, "source4": 4}
    pair = EXTENSION.load_pair_module()
    assert pair.STEPS == 12_000
    assert pair.BATCH_SIZE == 64
    assert pair.SEED == 731
