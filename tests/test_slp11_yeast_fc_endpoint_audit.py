import importlib.util
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "scripts/audit_slp11_yeast_fc_endpoint.py"
SPEC = importlib.util.spec_from_file_location("yeast_fc_endpoint_audit", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_context_statistics_use_only_supplied_rows_and_observed_values() -> None:
    targets = np.asarray([[1, 30, 999], [2, -25, 999], [999, 999, 999]], dtype=np.float32)
    observed = np.asarray([[1, 1, 0], [1, 1, 0], [1, 1, 1]], dtype=bool)
    result = MOD.context_statistics(targets, observed, np.asarray([0, 1]), np.asarray([10, 20, 1]))
    assert result["records"] == 2 and result["observed_values"] == 4
    assert result["value_maximum"] == 30 and result["value_minimum"] == -25
    assert result["absolute_threshold_fractions"]["gt_20"] == 0.5


def test_correlation_is_exact_for_affine_inputs() -> None:
    x = np.arange(5, dtype=np.float64)
    assert np.isclose(MOD.correlation(x, 3 * x + 2), 1)
    assert np.isclose(MOD.correlation(x, -x), -1)
