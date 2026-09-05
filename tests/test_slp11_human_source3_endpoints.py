from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

PATH = Path(__file__).parents[1] / "scripts/audit_slp11_human_source3_endpoints.py"
SPEC = importlib.util.spec_from_file_location("source3_endpoint_audit", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_centered_response_norm_removes_query_landscape_and_row_offset():
    landscape = np.asarray([10.0, -3.0, 7.0, 2.0])
    signal = np.asarray([[1.0, -1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]])
    targets = landscape[None] + np.asarray([40.0, -8.0])[:, None] + signal
    norms = MODULE.centered_response_norm(targets, np.ones_like(targets, bool))
    np.testing.assert_allclose(norms, np.sqrt(.5), atol=1e-12)


def test_constant_correlations_are_undefined_and_rank_ties_are_average():
    assert MODULE.stable_pearson(np.ones(5), np.arange(5)) is None
    assert MODULE.stable_spearman(np.ones(5), np.arange(5)) is None
    np.testing.assert_array_equal(MODULE.average_ranks(np.asarray([2, 1, 2, 4])), [1.5, 0, 1.5, 3])


def test_missing_query_mask_fails_closed():
    targets = np.ones((3, 4))
    observed = np.ones((3, 4), bool)
    observed[:, 2] = False
    try:
        MODULE.centered_response_norm(targets, observed)
    except ValueError as error:
        assert "unsupported query" in str(error)
    else:
        raise AssertionError("unsupported query did not fail closed")
