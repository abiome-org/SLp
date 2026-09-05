import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/score_slp11_k562_prior_rpe1_unfitted_context.py"
SPEC = importlib.util.spec_from_file_location("count_prior_rpe_scoring_test", PATH)
RUN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUN
SPEC.loader.exec_module(RUN)


def test_common_query_mapping_preserves_rpe_order(monkeypatch):
    monkeypatch.setattr(RUN, "COMMON_QUERIES", 3)
    common, k_rows, rpe_rows = RUN.common_query_indices(
        np.asarray(["q3", "q1", "q4", "q2"]),
        np.asarray(["x", "q2", "q1", "q3"]),
    )
    np.testing.assert_array_equal(common, ["q2", "q1", "q3"])
    np.testing.assert_array_equal(k_rows, [3, 1, 0])
    np.testing.assert_array_equal(rpe_rows, [1, 2, 3])


def test_common_query_mapping_rejects_duplicates(monkeypatch):
    monkeypatch.setattr(RUN, "COMMON_QUERIES", 1)
    with pytest.raises(RUN.TransferScoreError, match="unique"):
        RUN.common_query_indices(np.asarray(["q", "q"]), np.asarray(["q"]))


def test_per_gene_mse_is_equal_query_average():
    truth = np.asarray([[1.0, 2.0], [3.0, 5.0]])
    prediction = np.asarray([[0.0, 4.0], [4.0, 1.0]])
    np.testing.assert_array_equal(RUN.per_gene_mse(truth, prediction), [2.5, 8.5])


def test_protocol_freezes_source_comparators_before_fitting_moments():
    value = RUN.protocol()
    assert value["preOutcomeComparatorFreeze"]["RPEFittingMomentsRead"] is False
    assert value["scoring"]["geneStrata"] == {
        "all": 1666, "k562SeenAction": 1443, "rpe1OnlyAction": 223
    }
    assert "unequal" in value["supervisedRPEReferences"]
