import importlib.util
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "modules/slp-1-1-count-response-query-features-v1/response_query_features.py"
SPEC = importlib.util.spec_from_file_location("slp11_count_response_query_features_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@dataclass(frozen=True)
class FakePanel:
    source_id: str
    context_ids: np.ndarray
    query_ids: np.ndarray
    query_features: np.ndarray
    gene_action_features: np.ndarray
    sentinel: object

    def replace_features(self, query_features, gene_action_features):
        return replace(
            self,
            query_features=np.asarray(query_features, dtype=np.float32),
            gene_action_features=np.asarray(gene_action_features, dtype=np.float32),
        )


def pack(ids, raw):
    rms = np.sqrt(np.mean(np.square(raw), axis=0))
    scale = np.where(rms > 0, rms, 1.0)
    return {
        "schema": np.asarray("slp.human-count-response-query33/v1"),
        "source_id": np.asarray("k562"),
        "context_id": np.asarray("replogle-2022-k562-essential-day-6"),
        "query_ids": ids,
        "entity_taxon": np.full(len(ids), 9606, dtype=np.int64),
        "raw_response_query33": raw,
        "response_query33_rms": scale,
        "normalized_response_query33": raw / scale,
        "rank": np.asarray(32, dtype=np.int64),
        "alpha": np.asarray(1000.0),
        "rank_model_sha256": np.asarray("a" * 64),
        "query_ids_lf_sha256": np.asarray(MODULE.lf_roster_sha256(ids)),
        "fitting_outcome_derived": np.asarray(True),
        "development_outcomes_accessed": np.asarray(False),
        "test_outcomes_accessed": np.asarray(False),
    }


def test_response33_order_and_rms_without_centering():
    loading = np.arange(32 * 5, dtype=np.float64).reshape(32, 5) - 17
    intercept = np.arange(5, dtype=np.float64) + 100
    raw, _rms, normalized = MODULE.response_query33(loading, intercept)
    np.testing.assert_array_equal(raw[:, :32], loading.T)
    np.testing.assert_array_equal(raw[:, 32], intercept)
    np.testing.assert_allclose(np.sqrt(np.mean(np.square(normalized), axis=0)), 1.0)
    assert np.any(np.abs(normalized.mean(0)) > 0.1)


def test_matched_modes_preserve_input_and_append_action_zeros():
    rng = np.random.default_rng(5)
    ids = np.asarray(["ENSG1", "ENSG2", "ENSG3"])
    panel = FakePanel(
        "k562",
        np.asarray(["replogle-2022-k562-essential-day-6::gem-group:001"]),
        ids,
        rng.normal(size=(3, 577)).astype(np.float32),
        rng.normal(size=(2, 577)).astype(np.float32),
        object(),
    )
    query_before = panel.query_features.copy()
    action_before = panel.gene_action_features.copy()
    raw = rng.normal(size=(3, 33))
    response_pack = pack(ids, raw)
    zero = MODULE.augment_panel(panel, response_pack, "static-zero33")
    response = MODULE.augment_panel(panel, response_pack, "response33")
    assert zero.query_features.shape == response.query_features.shape == (3, 610)
    assert zero.gene_action_features.shape == response.gene_action_features.shape == (2, 610)
    np.testing.assert_array_equal(zero.query_features[:, :577], query_before)
    np.testing.assert_array_equal(response.query_features[:, :577], query_before)
    np.testing.assert_array_equal(zero.query_features[:, 577:], 0)
    np.testing.assert_allclose(response.query_features[:, 577:], response_pack["normalized_response_query33"], rtol=2e-7, atol=2e-7)
    np.testing.assert_array_equal(zero.gene_action_features[:, 577:], 0)
    np.testing.assert_array_equal(response.gene_action_features[:, 577:], 0)
    np.testing.assert_array_equal(panel.query_features, query_before)
    np.testing.assert_array_equal(panel.gene_action_features, action_before)
    assert zero.sentinel is panel.sentinel and response.sentinel is panel.sentinel


def test_exact_native_query_order_is_required():
    ids = np.asarray(["ENSG1", "ENSG2"])
    raw = np.ones((2, 33))
    panel = FakePanel(
        "k562",
        np.asarray(["replogle-2022-k562-essential-day-6::gem-group:001"]),
        ids[::-1],
        np.zeros((2, 577), np.float32),
        np.zeros((1, 577), np.float32),
        None,
    )
    with pytest.raises(ValueError, match="native panel order"):
        MODULE.augment_panel(panel, pack(ids, raw), "response33")


def test_wrong_source_context_or_alpha_cannot_cross_native_panels():
    ids = np.asarray(["ENSG1", "ENSG2"])
    response_pack = pack(ids, np.ones((2, 33)))
    panel = FakePanel(
        "rpe1",
        np.asarray(["replogle-2022-rpe1-essential-day-7::gem-group:001"]),
        ids,
        np.zeros((2, 577), np.float32),
        np.zeros((1, 577), np.float32),
        None,
    )
    with pytest.raises(ValueError, match="panel source"):
        MODULE.augment_panel(panel, response_pack, "response33")
    response_pack["alpha"] = np.asarray(999.0)
    with pytest.raises(ValueError, match="rank, alpha"):
        MODULE.validate_pack(response_pack)
