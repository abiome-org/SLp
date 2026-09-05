"""Attach fitting-derived source-native response descriptors to count panels."""

from __future__ import annotations

import hashlib

import numpy as np

MODES = ("static-zero33", "response33")
STATIC_WIDTH = 577
RESPONSE_WIDTH = 33
OUTPUT_WIDTH = STATIC_WIDTH + RESPONSE_WIDTH
EXPECTED_CONTEXT = {
    "k562": "replogle-2022-k562-essential-day-6",
    "rpe1": "replogle-2022-rpe1-essential-day-7",
}


def lf_roster_sha256(identifiers: np.ndarray) -> str:
    values = np.asarray(identifiers).astype(str)
    if values.ndim != 1 or len(set(values.tolist())) != len(values):
        raise ValueError("identifiers must be a unique vector")
    return hashlib.sha256(("\n".join(values.tolist()) + "\n").encode("ascii")).hexdigest()


def response_query33(query_loading: np.ndarray, intercept: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw, RMS and RMS-normalized [Q,33] response descriptors."""
    loading = np.asarray(query_loading, dtype=np.float64)
    mean = np.asarray(intercept, dtype=np.float64)
    if loading.ndim != 2 or loading.shape[0] != 32 or mean.shape != (loading.shape[1],):
        raise ValueError("rank-32 loading [32,Q] and intercept [Q] required")
    if not np.isfinite(loading).all() or not np.isfinite(mean).all():
        raise ValueError("response model parameters must be finite")
    raw = np.concatenate((loading.T, mean[:, None]), axis=1)
    rms = np.sqrt(np.mean(np.square(raw), axis=0, dtype=np.float64))
    scale = np.where(rms > 0, rms, 1.0)
    normalized = raw / scale
    return raw, scale, normalized


def validate_pack(pack: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    required = {
        "schema", "source_id", "context_id", "query_ids", "entity_taxon",
        "raw_response_query33", "response_query33_rms", "normalized_response_query33",
        "rank", "alpha", "rank_model_sha256", "query_ids_lf_sha256",
        "fitting_outcome_derived", "development_outcomes_accessed", "test_outcomes_accessed",
    }
    missing = required.difference(pack)
    if missing:
        raise ValueError(f"response-query pack missing keys: {sorted(missing)}")
    if str(np.asarray(pack["schema"]).item()) != "slp.human-count-response-query33/v1":
        raise ValueError("unsupported response-query pack schema")
    source_id = str(np.asarray(pack["source_id"]).item())
    context_id = str(np.asarray(pack["context_id"]).item())
    if source_id not in EXPECTED_CONTEXT or context_id != EXPECTED_CONTEXT[source_id]:
        raise ValueError("source/context identity is not an expected native panel")
    query = np.asarray(pack["query_ids"]).astype(str)
    taxon = np.asarray(pack["entity_taxon"])
    raw = np.asarray(pack["raw_response_query33"], dtype=np.float64)
    rms = np.asarray(pack["response_query33_rms"], dtype=np.float64)
    normalized = np.asarray(pack["normalized_response_query33"], dtype=np.float64)
    if query.ndim != 1 or len(set(query.tolist())) != len(query) or taxon.shape != query.shape or not np.all(taxon == 9606):
        raise ValueError("invalid stable query identity axis")
    if raw.shape != (len(query), RESPONSE_WIDTH) or normalized.shape != raw.shape or rms.shape != (RESPONSE_WIDTH,):
        raise ValueError("response-query arrays do not align")
    if not np.isfinite(raw).all() or not np.isfinite(normalized).all() or not np.isfinite(rms).all() or np.any(rms <= 0):
        raise ValueError("response-query arrays must be finite with positive RMS")
    np.testing.assert_allclose(normalized, raw / rms, rtol=2e-7, atol=2e-7)
    expected_rms = np.sqrt(np.mean(np.square(raw), axis=0, dtype=np.float64))
    nonzero = expected_rms > 0
    np.testing.assert_allclose(rms[nonzero], expected_rms[nonzero], rtol=0, atol=0)
    if (
        np.any(rms[~nonzero] != 1.0)
        or int(np.asarray(pack["rank"]).item()) != 32
        or float(np.asarray(pack["alpha"]).item()) != 1000.0
    ):
        raise ValueError("rank, alpha or zero-column RMS contract mismatch")
    model_sha = str(np.asarray(pack["rank_model_sha256"]).item())
    if len(model_sha) != 64 or any(character not in "0123456789abcdef" for character in model_sha):
        raise ValueError("rank model SHA-256 is malformed")
    if lf_roster_sha256(query) != str(np.asarray(pack["query_ids_lf_sha256"]).item()):
        raise ValueError("query roster hash mismatch")
    if not bool(np.asarray(pack["fitting_outcome_derived"]).item()):
        raise ValueError("pack must identify its fitting-derived provenance")
    if bool(np.asarray(pack["development_outcomes_accessed"]).item()) or bool(np.asarray(pack["test_outcomes_accessed"]).item()):
        raise ValueError("pack reports forbidden outcome access")
    return query, normalized.astype(np.float32)


def augment_panel(panel, pack: dict[str, np.ndarray], mode: str):
    """Return a PanelData replacement with one of two matched 610D feature arms."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    query_ids, response = validate_pack(pack)
    panel_queries = np.asarray(panel.query_ids).astype(str)
    query_static = np.asarray(panel.query_features)
    action_static = np.asarray(panel.gene_action_features)
    if not np.array_equal(query_ids, panel_queries):
        raise ValueError("response-query identifiers differ from native panel order")
    source_id = str(np.asarray(pack["source_id"]).item())
    context_id = str(np.asarray(pack["context_id"]).item())
    if panel.source_id != source_id:
        raise ValueError("response-query source differs from panel source")
    panel_contexts = np.asarray(panel.context_ids).astype(str)
    prefix = context_id + "::gem-group:"
    if panel_contexts.ndim != 1 or len(panel_contexts) == 0 or not all(item.startswith(prefix) for item in panel_contexts):
        raise ValueError("response-query context differs from panel contexts")
    if query_static.shape != (len(panel_queries), STATIC_WIDTH):
        raise ValueError("panel query features must be static577")
    if action_static.ndim != 2 or action_static.shape[1] != STATIC_WIDTH:
        raise ValueError("panel action features must be static577")
    appended_query = response if mode == "response33" else np.zeros_like(response)
    query_features = np.concatenate((query_static.astype(np.float32, copy=False), appended_query), axis=1)
    action_features = np.concatenate(
        (action_static.astype(np.float32, copy=False), np.zeros((len(action_static), RESPONSE_WIDTH), dtype=np.float32)),
        axis=1,
    )
    if query_features.shape[1] != OUTPUT_WIDTH or action_features.shape[1] != OUTPUT_WIDTH:
        raise AssertionError("feature augmentation width drift")
    return panel.replace_features(query_features, action_features)


__all__ = ["MODES", "OUTPUT_WIDTH", "augment_panel", "lf_roster_sha256", "response_query33", "validate_pack"]
