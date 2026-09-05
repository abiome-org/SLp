"""Explicit fitting-row objective weights for molecular transition training."""

from __future__ import annotations

import numpy as np

Array = np.ndarray

UNIFORM_ROW_V1 = "uniform-row-v1"
EQUAL_CONTEXT_GENE_V1 = "equal-context-gene-v1"
SUPPORTED_OBJECTIVES = (UNIFORM_ROW_V1, EQUAL_CONTEXT_GENE_V1)


def training_row_weights(
    context_index: Array,
    action_ids: Array,
    *,
    objective: str = UNIFORM_ROW_V1,
) -> Array:
    """Return fixed global fitting-row weights with arithmetic mean one.

    ``equal-context-gene-v1`` assigns each context equal total mass, each
    intervention gene within a context equal mass, and duplicate records of a
    gene equal shares of that gene's mass.  The weights are computed once over
    all fitting rows and are never renormalized within a minibatch.
    """

    contexts = np.asarray(context_index)
    actions = np.asarray(action_ids).astype(str)
    if (
        contexts.ndim != 1
        or contexts.dtype.kind not in "iu"
        or actions.shape != contexts.shape
        or contexts.size == 0
        or np.any(contexts < 0)
        or np.any(actions == "")
    ):
        raise ValueError("context_index and action_ids must be aligned non-empty rows")
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"unsupported training objective: {objective}")
    if objective == UNIFORM_ROW_V1:
        return np.ones(contexts.size, dtype=np.float64)

    unique_contexts = np.unique(contexts)
    weights = np.empty(contexts.size, dtype=np.float64)
    total_rows = contexts.size
    context_count = unique_contexts.size
    for context in unique_contexts:
        rows = np.flatnonzero(contexts == context)
        genes, inverse, record_counts = np.unique(
            actions[rows], return_inverse=True, return_counts=True
        )
        weights[rows] = total_rows / (
            context_count * genes.size * record_counts[inverse]
        )
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise RuntimeError("objective weighting produced invalid weights")
    if not np.isclose(weights.sum(), float(total_rows), rtol=1e-12, atol=1e-12):
        raise RuntimeError("objective weights do not have global mean one")
    return weights


def weighting_summary(context_index: Array, action_ids: Array, weights: Array) -> dict:
    """Summarize fixed weights without reading molecular outcomes."""

    contexts = np.asarray(context_index)
    actions = np.asarray(action_ids).astype(str)
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != contexts.shape or actions.shape != contexts.shape:
        raise ValueError("weight summary inputs must align")
    by_context = {}
    for context in np.unique(contexts):
        rows = contexts == context
        gene_mass = {}
        for action, weight in zip(actions[rows], values[rows]):
            gene_mass[action] = gene_mass.get(action, 0.0) + float(weight)
        by_context[str(int(context))] = {
            "rows": int(rows.sum()),
            "genes": len(gene_mass),
            "totalWeight": float(values[rows].sum()),
            "minimumGeneWeight": float(min(gene_mass.values())),
            "maximumGeneWeight": float(max(gene_mass.values())),
        }
    return {
        "rows": int(values.size),
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "normalization": "fixed globally; no minibatch renormalization",
        "contexts": by_context,
    }
