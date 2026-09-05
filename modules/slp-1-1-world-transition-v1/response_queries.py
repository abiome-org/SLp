"""Measured response descriptors for molecular queries, fitted on training only.

These are assay-derived descriptors, not static protein features. A new query
requires either a measurement-derived descriptor or a separately tested missing
descriptor policy. No fixed gene ID table is stored in the neural architecture.
"""
from __future__ import annotations

import numpy as np
from sklearn.utils.extmath import randomized_svd


def fit_query_response_descriptors(targets, context_index, reference, scale,
                                   rank=32, seed=731):
    """Accept training rows only; return standardized response-basis loadings."""
    y = np.asarray(targets, dtype=np.float64)
    context = np.asarray(context_index, dtype=np.int64)
    if y.ndim != 2 or context.shape != (len(y),) or not np.isfinite(y).all():
        raise ValueError("complete finite training measurements required")
    reference, scale = np.asarray(reference), np.asarray(scale)
    if reference.shape != scale.shape or reference.ndim != 2 or (scale <= 0).any():
        raise ValueError("context query references and positive scales required")
    residual = (y-reference[context])/scale[context]
    if not np.isfinite(residual).all() or not 0 < rank < min(residual.shape):
        raise ValueError("invalid standardized residuals or response rank")
    # Context baselines are removed before pooled response geometry is learned.
    _, singular, components = randomized_svd(residual, n_components=rank,
                                             n_iter=5, random_state=seed)
    descriptors = (components.T * np.sqrt(y.shape[1])).astype(np.float32)
    fraction = float(np.square(singular).sum()/np.square(residual).sum())
    return descriptors, {"rank": rank, "standardized_training_variance_fraction": fraction,
                         "rows_fitted": len(y), "query_count": y.shape[1]}
