"""Static physical-neighborhood summaries; no molecular response fitting."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def neighborhood_features(features, edges):
    """Deduplicate symmetric edges by maximum weight and average known neighbors."""
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("finite entity feature matrix required")
    unique = {}
    for left, right, weight in edges:
        if not 0 <= left < len(x) or not 0 <= right < len(x):
            raise ValueError("edge identity outside feature roster")
        if not np.isfinite(weight) or not 0 < weight <= 1:
            raise ValueError("physical confidence weight must be in (0,1]")
        if left == right:
            continue
        key = tuple(sorted((int(left), int(right))))
        unique[key] = max(unique.get(key, 0), float(weight))
    rows, cols, weights = [], [], []
    for (left, right), weight in sorted(unique.items()):
        rows.extend((left, right))
        cols.extend((right, left))
        weights.extend((weight, weight))
    adjacency = csr_matrix((np.asarray(weights, dtype=np.float32), (rows, cols)), shape=(len(x), len(x)))
    degree = np.diff(adjacency.indptr).astype(np.float32)
    mass = np.asarray(adjacency.sum(axis=1)).ravel()
    neighbor = (adjacency @ x) / np.maximum(mass[:, None], 1e-12)
    output = np.concatenate((x, neighbor, np.log1p(degree)[:, None], (degree > 0)[:, None]), axis=1).astype(np.float32)
    return output, {"edges":len(unique), "entities_with_neighbors":int((degree > 0).sum()),
                    "original_feature_dimensions":x.shape[1], "output_feature_dimensions":output.shape[1]}
