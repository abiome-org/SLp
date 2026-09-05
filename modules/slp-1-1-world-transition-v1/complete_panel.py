"""Select a shared measured assay panel from a development snapshot."""
from __future__ import annotations

import numpy as np

QUERY_ARRAYS = {
    "targets", "observed", "basal_control", "context_basal_expression",
    "control_targets", "control_observed",
}


def select_complete_panel(data):
    """Use fitting availability only; never impute molecular measurements."""
    if len(data["split_test"]):
        raise ValueError("only development snapshots are accepted")
    training = np.asarray(data["split_train"])
    if not len(training):
        raise ValueError("fitting records are required")
    observed = np.asarray(data["observed"], dtype=bool)
    keep = np.flatnonzero(observed[training].all(axis=0))
    if not len(keep):
        raise ValueError("no fully measured fitting queries")
    query_count = len(data["query_ids"])
    result = {}
    for name, value in data.items():
        value = np.asarray(value)
        if name == "query_ids":
            result[name] = value[keep]
        elif name in QUERY_ARRAYS:
            if value.ndim != 2 or value.shape[1] != query_count:
                raise ValueError(f"misaligned query array: {name}")
            result[name] = value[:, keep]
        else:
            result[name] = value.copy()
    return result, keep
