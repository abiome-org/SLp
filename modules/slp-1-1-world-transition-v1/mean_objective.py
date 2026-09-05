"""Fitting-only point objective utilities for heterogeneous molecular contexts."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch

Array = np.ndarray


class MeanObjectiveError(ValueError):
    """Raised when a mean-only fitting contract is invalid."""


def context_query_sd(
    targets: Array,
    observed: Array,
    context_index: Array,
    fitting_rows: Array,
    context_count: int,
    *,
    floor: float = 0.05,
) -> Array:
    """Return population SD per context/query from fitting rows only.

    Rows, including repeated constructs, receive equal mass. Missing values do
    not enter the moments. A query with fewer than two observed fitting rows is
    represented by the fixed floor rather than by an outcome-derived imputation.
    """

    values = np.asarray(targets)
    mask = np.asarray(observed)
    context = np.asarray(context_index)
    rows = np.asarray(fitting_rows)
    if (
        values.ndim != 2
        or mask.shape != values.shape
        or mask.dtype != np.bool_
        or context.shape != (values.shape[0],)
        or context.dtype.kind not in "iu"
        or rows.ndim != 1
        or rows.dtype.kind not in "iu"
        or rows.size == 0
        or np.any(rows < 0)
        or np.any(rows >= len(values))
        or context_count <= 0
        or not np.isfinite(floor)
        or floor <= 0
        or not np.isfinite(values[mask]).all()
    ):
        raise MeanObjectiveError("invalid fitting-only context scale inputs")
    if np.any(context[rows] < 0) or np.any(context[rows] >= context_count):
        raise MeanObjectiveError("fitting context index is out of range")

    scales = np.full((context_count, values.shape[1]), floor, dtype=np.float64)
    for index in range(context_count):
        selected = rows[context[rows] == index]
        if selected.size == 0:
            raise MeanObjectiveError("every requested context requires fitting rows")
        local_values = np.asarray(values[selected], dtype=np.float64)
        local_mask = mask[selected]
        counts = local_mask.sum(axis=0)
        total = np.where(local_mask, local_values, 0.0).sum(axis=0)
        mean = np.divide(total, counts, out=np.zeros_like(total), where=counts > 0)
        centered = np.where(local_mask, local_values - mean, 0.0)
        variance = np.divide(
            np.square(centered).sum(axis=0),
            counts,
            out=np.zeros_like(total),
            where=counts > 0,
        )
        eligible = counts >= 2
        scales[index, eligible] = np.maximum(
            np.sqrt(variance[eligible]), floor
        )
    return scales.astype(np.float32)


def masked_standardized_mse(
    prediction_mean: torch.Tensor,
    target: torch.Tensor,
    observed: torch.Tensor,
    query_scale: torch.Tensor,
    row_weight: torch.Tensor,
) -> torch.Tensor:
    """Return fixed-weight row-mean MSE in fitting-query-SD units.

    ``query_scale`` is one row per record in the batch. Row weights are assumed
    to have global mean one and are deliberately not renormalized in a batch.
    """

    if (
        prediction_mean.shape != target.shape
        or target.shape != observed.shape
        or target.shape != query_scale.shape
        or target.ndim != 2
        or observed.dtype != torch.bool
        or row_weight.shape != (target.shape[0],)
        or not torch.is_floating_point(row_weight)
    ):
        raise MeanObjectiveError("mean objective tensors do not align")
    safe_target = torch.where(observed, target, torch.zeros_like(target))
    safe_scale = torch.where(observed, query_scale, torch.ones_like(query_scale))
    if (
        not torch.isfinite(safe_target).all()
        or not torch.isfinite(safe_scale).all()
        or not (safe_scale > 0).all()
        or not torch.isfinite(row_weight).all()
        or not (row_weight > 0).all()
    ):
        raise MeanObjectiveError("mean objective requires finite positive inputs")
    counts = observed.sum(1)
    if not (counts > 0).all():
        raise MeanObjectiveError("every fitting record requires an observed query")
    squared = torch.where(
        observed,
        torch.square((prediction_mean - safe_target) / safe_scale),
        torch.zeros_like(target),
    )
    per_record = squared.sum(1) / counts
    return (per_record * row_weight.to(per_record.dtype)).mean()


def deterministic_shuffled_batches(
    fitting_rows: Array,
    *,
    batch_size: int,
    steps: int,
    seed: int,
) -> Iterator[Array]:
    """Yield exactly ``steps`` shuffled full batches, dropping each cycle's tail."""

    rows = np.asarray(fitting_rows)
    if (
        rows.ndim != 1
        or rows.dtype.kind not in "iu"
        or len(rows) < batch_size
        or len(np.unique(rows)) != len(rows)
        or batch_size <= 0
        or steps <= 0
    ):
        raise MeanObjectiveError("invalid deterministic batch contract")
    generator = np.random.default_rng(seed)
    emitted = 0
    while emitted < steps:
        order = generator.permutation(rows)
        complete = len(order) - (len(order) % batch_size)
        for offset in range(0, complete, batch_size):
            yield order[offset : offset + batch_size]
            emitted += 1
            if emitted == steps:
                return
