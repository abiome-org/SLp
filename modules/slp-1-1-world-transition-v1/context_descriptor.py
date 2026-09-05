"""Pooled control expression descriptor shared across assay contexts.

The descriptor for query ``q`` is

``log2(1 + 10000 * total_control_query_UMIs[q] / total_control_full_UMIs)``.

For deposited pseudobulk means, totals are reconstructed as the stored mean
times the number of filtered cells.  For single-cell rows, every row has
weight one.  The denominator must describe the complete retained source gene
matrix; it must never be recomputed from a selected query panel.

Only control measurements enter this statistic.  It is a measured context
input, not a perturbation outcome or a learned parameter.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray

VALUE_SPACE = "control-pooled-log2-1p-cp10k-full-library-v1"
FIXED_PANEL_VALUE_SPACE = "control-pooled-log2-1p-cp10k-fixed-shared-panel-v1"


class ContextDescriptorError(ValueError):
    """Raised when control aggregates violate the descriptor contract."""


def pooled_control_log2_cp10k(
    control_mean_counts: Array,
    control_full_mean_umi: Array,
    control_num_cells: Array,
    observed: Array | None = None,
    *,
    library_scale: float = 10_000.0,
) -> tuple[Array, Array]:
    """Compute a context descriptor from control rows or pseudobulk means.

    ``control_mean_counts`` is ``[P, Q]``.  Row ``p`` may be a single cell
    (weight one) or the arithmetic mean of ``control_num_cells[p]`` cells.
    ``control_full_mean_umi[p]`` is the corresponding mean full-library UMI
    count.  A missing query uses only rows where it is observed; a query with
    no support is returned as zero with ``descriptor_observed=False``.
    """

    values = np.asarray(control_mean_counts, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ContextDescriptorError("control_mean_counts must be a non-empty matrix")
    if observed is None:
        mask = np.isfinite(values)
    else:
        mask = np.asarray(observed)
        if mask.shape != values.shape or mask.dtype != np.bool_:
            raise ContextDescriptorError("observed must be a matching boolean matrix")
    if not np.isfinite(values[mask]).all() or np.any(values[mask] < 0.0):
        raise ContextDescriptorError("observed control means must be finite and nonnegative")

    full_mean = np.asarray(control_full_mean_umi, dtype=np.float64)
    weights = np.asarray(control_num_cells, dtype=np.float64)
    if full_mean.shape != (values.shape[0],):
        raise ContextDescriptorError("control_full_mean_umi must align with control rows")
    if weights.shape != (values.shape[0],):
        raise ContextDescriptorError("control_num_cells must align with control rows")
    if (
        not np.isfinite(full_mean).all()
        or np.any(full_mean <= 0.0)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
    ):
        raise ContextDescriptorError("full UMI means and cell counts must be positive")
    if np.any(weights != np.floor(weights)):
        raise ContextDescriptorError("control_num_cells must contain integer cell counts")
    if not np.isfinite(library_scale) or library_scale <= 0.0:
        raise ContextDescriptorError("library_scale must be finite and positive")

    stored = np.where(mask, values, 0.0)
    query_totals = (stored * weights[:, None]).sum(axis=0)
    denominator = (
        np.where(mask, full_mean[:, None] * weights[:, None], 0.0).sum(axis=0)
    )
    descriptor_observed = denominator > 0.0
    ratio = np.divide(
        query_totals,
        denominator,
        out=np.zeros(values.shape[1], dtype=np.float64),
        where=descriptor_observed,
    )
    descriptor = np.log2(1.0 + library_scale * ratio)
    if not np.isfinite(descriptor).all():
        raise ContextDescriptorError("descriptor contains non-finite values")
    return descriptor, descriptor_observed


def pooled_control_fixed_panel_log2_cp10k(
    control_mean_counts: Array,
    control_num_cells: Array,
    panel_mask: Array,
    observed: Array | None = None,
    *,
    library_scale: float = 10_000.0,
) -> tuple[Array, Array]:
    """Compute relative abundance on one explicit shared query panel.

    Both numerator and denominator use only queries selected by
    ``panel_mask``. Queries outside that panel are returned as masked zeros,
    even when a source measures them. This is appropriate when original
    full-library denominators cannot be established consistently across
    sources.
    """

    values = np.asarray(control_mean_counts, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ContextDescriptorError("control_mean_counts must be a non-empty matrix")
    selected = np.asarray(panel_mask)
    if selected.shape != (values.shape[1],) or selected.dtype != np.bool_:
        raise ContextDescriptorError("panel_mask must be one boolean per query")
    if not selected.any():
        raise ContextDescriptorError("panel_mask cannot be empty")
    if observed is None:
        source_observed = np.isfinite(values)
    else:
        source_observed = np.asarray(observed)
        if source_observed.shape != values.shape or source_observed.dtype != np.bool_:
            raise ContextDescriptorError("observed must be a matching boolean matrix")
    if not np.all(source_observed[:, selected]):
        raise ContextDescriptorError(
            "every control row must measure every fixed-panel query"
        )
    denominator_mean = values[:, selected].sum(axis=1)
    descriptor_mask = source_observed & selected[None, :]
    return pooled_control_log2_cp10k(
        values,
        denominator_mean,
        control_num_cells,
        descriptor_mask,
        library_scale=library_scale,
    )
