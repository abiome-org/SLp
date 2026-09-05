"""Control-only expression normalization for an unseen assay context.

This module implements the normalization used by the pinned Perturbseq_GI
source (commit ``3b25109a``): each cell is scaled to the experiment-wide
median control UMI count, then each query is centered and divided by the
sample standard deviation of controls from the same GEM group.  The transform
is linear in counts: it does not log-transform or clip values.

The fitted object is an SLp-computed, control-anchored transform.  It is not
the Nadig et al. author endpoint, which sums counts into GEM-group
pseudobulks and estimates log2 fold changes and standard errors with DESeq2.
Only non-targeting controls may be used to fit this object.

``full_umi_count`` is deliberately required.  It is the original full-cell
gene-expression UMI total recorded by the source, not the sum of a selected
query panel.  Counts outside the requested query panel therefore remain in
the library-size denominator without becoming model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray

VALUE_SPACE = "slp-per-gem-control-z-score-linear-umi-v1"
BASAL_VALUE_SPACE = "slp-control-mean-log2-1p-cp10k-full-library-v1"


class ControlNormalizationError(ValueError):
    """Raised when raw counts or control metadata violate the contract."""


def _raw_counts(values: Array, label: str) -> Array:
    counts = np.asarray(values, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[0] == 0 or counts.shape[1] == 0:
        raise ControlNormalizationError(f"{label} must be a non-empty matrix")
    if not np.isfinite(counts).all() or np.any(counts < 0.0):
        raise ControlNormalizationError(f"{label} must be finite and nonnegative")
    if np.any(counts != np.floor(counts)):
        raise ControlNormalizationError(f"{label} must contain raw integer UMI counts")
    return counts


def _full_depth(values: Array, rows: int, panel_sum: Array) -> Array:
    depth = np.asarray(values, dtype=np.float64)
    if depth.shape != (rows,):
        raise ControlNormalizationError("full_umi_count must have one value per cell")
    if not np.isfinite(depth).all() or np.any(depth <= 0.0):
        raise ControlNormalizationError("full_umi_count must be finite and positive")
    if np.any(depth != np.floor(depth)):
        raise ControlNormalizationError("full_umi_count must contain integer UMI totals")
    tolerance = np.finfo(np.float64).eps * np.maximum(depth, 1.0) * 8.0
    if np.any(panel_sum > depth + tolerance):
        raise ControlNormalizationError(
            "query-panel counts cannot exceed the original full-cell UMI count"
        )
    return depth


def _gem_groups(values: Array, rows: int) -> Array:
    groups = np.asarray(values)
    if groups.shape != (rows,) or groups.dtype.kind not in "iu":
        raise ControlNormalizationError("gem_group must be one integer per cell")
    return groups.astype(np.int64, copy=False)


def _scaled_counts(counts: Array, depth: Array, target_umi: float) -> Array:
    return counts * (target_umi / depth)[:, None]


@dataclass(frozen=True)
class GemGroupControlNormalizer:
    """Frozen per-GEM control location and scale in linear UMI space."""

    gem_groups_: Array
    target_umi_: float
    control_mean_: Array
    control_std_: Array
    control_observed_: Array
    control_counts_: Array
    value_space: str = VALUE_SPACE
    fit_provenance: str = "non-targeting-controls-only"
    author_endpoint_equivalent: bool = False

    def transform(
        self,
        counts: Array,
        full_umi_count: Array,
        gem_group: Array,
    ) -> tuple[Array, Array]:
        """Return per-cell z scores and their GEM/query support mask.

        The returned values are zero where a GEM/query pair has fewer than two
        controls or zero control variance.  ``observed`` distinguishes those
        placeholders from measured zeros.  Values are intentionally unclipped.
        """

        matrix = _raw_counts(counts, "counts")
        if matrix.shape[1] != self.control_mean_.shape[1]:
            raise ControlNormalizationError("counts have the wrong query dimension")
        depth = _full_depth(full_umi_count, matrix.shape[0], matrix.sum(axis=1))
        groups = _gem_groups(gem_group, matrix.shape[0])
        lookup = {int(group): i for i, group in enumerate(self.gem_groups_)}
        try:
            group_index = np.asarray([lookup[int(group)] for group in groups], dtype=np.int64)
        except KeyError as exc:
            raise ControlNormalizationError(
                f"gem_group {exc.args[0]} has no fitted controls"
            ) from None

        scaled = _scaled_counts(matrix, depth, self.target_umi_)
        mean = self.control_mean_[group_index]
        std = self.control_std_[group_index]
        observed = self.control_observed_[group_index]
        values = np.divide(
            scaled - mean,
            std,
            out=np.zeros_like(scaled),
            where=observed,
        )
        if not np.isfinite(values).all():
            raise ControlNormalizationError("normalization produced non-finite values")
        return values, observed.copy()


def fit_control_normalizer(
    control_counts: Array,
    control_full_umi_count: Array,
    control_gem_group: Array,
) -> GemGroupControlNormalizer:
    """Fit linear-UMI location and scale using non-targeting controls only.

    The sample standard deviation uses ``ddof=1``, matching pandas ``std`` in
    the pinned Perturbseq_GI implementation.  No perturbed expression or
    intervention identity is accepted by this API.
    """

    counts = _raw_counts(control_counts, "control_counts")
    depth = _full_depth(
        control_full_umi_count,
        counts.shape[0],
        counts.sum(axis=1),
    )
    groups = _gem_groups(control_gem_group, counts.shape[0])
    unique_groups = np.unique(groups)
    target_umi = float(np.median(depth))
    scaled = _scaled_counts(counts, depth, target_umi)

    shape = (unique_groups.size, counts.shape[1])
    mean = np.zeros(shape, dtype=np.float64)
    std = np.zeros(shape, dtype=np.float64)
    observed = np.zeros(shape, dtype=np.bool_)
    control_counts_by_group = np.zeros(unique_groups.size, dtype=np.int64)
    for index, group in enumerate(unique_groups):
        local = scaled[groups == group]
        control_counts_by_group[index] = local.shape[0]
        mean[index] = local.mean(axis=0)
        if local.shape[0] >= 2:
            std[index] = local.std(axis=0, ddof=1)
            observed[index] = np.isfinite(std[index]) & (std[index] > 0.0)

    return GemGroupControlNormalizer(
        gem_groups_=unique_groups,
        target_umi_=target_umi,
        control_mean_=mean,
        control_std_=std,
        control_observed_=observed,
        control_counts_=control_counts_by_group,
    )


def control_basal_expression(
    control_counts: Array,
    control_full_umi_count: Array,
    *,
    library_scale: float = 10_000.0,
) -> Array:
    """Return mean control log2(1 + CP10k) using full-cell UMI totals.

    This measured basal-state descriptor is separate from target z scoring.
    It averages cell-level transformed values and never uses perturbed cells.
    """

    counts = _raw_counts(control_counts, "control_counts")
    depth = _full_depth(
        control_full_umi_count,
        counts.shape[0],
        counts.sum(axis=1),
    )
    if not np.isfinite(library_scale) or library_scale <= 0.0:
        raise ControlNormalizationError("library_scale must be finite and positive")
    transformed = np.log2(1.0 + library_scale * counts / depth[:, None])
    return transformed.mean(axis=0)
