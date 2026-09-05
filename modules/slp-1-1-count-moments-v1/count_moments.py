"""Streaming moments of library-normalized RNA counts; no fitted parameters."""

from __future__ import annotations

import numpy as np
from scipy import sparse


class CountMoments:
    """Aggregate equal-cell log1p(CP10k) by caller-defined population.

    Input blocks are cells by source features. ``query_index`` maps source
    features to stable output queries (-1 omits an output). Many source rows
    may map to one query: their counts are summed before the nonlinear map.
    The explicit denominator mask may include unmapped biological features.
    Population assignment and eligibility must be fixed by the caller.
    """

    def __init__(
        self,
        query_index: np.ndarray,
        denominator_mask: np.ndarray,
        n_queries: int,
        n_groups: int,
    ) -> None:
        mapping = np.asarray(query_index)
        denominator = np.asarray(denominator_mask)
        if (
            mapping.ndim != 1
            or mapping.dtype.kind not in "iu"
            or denominator.shape != mapping.shape
            or denominator.dtype.kind != "b"
            or n_queries <= 0
            or n_groups <= 0
            or np.any(mapping < -1)
            or np.any(mapping >= n_queries)
            or not denominator.any()
        ):
            raise ValueError("invalid query mapping, denominator or dimensions")
        if np.any((mapping >= 0) & ~denominator):
            raise ValueError("queried count features must enter the denominator")
        self.denominator = denominator.copy()
        keep = np.flatnonzero(mapping >= 0)
        self.projection = sparse.csr_matrix(
            (np.ones(len(keep)), (keep, mapping[keep])),
            shape=(len(mapping), n_queries),
        )
        self.total_cells = np.zeros(n_groups, dtype=np.int64)
        self.cells = np.zeros(n_groups, dtype=np.int64)
        self.zero_library_cells = np.zeros(n_groups, dtype=np.int64)
        self.sums = np.zeros((n_groups, n_queries), dtype=np.float64)
        self.squares = np.zeros_like(self.sums)

    def update(self, counts: sparse.spmatrix, groups: np.ndarray) -> np.ndarray:
        """Consume one bounded block; return its positive-library eligibility."""
        if not sparse.issparse(counts) or counts.ndim != 2:
            raise ValueError("counts must be a sparse cell-by-feature matrix")
        groups = np.asarray(groups)
        if (
            counts.shape[1] != self.projection.shape[0]
            or groups.shape != (counts.shape[0],)
            or groups.dtype.kind not in "iu"
            or np.any(groups < 0)
            or np.any(groups >= len(self.cells))
        ):
            raise ValueError("count shape or population assignment mismatch")
        raw = counts.tocsr().astype(np.float64, copy=True)
        if (
            not np.isfinite(raw.data).all()
            or np.any(raw.data < 0)
            or np.any(raw.data != np.floor(raw.data))
        ):
            raise ValueError("raw RNA counts must be finite nonnegative integers")
        raw.sum_duplicates()
        library = np.asarray(raw[:, self.denominator].sum(axis=1)).ravel()
        if not np.isfinite(library).all():
            raise ValueError("nonfinite library size")
        valid = library > 0
        projected = (raw[valid] @ self.projection).tocsr()
        projected.sum_duplicates()
        projected.eliminate_zeros()
        rows = np.repeat(np.arange(valid.sum()), np.diff(projected.indptr))
        values = np.log1p(projected.data * (10000.0 / library[valid][rows]))
        if not np.isfinite(values).all():
            raise ValueError("nonfinite normalized count")
        selected_groups = groups[valid]
        np.add.at(self.total_cells, groups, 1)
        np.add.at(self.cells, selected_groups, 1)
        np.add.at(self.zero_library_cells, groups[~valid], 1)
        index = (selected_groups[rows], projected.indices)
        np.add.at(self.sums, index, values)
        np.add.at(self.squares, index, values * values)
        return valid

    def summary(self) -> dict[str, np.ndarray]:
        """Return means and unbiased cell variances, with explicit support.

        Empty populations and variance for fewer than two cells are NaN.
        An observed zero is included in the population count and moments.
        """
        means = np.full_like(self.sums, np.nan)
        variances = np.full_like(self.sums, np.nan)
        observed = self.cells > 0
        replicated = self.cells > 1
        means[observed] = self.sums[observed] / self.cells[observed, None]
        centered_ss = (
            self.squares[replicated]
            - self.sums[replicated] ** 2 / self.cells[replicated, None]
        )
        variances[replicated] = np.maximum(centered_ss, 0) / (
            self.cells[replicated, None] - 1
        )
        return {
            "mean": means,
            "cell_variance": variances,
            "num_cells": self.cells.copy(),
            "total_cells": self.total_cells.copy(),
            "zero_library_cells": self.zero_library_cells.copy(),
            "mean_observed": observed,
            "variance_observed": replicated,
        }
