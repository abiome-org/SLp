"""Stratified AUROC: P(score(pos) > score(neg)) over pos/neg pairs that share a stratum.

Comparing only within a stratum (a cell line / strain, or a cell line + query gene) means a
model cannot score well by learning which screens or genes have high hit rates.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def stratified_auc(strata: np.ndarray, y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> tuple[float, float]:
    """Returns (AUROC, number of weighted pos/neg comparisons). Ties count 1/2.

    strata: int codes; y: 0/1; s: scores; w: optional per-example weights (for bootstrap).
    """
    if w is None:
        w = np.ones(len(y))
    order = np.lexsort((s, strata))
    strata, y, s, w = strata[order], y[order], s[order], w[order]
    bounds = np.flatnonzero(np.diff(strata)) + 1
    num = den = 0.0
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(y)]):
        yy, ss, ww = y[a:b], s[a:b], w[a:b]
        wp, wn = ww[yy == 1].sum(), ww[yy == 0].sum()
        if wp == 0 or wn == 0:
            continue
        num += _weighted_u(ss, yy, ww)
        den += wp * wn
    return (num / den if den else float("nan")), den


def _weighted_u(s: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted Mann-Whitney U for positives: sum over pos i, neg j of w_i w_j [s_i > s_j] + 0.5 ties."""
    if np.all(w == 1):
        r = rankdata(s)
        n1 = y.sum()
        return float(r[y == 1].sum() - n1 * (n1 + 1) / 2)
    # s is sorted within the stratum; group ties
    uniq, inv = np.unique(s, return_inverse=True)
    wneg = np.bincount(inv, weights=w * (y == 0), minlength=len(uniq))
    wpos = np.bincount(inv, weights=w * (y == 1), minlength=len(uniq))
    cum_neg_below = np.cumsum(wneg) - wneg
    return float((wpos * (cum_neg_below + 0.5 * wneg)).sum())


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())
