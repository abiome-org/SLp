"""Exact binary benchmark metrics and predeclared inner-fold selection."""

from collections import defaultdict
import math

import numpy as np


def binary_metrics(labels, scores):
    y, score = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if (
        y.ndim != 1
        or score.shape != y.shape
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(score).all()
    ):
        raise ValueError("Metrics require finite scores for every binary label")
    positive = int(y.sum())
    negative = len(y) - positive
    if not positive or not negative:
        raise ValueError("Both classes required; report the degenerate fold explicitly")
    order = np.argsort(-score, kind="stable")
    sorted_score, sorted_y = score[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_score)), len(y) - 1]
    tp = np.cumsum(sorted_y)[ends]
    fp = ends + 1 - tp
    recall, precision = tp / positive, tp / (tp + fp)
    # AP groups tied scores and uses the step integral. The separate trapezoid
    # is reported under its own name; neither is silently substituted.
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    pr_auc = float(np.trapezoid(np.r_[1.0, precision], np.r_[0.0, recall]))
    auroc = float(np.trapezoid(np.r_[0.0, tp / positive], np.r_[0.0, fp / negative]))
    return {
        "n": len(y),
        "positives": positive,
        "prevalence": positive / len(y),
        "average_precision": ap,
        "trapezoidal_pr_auc": pr_auc,
        "auroc": auroc,
    }


def select_checkpoint(reports, expected_benchmarks, expected_inner_folds):
    """Equal benchmark weight after averaging that benchmark's inner-fold APs."""
    folds = (
        expected_inner_folds
        if isinstance(expected_inner_folds, dict)
        else {b: expected_inner_folds for b in expected_benchmarks}
    )
    if set(folds) != set(expected_benchmarks) or any(not f for f in folds.values()):
        raise ValueError("Invalid benchmark-specific fold protocol")
    expected = {(b, f) for b in expected_benchmarks for f in folds[b]}
    if not expected:
        raise ValueError("Selection protocol is empty")
    grouped = defaultdict(dict)
    for report in reports:
        if report["partition"] != "inner" or report["forbidden_human_exposures"] != 0:
            raise ValueError("Selection requires clean inner-CV3 evidence")
        key = (report["benchmark"], report["fold"])
        candidate = report["checkpoint"]
        if key in grouped[candidate]:
            raise ValueError("Duplicate benchmark/fold evidence")
        value = report["average_precision"]
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid AP")
        grouped[candidate][key] = value
    scores = {}
    for candidate, values in grouped.items():
        if set(values) != expected:
            raise ValueError("Incomplete or changed benchmark suite for " + candidate)
        means = {
            b: sum(values[b, f] for f in folds[b]) / len(folds[b])
            for b in expected_benchmarks
        }
        scores[candidate] = {
            "equal_benchmark_mean_ap": sum(means.values()) / len(means),
            "benchmarks": means,
        }
    if not scores:
        raise ValueError("No candidate evidence")
    winner = max(sorted(scores), key=lambda c: scores[c]["equal_benchmark_mean_ap"])
    return {
        "selected": winner,
        "scores": scores,
        "selector": "equal-benchmark mean of inner-fold average precision",
    }
