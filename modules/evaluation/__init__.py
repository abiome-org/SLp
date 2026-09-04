"""Read-only molecular and benchmark evaluation entry points."""

from .generalization import (
    PROTOCOLS,
    EvidenceRequirements,
    GeneralizationSplit,
    GeneralizationTable,
    additive_single_baseline,
    cardinality_mean_baseline,
    make_split,
    make_suite,
    regression_metrics,
)
from .protocol import require_explicit_benchmark

__all__ = [
    "PROTOCOLS",
    "EvidenceRequirements",
    "GeneralizationSplit",
    "GeneralizationTable",
    "additive_single_baseline",
    "cardinality_mean_baseline",
    "make_split",
    "make_suite",
    "regression_metrics",
    "require_explicit_benchmark",
]
