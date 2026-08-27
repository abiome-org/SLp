"""Application-level decoders over frozen world-model predictions."""

from .continuous import (
    DependencyInteraction,
    RankedResidualInteraction,
    ResidualInteraction,
    interaction_depletion_head,
    interaction_head,
    tolerance_head,
)

__all__ = [
    "DependencyInteraction",
    "RankedResidualInteraction",
    "ResidualInteraction",
    "interaction_depletion_head",
    "interaction_head",
    "tolerance_head",
]
