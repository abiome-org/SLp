"""Query-feature molecular cell encoder and linear state observations.

Inputs and predictions are in caller-specified, fitted measurement units.
Gene and assay identities never index learned embedding tables.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    rna_features: int
    protein_features: int
    key_dim: int = 64
    state_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.1


def mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, outputs))


class CellState(nn.Module):
    """Pool molecular values through supplied query features into a shared state.

    RNA and protein panels can change size and order. Missing values require
    explicit Boolean masks; a missing modality contributes zero moments and
    a false availability flag. A completely unobserved cell is invalid.
    """

    def __init__(self, config: Config):
        super().__init__()
        if min(config.rna_features, config.protein_features, config.key_dim,
               config.state_dim, config.hidden_dim) <= 0:
            raise ValueError("all dimensions must be positive")
        if not 0 <= config.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.config = config
        self.keys = nn.ModuleDict({
            "rna": mlp(config.rna_features, config.hidden_dim, config.key_dim),
            "protein": mlp(config.protein_features, config.hidden_dim, config.key_dim),
        })
        self.encoder = nn.Sequential(
            nn.Linear(4 * config.key_dim + 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.state_dim),
        )
        self.observation = nn.ModuleDict({
            "rna": mlp(config.rna_features, config.hidden_dim, config.state_dim + 1),
            "protein": mlp(config.protein_features, config.hidden_dim, config.state_dim + 1),
        })

    def _features(self, head: str, features: torch.Tensor) -> None:
        if head not in ("rna", "protein"):
            raise ValueError("unknown molecular head")
        expected = self.config.rna_features if head == "rna" else self.config.protein_features
        if features.ndim != 2 or features.shape[1] != expected:
            raise ValueError("query feature dimensions do not match head")
        if not torch.is_floating_point(features) or not torch.isfinite(features).all():
            raise ValueError("query features must be finite floating point")

    def _moments(
        self, head: str, features: torch.Tensor, values: torch.Tensor, observed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._features(head, features)
        if values.ndim != 2 or values.shape[1] != len(features):
            raise ValueError("values must align with molecular queries")
        if observed.shape != values.shape or observed.dtype != torch.bool:
            raise ValueError("observed mask must be Boolean and align with values")
        if not (values.device == features.device == observed.device) or values.dtype != features.dtype:
            raise ValueError("values, features and masks must share device; values and features need one dtype")
        safe = torch.where(observed, values, 0.0)
        if not torch.isfinite(safe).all():
            raise ValueError("observed molecular values must be finite")
        keys = self.keys[head](features)
        counts = observed.sum(1, keepdim=True)
        available = counts > 0
        # sqrt(panel size) keeps a standardized sum from growing linearly
        # with panel size; the composition moment records the measured panel.
        value_moment = safe @ keys / counts.clamp_min(1).sqrt()
        panel_moment = observed.to(values.dtype) @ keys / counts.clamp_min(1)
        return value_moment, panel_moment, available.to(values.dtype)

    def encode(
        self,
        rna_features: torch.Tensor,
        rna_values: torch.Tensor,
        rna_observed: torch.Tensor,
        protein_features: torch.Tensor,
        protein_values: torch.Tensor,
        protein_observed: torch.Tensor,
    ) -> torch.Tensor:
        rna = self._moments("rna", rna_features, rna_values, rna_observed)
        protein = self._moments("protein", protein_features, protein_values, protein_observed)
        if rna_values.shape[0] != protein_values.shape[0]:
            raise ValueError("paired molecular cell rows must align")
        if not ((rna[2] + protein[2]) > 0).all():
            raise ValueError("every cell needs at least one observed modality")
        return self.encoder(torch.cat((*rna, *protein), dim=1))

    def observe(self, state: torch.Tensor, features: torch.Tensor, head: str) -> torch.Tensor:
        """Observe a state in standardized molecular units, with query chunking.

        Observation is affine in state. Therefore averaging state predictions
        and averaging molecular predictions commute up to floating point error.
        """
        self._features(head, features)
        if state.ndim != 2 or state.shape[1] != self.config.state_dim:
            raise ValueError("state must be [cells,state_dim]")
        if state.device != features.device or state.dtype != features.dtype or not torch.isfinite(state).all():
            raise ValueError("state and query features must share finite dtype/device")
        parameters = self.observation[head](features)
        return state @ parameters[:, :-1].T / math.sqrt(self.config.state_dim) + parameters[:, -1]

    def observe_delta(
        self, state_delta: torch.Tensor, features: torch.Tensor, head: str,
        control_mean: torch.Tensor, measurement_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Map a later learned state change to an explicitly supplied control.

        This is an algebraic interface, not an already fitted intervention
        transition. Zero state change gives exact control identity.
        """
        self._features(head, features)
        if state_delta.ndim != 2 or state_delta.shape[1] != self.config.state_dim:
            raise ValueError("state delta must be [cells,state_dim]")
        if control_mean.shape != (len(state_delta), len(features)) or measurement_scale.shape != (len(features),):
            raise ValueError("control/measurement scale must align with states and queries")
        tensors = (features, state_delta, control_mean, measurement_scale)
        if any(x.device != features.device or x.dtype != features.dtype or not torch.isfinite(x).all() for x in tensors):
            raise ValueError("observation tensors must share finite dtype and device")
        if not (measurement_scale > 0).all():
            raise ValueError("measurement scales must be positive")
        basis = self.observation[head](features)[:, :-1]
        change = state_delta @ basis.T / math.sqrt(self.config.state_dim)
        return control_mean + change * measurement_scale


def balanced_reconstruction_loss(
    rna_prediction: torch.Tensor, rna_target: torch.Tensor, rna_observed: torch.Tensor,
    protein_prediction: torch.Tensor, protein_target: torch.Tensor, protein_observed: torch.Tensor,
) -> torch.Tensor:
    """Equal RNA/protein mean squared error; targets use training-fitted scales."""
    losses = []
    for predicted, target, observed in (
        (rna_prediction, rna_target, rna_observed),
        (protein_prediction, protein_target, protein_observed),
    ):
        if predicted.shape != target.shape or observed.shape != target.shape or observed.dtype != torch.bool:
            raise ValueError("reconstruction targets, predictions and masks must align")
        if not (observed.sum(1) > 0).all():
            raise ValueError("balanced loss requires both measured target modalities")
        safe_target = torch.where(observed, target, 0.0)
        if not torch.isfinite(predicted).all() or not torch.isfinite(safe_target).all():
            raise ValueError("reconstruction values must be finite when observed")
        squared = torch.where(observed, (predicted - safe_target).square(), 0.0)
        losses.append((squared.sum(1) / observed.sum(1)).mean())
    return 0.5 * (losses[0] + losses[1])
