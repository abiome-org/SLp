"""Feature-conditioned residual transition in a supplied continuous state space.

This module contains no dataset, split, assay, species, or application logic.
It accepts one fixed-width action feature vector and one measured control-state
vector per record. Multiple-action combinations are intentionally unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    action_features: int
    state_dim: int
    hidden_dim: int = 128
    dropout: float = 0.2


class ResidualStateTransition(nn.Module):
    """Add a nonlinear residual to a caller-supplied base state transition."""

    combination_supported = False

    def __init__(self, config: Config):
        super().__init__()
        if min(config.action_features, config.state_dim, config.hidden_dim) <= 0:
            raise ValueError("all dimensions must be positive")
        if not 0 <= config.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.config = config
        self.hidden = nn.Sequential(
            nn.Linear(config.action_features + config.state_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.residual = nn.Linear(config.hidden_dim, config.state_dim)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(
        self,
        action_features: torch.Tensor,
        control_state: torch.Tensor,
        base_delta: torch.Tensor,
        has_action: torch.Tensor,
    ) -> torch.Tensor:
        if action_features.ndim != 2 or action_features.shape[1] != self.config.action_features:
            raise ValueError("action features must be [records,action_features]")
        expected_state = (len(action_features), self.config.state_dim)
        if control_state.shape != expected_state or base_delta.shape != expected_state:
            raise ValueError("control state and base delta must align with action records")
        if has_action.shape != (len(action_features),) or has_action.dtype != torch.bool:
            raise ValueError("has_action must be a Boolean record mask")
        tensors = (action_features, control_state, base_delta)
        if any(
            value.device != action_features.device
            or value.dtype != action_features.dtype
            or not torch.isfinite(value).all()
            for value in tensors
        ):
            raise ValueError("transition inputs must share finite dtype and device")
        residual = self.residual(self.hidden(torch.cat((action_features, control_state), dim=1)))
        return torch.where(has_action[:, None], base_delta + residual, torch.zeros_like(base_delta))
