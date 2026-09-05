"""Control-anchored transition with externally supplied fixed query coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    action_feature_dim: int
    basal_feature_dim: int
    state_dim: int = 128
    hidden_dim: int = 128
    dropout: float = 0.2


class FixedQueryTransition(nn.Module):
    """Predict query deltas in a supplied, non-parametric output basis."""

    def __init__(self, config: Config):
        super().__init__()
        if min(
            config.action_feature_dim,
            config.basal_feature_dim,
            config.state_dim,
            config.hidden_dim,
        ) <= 0:
            raise ValueError("model dimensions must be positive")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        self.config = config
        hidden, state = config.hidden_dim, config.state_dim
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, state),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(config.basal_feature_dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, state),
        )
        self.transition = nn.Sequential(
            nn.Linear(2 * state, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, state),
        )
        self.mean_state = nn.Linear(state, state, bias=False)
        nn.init.normal_(self.mean_state.weight, std=0.002)

    @staticmethod
    def _query_matrix(values: torch.Tensor, batch: int, queries: int, label: str) -> torch.Tensor:
        if values.ndim == 1 and values.shape[0] == queries:
            values = values.unsqueeze(0).expand(batch, -1)
        if values.shape != (batch, queries) or not torch.isfinite(values).all():
            raise ValueError(f"{label} must be finite [Q] or [B,Q]")
        return values

    def _encode_basal(
        self,
        batch: int,
        features: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim == 2:
            features = features.unsqueeze(0).expand(batch, -1, -1)
        if (
            features.ndim != 3
            or features.shape[0] != batch
            or features.shape[2] != self.config.basal_feature_dim
            or values.shape != features.shape[:2]
            or mask.shape != features.shape[:2]
            or mask.dtype != torch.bool
        ):
            raise ValueError("basal feature, value, and mask shapes disagree")
        safe_features = torch.where(mask[..., None], features, torch.zeros_like(features))
        safe_values = torch.where(mask, values, torch.zeros_like(values))
        if not torch.isfinite(safe_features).all() or not torch.isfinite(safe_values).all():
            raise ValueError("unmasked basal entries must be finite")
        encoded = self.context_encoder(torch.cat((safe_features, safe_values[..., None]), -1))
        return (encoded * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)

    def _encode_actions(
        self, actions: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if actions.ndim == 2:
            actions = actions[:, None, :]
        if actions.ndim != 3 or actions.shape[2] != self.config.action_feature_dim:
            raise ValueError("actions must be [B,A,F] or [B,F]")
        if mask is None:
            mask = torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
        if mask.shape != actions.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("action_mask must be Boolean [B,A]")
        safe = torch.where(mask[..., None], actions, torch.zeros_like(actions))
        if not torch.isfinite(safe).all():
            raise ValueError("unmasked action entries must be finite")
        encoded = self.action_encoder(safe)
        return (encoded * mask[..., None]).sum(1), mask.any(1)

    def forward(
        self,
        actions: torch.Tensor,
        query_coordinates: torch.Tensor,
        control_mean: torch.Tensor,
        delta_amplitude: torch.Tensor,
        observation_scale: torch.Tensor,
        basal_features: torch.Tensor,
        basal_values: torch.Tensor,
        basal_mask: torch.Tensor,
        *,
        action_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if actions.ndim not in (2, 3):
            raise ValueError("actions must be [B,A,F] or [B,F]")
        batch = actions.shape[0]
        if query_coordinates.ndim != 2 or query_coordinates.shape[1] != self.config.state_dim:
            raise ValueError("query_coordinates must be [Q,state_dim]")
        if not torch.isfinite(query_coordinates).all():
            raise ValueError("query_coordinates must be finite")
        queries = query_coordinates.shape[0]
        control_mean = self._query_matrix(control_mean, batch, queries, "control_mean")
        observation_scale = self._query_matrix(
            observation_scale, batch, queries, "observation_scale"
        )
        if not (observation_scale > 0).all():
            raise ValueError("observation_scale must be positive")
        if delta_amplitude.shape != (queries,) or not torch.isfinite(delta_amplitude).all():
            raise ValueError("delta_amplitude must be finite [Q]")
        if not (delta_amplitude > 0).all():
            raise ValueError("delta_amplitude must be positive")
        basal_state = self._encode_basal(batch, basal_features, basal_values, basal_mask)
        action_state, has_action = self._encode_actions(actions, action_mask)
        raw_delta = action_state + self.transition(torch.cat((basal_state, action_state), -1))
        intervention_delta = torch.where(
            has_action[:, None], raw_delta, torch.zeros_like(raw_delta)
        )
        state = basal_state + intervention_delta
        standardized_delta = self.mean_state(state) @ query_coordinates.T / math.sqrt(
            self.config.state_dim
        )
        molecular_delta = torch.where(
            has_action[:, None],
            delta_amplitude[None, :] * standardized_delta,
            torch.zeros_like(control_mean),
        )
        mean = torch.where(has_action[:, None], control_mean + molecular_delta, control_mean)
        return {
            "mean": mean,
            "delta": molecular_delta,
            "state": state,
            "basal_state": basal_state,
            "intervention_delta": intervention_delta,
        }
