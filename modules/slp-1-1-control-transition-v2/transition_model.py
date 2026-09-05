"""Minimal control-anchored transition with shared decoder amplitude.

This module deliberately matches the original transition candidate's action,
context, query, and transition topology. It adds only algebraic empty-action
gates and separates decoder amplitude from observation uncertainty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    action_feature_dim: int
    query_feature_dim: int
    hidden_dim: int = 128
    state_dim: int = 64
    dropout: float = 0.1


class MinimalControlTransition(nn.Module):
    """Predict query-wise deltas around an externally supplied control mean."""

    def __init__(self, config: Config):
        super().__init__()
        if min(
            config.action_feature_dim,
            config.query_feature_dim,
            config.hidden_dim,
            config.state_dim,
        ) <= 0:
            raise ValueError("model dimensions must be positive")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.config = config
        h, s = config.hidden_dim, config.state_dim
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_feature_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, s),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(config.query_feature_dim + 1, h),
            nn.GELU(),
            nn.Linear(h, s),
        )
        self.transition = nn.Sequential(
            nn.Linear(2 * s, h),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, s),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(config.query_feature_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, s),
        )
        self.mean_state = nn.Linear(s, s, bias=False)
        nn.init.normal_(self.mean_state.weight, std=0.002)

    @staticmethod
    def _query_matrix(
        values: torch.Tensor, batch: int, queries: int, label: str
    ) -> torch.Tensor:
        if values.ndim == 1 and values.shape[0] == queries:
            values = values.unsqueeze(0).expand(batch, -1)
        if values.shape != (batch, queries):
            raise ValueError(f"{label} must be [Q] or [B,Q]")
        if not torch.isfinite(values).all():
            raise ValueError(f"{label} must be finite")
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
            or features.shape[2] != self.config.query_feature_dim
            or values.shape != features.shape[:2]
            or mask.shape != features.shape[:2]
            or mask.dtype != torch.bool
        ):
            raise ValueError("basal feature, value, and mask shapes disagree")
        safe_features = torch.where(mask[..., None], features, torch.zeros_like(features))
        safe_values = torch.where(mask, values, torch.zeros_like(values))
        if not torch.isfinite(safe_features).all() or not torch.isfinite(
            safe_values
        ).all():
            raise ValueError("unmasked basal entries must be finite")
        encoded = self.context_encoder(
            torch.cat((safe_features, safe_values[..., None]), dim=-1)
        )
        total = (encoded * mask[..., None]).sum(1)
        return total / mask.sum(1, keepdim=True).clamp_min(1)

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
        queries: torch.Tensor,
        control_mean: torch.Tensor,
        delta_amplitude: torch.Tensor,
        observation_scale: torch.Tensor,
        basal_features: torch.Tensor,
        basal_values: torch.Tensor,
        basal_mask: torch.Tensor,
        *,
        action_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return a diagonal aggregate Gaussian and explicit latent deltas.

        ``delta_amplitude`` is intentionally restricted to ``[Q]``. The same
        per-query amplitude must therefore be used for every batch/context.
        Observation scales may vary by record but do not enter the mean/state.
        """
        if actions.ndim not in (2, 3):
            raise ValueError("actions must be [B,A,F] or [B,F]")
        batch = actions.shape[0]
        if queries.ndim != 2 or queries.shape[1] != self.config.query_feature_dim:
            raise ValueError("queries must be [Q,Fq]")
        if not torch.isfinite(queries).all():
            raise ValueError("queries must be finite")
        query_count = queries.shape[0]
        control_mean = self._query_matrix(control_mean, batch, query_count, "control_mean")
        observation_scale = self._query_matrix(
            observation_scale, batch, query_count, "observation_scale"
        )
        if not (observation_scale > 0).all():
            raise ValueError("observation_scale must be positive")
        if delta_amplitude.shape != (query_count,) or not torch.isfinite(
            delta_amplitude
        ).all():
            raise ValueError("delta_amplitude must be one finite shared [Q] vector")
        if not (delta_amplitude > 0).all():
            raise ValueError("delta_amplitude must be positive")

        basal_state = self._encode_basal(
            batch, basal_features, basal_values, basal_mask
        )
        action_state, has_action = self._encode_actions(actions, action_mask)
        raw_intervention = action_state + self.transition(
            torch.cat((basal_state, action_state), dim=-1)
        )
        intervention_delta = torch.where(
            has_action[:, None], raw_intervention, torch.zeros_like(raw_intervention)
        )
        state = basal_state + intervention_delta
        query_state = self.query_encoder(queries)
        standardized_delta = (
            self.mean_state(state) @ query_state.T
            / math.sqrt(self.config.state_dim)
        )
        molecular_delta = torch.where(
            has_action[:, None],
            delta_amplitude[None, :] * standardized_delta,
            torch.zeros_like(control_mean),
        )
        mean = torch.where(
            has_action[:, None], control_mean + molecular_delta, control_mean
        )
        return {
            "mean": mean,
            "scale": observation_scale,
            "delta": molecular_delta,
            "state": state,
            "basal_state": basal_state,
            "intervention_delta": intervention_delta,
        }


def gaussian_loss(
    prediction: dict[str, torch.Tensor],
    target: torch.Tensor,
    observed: torch.Tensor,
    row_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean diagonal-Gaussian NLL with optional fixed global row weights."""
    if target.shape != observed.shape or observed.dtype != torch.bool:
        raise ValueError("targets and observation masks disagree")
    if target.shape != prediction["mean"].shape:
        raise ValueError("target and prediction shapes disagree")
    safe_target = torch.where(observed, target, torch.zeros_like(target))
    if not torch.isfinite(safe_target).all():
        raise ValueError("unmasked targets must be finite")
    counts = observed.sum(1)
    if not (counts > 0).all():
        raise ValueError("each record requires an observed target")
    error = torch.where(
        observed, safe_target - prediction["mean"], torch.zeros_like(target)
    )
    variance = prediction["scale"].square().clamp_min(1e-8)
    terms = torch.where(
        observed,
        0.5 * (math.log(2 * math.pi) + variance.log() + error.square() / variance),
        torch.zeros_like(target),
    )
    per_record = terms.sum(1) / counts
    if row_weight is None:
        return per_record.mean()
    weights = torch.as_tensor(row_weight, device=per_record.device)
    if (
        weights.shape != per_record.shape
        or not torch.is_floating_point(weights)
        or not torch.isfinite(weights).all()
        or not (weights > 0).all()
    ):
        raise ValueError("row_weight must be one finite positive weight per record")
    # Weights have global mean one by construction.  Dividing by minibatch size
    # preserves that fixed objective instead of renormalizing each minibatch.
    return (per_record * weights.to(per_record.dtype)).mean()
