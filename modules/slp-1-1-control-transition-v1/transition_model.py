"""Control-anchored molecular transition with an algebraic empty-set identity.

The model consumes feature vectors supplied by the caller.  It has no learned
gene, assay, perturbation-mode, or species identifiers.  Queries decode
independently from a shared state, and the likelihood is diagonal Gaussian over
aggregate molecular measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    """Dimensions for :class:`ControlTransition`.

    ``assay_feature_dim`` covers any caller-defined assay, perturbation-mode,
    species, or context descriptors.  They are ordinary supplied features,
    never integer IDs embedded by this module.
    """

    action_feature_dim: int
    query_feature_dim: int
    assay_feature_dim: int = 0
    hidden_dim: int = 128
    state_dim: int = 64
    dropout: float = 0.1
    learn_scale: bool = True


class ControlTransition(nn.Module):
    """Predict a molecular delta relative to a supplied control baseline.

    ``state = basal_state + intervention_delta``.  The intervention delta is
    multiplied by an exact Boolean presence gate.  Rows with no valid action
    consequently return zero latent and molecular deltas and select the
    supplied control mean and scale directly.
    """

    def __init__(self, config: Config):
        super().__init__()
        dimensions = (
            config.action_feature_dim,
            config.query_feature_dim,
            config.hidden_dim,
            config.state_dim,
        )
        if min(dimensions) <= 0:
            raise ValueError("model dimensions must be positive")
        if config.assay_feature_dim < 0:
            raise ValueError("assay_feature_dim cannot be negative")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.config = config
        h = config.hidden_dim
        s = config.state_dim
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_feature_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, s),
        )
        self.basal_encoder = nn.Sequential(
            nn.Linear(config.query_feature_dim + 1, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, s),
        )
        self.assay_encoder = (
            nn.Sequential(
                nn.Linear(config.assay_feature_dim, h),
                nn.GELU(),
                nn.Linear(h, s),
            )
            if config.assay_feature_dim
            else None
        )
        self.intervention_input = nn.Linear(2 * s, h)
        self.interaction_projection = nn.Linear(s, h, bias=False)
        self.intervention_output = nn.Sequential(
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, s),
        )
        self.response_state = nn.Sequential(
            nn.Linear(2 * s, h),
            nn.GELU(),
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
        self.scale_state = nn.Linear(s, s, bias=False)
        nn.init.zeros_(self.scale_state.weight)
        if not config.learn_scale:
            self.scale_state.requires_grad_(False)

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

    def _basal_state(
        self,
        batch: int,
        basal_features: torch.Tensor,
        basal_values: torch.Tensor,
        basal_mask: torch.Tensor,
        assay_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if basal_features.ndim == 2:
            basal_features = basal_features.unsqueeze(0).expand(batch, -1, -1)
        if (
            basal_features.ndim != 3
            or basal_features.shape[0] != batch
            or basal_features.shape[2] != self.config.query_feature_dim
            or basal_values.shape != basal_features.shape[:2]
            or basal_mask.shape != basal_features.shape[:2]
            or basal_mask.dtype != torch.bool
        ):
            raise ValueError("basal features, values, and mask have incompatible shapes")
        safe_values = torch.where(
            basal_mask, basal_values, torch.zeros_like(basal_values)
        )
        safe_features = torch.where(
            basal_mask[..., None], basal_features, torch.zeros_like(basal_features)
        )
        if not torch.isfinite(safe_values).all() or not torch.isfinite(
            safe_features
        ).all():
            raise ValueError("unmasked basal entries must be finite")
        tokens = self.basal_encoder(
            torch.cat((safe_features, safe_values[..., None]), dim=-1)
        )
        token_sum = (tokens * basal_mask[..., None]).sum(1)
        count = basal_mask.sum(1, keepdim=True)
        state = token_sum / count.clamp_min(1)
        if self.assay_encoder is None:
            if assay_features is not None:
                raise ValueError("assay features supplied to a zero-dimension config")
        else:
            expected = (batch, self.config.assay_feature_dim)
            if assay_features is None or assay_features.shape != expected:
                raise ValueError(f"assay_features must be {expected}")
            if not torch.isfinite(assay_features).all():
                raise ValueError("assay_features must be finite")
            state = state + self.assay_encoder(assay_features)
        return state

    def _action_summary(
        self, actions: torch.Tensor, action_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if actions.ndim == 2:
            actions = actions[:, None, :]
        if actions.ndim != 3 or actions.shape[2] != self.config.action_feature_dim:
            raise ValueError("actions must be [B,A,F] or [B,F]")
        if action_mask is None:
            action_mask = torch.ones(
                actions.shape[:2], dtype=torch.bool, device=actions.device
            )
        if action_mask.shape != actions.shape[:2] or action_mask.dtype != torch.bool:
            raise ValueError("action_mask must be Boolean [B,A]")
        safe_actions = torch.where(
            action_mask[..., None], actions, torch.zeros_like(actions)
        )
        if not torch.isfinite(safe_actions).all():
            raise ValueError("unmasked action entries must be finite")
        encoded = self.action_encoder(safe_actions)
        masked = encoded * action_mask[..., None]
        count = action_mask.sum(1, keepdim=True)
        total = masked.sum(1)
        mean = total / count.clamp_min(1)

        # Average elementwise product across distinct ordered token pairs.
        # This is symmetric in token order and is exactly zero for <2 actions.
        pair_numerator = total.square() - masked.square().sum(1)
        pair_denominator = count * (count - 1)
        pair_mean = torch.where(
            count > 1,
            pair_numerator / pair_denominator.clamp_min(1),
            torch.zeros_like(pair_numerator),
        )
        return mean, pair_mean, count.squeeze(1)

    def forward(
        self,
        actions: torch.Tensor,
        queries: torch.Tensor,
        control_mean: torch.Tensor,
        control_scale: torch.Tensor,
        basal_features: torch.Tensor,
        basal_values: torch.Tensor,
        basal_mask: torch.Tensor,
        *,
        action_mask: torch.Tensor | None = None,
        assay_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return aggregate Gaussian parameters and explicit transition state."""
        if actions.ndim not in (2, 3):
            raise ValueError("actions must be [B,A,F] or [B,F]")
        batch = actions.shape[0]
        if queries.ndim != 2 or queries.shape[1] != self.config.query_feature_dim:
            raise ValueError("queries must be [Q,Fq]")
        if not torch.isfinite(queries).all():
            raise ValueError("queries must be finite")
        query_count = queries.shape[0]
        control_mean = self._query_matrix(
            control_mean, batch, query_count, "control_mean"
        )
        control_scale = self._query_matrix(
            control_scale, batch, query_count, "control_scale"
        )
        if not (control_scale > 0).all():
            raise ValueError("control_scale must be positive")

        basal_state = self._basal_state(
            batch,
            basal_features,
            basal_values,
            basal_mask,
            assay_features,
        )
        action_mean, action_pair_mean, action_count = self._action_summary(
            actions, action_mask
        )
        has_action = action_count > 0
        intervention_hidden = self.intervention_input(
            torch.cat((basal_state, action_mean), dim=-1)
        ) + self.interaction_projection(action_pair_mean)
        raw_intervention = self.intervention_output(
            intervention_hidden
        )
        intervention_delta = torch.where(
            has_action[:, None], raw_intervention, torch.zeros_like(raw_intervention)
        )
        state = basal_state + intervention_delta
        response = self.response_state(torch.cat((state, intervention_delta), dim=-1))
        query_state = self.query_encoder(queries)
        standardized_delta = (
            self.mean_state(response) @ query_state.T
            / math.sqrt(self.config.state_dim)
        )
        molecular_delta = torch.where(
            has_action[:, None],
            control_scale * standardized_delta,
            torch.zeros_like(control_mean),
        )
        proposed_mean = control_mean + molecular_delta
        mean = torch.where(has_action[:, None], proposed_mean, control_mean)

        log_scale_delta = (
            self.scale_state(response) @ query_state.T
            / math.sqrt(self.config.state_dim)
        ).clamp(-2.0, 2.0)
        proposed_scale = control_scale * log_scale_delta.exp()
        scale = torch.where(has_action[:, None], proposed_scale, control_scale)
        return {
            "mean": mean,
            "scale": scale,
            "delta": molecular_delta,
            "state": state,
            "basal_state": basal_state,
            "intervention_delta": intervention_delta,
            "action_count": action_count,
        }


def gaussian_loss(
    prediction: dict[str, torch.Tensor],
    target: torch.Tensor,
    observed: torch.Tensor,
) -> torch.Tensor:
    """Mean diagonal-Gaussian NLL; masked nonfinite targets are inert."""
    if target.shape != observed.shape or observed.dtype != torch.bool:
        raise ValueError("targets and observation masks disagree")
    if target.shape != prediction["mean"].shape:
        raise ValueError("targets and prediction shapes disagree")
    safe_target = torch.where(observed, target, torch.zeros_like(target))
    if not torch.isfinite(safe_target).all():
        raise ValueError("unmasked targets must be finite")
    count = observed.sum(1)
    if not (count > 0).all():
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
    return (terms.sum(1) / count).mean()
