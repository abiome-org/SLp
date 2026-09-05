"""Feature-conditioned molecular transition with optional shared uncertainty.

Identifiers and fitting references are supplied by the caller, never learned
as a gene vocabulary. Queries decode independently from the same latent state.
The Gaussian factor output defines a coherent low-rank joint distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    feature_dim: int
    hidden: int = 128
    state_dim: int = 64
    covariance_rank: int = 0
    dropout: float = 0.1
    learn_scale: bool = True
    query_feature_dim: int | None = None


class TransitionWorld(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        h, s = config.hidden, config.state_dim
        qf = config.query_feature_dim or config.feature_dim
        if min(config.feature_dim, h, s) <= 0 or config.covariance_rank < 0:
            raise ValueError("invalid model dimensions")
        self.action_encoder = nn.Sequential(
            nn.Linear(config.feature_dim, h), nn.LayerNorm(h), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(h, s),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(qf + 1, h), nn.GELU(), nn.Linear(h, s),
        )
        self.transition = nn.Sequential(
            nn.Linear(2 * s, h), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(h, s),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(qf, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, s),
        )
        self.mean_state = nn.Linear(s, s, bias=False)
        nn.init.normal_(self.mean_state.weight, std=0.002)
        self.scale_state = nn.Linear(s, s, bias=False)
        nn.init.zeros_(self.scale_state.weight)
        if not config.learn_scale:
            self.scale_state.requires_grad_(False)
        if config.covariance_rank:
            self.factor_query = nn.Linear(s, config.covariance_rank, bias=False)
            nn.init.normal_(self.factor_query.weight, std=0.005)
            self.factor_state = nn.Linear(s, config.covariance_rank)

    def forward(
        self, actions: torch.Tensor, queries: torch.Tensor,
        reference: torch.Tensor, reference_scale: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        context_features: torch.Tensor | None = None,
        context_values: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if actions.ndim == 2:
            actions = actions[:, None, :]
        if actions.ndim != 3 or queries.ndim != 2:
            raise ValueError("actions must be [B,A,F], queries [Q,F]")
        if action_mask is None:
            action_mask = torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
        if action_mask.shape != actions.shape[:2] or not action_mask.any(1).all():
            raise ValueError("each record requires a nonempty action set")
        actions = torch.where(action_mask[..., None], actions, torch.zeros_like(actions))
        encoded = self.action_encoder(actions)
        action_state = (encoded * action_mask[..., None]).sum(1)
        context_state = torch.zeros_like(action_state)
        if context_features is not None:
            if context_values is None or context_mask is None:
                raise ValueError("context values and mask must accompany features")
            if context_features.shape[:2] != context_values.shape or context_values.shape != context_mask.shape:
                raise ValueError("context shapes disagree")
            safe_values = torch.where(context_mask, context_values, torch.zeros_like(context_values))
            safe_features = torch.where(context_mask[..., None], context_features, torch.zeros_like(context_features))
            context = self.context_encoder(torch.cat((safe_features, safe_values[..., None]), dim=-1))
            context_state = (context * context_mask[..., None]).sum(1)
            context_state = context_state / context_mask.sum(1, keepdim=True).clamp_min(1)
        state = action_state + self.transition(torch.cat((context_state, action_state), dim=-1))
        query = self.query_encoder(queries)
        residual = self.mean_state(state) @ query.T / math.sqrt(self.config.state_dim)
        log_scale_ratio = (self.scale_state(state) @ query.T / math.sqrt(self.config.state_dim)).clamp(-2.0, 2.0)
        mean = reference + reference_scale * residual
        scale = reference_scale * log_scale_ratio.exp()
        result = {"mean": mean, "scale": scale, "state": state}
        if self.config.covariance_rank:
            factor = self.factor_query(query)[None, :, :] * torch.sigmoid(self.factor_state(state))[:, None, :]
            result["factor"] = reference_scale.unsqueeze(-1) * factor
        return result


def gaussian_loss(prediction: dict[str, torch.Tensor], target: torch.Tensor,
                  observed: torch.Tensor, *, joint: bool = False) -> torch.Tensor:
    """Mean record NLL per observed target; missing values never enter arithmetic."""
    if target.shape != observed.shape or observed.dtype != torch.bool:
        raise ValueError("targets and observation masks disagree")
    counts = observed.sum(1)
    if not (counts > 0).all():
        raise ValueError("each record needs an observed target")
    error = torch.where(observed, target - prediction["mean"], torch.zeros_like(target))
    variance = prediction["scale"].square().clamp_min(1e-8)
    inverse = observed / variance
    logdet = torch.where(observed, variance.log(), torch.zeros_like(variance)).sum(1)
    quadratic = (error.square() * inverse).sum(1)
    if joint and "factor" in prediction:
        factor = prediction["factor"]
        rank = factor.shape[-1]
        identity = torch.eye(rank, device=target.device, dtype=target.dtype)
        precision = identity + factor.transpose(1, 2) @ (inverse[..., None] * factor)
        chol = torch.linalg.cholesky(precision)
        rhs = factor.transpose(1, 2) @ (error * inverse)[..., None]
        quadratic = quadratic - (rhs * torch.cholesky_solve(rhs, chol)).sum((1, 2))
        logdet = logdet + 2 * torch.diagonal(chol, dim1=-2, dim2=-1).log().sum(1)
    terms = 0.5 * (counts * math.log(2 * math.pi) + logdet + quadratic)
    return (terms / counts).mean()
