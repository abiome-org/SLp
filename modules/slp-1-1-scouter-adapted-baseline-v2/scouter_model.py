"""LayerNorm Scouter compressor-generator adapted to stable static features.

The architecture follows Zhu and Li's Scouter implementation, while accepting
caller-supplied continuous action features instead of an embedding lookup. The
fixed output panel and pseudobulk Gaussian objective are study adaptations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Config:
    query_dim: int
    action_feature_dim: int
    control_hidden: tuple[int, ...] = (2048, 512)
    control_state_dim: int = 64
    generator_hidden: tuple[int, ...] = (2048,)
    batch_norm: bool = False
    layer_norm: bool = True
    dropout: float = 0.0

    def __post_init__(self) -> None:
        dimensions = (
            self.query_dim,
            self.action_feature_dim,
            self.control_state_dim,
            *self.control_hidden,
            *self.generator_hidden,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("all model dimensions must be positive integers")
        if self.batch_norm and self.layer_norm:
            raise ValueError("Scouter uses at most one normalization family")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")


def _mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    *,
    batch_norm: bool,
    layer_norm: bool,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(input_dim, hidden_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SELU())
        if dropout > 0.0:
            layers.append(nn.AlphaDropout(dropout))
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class ScouterAdaptedBaseline(nn.Module):
    """Control compressor plus action-conditioned full-panel generator."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        common = {
            "batch_norm": config.batch_norm,
            "layer_norm": config.layer_norm,
            "dropout": config.dropout,
        }
        self.control_encoder = _mlp(
            config.query_dim,
            config.control_hidden,
            config.control_state_dim,
            **common,
        )
        self.generator = _mlp(
            config.control_state_dim + config.action_feature_dim,
            config.generator_hidden,
            config.query_dim,
            **common,
        )

    def forward(
        self,
        action_features: Tensor,
        control_expression: Tensor,
        action_mask: Tensor | None = None,
    ) -> Tensor:
        if action_features.ndim != 3:
            raise ValueError("action_features must have shape [batch, actions, features]")
        batch, actions, features = action_features.shape
        if features != self.config.action_feature_dim:
            raise ValueError("action feature dimension mismatch")
        if control_expression.shape != (batch, self.config.query_dim):
            raise ValueError("control_expression shape mismatch")
        if action_mask is None:
            action_mask = torch.ones(
                (batch, actions), dtype=torch.bool, device=action_features.device
            )
        if action_mask.shape != (batch, actions) or action_mask.dtype != torch.bool:
            raise ValueError("action_mask must be boolean with shape [batch, actions]")
        safe_actions = torch.where(
            action_mask.unsqueeze(-1), action_features, torch.zeros_like(action_features)
        )
        if not torch.isfinite(safe_actions).all() or not torch.isfinite(
            control_expression
        ).all():
            raise ValueError("unmasked action and control inputs must be finite")
        action_sum = safe_actions.sum(dim=1)
        control_state = self.control_encoder(control_expression)
        return self.generator(torch.cat((action_sum, control_state), dim=-1))


def gaussian_loss(
    prediction: Tensor,
    target: Tensor,
    observed: Tensor,
    scale: Tensor,
) -> Tensor:
    """Uniform-row fixed-scale Gaussian NLL used by the matched comparison."""

    if (
        prediction.ndim != 2
        or target.shape != prediction.shape
        or observed.shape != prediction.shape
        or scale.shape != prediction.shape
        or observed.dtype != torch.bool
    ):
        raise ValueError("Gaussian loss inputs must be aligned [batch, query] tensors")
    safe_target = torch.where(observed, target, prediction.detach())
    safe_scale = torch.where(observed, scale, torch.ones_like(scale))
    if (
        not torch.isfinite(prediction[observed]).all()
        or not torch.isfinite(safe_target).all()
        or not torch.isfinite(safe_scale).all()
        or torch.any(safe_scale <= 0.0)
    ):
        raise ValueError("observed Gaussian inputs must be finite with positive scale")
    count = observed.sum(dim=1)
    if torch.any(count == 0):
        raise ValueError("every training row must contain an observed query")
    nll = 0.5 * (
        math.log(2.0 * math.pi)
        + 2.0 * torch.log(safe_scale)
        + ((prediction - safe_target) / safe_scale).square()
    )
    return (torch.where(observed, nll, 0.0).sum(dim=1) / count).mean()
