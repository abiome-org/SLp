"""Permutation-invariant operator over observable molecular states and actions.

The input state is an observed endpoint representation, not a claim about a
temporally ordered biological state.  Callers own feature standardization and
the construction of matched control, single, and double endpoint examples.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    state_dim: int = 32
    action_dim: int = 577
    width: int = 64
    latent: int = 32
    layers: int = 2
    heads: int = 4
    dropout: float = 0.0
    zero_init_delta: bool = True


class CompositionalStateOperator(nn.Module):
    """Predict a residual update from a molecular state and an action set.

    Action order is deliberately unrepresented.  Each action contributes one
    token per fixed modality (ESM, coverage presence, and GO), and padding is
    supplied only through ``mask``.  Thus two active actions are exchangeable.
    """

    ESM = slice(0, 320)
    PRESENCE = slice(320, 321)
    GO = slice(321, 577)

    def __init__(self, config: Config = Config()):
        super().__init__()
        if config.state_dim != 32 or config.action_dim != 577:
            raise ValueError("v1 requires state_dim=32 and action_dim=577")
        if min(config.width, config.latent, config.layers, config.heads) <= 0:
            raise ValueError("model dimensions must be positive")
        if config.width % config.heads:
            raise ValueError("width must be divisible by heads")
        self.config = config
        w = config.width
        self.state_projection = nn.Linear(config.state_dim, w)
        self.modality_projection = nn.ModuleList(
            (nn.Linear(320, w), nn.Linear(1, w), nn.Linear(256, w))
        )
        self.modality_embedding = nn.Parameter(torch.zeros(3, w))
        self.state_embedding = nn.Parameter(torch.zeros(w))
        layer = nn.TransformerEncoderLayer(
            d_model=w,
            nhead=config.heads,
            dim_feedforward=4 * w,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.core = nn.TransformerEncoder(layer, config.layers, nn.LayerNorm(w))
        self.delta_head = nn.Sequential(
            nn.Linear(w, config.latent), nn.GELU(), nn.Linear(config.latent, config.state_dim)
        )
        if config.zero_init_delta:
            nn.init.zeros_(self.delta_head[-1].weight)
            nn.init.zeros_(self.delta_head[-1].bias)

    def forward(
        self, state: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.config.state_dim:
            raise ValueError("state must be [B,32]")
        if actions.ndim != 3 or actions.shape[0] != state.shape[0] or actions.shape[1:] != (2, 577):
            raise ValueError("actions must be [B,2,577]")
        if mask.shape != actions.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean [B,2]")

        state_token = self.state_projection(state) + self.state_embedding
        pieces = (actions[..., self.ESM], actions[..., self.PRESENCE], actions[..., self.GO])
        modality_tokens = [
            projection(values) + self.modality_embedding[index]
            for index, (projection, values) in enumerate(zip(self.modality_projection, pieces))
        ]
        # [B, action, modality, width], flattened without positional encodings.
        action_tokens = torch.stack(modality_tokens, dim=2).flatten(1, 2)
        token_mask = mask[..., None].expand(-1, -1, 3).reshape(mask.shape[0], -1)
        action_tokens = torch.where(token_mask[..., None], action_tokens, torch.zeros_like(action_tokens))
        padding = torch.cat(
            (torch.zeros((len(state), 1), dtype=torch.bool, device=state.device), ~token_mask), dim=1
        )
        encoded = self.core(torch.cat((state_token[:, None], action_tokens), dim=1),
                            src_key_padding_mask=padding)[:, 0]
        delta = self.delta_head(encoded)
        # This gate makes the empty action set an exact identity independently
        # of learned biases or normalization behavior.
        active = mask.any(1, keepdim=True).to(delta.dtype)
        return state + active * delta


Operator = CompositionalStateOperator
