"""The frozen v1 intervention-conditioned cellular world model.

This module deliberately depends only on PyTorch.  It does not read data,
know benchmark names, train a decoder, or score synthetic lethality.
"""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from .contracts import ActionSet, WorldContext, WorldPrediction


MODEL_VERSION = "v1"


@dataclass(frozen=True)
class WorldModelConfig:
    d: int = 384
    latent: int = 128
    layers: int = 6
    contexts: int = 28
    outcomes: int = 2
    state_dim: int = 1816
    context_dim: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class SLPredict(nn.Module):
    """v1 state transition model for encoded genes and intervention sets.

    The existing ``encode`` and ``transition`` methods are preserved for
    checkpoint compatibility.  ``rollout`` is the stable public interface for
    a stochastic prediction and never applies an application-level score.
    """

    def __init__(
        self,
        d: int = 384,
        latent: int = 128,
        layers: int = 6,
        contexts: int = 28,
        outcomes: int = 2,
        state_dim: int = 1816,
        context_dim: int = 0,
    ):
        super().__init__()
        self.config = WorldModelConfig(
            d=d,
            latent=latent,
            layers=layers,
            contexts=contexts,
            outcomes=outcomes,
            state_dim=state_dim,
            context_dim=context_dim,
        )
        anchor = (state_dim - 1624) // 6
        self.slices = (
            (0, 768),
            (768, 1024),
            (1024, 1424),
            (1424, 1624),
            *((1624 + i * anchor, 1624 + (i + 1) * anchor) for i in range(6)),
        )
        self.proj = nn.ModuleList(nn.Linear(hi - lo, d) for lo, hi in self.slices)
        self.cls = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.basal = nn.Parameter(torch.zeros(d))
        self.context = nn.Parameter(torch.zeros(d))
        self.cell = nn.Embedding(contexts, d)
        self.context_proj = nn.Linear(context_dim, d, bias=False) if context_dim else None
        self.time = nn.Parameter(torch.zeros(d))
        self.state_up = nn.Linear(latent, d)
        self.token_type = nn.Embedding(5, d)
        layer = nn.TransformerEncoderLayer(
            d,
            6,
            4 * d,
            0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.core = nn.TransformerEncoder(layer, layers, nn.LayerNorm(d))
        self.gene = nn.Linear(d, latent)
        self.dist = nn.Linear(d, 2 * latent)
        self.reconstruct = nn.Linear(latent, state_dim)
        self.decode_state = nn.Linear(latent, 6 * anchor)
        self.relation = nn.Sequential(nn.Linear(3 * latent, 256), nn.GELU(), nn.Linear(256, 6))
        self.outcome = nn.Linear(latent, outcomes)

    def encode(self, x: torch.Tensor, mask: float = 0.0) -> torch.Tensor:
        tok = torch.stack([p(x[:, lo:hi]) for p, (lo, hi) in zip(self.proj, self.slices)], 1)
        if self.training and mask:
            tok = tok * (torch.rand(tok.shape[:2], device=x.device) > mask)[..., None]
        cls = self.cls[None, None].expand(len(x), 1, -1) + self.token_type.weight[0]
        return self.gene(self.core(torch.cat((cls, tok + self.token_type.weight[1]), 1))[:, 0])

    def transition(
        self,
        action: torch.Tensor,
        second: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = len(action)
        start = self.basal[None].expand(batch, -1) if state is None else self.state_up(state)
        sequence = [start + self.token_type.weight[2], self.state_up(action) + self.token_type.weight[3]]
        if second is not None:
            sequence.append(self.state_up(second) + self.token_type.weight[3])
        if context_state is not None:
            if self.context_proj is None:
                raise ValueError("This checkpoint does not accept encoded context features.")
            ctx = self.context[None] + self.context_proj(context_state)
        elif context is None:
            ctx = self.context[None].expand(batch, -1)
        else:
            ctx = self.cell(context.clamp_min(0))
            ctx = torch.where((context >= 0)[:, None], ctx, self.context[None])
        sequence += [self.time[None].expand(batch, -1) + self.token_type.weight[4], ctx]
        hidden = self.core(torch.stack(sequence, 1))[:, 0]
        mean, log_std = self.dist(hidden).chunk(2, 1)
        return mean, log_std.clamp(-5, 2)

    def transition_set(
        self,
        actions: torch.Tensor,
        mask: torch.Tensor,
        state: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = len(actions)
        start = self.basal[None].expand(batch, -1) if state is None else self.state_up(state)
        action_tokens = self.state_up(actions) + self.token_type.weight[3]
        if context_state is not None:
            if self.context_proj is None:
                raise ValueError("This checkpoint does not accept encoded context features.")
            ctx = self.context[None] + self.context_proj(context_state)
        elif context is None:
            ctx = self.context[None].expand(batch, -1)
        else:
            ctx = torch.where(
                (context >= 0)[:, None], self.cell(context.clamp_min(0)), self.context[None]
            )
        sequence = torch.cat(
            (
                start[:, None] + self.token_type.weight[2],
                action_tokens,
                self.time[None, None].expand(batch, 1, -1) + self.token_type.weight[4],
                ctx[:, None],
            ),
            1,
        )
        padding = torch.cat(
            (
                torch.zeros((batch, 1), device=mask.device, dtype=torch.bool),
                ~mask,
                torch.zeros((batch, 2), device=mask.device, dtype=torch.bool),
            ),
            1,
        )
        hidden = self.core(sequence, src_key_padding_mask=padding)[:, 0]
        mean, log_std = self.dist(hidden).chunk(2, 1)
        return mean, log_std.clamp(-5, 2)

    def rollout(
        self,
        actions: ActionSet,
        *,
        state: torch.Tensor | None = None,
        context: WorldContext | None = None,
    ) -> WorldPrediction:
        """Simulate a set of encoded interventions without applying a score."""

        context_ids = None if context is None else context.context_ids
        context_features = None if context is None else context.features
        mean, log_std = self.transition_set(
            actions.actions,
            actions.mask,
            state=state,
            context=context_ids,
            context_state=context_features,
        )
        return WorldPrediction(
            mean=mean,
            log_std=log_std,
            model_version=MODEL_VERSION,
            metadata={"action_count": actions.mask.sum(1).detach().cpu().tolist()},
        )

    def relation_score(self, a: torch.Tensor, b: torch.Tensor, joint: torch.Tensor) -> torch.Tensor:
        return self.relation(torch.cat((a * b, (a - b).abs(), joint), 1))

    def count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
