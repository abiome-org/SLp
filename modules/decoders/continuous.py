"""Continuous application decoders, intentionally outside ``model/v1``.

These modules translate a frozen simulated latent state into assay-specific
fitness or depletion targets. They are not part of a world-model checkpoint.
"""

import torch
from torch import nn


def tolerance_head(latent: int) -> nn.Module:
    return nn.Sequential(nn.LayerNorm(latent), nn.Linear(latent, latent), nn.GELU(), nn.Linear(latent, 1))


def interaction_head(latent: int) -> nn.Module:
    return nn.Sequential(nn.LayerNorm(latent), nn.Linear(latent, latent), nn.GELU(), nn.Linear(latent, 2))


def interaction_depletion_head(latent: int) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(5 * latent),
        nn.Linear(5 * latent, 2 * latent),
        nn.GELU(),
        nn.Linear(2 * latent, 1),
    )


class ResidualInteraction(nn.Module):
    """A small assay decoder correction over a frozen continuous decoder."""

    def __init__(self, base_head: nn.Module, dim: int = 128):
        super().__init__()
        self.base_head = base_head.requires_grad_(False)
        self.correction = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 2)
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, latent: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.base_head(latent) + self.correction(residual)


class RankedResidualInteraction(nn.Module):
    """Optional ranking correction over an otherwise frozen assay decoder."""

    def __init__(self, base_head: nn.Module, dim: int = 128):
        super().__init__()
        self.base_head = base_head.requires_grad_(False)
        self.rank = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 16), nn.GELU(), nn.Linear(16, 1)
        )
        nn.init.zeros_(self.rank[-1].weight)
        nn.init.zeros_(self.rank[-1].bias)

    def forward(self, latent: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        base = self.base_head(latent, residual)
        return torch.cat((base[:, :1] + self.rank(residual), base[:, 1:]), 1)


class DependencyInteraction(nn.Module):
    """A dependency-state correction over a frozen continuous decoder."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.correction = nn.Sequential(
            nn.LayerNorm(32), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 2)
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(
        self, latent: torch.Tensor, residual: torch.Tensor, dependency: torch.Tensor
    ) -> torch.Tensor:
        return self.base(latent, residual) + self.correction(dependency)
