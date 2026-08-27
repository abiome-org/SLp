"""Data-free public types for the v1 world-model API."""

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class Perturbation:
    """A declarative intervention specification.

    v1 training currently provides encoded gene actions.  The identifiers and
    metadata are deliberately retained here so future action encoders can be
    added without changing the rollout contract.
    """

    target: str
    mode: str = "knockout"
    dose: float | None = None
    time_hours: float | None = None


@dataclass(frozen=True)
class WorldContext:
    """Optional encoded basal context passed to a rollout."""

    features: torch.Tensor | None = None
    context_ids: torch.Tensor | None = None


@dataclass(frozen=True)
class ActionSet:
    """Encoded perturbation actions and their validity mask.

    `actions` has shape ``[batch, actions, latent]`` and `mask` has shape
    ``[batch, actions]``.  The model is permutation-invariant over valid
    action positions when using :meth:`SLPredict.transition_set`.
    """

    actions: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True)
class WorldPrediction:
    """Distribution over a simulated endpoint state."""

    mean: torch.Tensor
    log_std: torch.Tensor
    model_version: str
    metadata: Mapping[str, object] | None = None

    @property
    def std(self) -> torch.Tensor:
        return self.log_std.exp()

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(
            self.mean.shape,
            device=self.mean.device,
            dtype=self.mean.dtype,
            generator=generator,
        )
        return self.mean + self.std * noise
