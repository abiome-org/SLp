"""An intervention-conditioned endpoint state with separate observation heads.

No data loading, gene vocabulary, application scores or assay normalization
is hidden in this module. Callers supply quantitative controls and fixed
features. RNA and antibody channels have distinct measurement decoders.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    action_features: int
    rna_features: int
    protein_features: int
    hidden: int = 64
    state: int = 32
    decoder_hidden: int = 32
    dropout: float = 0.2


class ObservationHead(nn.Module):
    def __init__(self, features: int, config: Config):
        super().__init__()
        self.features = features
        self.context_encoder = nn.Sequential(
            nn.Linear(features + 1, config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, config.state),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(features, config.hidden),
            nn.LayerNorm(config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, config.decoder_hidden),
        )
        self.state_projection = nn.Linear(
            config.state, config.decoder_hidden, bias=False
        )
        self.readout = nn.Linear(config.decoder_hidden, 1, bias=False)
        nn.init.normal_(self.readout.weight, std=0.002)

    def encode_control(self, features, values, observed):
        if features.ndim != 2 or features.shape[1] != self.features:
            raise ValueError("control features must be [Q,F]")
        if values.ndim != 2 or values.shape[1] != len(features):
            raise ValueError("control values must be [B,Q]")
        if observed.shape != values.shape or observed.dtype != torch.bool:
            raise ValueError("control mask must be Boolean [B,Q]")
        expanded = features[None].expand(len(values), -1, -1)
        safe_features = torch.where(observed[..., None], expanded, 0.0)
        safe_values = torch.where(observed, values, 0.0)
        if (
            not torch.isfinite(safe_features).all()
            or not torch.isfinite(safe_values).all()
        ):
            raise ValueError("observed control entries must be finite")
        encoded = self.context_encoder(
            torch.cat((safe_features, safe_values[..., None]), -1)
        )
        pooled = (encoded * observed[..., None]).sum(1)
        return pooled / observed.sum(1, keepdim=True).clamp_min(1), observed.any(1)

    def decode(self, state, basal_state, queries, control_mean, amplitude):
        if queries.ndim != 2 or queries.shape[1] != self.features:
            raise ValueError("query features must be [Q,F]")
        if not torch.isfinite(queries).all():
            raise ValueError("query features must be finite")
        if (
            control_mean.shape != (len(state), len(queries))
            or not torch.isfinite(control_mean).all()
        ):
            raise ValueError("control mean must be finite [B,Q]")
        if (
            amplitude.shape != (len(queries),)
            or not torch.isfinite(amplitude).all()
            or not (amplitude > 0).all()
        ):
            raise ValueError("amplitude must be finite positive shared [Q]")
        q = self.query_encoder(queries)[None]
        after = torch.nn.functional.gelu(self.state_projection(state)[:, None] + q)
        before = torch.nn.functional.gelu(
            self.state_projection(basal_state)[:, None] + q
        )
        delta = self.readout(after - before).squeeze(-1) * amplitude[None]
        return {"mean": control_mean + delta, "delta": delta}


class PairedStateModel(nn.Module):
    """Shared endpoint transition, modality-specific controls and observations."""

    def __init__(self, config: Config):
        super().__init__()
        if (
            min(
                config.action_features,
                config.rna_features,
                config.protein_features,
                config.hidden,
                config.state,
                config.decoder_hidden,
            )
            <= 0
        ):
            raise ValueError("dimensions must be positive")
        if not 0 <= config.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.config = config
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_features, config.hidden),
            nn.LayerNorm(config.hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, config.state),
        )
        self.transition = nn.Sequential(
            nn.Linear(config.state * 2, config.hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, config.state),
        )
        self.heads = nn.ModuleDict(
            {
                "rna": ObservationHead(config.rna_features, config),
                "protein": ObservationHead(config.protein_features, config),
            }
        )

    def encode(self, actions, action_mask, controls):
        if actions.ndim != 3 or actions.shape[-1] != self.config.action_features:
            raise ValueError("actions must be [B,A,F]")
        if action_mask.shape != actions.shape[:2] or action_mask.dtype != torch.bool:
            raise ValueError("action mask must be Boolean [B,A]")
        safe = torch.where(action_mask[..., None], actions, 0.0)
        if not torch.isfinite(safe).all():
            raise ValueError("unmasked action features must be finite")
        action = (self.action_encoder(safe) * action_mask[..., None]).sum(1)
        states, availability = [], []
        for modality in self.heads:
            item = controls[modality]
            state, available = self.heads[modality].encode_control(
                item["features"],
                item["values"],
                item["observed"],
            )
            if len(state) != len(actions):
                raise ValueError("control and action batches must align")
            states.append(state * available[:, None])
            availability.append(available)
        count = torch.stack(availability).sum(0)
        if not (count > 0).all():
            raise ValueError(
                "each record requires at least one observed control modality"
            )
        basal = torch.stack(states).sum(0) / count[:, None]
        raw_delta = action + self.transition(torch.cat((basal, action), -1))
        delta = torch.where(action_mask.any(1)[:, None], raw_delta, 0.0)
        return {
            "basal_state": basal,
            "intervention_delta": delta,
            "state": basal + delta,
        }

    def observe(self, encoded, modality, queries, control_mean, amplitude):
        if modality not in self.heads:
            raise ValueError("unknown observation modality")
        return self.heads[modality].decode(
            encoded["state"],
            encoded["basal_state"],
            queries,
            control_mean,
            amplitude,
        )


def scaled_mse(prediction, target, observed, scale):
    if (
        target.shape != prediction.shape
        or observed.shape != target.shape
        or observed.dtype != torch.bool
    ):
        raise ValueError("prediction, target and Boolean mask must align")
    if (
        scale.shape != target.shape
        or not torch.isfinite(scale).all()
        or not (scale > 0).all()
    ):
        raise ValueError("loss scale must be finite positive [B,Q]")
    safe_target = torch.where(observed, target, 0.0)
    if not torch.isfinite(safe_target).all() or not torch.isfinite(prediction).all():
        raise ValueError("prediction and unmasked targets must be finite")
    count = observed.sum(1)
    if not (count > 0).all():
        raise ValueError("each record requires observed outcomes")
    residual = torch.where(observed, (prediction - safe_target) / scale, 0.0)
    return (residual.square().sum(1) / count).mean()
