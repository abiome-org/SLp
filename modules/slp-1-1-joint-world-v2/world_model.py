"""Shared observation encoder, action operator, and queried assay decoder."""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn

MODE_CRISPRI = 0
MODE_CRISPRA = 1
ASSAY_LN1P_MEAN_CP10K = 0
ASSAY_CONTROL_Z = 1


@dataclass(frozen=True)
class Config:
    feature_dim: int = 577
    width: int = 128
    state_slots: int = 4
    heads: int = 4
    bind_observation_values: bool = True
    control_context: bool = False
    mode_count: int = 2
    assay_count: int = 2


class SharedWorldModel(nn.Module):
    """Application-neutral molecular population state model."""

    def __init__(self, config: Config):
        super().__init__()
        if min(config.feature_dim, config.width, config.state_slots, config.heads) <= 0:
            raise ValueError("model dimensions must be positive")
        if config.width % config.heads:
            raise ValueError("width must be divisible by heads")
        self.config = config
        w = config.width
        self.query_features = nn.Sequential(nn.Linear(config.feature_dim, w), nn.LayerNorm(w), nn.GELU())
        self.action_features = nn.Sequential(nn.Linear(config.feature_dim, w), nn.LayerNorm(w), nn.GELU())
        self.value_features = nn.Sequential(nn.Linear(5 if config.control_context else 3, w), nn.GELU(), nn.Linear(w, w))
        self.mode = nn.Embedding(config.mode_count, w)
        self.assay = nn.Embedding(config.assay_count, w)
        self.slots = nn.Parameter(torch.randn(config.state_slots, w) / math.sqrt(w))
        self.observe_attention = nn.MultiheadAttention(w, config.heads, batch_first=True)
        self.observe_norm = nn.LayerNorm(w)
        self.observe_ff = nn.Sequential(nn.Linear(w, 4*w), nn.GELU(), nn.Linear(4*w, w))
        self.observe_ff_norm = nn.LayerNorm(w)

        self.action_attention = nn.MultiheadAttention(w, config.heads, batch_first=True)
        self.transition_norm = nn.LayerNorm(w)
        self.transition_body = nn.Sequential(nn.Linear(w, 4*w), nn.GELU(), nn.Linear(4*w, w), nn.GELU())
        self.transition_output = nn.Linear(w, w)
        nn.init.zeros_(self.transition_output.weight)
        nn.init.zeros_(self.transition_output.bias)

        self.decode_query = nn.Sequential(nn.Linear(config.feature_dim, w), nn.LayerNorm(w), nn.GELU())
        self.decode_attention = nn.MultiheadAttention(w, config.heads, batch_first=True)
        self.decode_norm = nn.LayerNorm(w)
        self.assay_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(2*w, w), nn.GELU(), nn.Linear(w, 1))
            for _ in range(config.assay_count)
        ])

    def _ids(self, values, batch, count, label):
        if values.shape != (batch,) or values.dtype != torch.long:
            raise ValueError(f"{label} must be int64 [B]")
        if (values < 0).any() or (values >= count).any():
            raise ValueError(f"{label} is out of range")

    def _query_features(self, values, batch, queries):
        if values.ndim == 2:
            values = values.unsqueeze(0).expand(batch, -1, -1)
        if values.shape != (batch, queries, self.config.feature_dim):
            raise ValueError("query_features must be [Q,F] or [B,Q,F]")
        if not torch.isfinite(values).all():
            raise ValueError("query features must be finite")
        return values

    def encode(self, observed_values, basal_values, query_features, observed_mask, mode_ids, assay_ids,
               control_context_values=None, control_context_mask=None):
        if observed_values.ndim != 2 or basal_values.shape != observed_values.shape:
            raise ValueError("observed and basal values must align [B,Q]")
        b, q = observed_values.shape
        if observed_mask.shape != (b, q) or observed_mask.dtype != torch.bool:
            raise ValueError("observed_mask must be Boolean [B,Q]")
        if not observed_mask.any(1).all():
            raise ValueError("each observation requires at least one measured query")
        self._ids(mode_ids, b, self.config.mode_count, "mode_ids")
        self._ids(assay_ids, b, self.config.assay_count, "assay_ids")
        features = self._query_features(query_features, b, q)
        safe_observed = torch.where(observed_mask, observed_values, 0.)
        safe_basal = torch.where(observed_mask, basal_values, 0.)
        if not torch.isfinite(safe_observed).all() or not torch.isfinite(safe_basal).all():
            raise ValueError("observed molecular values must be finite where measured")
        values = torch.stack((safe_observed, safe_basal, safe_observed-safe_basal), -1)
        context = torch.zeros_like(safe_observed)
        if self.config.control_context:
            if (control_context_values is None) != (control_context_mask is None):
                raise ValueError("control context values and mask must be supplied together")
            context_mask = torch.zeros_like(observed_mask)
            if control_context_values is not None:
                if control_context_values.shape != (b, q) or control_context_mask.shape != (b, q) or control_context_mask.dtype != torch.bool:
                    raise ValueError("control context must be values and Boolean mask [B,Q]")
                context_mask = control_context_mask & observed_mask
                context = torch.where(context_mask, control_context_values, 0.)
                if not torch.isfinite(context).all():
                    raise ValueError("measured control context must be finite")
            values = torch.cat((values, context[..., None], context_mask.to(values.dtype)[..., None]), -1)
        gene_tokens = self.query_features(features)
        tokens = gene_tokens + self.value_features(values)
        if self.config.bind_observation_values:
            response = safe_observed - safe_basal
            tokens = tokens + gene_tokens * torch.tanh(response[..., None])
            if self.config.control_context:
                tokens = tokens + gene_tokens * torch.tanh(context[..., None])
        tokens = tokens + self.mode(mode_ids)[:,None] + self.assay(assay_ids)[:,None]
        slots = self.slots.unsqueeze(0).expand(b, -1, -1)
        update, _ = self.observe_attention(slots, tokens, tokens, key_padding_mask=~observed_mask, need_weights=False)
        state = self.observe_norm(slots + update)
        return self.observe_ff_norm(state + self.observe_ff(state))

    def transition(self, state, action_features, action_mask, mode_ids, assay_ids):
        if state.ndim != 3 or state.shape[1:] != (self.config.state_slots, self.config.width):
            raise ValueError("state must be [B,S,W]")
        b = len(state)
        if action_features.ndim != 3 or action_features.shape[0] != b or action_features.shape[2] != self.config.feature_dim:
            raise ValueError("action_features must be [B,A,F]")
        if action_mask.shape != action_features.shape[:2] or action_mask.dtype != torch.bool:
            raise ValueError("action_mask must be Boolean [B,A]")
        self._ids(mode_ids, b, self.config.mode_count, "mode_ids")
        self._ids(assay_ids, b, self.config.assay_count, "assay_ids")
        safe = torch.where(action_mask[...,None], action_features, 0.)
        if not torch.isfinite(safe).all():
            raise ValueError("active action features must be finite")
        tokens = self.action_features(safe) + self.mode(mode_ids)[:,None] + self.assay(assay_ids)[:,None]
        active = action_mask.any(1)
        # MultiheadAttention cannot consume a row whose keys are all masked.
        attention_mask = action_mask.clone()
        if (~active).any():
            attention_mask[~active, 0] = True
        update, _ = self.action_attention(state, tokens, tokens, key_padding_mask=~attention_mask, need_weights=False)
        hidden = self.transition_body(self.transition_norm(state + update))
        delta = self.transition_output(hidden)
        return torch.where(active[:,None,None], state + delta, state)

    def decode(self, state, query_features, assay_ids):
        if state.ndim != 3 or state.shape[1:] != (self.config.state_slots, self.config.width):
            raise ValueError("state must be [B,S,W]")
        b = len(state); q = query_features.shape[-2]
        self._ids(assay_ids, b, self.config.assay_count, "assay_ids")
        features = self._query_features(query_features, b, q)
        query = self.decode_query(features) + self.assay(assay_ids)[:,None]
        readout, _ = self.decode_attention(query, state, state, need_weights=False)
        joined = torch.cat((query, self.decode_norm(query + readout)), -1)
        outputs = torch.empty((b, q), dtype=state.dtype, device=state.device)
        for assay, head in enumerate(self.assay_heads):
            take = assay_ids == assay
            if take.any(): outputs[take] = head(joined[take]).squeeze(-1)
        return outputs

    def forward(self, observed_values, basal_values, observation_query_features, observed_mask,
                action_features, action_mask, decode_query_features, mode_ids, assay_ids,
                control_context_values=None, control_context_mask=None):
        state = self.encode(observed_values, basal_values, observation_query_features,
                            observed_mask, mode_ids, assay_ids, control_context_values, control_context_mask)
        changed = self.transition(state, action_features, action_mask, mode_ids, assay_ids)
        return self.decode(changed, decode_query_features, assay_ids) - self.decode(state, decode_query_features, assay_ids)

    def reconstruct(self, observed_values, basal_values, query_features, observed_mask, mode_ids, assay_ids,
                    control_context_values=None, control_context_mask=None):
        state = self.encode(observed_values, basal_values, query_features, observed_mask, mode_ids, assay_ids,
                            control_context_values, control_context_mask)
        return self.decode(state, query_features, assay_ids)
