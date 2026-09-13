"""Inductive experimental predictor with independent measurement queries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class Config:
    width: int = 768
    layers: int = 16
    heads: int = 12
    ff_multiplier: float = 2.75
    sequence_dim: int = 1280
    annotation_dim: int = 64
    context_dim: int = 128
    assay_count: int = 64
    taxon_count: int = 8
    mechanism_count: int = 16
    method_count: int = 32
    scope_count: int = 8
    measurement_count: int = 5  # RNA, protein, fitness, interaction, human SL
    dropout: float = 0.0
    activation_checkpointing: bool = True

    def __post_init__(self):
        if self.width < 8 or self.width % self.heads or self.layers < 1:
            raise ValueError("Invalid transformer dimensions")
        if not 0 <= self.dropout < 1:
            raise ValueError("Invalid dropout")
        if min(self.sequence_dim, self.annotation_dim, self.context_dim) < 1:
            raise ValueError("Feature dimensions must be positive")


@dataclass
class ContextCache:
    owner: int
    versions: tuple
    inputs: dict
    layers: list
    valid: torch.Tensor


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.dropout = c.heads, c.dropout
        self.attention_norm = nn.RMSNorm(c.width)
        self.qkv = nn.Linear(c.width, 3 * c.width, bias=False)
        self.out = nn.Linear(c.width, c.width, bias=False)
        self.ff_norm = nn.RMSNorm(c.width)
        hidden = math.ceil(c.width * c.ff_multiplier / 64) * 64
        self.gate_up = nn.Linear(c.width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, c.width, bias=False)

    def project(self, x):
        b, n, w = x.shape
        return tuple(
            v.transpose(1, 2)
            for v in self.qkv(self.attention_norm(x))
            .reshape(b, n, 3, self.heads, w // self.heads)
            .unbind(2)
        )

    def attend(self, x, q, k, v, mask):
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        x = x + self.out(y.transpose(1, 2).reshape_as(x))
        gate, value = self.gate_up(self.ff_norm(x)).chunk(2, -1)
        return x + self.down(F.silu(gate) * value)

    def forward(self, x, mask):
        return self.attend(x, *self.project(x), mask)


class WorldModel(nn.Module):
    """IDs never enter this module; they only join records to static features.

    A context token is always available. Padding is removed as attention keys.
    Output queries attend conditioning and themselves, but no other queries.
    """

    CONTEXT_KEYS = (
        "context",
        "context_known",
        "assay",
        "taxon",
        "scope",
        "observation_sequence",
        "observation_annotation",
        "observation_known",
        "observation_values",
        "observation_kind",
        "observation_mask",
    )

    def __init__(self, config=Config()):
        super().__init__()
        self.config = c = config
        w = c.width
        self.identity = nn.Sequential(
            nn.Linear(c.sequence_dim + c.annotation_dim + 2, w),
            nn.RMSNorm(w),
            nn.SiLU(),
            nn.Linear(w, w),
        )
        self.context = nn.Sequential(
            nn.Linear(c.context_dim + 1, w), nn.SiLU(), nn.Linear(w, w)
        )
        self.numeric = nn.Sequential(nn.Linear(8, w), nn.SiLU(), nn.Linear(w, w))
        self.role = nn.Embedding(4, w)
        self.assay = nn.Embedding(c.assay_count, w)
        self.taxon = nn.Embedding(c.taxon_count, w)
        self.scope = nn.Embedding(c.scope_count, w)
        self.kind = nn.Embedding(c.measurement_count, w)
        self.mechanism = nn.Embedding(c.mechanism_count, w)
        self.method = nn.Embedding(c.method_count, w)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.norm = nn.RMSNorm(w)
        self.output = nn.Linear(w, 3)  # location, log scale, binary SL logit
        self.apply(self._initialize)
        with torch.no_grad():
            self.output.weight.mul_(0.1)
            for block in self.blocks:
                block.out.weight.div_(math.sqrt(2 * c.layers))
                block.down.weight.div_(math.sqrt(2 * c.layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def gene(self, batch, prefix):
        known = batch[prefix + "_known"].bool() & batch[prefix + "_mask"][..., None]
        sequence = torch.where(known[..., :1], batch[prefix + "_sequence"], 0.0)
        annotation = torch.where(known[..., 1:2], batch[prefix + "_annotation"], 0.0)
        return self.identity(
            torch.cat((sequence, annotation, known.to(sequence.dtype)), -1)
        )

    def common(self, b):
        return self.assay(b["assay"]) + self.taxon(b["taxon"]) + self.scope(b["scope"])

    def conditioning(self, b):
        common = self.common(b)[:, None]
        known = b["context_known"][:, None]
        context = self.context(
            torch.cat(
                (torch.where(known, b["context"], 0.0), known.to(b["context"].dtype)),
                -1,
            )
        )
        context = context[:, None] + common + self.role.weight[0]
        values = torch.where(b["observation_mask"], b["observation_values"], 0.0)
        zero = torch.zeros_like(values)
        numeric = torch.stack(
            (
                values,
                torch.asinh(values),
                b["observation_mask"].to(values.dtype),
                zero,
                zero,
                zero,
                zero,
                zero,
            ),
            -1,
        )
        observations = (
            self.gene(b, "observation")
            + self.numeric(numeric)
            + self.kind(b["observation_kind"])
            + common
            + self.role.weight[1]
        )
        valid = torch.cat(
            (
                torch.ones((len(context), 1), device=context.device, dtype=torch.bool),
                b["observation_mask"],
            ),
            1,
        )
        return torch.cat((context, observations), 1), valid

    def actions_queries(self, b):
        common = self.common(b)[:, None]
        ak = b["action_numeric_known"] & b["action_mask"][..., None]
        values = torch.where(ak, b["action_values"], 0.0)
        numeric = torch.cat(
            (
                values,
                ak.to(values.dtype),
                b["action_mask"][..., None].to(values.dtype),
                torch.zeros_like(values[..., :1]),
            ),
            -1,
        )
        actions = (
            self.gene(b, "action")
            + self.numeric(numeric)
            + self.mechanism(b["action_mechanism"])
            + self.method(b["action_method"])
            + common
            + self.role.weight[2]
        )
        queries = (
            self.gene(b, "query")
            + self.kind(b["query_kind"])
            + common
            + self.role.weight[3]
        )
        return actions, queries

    @staticmethod
    def mask(valid, context_count, action_count, *, cached=False):
        n = valid.shape[1]
        index = torch.arange(n, device=valid.device)
        end = context_count + action_count
        # Basal/context cannot see actions; neither can see outputs.
        allowed = (index[:, None] < context_count) & (index[None, :] < context_count)
        allowed |= (
            (index[:, None] >= context_count)
            & (index[:, None] < end)
            & (index[None, :] < end)
        )
        allowed |= (index[:, None] >= end) & (
            (index[None, :] < end) | (index[:, None] == index[None, :])
        )
        if cached:
            allowed = allowed[context_count:]
        return allowed[None, None] & valid[:, None, None, :]

    def _versions(self):
        return tuple(
            (p._version, str(p.device), str(p.dtype), p.data_ptr())
            for p in self.parameters()
        )

    @torch.no_grad()
    def prepare_context(self, batch):
        if self.training:
            raise ValueError("Context caching requires eval mode")
        x, valid = self.conditioning(batch)
        layers = []
        for block in self.blocks:
            q, k, v = block.project(x)
            layers.append((k, v))
            x = block.attend(x, q, k, v, valid[:, None, None, :])
        return ContextCache(
            id(self),
            self._versions(),
            {k: batch[k].detach().clone() for k in self.CONTEXT_KEYS},
            layers,
            valid,
        )

    def forward(self, batch, cache=None):
        if batch["query_mask"].shape[1] < 1:
            raise ValueError("At least one query is required")
        actions, queries = self.actions_queries(batch)
        if cache is None:
            context, cv = self.conditioning(batch)
            x = torch.cat((context, actions, queries), 1)
        else:
            if self.training or torch.is_grad_enabled():
                raise ValueError("Cached inference requires eval and no_grad")
            if cache.owner != id(self) or cache.versions != self._versions():
                raise ValueError("Context cache belongs to a different model state")
            if any(
                not torch.equal(batch[k], cache.inputs[k]) for k in self.CONTEXT_KEYS
            ):
                raise ValueError("Context inputs changed")
            cv = cache.valid
            x = torch.cat((actions, queries), 1)
        valid = torch.cat((cv, batch["action_mask"], batch["query_mask"]), 1)
        mask = self.mask(valid, cv.shape[1], actions.shape[1], cached=cache is not None)
        for i, block in enumerate(self.blocks):
            if cache is not None:
                q, k, v = block.project(x)
                ck, cvalue = cache.layers[i]
                if ck.dtype != k.dtype or ck.device != k.device:
                    raise ValueError(
                        "Cache and query inference precision/device differ"
                    )
                x = block.attend(
                    x, q, torch.cat((ck, k), 2), torch.cat((cvalue, v), 2), mask
                )
            elif self.training and self.config.activation_checkpointing:
                x = checkpoint(block, x, mask, use_reentrant=False)
            else:
                x = block(x, mask)
        hidden = self.norm(x[:, -queries.shape[1] :])
        out = self.output(hidden).float()
        return {
            "location": out[..., 0],
            "log_scale": out[..., 1].clamp(-6, 5),
            "sl_logit": out[..., 2],
            "features": hidden,
        }

    def configuration(self):
        return asdict(self.config)
