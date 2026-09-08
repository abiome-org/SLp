"""SLp-1.2: a shared, experiment-conditioned set transformer.

Every intervention remains an individual token. Molecular and fitness queries
use the same transformer and trainable entity representation. There is no
frozen molecular simulation bridge, latent action addition, or SL classifier.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class Config:
    width: int = 768
    layers: int = 16
    heads: int = 12
    ff_multiplier: float = 2.75
    descriptor_dim: int = 702
    context_dim: int = 128
    gene_count: int = 0
    assay_count: int = 16
    taxon_count: int = 2
    mechanism_count: int = 8
    id_dropout: float = .25
    dropout: float = 0.
    activation_checkpointing: bool = True

    def __post_init__(self):
        if self.width < 8 or self.width % self.heads or self.layers < 1:
            raise ValueError('Invalid transformer dimensions')
        if not 0 <= self.id_dropout < 1 or not 0 <= self.dropout < 1:
            raise ValueError('Invalid dropout probability')


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.heads = c.heads
        self.dropout = c.dropout
        self.attention_norm = nn.RMSNorm(c.width)
        self.qkv = nn.Linear(c.width, 3 * c.width, bias=False)
        self.out = nn.Linear(c.width, c.width, bias=False)
        self.ff_norm = nn.RMSNorm(c.width)
        hidden = math.ceil(c.width * c.ff_multiplier / 64) * 64
        self.gate_up = nn.Linear(c.width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, c.width, bias=False)

    def forward(self, x, valid):
        b, n, width = x.shape
        q, k, v = self.qkv(self.attention_norm(x)).reshape(
            b, n, 3, self.heads, width // self.heads).unbind(2)
        q, k, v = (a.transpose(1, 2) for a in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None if valid is None else valid[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.)
        x = x + self.out(y.transpose(1, 2).reshape(b, n, width))
        gate, value = self.gate_up(self.ff_norm(x)).chunk(2, -1)
        return x + self.down(F.silu(gate) * value)


class WorldModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        c, w = config, config.width
        self.descriptor = nn.Sequential(nn.Linear(c.descriptor_dim, w), nn.RMSNorm(w), nn.SiLU(), nn.Linear(w, w))
        # Index zero is always the unknown-entity fallback. IDs are optional.
        self.entity = nn.Embedding(c.gene_count + 1, w, padding_idx=0)
        self.role = nn.Embedding(4, w)  # context, observation, action, query
        self.modality = nn.Embedding(3, w)  # RNA, protein, fitness
        self.assay = nn.Embedding(c.assay_count, w)
        self.taxon = nn.Embedding(c.taxon_count, w)
        self.mechanism = nn.Embedding(c.mechanism_count, w)
        self.numerics = nn.Sequential(nn.Linear(8, w), nn.SiLU(), nn.Linear(w, w))
        self.context = nn.Sequential(nn.Linear(c.context_dim + 1, w), nn.SiLU(), nn.Linear(w, w))
        self.flow_time = nn.Sequential(nn.Linear(4, w), nn.SiLU(), nn.Linear(w, w))
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm = nn.RMSNorm(w)
        # Mean residual and rectified-flow velocity have separate output rows;
        # both see the full shared trunk. Variance is fitted as an observation.
        self.output = nn.Linear(w, 3)
        self.apply(self._initialize)
        with torch.no_grad():
            self.entity.weight[0].zero_()
            self.output.weight.mul_(.1)
            self.output.bias.zero_()
            for block in self.blocks:
                block.out.weight.div_(math.sqrt(2 * c.layers))
                block.down.weight.div_(math.sqrt(2 * c.layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def identity(self, features, ids):
        if self.training and self.config.id_dropout:
            ids = torch.where(torch.rand_like(ids, dtype=torch.float32) < self.config.id_dropout, 0, ids)
        # Even if optimizer decay changes row zero, unknown IDs contribute zero.
        return self.descriptor(features) + self.entity(ids) * (ids != 0)[..., None]

    def forward(self, batch, noisy_residual=None, time=None):
        """Targets are never read here. ``noisy_residual`` is flow-mode only.

        Shapes are [B,O,*] observations, [B,A,*] actions, [B,Q,*] queries.
        Masks are True for available tokens. No arbitrary positional encoding
        is used: permuting observations/actions/queries preserves semantics.
        """
        b = batch['query_features'].shape[0]
        device = batch['query_features'].device
        flow = noisy_residual is not None
        if time is None:
            time = torch.zeros(b, device=device)
        elif not flow:
            raise ValueError('Time argument is reserved for computational flow time')
        clock = torch.stack((time, time.sin(), time.cos(), torch.full_like(time, float(flow))), -1)
        common = self.assay(batch['assay']) + self.taxon(batch['taxon']) + self.flow_time(clock)
        context = self.context(torch.cat((batch['context'], batch['context_known'][:, None].float()), -1))
        context = context[:, None] + common[:, None] + self.role.weight[0]

        of = torch.where(batch['observation_mask'][..., None], batch['observation_features'], 0.)
        ov = torch.where(batch['observation_mask'], batch['observation_values'], 0.)
        om = batch['observation_mask'].float()
        on = torch.stack((ov, torch.asinh(ov), om, torch.zeros_like(ov),
                          torch.zeros_like(ov), torch.zeros_like(ov), torch.zeros_like(ov), torch.zeros_like(ov)), -1)
        observations = (self.identity(of, batch['observation_ids']) + self.numerics(on)
                        + self.modality(batch['observation_modality']) + self.role.weight[1] + common[:, None])

        af = torch.where(batch['action_mask'][..., None], batch['action_features'], 0.)
        # Dose, delay and duration have explicit known flags. Missing is not zero dose.
        an = torch.cat((batch['action_values'], batch['action_known'].float(),
                        batch['action_mask'][..., None].float(), torch.zeros_like(batch['action_mask'][..., None], dtype=af.dtype)), -1)
        an = torch.where(batch['action_mask'][..., None], an, 0.)
        actions = (self.identity(af, batch['action_ids']) + self.numerics(an)
                   + self.mechanism(batch['action_mechanism']) + self.role.weight[2] + common[:, None])

        qm = batch['query_mask']
        anchor = torch.where(qm, batch['query_anchor'], 0.)
        noisy = torch.zeros_like(anchor) if noisy_residual is None else torch.where(qm, noisy_residual, 0.)
        qn = torch.stack((anchor, torch.asinh(anchor), batch['query_anchor_known'].float(),
                          noisy, torch.asinh(noisy), torch.full_like(anchor, float(flow)),
                          torch.zeros_like(anchor), torch.zeros_like(anchor)), -1)
        qf = torch.where(qm[..., None], batch['query_features'], 0.)
        queries = (self.identity(qf, batch['query_ids']) + self.numerics(qn)
                   + self.modality(batch['query_modality']) + self.role.weight[3] + common[:, None])
        x = torch.cat((context, observations, actions, queries), 1)
        valid = torch.cat((torch.ones((b, 1), device=device, dtype=torch.bool),
                           batch['observation_mask'], batch['action_mask'], batch['query_mask']), 1)
        # Passing no mask for a complete panel enables the fastest SDPA kernel.
        # Padding still uses the explicit availability mask.
        if bool(valid.all()):
            valid = None
        for block in self.blocks:
            if self.training and self.config.activation_checkpointing:
                x = checkpoint(block, x, valid, use_reentrant=False)
            else:
                x = block(x, valid)
        hidden = self.norm(x[:, -queries.shape[1]:])
        out = self.output(hidden).float()
        return {'mean': anchor + out[..., 0], 'log_variance': out[..., 1].clamp(-6, 5),
                'velocity': out[..., 2], 'features': hidden}

    @torch.inference_mode()
    def generate(self, batch, *, steps=32, seed=731, noise_scale=1.):
        if self.training:
            raise ValueError('Call eval() before generation')
        if steps < 1:
            raise ValueError('steps must be positive')
        anchor = batch['query_anchor']
        # Draw on CPU so the same seed is portable across CPU/CUDA generators.
        generator = torch.Generator(device='cpu').manual_seed(seed)
        residual = torch.randn(anchor.shape, generator=generator, dtype=anchor.dtype).to(anchor.device) * noise_scale
        for i in range(steps):
            t = torch.full((len(anchor),), i / steps, device=anchor.device)
            residual = residual + self(batch, residual, t)['velocity'] / steps
        return anchor + residual

    def configuration(self):
        return asdict(self.config)
