"""A molecular state encoder, shared intervention dynamics and conditional latent flow."""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    gene_features: int = 642
    measurement_features: int = 60
    width: int = 256
    slots: int = 32
    heads: int = 8
    encoder_layers: int = 3
    dynamics_layers: int = 4
    flow_layers: int = 4
    assay_count: int = 8
    taxon_count: int = 2
    mechanism_count: int = 3
    observation_likelihood: bool = False


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))

    def forward(self, value):
        q = self.norm(value)
        value = value + self.attention(q, q, q, need_weights=False)[0]
        return value + self.ff(self.ff_norm(value))


class ConditionedBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.block = Block(width, heads)
        self.norm = nn.LayerNorm(width)
        self.cross = nn.MultiheadAttention(width, heads, batch_first=True)
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(width, 2*width))

    def forward(self, state, tokens, pooled):
        shift, scale = self.condition(pooled).chunk(2, -1)
        q = self.norm(state)*(1+.1*torch.tanh(scale[:, None])) + shift[:, None]
        state = state + self.cross(q, tokens, tokens, need_weights=False)[0]
        return self.block(state)


class CellWorld(nn.Module):
    """Gene identity is supplied by descriptors; there is no learned gene vocabulary."""
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        c, w = config, config.width
        if w % c.heads or min(w,c.slots,c.heads) <= 0:
            raise ValueError('invalid state dimensions')
        self.query = nn.Sequential(nn.Linear(c.gene_features+c.measurement_features,w),
                                   nn.LayerNorm(w),nn.GELU(),nn.Linear(w,w))
        self.action = nn.Sequential(nn.Linear(c.gene_features,w),nn.LayerNorm(w),nn.GELU(),nn.Linear(w,w))
        self.assay = nn.Embedding(c.assay_count,w)
        self.taxon = nn.Embedding(c.taxon_count,w)
        self.mechanism = nn.Embedding(c.mechanism_count,w)
        self.modality = nn.Embedding(2,w)
        self.values = nn.Sequential(nn.Linear(6,w),nn.GELU(),nn.Linear(w,w))
        self.slots = nn.Parameter(torch.randn(c.slots,w)/math.sqrt(w))
        self.observe = nn.MultiheadAttention(w,c.heads,batch_first=True)
        self.encoder = nn.ModuleList([Block(w,c.heads) for _ in range(c.encoder_layers)])
        self.encoder_norm = nn.LayerNorm(w)
        self.dynamics = nn.ModuleList([ConditionedBlock(w,c.heads) for _ in range(c.dynamics_layers)])
        self.action_pool = nn.Sequential(nn.Linear(2*w,w),nn.SiLU(),nn.Linear(w,w))
        self.delta = nn.Linear(w,w)
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)
        self.decode_attention = nn.MultiheadAttention(w,c.heads,batch_first=True)
        self.decode_norm = nn.LayerNorm(w)
        self.decoder = nn.Sequential(nn.Linear(2*w,2*w),nn.GELU(),nn.Linear(2*w,w),nn.GELU(),nn.Linear(w,2))
        if c.observation_likelihood:
            self.query_control=nn.Sequential(nn.Linear(2,w),nn.GELU(),nn.Linear(w,w))
            nn.init.zeros_(self.query_control[2].weight);nn.init.zeros_(self.query_control[2].bias)
            self.observation_head=nn.Sequential(nn.Linear(w+2,w),nn.SiLU(),nn.Linear(w,3))
            nn.init.zeros_(self.observation_head[2].weight)
            nn.init.constant_(self.observation_head[2].bias,-1.)
        self.flow_input = nn.Linear(3*w,w)
        self.time = nn.Sequential(nn.Linear(w,w),nn.SiLU(),nn.Linear(w,w))
        self.flow = nn.ModuleList([ConditionedBlock(w,c.heads) for _ in range(c.flow_layers)])
        self.velocity = nn.Linear(w,w)
        nn.init.zeros_(self.velocity.weight); nn.init.zeros_(self.velocity.bias)

    def context(self, assay, taxon, mechanism):
        return self.assay(assay)+self.taxon(taxon)+self.mechanism(mechanism)

    def encode(self, values, basal, descriptors, modality, mask, assay, taxon, mechanism, control=None, control_mask=None):
        if values.ndim!=2 or basal.shape!=values.shape or mask.shape!=values.shape or mask.dtype!=torch.bool:
            raise ValueError('values, basal and Boolean mask must align [B,Q]')
        if not mask.any(1).all():
            raise ValueError('every state needs an observed molecular coordinate')
        if descriptors.shape!=(values.shape[1],self.config.gene_features+self.config.measurement_features):
            raise ValueError('query descriptors do not align')
        safe_values=torch.where(mask,values,0.)
        safe_basal=torch.where(mask,basal,0.)
        if not torch.isfinite(safe_values).all() or not torch.isfinite(safe_basal).all():
            raise ValueError('observed values must be finite')
        query=self.query(descriptors)+self.modality(modality)
        residual=safe_values-safe_basal
        control=safe_basal if control is None else control
        control_mask=mask if control_mask is None else control_mask
        control=torch.where(control_mask,control,0.)
        if control.shape!=values.shape or not torch.isfinite(control).all():raise ValueError('invalid measured control context')
        numerics=torch.stack((torch.asinh(safe_values),torch.asinh(safe_basal),
                              torch.asinh(residual),mask.to(values.dtype),torch.asinh(control),control_mask.to(values.dtype)),-1)
        tokens=query[None]+self.values(numerics)+query[None]*torch.tanh(residual[...,None])
        tokens=tokens+self.context(assay,taxon,mechanism)[:,None]
        slots=self.slots[None].expand(len(values),-1,-1)
        state=slots+self.observe(slots,tokens,tokens,key_padding_mask=~mask,need_weights=False)[0]
        for block in self.encoder: state=block(state)
        return self.encoder_norm(state)

    def condition(self, actions, action_mask, assay, taxon, mechanism):
        if actions.ndim!=3 or actions.shape[-1]!=self.config.gene_features or action_mask.shape!=actions.shape[:2] or action_mask.dtype!=torch.bool:
            raise ValueError('actions and Boolean action mask must align [B,A,F]')
        safe=torch.where(action_mask[...,None],actions,0.)
        if not torch.isfinite(safe).all(): raise ValueError('active actions must be finite')
        context=self.context(assay,taxon,mechanism)
        tokens=self.action(safe)*action_mask[...,None]
        # The aggregate is invariant to action order, preserves multiplicity, and
        # gives attention no padding keys whose number would change its normalization.
        summed=tokens.sum(1)
        pooled=self.action_pool(torch.cat((summed,context),-1))
        return torch.stack((pooled,context),1),pooled

    def transition(self,state,actions,action_mask,assay,taxon,mechanism):
        tokens,pooled=self.condition(actions,action_mask,assay,taxon,mechanism)
        changed=state
        for block in self.dynamics: changed=block(changed,tokens,pooled)
        changed=state+self.delta(changed)
        return torch.where(action_mask.any(1)[:,None,None],changed,state)

    def decoded_hidden(self,state,descriptors,modality,assay,taxon,mechanism,control=None):
        query=self.query(descriptors)+self.modality(modality)
        query=query[None].expand(len(state),-1,-1)+self.context(assay,taxon,mechanism)[:,None]
        if self.config.observation_likelihood and control is not None:
            query=query+self.query_control(torch.stack((control,torch.asinh(control)),-1))
        attended=self.decode_attention(query,state,state,need_weights=False)[0]
        return self.decoder[:4](torch.cat((query,self.decode_norm(attended+query)),-1))

    def decode(self,state,descriptors,modality,assay,taxon,mechanism,control=None):
        value=self.decoder[4](self.decoded_hidden(state,descriptors,modality,assay,taxon,mechanism,control))
        # Log variance describes observation reconstruction in normalized units.
        # It is not asserted to be calibrated biological intervention uncertainty.
        return value[...,0],value[...,1].clamp(-5,4)

    def observation_parameters(self,state,descriptors,modality,assay,taxon,mechanism,control=None):
        """RNA detection Bernoulli and lognormal positive-expression parameters."""
        if not self.config.observation_likelihood:raise ValueError('checkpoint has no sparse RNA observation head')
        hidden=self.decoded_hidden(state,descriptors,modality,assay,taxon,mechanism,control)
        control=torch.zeros(hidden.shape[:2],device=hidden.device,dtype=hidden.dtype) if control is None else control
        value=self.observation_head(torch.cat((hidden,control[...,None],torch.asinh(control)[...,None]),-1))
        return value[...,0],value[...,1].clamp(-8,3),value[...,2].clamp(-5,2)

    def flow_velocity(self,residual,t,initial,mean,actions,action_mask,assay,taxon,mechanism):
        w=self.config.width
        freq=torch.exp(-math.log(10000)*torch.arange(w//2,device=t.device,dtype=t.dtype)/(w//2-1))
        angle=t[:,None]*freq[None]*1000
        time=self.time(torch.cat((angle.sin(),angle.cos()),-1))
        tokens,pooled=self.condition(actions,action_mask,assay,taxon,mechanism)
        pooled=pooled+time
        value=self.flow_input(torch.cat((residual,initial,mean),-1))+time[:,None]
        for block in self.flow: value=block(value,tokens,pooled)
        return self.velocity(value)

    @torch.no_grad()
    def sample_state(self,initial,actions,action_mask,assay,taxon,mechanism,*,steps=16,noise=None,generator=None):
        if steps<1: raise ValueError('flow integration requires positive steps')
        mean=self.transition(initial,actions,action_mask,assay,taxon,mechanism)
        residual=(torch.randn(initial.shape,device=initial.device,dtype=initial.dtype,generator=generator)*.3
                  if noise is None else noise.clone())
        if residual.shape!=initial.shape or not torch.isfinite(residual).all():raise ValueError('noise must match state and be finite')
        for i in range(steps):
            t=torch.full((len(initial),),(i+.5)/steps,device=initial.device,dtype=initial.dtype)
            residual=residual+self.flow_velocity(residual,t,initial,mean,actions,action_mask,assay,taxon,mechanism)/steps
        return torch.where(action_mask.any(1)[:,None,None],mean+residual,initial)


def gaussian_loss(mean,logvar,target):
    return .5*((target-mean).square()*torch.exp(-logvar)+logvar).mean()
