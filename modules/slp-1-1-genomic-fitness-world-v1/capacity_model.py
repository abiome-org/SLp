"""A shared nonlinear viability landscape over persistent functional state.

Interventions compose in a common action space. Conditional fitness is a
finite difference of one state observation function, rather than an independent
pair/query regressor. Signed curvature supports both aggravating and rescuing
interactions. Endpoint composition is commutative; no time-course claim is made.
"""
from dataclasses import dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class State:
    latent: torch.Tensor
    assay: torch.Tensor


class FitnessWorld(nn.Module):
    def __init__(self,inputs=930,context=128,width=256):
        super().__init__();self.width=width
        self.gene=nn.Sequential(nn.Linear(inputs,512),nn.GELU(),nn.Dropout(.1),nn.Linear(512,width),nn.LayerNorm(width))
        self.contexts=nn.ModuleList([nn.Sequential(nn.Linear(context,width),nn.GELU(),nn.Linear(width,width)) for _ in range(2)])
        self.linear=nn.Linear(width,1,bias=False)
        self.curvature=nn.Parameter(torch.randn(width)*.001)
        self.landscape=nn.Sequential(nn.Linear(width,128),nn.GELU(),nn.Linear(128,1,bias=False))
        self.assay_scale=nn.Parameter(torch.zeros(2))
        self.reconstruct_gene=nn.Linear(width,642)

    def encode(self,context,assay):
        a=torch.as_tensor(assay,dtype=torch.long,device=context.device).expand(len(context))
        if torch.any((a<0)|(a>1)):raise ValueError('unknown quantitative fitness assay')
        options=torch.stack([f(context) for f in self.contexts],1)
        return State(options[torch.arange(len(a),device=a.device),a],a)

    def action(self,descriptors):return self.gene(descriptors)

    def intervene(self,state,actions,mask=None):
        if actions.ndim==2:actions=actions[:,None,:]
        if mask is None:mask=torch.ones(actions.shape[:2],dtype=torch.bool,device=actions.device)
        return State(state.latent+(actions*mask[:,:,None]).sum(1),state.assay)

    def viability(self,state):
        z=state.latent
        value=self.linear(z).squeeze(-1)+(z.square()*self.curvature).sum(-1)/(self.width**.5)+self.landscape(z).squeeze(-1)
        return value*self.assay_scale[state.assay].exp()

    def decode(self,state,control):return self.viability(state)-self.viability(control)

    def observe(self,state,query):return self.decode(self.intervene(state,query),state)

    def conditional(self,state,action,query):return self.observe(self.intervene(state,action),query)
