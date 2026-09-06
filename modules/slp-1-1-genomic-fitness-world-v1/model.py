"""Species-aware functional state dynamics and continuous fitness observations.

No gene identity embedding, pair labels, application score, or benchmark imports.
The context is an observed basal molecular summary. A state may receive any
number of interventions; the observation is the effect of a queried next action.
Endpoint training constrains one-action transitions, not temporal trajectories.
"""
from dataclasses import dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class State:
    latent: torch.Tensor
    assay: torch.Tensor


class FitnessWorld(nn.Module):
    def __init__(self, inputs=930, context=128, width=256):
        super().__init__();self.width=width
        self.gene=nn.Sequential(nn.Linear(inputs,512),nn.GELU(),nn.Dropout(.1),nn.Linear(512,width),nn.LayerNorm(width))
        # Different observed-context coordinates and assay scales, shared dynamics.
        self.contexts=nn.ModuleList([nn.Sequential(nn.Linear(context,width),nn.GELU(),nn.Linear(width,width)) for _ in range(2)])
        self.transition=nn.Sequential(nn.Linear(width*2,width),nn.GELU(),nn.Linear(width,width))
        self.query=nn.Sequential(nn.Linear(width*3,256),nn.GELU(),nn.Linear(256,64),nn.GELU(),nn.Linear(64,1))
        self.assay_scale=nn.Parameter(torch.zeros(2));self.assay_bias=nn.Parameter(torch.zeros(2))
        self.reconstruct_gene=nn.Linear(width,642)

    def encode(self, context, assay):
        a=torch.as_tensor(assay,dtype=torch.long,device=context.device).expand(len(context))
        if torch.any((a<0)|(a>1)):raise ValueError('assay must be human CRISPR gene effect (0) or yeast log relative fitness (1)')
        options=torch.stack([f(context) for f in self.contexts],1)
        return State(options[torch.arange(len(a),device=a.device),a],a)

    def action(self, descriptors):return self.gene(descriptors)

    def intervene(self, state, actions, mask=None):
        """Return a new state; the input is immutable and an empty action is identity."""
        if actions.ndim==2:actions=actions[:,None,:]
        if mask is None:mask=torch.ones(actions.shape[:2],dtype=torch.bool,device=actions.device)
        latent=state.latent
        for j in range(actions.shape[1]):
            effect=actions[:,j]+self.transition(torch.cat((latent,actions[:,j]),-1))
            latent=latent+effect*mask[:,j,None]
        return State(latent,state.assay)

    def observe(self, state, query):
        p=self.query(torch.cat((state.latent,query,state.latent*query),-1)).squeeze(-1)
        return p*self.assay_scale[state.assay].exp()+self.assay_bias[state.assay]

    def conditional(self, state, action, query):
        return self.observe(self.intervene(state,action),query)
