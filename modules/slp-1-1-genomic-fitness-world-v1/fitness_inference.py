"""Application-neutral, descriptor-driven functional world inference."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

_spec=importlib.util.spec_from_file_location('_slp_genomic_fitness_model',Path(__file__).with_name('model.py'))
_model=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_model;_spec.loader.exec_module(_model)
FitnessWorld,State=_model.FitnessWorld,_model.State


class FunctionalWorld:
    def __init__(self,root,device='cpu',untrained=False):
        root=Path(root);self.device=device
        architecture=json.loads((root/'training.json').read_text()).get('architecture','conditional')
        cls=FitnessWorld
        if architecture=='capacity':
            spec=importlib.util.spec_from_file_location('_slp_capacity_model',Path(__file__).with_name('capacity_model.py'))
            module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);cls=module.FitnessWorld
        torch.manual_seed(1731);self.model=cls().to(device).eval()
        if not untrained:self.model.load_state_dict(load_file(str(root/'model.safetensors'),device=device))
        with np.load(root/'normalizer.npz') as z:self.norm={k:z[k] for k in z.files}

    def tensor(self,x):return torch.as_tensor(x,device=self.device,dtype=torch.float32)

    @torch.inference_mode()
    def actions(self,raw,molecular_signatures):
        raw=np.asarray(raw,np.float32);signatures=np.asarray(molecular_signatures,np.float32)
        if raw.ndim!=2 or raw.shape[1]!=642 or signatures.shape!=(len(raw),288):raise ValueError('expected raw gene descriptors [N,642] and molecular signatures [N,288]')
        if not np.isfinite(raw).all() or not np.isfinite(signatures).all():raise ValueError('nonfinite observations')
        n=self.norm;x=np.concatenate(((raw-n['feature_mean'])/n['feature_scale'],(signatures-n['signature_mean'])/n['signature_scale']),1)
        return self.model.action(self.tensor(x))

    @torch.inference_mode()
    def encode(self,context,taxon=9606):
        c=np.asarray(context,np.float32)
        if c.ndim!=2 or c.shape[1]!=128 or not np.isfinite(c).all():raise ValueError('finite [contexts,128] molecular observations required')
        if taxon==9606:c=(c-self.norm['context_mean'])/self.norm['context_scale'];assay=0
        elif taxon==4932:assay=1
        else:raise ValueError('only trained human/yeast assay coordinates supported')
        return self.model.encode(self.tensor(c),assay)

    @torch.inference_mode()
    def intervene(self,state,actions,mask=None):return self.model.intervene(state,actions,mask)

    @torch.inference_mode()
    def observe(self,state,query_actions):return self.model.observe(state,query_actions)

    @torch.inference_mode()
    def decode(self,state,control_state):
        if not hasattr(self.model,'decode'):raise ValueError('state viability decoding requires the capacity architecture')
        return self.model.decode(state,control_state)

    @staticmethod
    def expand(state,n):return State(state.latent.expand(n,-1),state.assay.expand(n))
