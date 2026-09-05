"""Portable inference bundle for the joint population world model."""
from __future__ import annotations
import argparse, contextlib, json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from response_model import load as load_prior
from world_model import Config, SharedWorldModel

@contextlib.contextmanager
def _portable_attention():
    previous=torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try: yield
    finally: torch.backends.mha.set_fastpath_enabled(previous)

class JointWorldBundle:
    def __init__(self, modeldir, checkpoint='step-006000.safetensors', device='cpu'):
        self.root=Path(modeldir); self.device=torch.device(device)
        payload=json.loads((self.root/'config.json').read_text(encoding='utf-8'))
        self.settings=payload; self.config=Config(**payload['config'])
        self.model=SharedWorldModel(self.config).to(self.device)
        self.model.load_state_dict(load_file(self.root/'checkpoints'/checkpoint,device=str(self.device)),strict=True)
        self.model.eval()
        with np.load(self.root/'normalizer.npz',allow_pickle=False) as a:
            self.feature_mean=np.asarray(a['feature_mean'],np.float64)
            self.feature_scale=np.asarray(a['feature_scale'],np.float64)
        if self.feature_mean.shape!=(self.config.feature_dim,) or self.feature_scale.shape!=self.feature_mean.shape or np.any(self.feature_scale<=0): raise ValueError('invalid feature normalizer')
        self.adapters={}; self.priors={}
        for context, record in payload['contexts'].items():
            with np.load(self.root/'adapters'/f'{context}.npz',allow_pickle=False) as a:
                adapter={k:np.asarray(a[k]) for k in a.files}
            q=adapter['query_ids'].astype(str); f=np.asarray(adapter['query_features'],np.float32); observed=np.asarray(adapter['observed_query_mask'])
            indices=np.asarray(adapter['observation_indices'])
            if q.ndim!=1 or f.shape!=(len(q),self.config.feature_dim) or observed.shape!=(len(q),) or observed.dtype!=np.bool_ or indices.ndim!=1 or not np.issubdtype(indices.dtype,np.integer) or np.any(indices<0) or np.any(indices>=len(q)) or not observed[indices].all(): raise ValueError(f'invalid adapter for {context}')
            self.adapters[context]={'query_ids':q,'query_features':f,'observation_indices':indices.astype(np.int64),
                                    'observed_query_mask':observed.copy()}
            if 'control_context_values' in adapter:
                self.adapters[context]['control_context_values']=adapter['control_context_values']
                self.adapters[context]['control_context_mask']=adapter['control_context_mask']
            self.priors[context]=load_prior(self.root/'priors'/f'{context}.npz')

    def query_ids(self, context): return self.adapters[context]['query_ids'].copy()
    def supported_query_mask(self, context): return self.adapters[context]['observed_query_mask'].copy()
    def _normalize(self, values): return (np.asarray(values,np.float64)-self.feature_mean)/self.feature_scale

    @torch.inference_mode()
    def predict(self, context, action_features, action_mask, basal, observed=None, *, batch_size=16, query_chunk=512,
                control_context_values=None, control_context_mask=None):
        with _portable_attention():
            return self._predict(context,action_features,action_mask,basal,observed,batch_size=batch_size,query_chunk=query_chunk,
                control_context_values=control_context_values,control_context_mask=control_context_mask)

    def _predict(self, context, action_features, action_mask, basal, observed, *, batch_size, query_chunk,
                 control_context_values, control_context_mask):
        record=self.settings['contexts'][context]; adapter=self.adapters[context]; raw_actions=np.asarray(action_features,np.float64); mask=np.asarray(action_mask)
        base=np.asarray(basal,np.float64); obs=base if observed is None else np.asarray(observed,np.float64)
        if raw_actions.ndim!=3 or raw_actions.shape[2]!=self.config.feature_dim or mask.shape!=raw_actions.shape[:2] or mask.dtype!=np.bool_: raise ValueError('actions/mask must be [B,A,F]/Boolean [B,A]')
        b=len(raw_actions); q=len(adapter['query_ids'])
        if base.shape!=(b,q) or obs.shape!=(b,q) or not np.isfinite(raw_actions).all() or not np.isfinite(base).all() or not np.isfinite(obs).all(): raise ValueError('context-aligned finite basal/observed [B,Q] required')
        if batch_size <= 0 or query_chunk <= 0: raise ValueError('batch and query chunk sizes must be positive')
        if (control_context_values is None) != (control_context_mask is None): raise ValueError('control context values and mask are required together')
        if control_context_values is None:
            if int(record['assay']) == 0:
                control_context_values=base/np.log(2.)
                control_context_mask=np.ones_like(base,dtype=bool)
            elif 'control_context_values' in adapter:
                control_context_values=np.broadcast_to(adapter['control_context_values'],(b,q))
                control_context_mask=np.broadcast_to(adapter['control_context_mask'],(b,q))
            else:
                control_context_values=np.zeros_like(base)
                control_context_mask=np.zeros_like(base,dtype=bool)
        control_context_values=np.asarray(control_context_values,np.float32)
        control_context_mask=np.asarray(control_context_mask)
        if control_context_values.shape!=(b,q) or control_context_mask.shape!=(b,q) or control_context_mask.dtype!=np.bool_:
            raise ValueError('control context requires values and Boolean mask [B,Q]')
        if not np.isfinite(control_context_values[control_context_mask]).all(): raise ValueError('measured control context must be finite')
        scale=float(record['response_scale'])
        if not np.isfinite(scale) or scale<=0: raise ValueError('response_scale must be positive')
        flat=self.priors[context].predict(raw_actions.reshape(-1,self.config.feature_dim)).reshape(b,raw_actions.shape[1],q)
        prior=(flat*mask[...,None]).sum(1)
        normalized_actions=self._normalize(raw_actions).astype(np.float32)
        normalized_queries=self._normalize(adapter['query_features']).astype(np.float32)
        indices=adapter['observation_indices']; result=np.empty((b,q),np.float64)
        for start in range(0,b,batch_size):
            stop=min(start+batch_size,b); sl=slice(start,stop); n=stop-start
            modes=torch.full((n,),int(record['mode']),dtype=torch.long,device=self.device); assays=torch.full((n,),int(record['assay']),dtype=torch.long,device=self.device)
            state=self.model.encode(torch.as_tensor(obs[sl][:,indices]/scale,dtype=torch.float32,device=self.device),torch.as_tensor(base[sl][:,indices]/scale,dtype=torch.float32,device=self.device),torch.as_tensor(normalized_queries[indices],device=self.device),torch.ones((n,len(indices)),dtype=torch.bool,device=self.device),modes,assays,
                torch.as_tensor(control_context_values[sl][:,indices],device=self.device),torch.as_tensor(control_context_mask[sl][:,indices],device=self.device))
            changed=self.model.transition(state,torch.as_tensor(normalized_actions[sl],device=self.device),torch.as_tensor(mask[sl],device=self.device),modes,assays)
            pieces=[]
            for qs in range(0,q,query_chunk):
                qe=min(qs+query_chunk,q); features=torch.as_tensor(normalized_queries[qs:qe],device=self.device)
                pieces.append((self.model.decode(changed,features,assays)-self.model.decode(state,features,assays)).cpu().numpy())
            result[sl]=obs[sl].astype(np.float64)+prior[sl]+np.concatenate(pieces,1)*scale
        inactive=~mask.any(1); result[inactive]=obs[inactive]
        if not np.isfinite(result).all(): raise ValueError('nonfinite prediction')
        return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',type=Path,required=True); p.add_argument('--checkpoint',default='step-006000.safetensors'); p.add_argument('--context',required=True); p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    with np.load(a.input,allow_pickle=False) as x: values={k:np.asarray(x[k]) for k in x.files}
    if not {'actions','action_mask','basal'}<=set(values) or set(values)-{'actions','action_mask','basal','observed','control_context_values','control_context_mask'}: raise ValueError('invalid request arrays')
    bundle=JointWorldBundle(a.model,a.checkpoint); prediction=bundle.predict(a.context,values['actions'],values['action_mask'],values['basal'],values.get('observed'),control_context_values=values.get('control_context_values'),control_context_mask=values.get('control_context_mask'))
    np.savez_compressed(a.output,predictions=prediction,query_ids=bundle.query_ids(a.context),
                        prediction_supported=bundle.supported_query_mask(a.context))
if __name__=='__main__': main()
