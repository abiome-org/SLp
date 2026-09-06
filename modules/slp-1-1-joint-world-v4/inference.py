"""Portable inference bundle for the joint population world model."""
from __future__ import annotations
import argparse, contextlib, json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from response_model import load as load_prior, apply_prior_increment
from world_model import Config, SharedWorldModel

def load_file(path, device='cpu'):
    with safe_open(str(path), framework='pt', device=str(device)) as archive:
        return {name: archive.get_tensor(name) for name in archive.keys()}

@contextlib.contextmanager
def _portable_attention():
    previous=torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try: yield
    finally: torch.backends.mha.set_fastpath_enabled(previous)

class JointWorldBundle:
    def __init__(self, modeldir, checkpoint=None, device='cpu'):
        self.root=Path(modeldir); self.device=torch.device(device)
        payload=json.loads((self.root/'config.json').read_text(encoding='utf-8'))
        if checkpoint is None:
            manifest_path=self.root/'manifest.json'
            if manifest_path.is_file(): checkpoint=json.loads(manifest_path.read_text(encoding='utf-8')).get('selectedCheckpoint')
            if checkpoint is None:
                completed=int(payload.get('training',{}).get('steps_completed',0))
                if completed<=0: raise ValueError('specify a checkpoint for an unfinished training run')
                checkpoint=f'step-{completed:06d}.safetensors'
        if Path(checkpoint).name!=checkpoint or not checkpoint.endswith('.safetensors'): raise ValueError('checkpoint must be a safetensor filename')
        self.checkpoint=checkpoint; self.settings=payload; self.config=Config(**payload['config'])
        self.model=SharedWorldModel(self.config).to(self.device)
        self.model.load_state_dict(load_file(self.root/'checkpoints'/checkpoint,device=str(self.device)),strict=True); self.model.eval()
        with np.load(self.root/'normalizer.npz',allow_pickle=False) as a:
            self.feature_mean=np.asarray(a['feature_mean'],np.float64); self.feature_scale=np.asarray(a['feature_scale'],np.float64)
        if self.feature_mean.shape!=(self.config.feature_dim,) or self.feature_scale.shape!=self.feature_mean.shape or np.any(self.feature_scale<=0): raise ValueError('invalid feature normalizer')
        self.adapters={}; self.priors={}
        for context,record in payload['contexts'].items():
            with np.load(self.root/'adapters'/f'{context}.npz',allow_pickle=False) as a: adapter={k:np.asarray(a[k]) for k in a.files}
            q=adapter['query_ids'].astype(str); f=np.asarray(adapter['query_features'],np.float32); observed=np.asarray(adapter['observed_query_mask']); indices=np.asarray(adapter['observation_indices'])
            if q.ndim!=1 or f.shape!=(len(q),self.config.feature_dim) or observed.shape!=(len(q),) or observed.dtype!=np.bool_ or indices.ndim!=1 or not np.issubdtype(indices.dtype,np.integer) or np.any(indices<0) or np.any(indices>=len(q)) or not observed[indices].all(): raise ValueError(f'invalid adapter for {context}')
            self.adapters[context]={'query_ids':q,'query_features':f,'observation_indices':indices.astype(np.int64),'observed_query_mask':observed.copy()}
            if 'control_context_values' in adapter:
                self.adapters[context]['control_context_values']=adapter['control_context_values']; self.adapters[context]['control_context_mask']=adapter['control_context_mask']
            self.priors[context]=load_prior(self.root/'priors'/f'{context}.npz')

    def query_ids(self,context): return self.adapters[context]['query_ids'].copy()
    def supported_query_mask(self,context): return self.adapters[context]['observed_query_mask'].copy()
    def _normalize(self,values): return (np.asarray(values,np.float64)-self.feature_mean)/self.feature_scale

    def _inputs(self,context,action_features,action_mask,basal,observed,control_context_values,control_context_mask,batch_size,query_chunk):
        if context not in self.adapters: raise ValueError(f'unknown context: {context}')
        record=self.settings['contexts'][context]; adapter=self.adapters[context]
        raw=np.asarray(action_features,np.float64); mask=np.asarray(action_mask); base=np.asarray(basal,np.float64); obs=base if observed is None else np.asarray(observed,np.float64)
        if raw.ndim!=3 or raw.shape[2]!=self.config.feature_dim or mask.shape!=raw.shape[:2] or mask.dtype!=np.bool_: raise ValueError('actions/mask must be [B,A,F]/Boolean [B,A]')
        b=len(raw); q=len(adapter['query_ids'])
        if base.shape!=(b,q) or obs.shape!=(b,q) or not np.isfinite(raw).all() or not np.isfinite(base).all() or not np.isfinite(obs).all(): raise ValueError('context-aligned finite basal/observed [B,Q] required')
        if batch_size<=0 or query_chunk<=0: raise ValueError('batch and query chunk sizes must be positive')
        if (control_context_values is None)!=(control_context_mask is None): raise ValueError('control context values and mask are required together')
        if control_context_values is None:
            if int(record['assay'])==0: control_context_values=base/np.log(2.); control_context_mask=np.ones_like(base,dtype=bool)
            elif 'control_context_values' in adapter:
                control_context_values=np.broadcast_to(adapter['control_context_values'],(b,q)); control_context_mask=np.broadcast_to(adapter['control_context_mask'],(b,q))
            else: control_context_values=np.zeros_like(base); control_context_mask=np.zeros_like(base,dtype=bool)
        cv=np.asarray(control_context_values,np.float32); cm=np.asarray(control_context_mask)
        if cv.shape!=(b,q) or cm.shape!=(b,q) or cm.dtype!=np.bool_: raise ValueError('control context requires values and Boolean mask [B,Q]')
        if not np.isfinite(cv[cm]).all(): raise ValueError('measured control context must be finite')
        scale=float(record['response_scale'])
        if not np.isfinite(scale) or scale<=0: raise ValueError('response_scale must be positive')
        return record,adapter,raw,mask,base,obs,cv,cm,scale

    @torch.inference_mode()
    def predict(self,context,action_features,action_mask,basal,observed=None,*,batch_size=16,query_chunk=512,control_context_values=None,control_context_mask=None):
        with _portable_attention():
            return self._predict(context,action_features,action_mask,basal,observed,batch_size=batch_size,query_chunk=query_chunk,control_context_values=control_context_values,control_context_mask=control_context_mask,sequential=False)

    @torch.inference_mode()
    def predict_latent_rollout(self,context,actions,action_mask,basal,observed=None,control_context_values=None,control_context_mask=None,batch_size=16,query_chunk=512):
        """Apply active action slots in their supplied order without RNA re-encoding."""
        with _portable_attention():
            return self._predict(context,actions,action_mask,basal,observed,batch_size=batch_size,query_chunk=query_chunk,control_context_values=control_context_values,control_context_mask=control_context_mask,sequential=True)

    def _predict(self,context,action_features,action_mask,basal,observed,*,batch_size,query_chunk,control_context_values,control_context_mask,sequential):
        record,adapter,raw,mask,base,obs,cv,cm,scale=self._inputs(context,action_features,action_mask,basal,observed,control_context_values,control_context_mask,batch_size,query_chunk)
        b,a=mask.shape; q=len(adapter['query_ids'])
        flat=self.priors[context].predict(raw.reshape(-1,self.config.feature_dim)).reshape(b,a,q)
        from_control=np.all(obs==base,axis=1)
        prior=apply_prior_increment(flat,self.priors[context].intercept,mask,from_control,float(record.get('template_saturation',0.)))
        na=self._normalize(raw).astype(np.float32); nq=self._normalize(adapter['query_features']).astype(np.float32); indices=adapter['observation_indices']; result=np.empty((b,q),np.float64)
        for start in range(0,b,batch_size):
            stop=min(start+batch_size,b); sl=slice(start,stop); n=stop-start
            modes=torch.full((n,),int(record['mode']),dtype=torch.long,device=self.device); assays=torch.full((n,),int(record['assay']),dtype=torch.long,device=self.device)
            state=self.model.encode(torch.as_tensor(obs[sl][:,indices]/scale,dtype=torch.float32,device=self.device),torch.as_tensor(base[sl][:,indices]/scale,dtype=torch.float32,device=self.device),torch.as_tensor(nq[indices],device=self.device),torch.ones((n,len(indices)),dtype=torch.bool,device=self.device),modes,assays,torch.as_tensor(cv[sl][:,indices],device=self.device),torch.as_tensor(cm[sl][:,indices],device=self.device))
            actions=torch.as_tensor(na[sl],device=self.device); action_mask_t=torch.as_tensor(mask[sl],device=self.device)
            if sequential:
                order=torch.arange(a,dtype=torch.long,device=self.device)[None].expand(n,-1)
                changed=self.model.transition_chain(state,actions,action_mask_t,order,modes,assays)
            else: changed=self.model.transition(state,actions,action_mask_t,modes,assays)
            pieces=[]
            for qs in range(0,q,query_chunk):
                qe=min(qs+query_chunk,q); features=torch.as_tensor(nq[qs:qe],device=self.device)
                pieces.append((self.model.decode(changed,features,assays)-self.model.decode(state,features,assays)).cpu().numpy())
            result[sl]=obs[sl]+prior[sl]+np.concatenate(pieces,1)*scale
        inactive=~mask.any(1); result[inactive]=obs[inactive]
        if not np.isfinite(result).all(): raise ValueError('nonfinite prediction')
        return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',type=Path,required=True); p.add_argument('--checkpoint'); p.add_argument('--context',required=True); p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--rollout',choices=('direct','latent'),default='direct'); a=p.parse_args()
    with np.load(a.input,allow_pickle=False) as x: values={k:np.asarray(x[k]) for k in x.files}
    if not {'actions','action_mask','basal'}<=set(values) or set(values)-{'actions','action_mask','basal','observed','control_context_values','control_context_mask'}: raise ValueError('invalid request arrays')
    bundle=JointWorldBundle(a.model,a.checkpoint)
    predictor=bundle.predict_latent_rollout if a.rollout=='latent' else bundle.predict
    prediction=predictor(a.context,values['actions'],values['action_mask'],values['basal'],values.get('observed'),control_context_values=values.get('control_context_values'),control_context_mask=values.get('control_context_mask'))
    np.savez_compressed(a.output,predictions=prediction,query_ids=bundle.query_ids(a.context),prediction_supported=bundle.supported_query_mask(a.context))
if __name__=='__main__': main()
