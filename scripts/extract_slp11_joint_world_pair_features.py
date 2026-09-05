"""Extract label-free pair features from a frozen joint-world model bundle."""
from __future__ import annotations
import argparse, contextlib, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import torch

ALLOWED_INPUTS={'pair_ids','gene_indices','action_features','basal','observed','control_context_values','control_context_mask'}

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def load_module(module_dir):
    module_dir=Path(module_dir).resolve();sys.path.insert(0,str(module_dir))
    spec=importlib.util.spec_from_file_location('joint_world_pair_bridge_inference',module_dir/'inference.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module

def canonicalize(pair_ids,features):
    ids=np.asarray(pair_ids).astype(str);features=np.asarray(features)
    if ids.ndim!=2 or ids.shape[1]!=2 or features.ndim!=3 or features.shape[:2]!=ids.shape:
        raise ValueError('pair_ids/action_features must be [N,2] and [N,2,F]')
    if np.any(ids=='') or np.any(ids[:,0]==ids[:,1]):raise ValueError('pairs require two distinct stable IDs')
    swap=ids[:,0]>ids[:,1];ids=ids.copy();features=features.copy()
    ids[swap]=ids[swap,::-1];features[swap]=features[swap,::-1]
    return ids,features

def rows(values,n,q,label,dtype=np.float64):
    value=np.asarray(values,dtype=dtype)
    if value.shape==(q,):value=np.broadcast_to(value,(n,q))
    if value.shape!=(n,q):raise ValueError(f'{label} must be [Q] or [N,Q]')
    return value

def prediction_summary(single_a,single_b,pair,sequential_a_b,sequential_b_a,anchor,supported):
    supported=np.asarray(supported,bool);arrays=[np.asarray(x,np.float64)[:,supported] for x in
        (single_a,single_b,pair,sequential_a_b,sequential_b_a,anchor)]
    a,b,p,ab,ba,base=arrays;add=a+b-base;interaction=p-add;seq=.5*(ab+ba);seq_interaction=seq-add
    def stats(x):
        return np.stack((x.mean(1),np.sqrt(np.square(x).mean(1)),np.abs(x).mean(1),
                         np.abs(x).max(1),(x>1e-3).mean(1)),1)
    stats_a=stats(a-base);stats_b=stats(b-base)
    return np.concatenate((stats_a+stats_b,np.abs(stats_a-stats_b),stats(p-base),stats(interaction),
                           stats(seq_interaction),stats(np.abs(ab-ba))),1).astype(np.float32)

def latent_pair_features(delta_a,delta_b,delta_pair,second_after_a,second_after_b):
    route_ab=delta_a+second_after_a;route_ba=delta_b+second_after_b
    sequential=.5*(route_ab+route_ba)
    return np.concatenate((delta_a+delta_b,np.abs(delta_a-delta_b),delta_a*delta_b,delta_pair,
        delta_pair-delta_a-delta_b,sequential,sequential-delta_a-delta_b,np.abs(route_ab-route_ba)),1)

class WorldPairFeatureBridge:
    def __init__(self,bundle,context):
        if context not in bundle.settings['contexts']:raise KeyError(f'unknown model context {context}')
        self.bundle=bundle;self.context=context

    def _states(self,actions,mask,basal,observed,control,control_mask):
        b=len(actions);bundle=self.bundle;adapter=bundle.adapters[self.context];record=bundle.settings['contexts'][self.context]
        q=len(adapter['query_ids']);indices=adapter['observation_indices'];scale=float(record['response_scale'])
        normalized_actions=bundle._normalize(actions).astype(np.float32)
        normalized_queries=bundle._normalize(adapter['query_features']).astype(np.float32)
        modes=torch.full((b,),int(record['mode']),dtype=torch.long,device=bundle.device)
        assays=torch.full((b,),int(record['assay']),dtype=torch.long,device=bundle.device)
        with torch.inference_mode():
            state=bundle.model.encode(torch.as_tensor(observed[:,indices]/scale,dtype=torch.float32,device=bundle.device),
                torch.as_tensor(basal[:,indices]/scale,dtype=torch.float32,device=bundle.device),
                torch.as_tensor(normalized_queries[indices],device=bundle.device),
                torch.ones((b,len(indices)),dtype=torch.bool,device=bundle.device),modes,assays,
                torch.as_tensor(control[:,indices],dtype=torch.float32,device=bundle.device),
                torch.as_tensor(control_mask[:,indices],dtype=torch.bool,device=bundle.device))
            changed=bundle.model.transition(state,torch.as_tensor(normalized_actions,device=bundle.device),
                torch.as_tensor(mask,device=bundle.device),modes,assays)
        return state,changed

    def extract(self,pair_ids,action_features,basal,observed=None,control_context_values=None,
                control_context_mask=None,batch_size=16):
        pair_ids,action_features=canonicalize(pair_ids,action_features);n=len(pair_ids)
        q=len(self.bundle.query_ids(self.context));basal=rows(basal,n,q,'basal')
        observed=basal if observed is None else rows(observed,n,q,'observed')
        if (control_context_values is None)!=(control_context_mask is None):raise ValueError('control context values and mask are required together')
        if control_context_values is None:
            record=self.bundle.settings['contexts'][self.context];adapter=self.bundle.adapters[self.context]
            if int(record['assay'])==0:
                control=basal/np.log(2.);control_mask=np.ones((n,q),bool)
            elif 'control_context_values' in adapter:
                control=rows(adapter['control_context_values'],n,q,'adapter control context')
                control_mask=rows(adapter['control_context_mask'],n,q,'adapter control mask',bool)
            else:control=np.zeros((n,q));control_mask=np.zeros((n,q),bool)
        else:
            control=rows(control_context_values,n,q,'control context')
            control_mask=rows(control_context_mask,n,q,'control context mask',bool)
        output=[]
        for start in range(0,n,batch_size):
            stop=min(start+batch_size,n);sl=slice(start,stop);f=action_features[sl];base=basal[sl];obs=observed[sl]
            one=np.ones((stop-start,1),bool);two=np.ones((stop-start,2),bool)
            kw={'control_context_values':control[sl],'control_context_mask':control_mask[sl]}
            pred_a=self.bundle.predict(self.context,f[:,:1],one,base,observed=obs,**kw)
            pred_b=self.bundle.predict(self.context,f[:,1:],one,base,observed=obs,**kw)
            pred_pair=self.bundle.predict(self.context,f,two,base,observed=obs,**kw)
            pred_ab=self.bundle.predict(self.context,f[:,1:],one,base,observed=pred_a,**kw)
            pred_ba=self.bundle.predict(self.context,f[:,:1],one,base,observed=pred_b,**kw)
            z0,za=self._states(f[:,:1],one,base,obs,control[sl],control_mask[sl]);_,zb=self._states(f[:,1:],one,base,obs,control[sl],control_mask[sl])
            _,zp=self._states(f,two,base,obs,control[sl],control_mask[sl])
            zab0,zab=self._states(f[:,1:],one,base,pred_a,control[sl],control_mask[sl]);zba0,zba=self._states(f[:,:1],one,base,pred_b,control[sl],control_mask[sl])
            pool=lambda x:x.mean(1).cpu().numpy();d_a=pool(za-z0);d_b=pool(zb-z0);d_pair=pool(zp-z0)
            latent=latent_pair_features(d_a,d_b,d_pair,pool(zab-zab0),pool(zba-zba0))
            summaries=prediction_summary(pred_a,pred_b,pred_pair,pred_ab,pred_ba,obs,self.bundle.supported_query_mask(self.context))
            output.append(np.concatenate((latent.astype(np.float32),summaries),1))
        features=np.concatenate(output,0) if output else np.empty((0,8*self.bundle.config.width+30),np.float32)
        if not np.isfinite(features).all():raise ValueError('extracted pair features are nonfinite')
        return pair_ids,features

    def extract_cached(self,pair_ids,gene_indices,gene_features,basal,control,control_mask,batch_size=256,query_chunk=512):
        """Exact shared-control extraction with each unique single encoded once."""
        pair_ids=np.asarray(pair_ids).astype(str);indices=np.asarray(gene_indices,np.int64);gene_features=np.asarray(gene_features,np.float32)
        if indices.shape!=pair_ids.shape or indices.ndim!=2 or np.any(indices<0) or np.any(indices>=len(gene_features)):raise ValueError('invalid compact pair indices')
        n=len(pair_ids);q=len(self.bundle.query_ids(self.context));base=rows(basal,1,q,'shared basal')[0]
        control=rows(control,1,q,'shared control')[0];control_mask=rows(control_mask,1,q,'shared control mask',bool)[0]
        used=np.unique(indices);local={gene:i for i,gene in enumerate(used)};f=gene_features[used]
        pos=np.vectorize(local.__getitem__)(indices);record=self.bundle.settings['contexts'][self.context];scale=float(record['response_scale'])
        adapter=self.bundle.adapters[self.context];normalized_queries=self.bundle._normalize(adapter['query_features']).astype(np.float32)
        assay=int(record['assay']);mode=int(record['mode']);saturation=float(record.get('template_saturation',0.));prior_model=self.bundle.priors[self.context]
        with torch.no_grad():
            projected_query=self.bundle.model.decode_query(torch.as_tensor(normalized_queries,device=self.bundle.device))
            projected_query=projected_query+self.bundle.model.assay.weight[assay][None,:]
        @torch.inference_mode()
        def decode_state(state,assays):
            pieces=[];count=len(state)
            for qs in range(0,q,query_chunk):
                query=projected_query[qs:qs+query_chunk].unsqueeze(0).expand(count,-1,-1)
                readout,_=self.bundle.model.decode_attention(query,state,state,need_weights=False)
                joined=torch.cat((query,self.bundle.model.decode_norm(query+readout)),-1)
                pieces.append(self.bundle.model.assay_heads[assay](joined).squeeze(-1).cpu().numpy())
            return np.concatenate(pieces,1)
        single_predictions=[];single_states=[];parent_states=[];parent_decodes=[];single_priors=[];base_state=None;base_decode=None
        for start in range(0,len(f),batch_size):
            chunk=f[start:start+batch_size];count=len(chunk);one=np.ones((count,1),bool)
            b=np.broadcast_to(base,(count,q));c=np.broadcast_to(control,(count,q));cm=np.broadcast_to(control_mask,(count,q))
            z0,z1=self._states(chunk[:,None],one,b,b,c,cm)
            if base_state is None:
                base_state=z0[:1].detach().cpu();base_decode=decode_state(z0[:1],torch.full((1,),assay,dtype=torch.long,device=self.bundle.device))[0]
            prior=prior_model.predict(chunk);prediction=base+prior+(decode_state(z1,torch.full((count,),assay,dtype=torch.long,device=self.bundle.device))-base_decode)*scale
            zp,_=self._states(chunk[:,None],one,b,prediction,c,cm)
            single_predictions.append(prediction.astype(np.float32));single_priors.append(prior);single_states.append(z1.detach().cpu());parent_states.append(zp.detach().cpu())
            parent_decodes.append(decode_state(zp,torch.full((count,),assay,dtype=torch.long,device=self.bundle.device)).astype(np.float32))
        single_predictions=np.concatenate(single_predictions);single_priors=np.concatenate(single_priors)
        single_states=torch.cat(single_states);parent_states=torch.cat(parent_states);parent_decodes=np.concatenate(parent_decodes)
        output=[];started=time.monotonic()
        with torch.inference_mode():
            for start in range(0,n,batch_size):
                stop=min(start+batch_size,n);take=pos[start:stop];count=len(take);raw=gene_features[indices[start:stop]]
                actions=torch.as_tensor(self.bundle._normalize(raw).astype(np.float32),device=self.bundle.device);mask=torch.ones((count,2),dtype=torch.bool,device=self.bundle.device)
                modes=torch.full((count,),mode,dtype=torch.long,device=self.bundle.device);assays=torch.full((count,),assay,dtype=torch.long,device=self.bundle.device)
                z0=base_state.to(self.bundle.device).expand(count,-1,-1);za=single_states[take[:,0]].to(self.bundle.device);zb=single_states[take[:,1]].to(self.bundle.device)
                zpa=parent_states[take[:,0]].to(self.bundle.device);zpb=parent_states[take[:,1]].to(self.bundle.device)
                zp=self.bundle.model.transition(z0,actions,mask,modes,assays)
                zab=self.bundle.model.transition(zpa,actions[:,1:2],mask[:,:1],modes,assays);zba=self.bundle.model.transition(zpb,actions[:,:1],mask[:,:1],modes,assays)
                flat=single_priors[take]
                pair_prior=flat.sum(1)-saturation*prior_model.intercept
                conditional_a=flat[:,0]-saturation*prior_model.intercept;conditional_b=flat[:,1]-saturation*prior_model.intercept
                pred_a=single_predictions[take[:,0]].astype(np.float64);pred_b=single_predictions[take[:,1]].astype(np.float64)
                pred_pair=base+pair_prior+(decode_state(zp,assays)-base_decode)*scale
                pred_ab=pred_a+conditional_b+(decode_state(zab,assays)-parent_decodes[take[:,0]])*scale
                pred_ba=pred_b+conditional_a+(decode_state(zba,assays)-parent_decodes[take[:,1]])*scale
                pool=lambda x:x.mean(1).cpu().numpy();da=pool(za-z0);db=pool(zb-z0);dp=pool(zp-z0);sab=pool(zab-zpa);sba=pool(zba-zpb)
                latent=latent_pair_features(da,db,dp,sab,sba)
                summary=prediction_summary(pred_a,pred_b,pred_pair,pred_ab,pred_ba,np.broadcast_to(base,(count,q)),self.bundle.supported_query_mask(self.context))
                output.append(np.concatenate((latent.astype(np.float32),summary),1))
                if stop==n or stop%4096<batch_size:print(json.dumps({'pairsCompleted':stop,'pairsTotal':n,'elapsedSeconds':time.monotonic()-started}),flush=True)
        result=np.concatenate(output) if output else np.empty((0,8*self.bundle.config.width+30),np.float32)
        if not np.isfinite(result).all():raise ValueError('cached pair features are nonfinite')
        return pair_ids,result

def main():
    p=argparse.ArgumentParser();p.add_argument('--module',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--checkpoint');p.add_argument('--context',required=True);p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--genes',type=Path,help='compact gene feature table used with pair gene_indices')
    p.add_argument('--context-request',type=Path,help='label-free basal/control arrays kept separate from the pair roster')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--cached-shared-control',action='store_true');p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    with np.load(a.pairs,allow_pickle=False) as archive:
        unknown=set(archive.files)-ALLOWED_INPUTS
        if unknown:raise ValueError(f'pair roster contains forbidden or unknown arrays: {sorted(unknown)}')
        values={key:np.asarray(archive[key]) for key in archive.files}
    gene_features=None
    if 'action_features' not in values:
        if a.genes is None or 'gene_indices' not in values:raise ValueError('action_features or --genes plus gene_indices is required')
        with np.load(a.genes,allow_pickle=False) as genes:
            if set(genes.files)-{'gene_ids','action_features','ncbi_taxon'}:raise ValueError('invalid compact gene table arrays')
            gene_ids=genes['gene_ids'].astype(str);gene_features=np.asarray(genes['action_features'],np.float32)
        indices=np.asarray(values['gene_indices'])
        if indices.shape!=values['pair_ids'].shape or np.any(indices<0) or np.any(indices>=len(gene_ids)):
            raise ValueError('pair gene_indices do not align with the gene table')
        if not np.array_equal(gene_ids[indices],values['pair_ids'].astype(str)):raise ValueError('pair IDs disagree with compact gene indices')
        if not a.cached_shared_control:values['action_features']=gene_features[indices]
    request_query_ids=None
    if a.context_request is not None:
        with np.load(a.context_request,allow_pickle=False) as request:
            allowed={'query_ids','basal','observed','control_context_values','control_context_mask'}
            if set(request.files)-allowed:raise ValueError('invalid context request arrays')
            if 'query_ids' in request:request_query_ids=request['query_ids'].astype(str)
            for key in request.files:
                if key!='query_ids':values[key]=np.asarray(request[key])
    required={'pair_ids','basal'}|({'gene_indices'} if a.cached_shared_control else {'action_features'})
    if not required<=set(values):raise ValueError(f'missing required arrays: {sorted(required-set(values))}')
    module=load_module(a.module);bundle=module.JointWorldBundle(a.model,a.checkpoint,a.device)
    if request_query_ids is not None and not np.array_equal(request_query_ids,bundle.query_ids(a.context)):
        raise ValueError('context request query axis disagrees with model adapter')
    total=len(values['pair_ids']);stop=total if a.stop is None else a.stop
    if not 0<=a.start<=stop<=total:raise ValueError('start/stop outside pair roster')
    take=slice(a.start,stop);bridge=WorldPairFeatureBridge(bundle,a.context)
    if a.cached_shared_control:
        if gene_features is None or 'gene_indices' not in values:raise ValueError('cached extraction requires compact --genes input')
        if values.get('observed') is not None:raise ValueError('cached shared-control extraction requires observed=basal')
        previous=torch.backends.mha.get_fastpath_enabled();torch.backends.mha.set_fastpath_enabled(False)
        try:
            pair_ids,features=bridge.extract_cached(values['pair_ids'][take],values['gene_indices'][take],gene_features,
                values['basal'],values['control_context_values'],values['control_context_mask'],a.batch_size)
        finally:torch.backends.mha.set_fastpath_enabled(previous)
    else:
        pair_ids,features=bridge.extract(values['pair_ids'][take],values['action_features'][take],values['basal'],values.get('observed'),
            values.get('control_context_values'),values.get('control_context_mask'),a.batch_size)
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,pair_ids=pair_ids,features=features,
        feature_schema=np.asarray('slp.joint-world-symmetric-pair-features/v1'),
        feature_description=np.asarray('8W symmetric pooled latent coordinates: single sum, absolute difference, product, direct pair, direct nonadditivity, sequential average, sequential nonadditivity, absolute order difference; then 30 prediction-summary coordinates: single-stat sum, single-stat absolute difference, direct pair, direct nonadditivity, sequential nonadditivity, absolute order difference'),
        context=np.asarray(a.context),checkpoint=np.asarray(bundle.checkpoint),model_checkpoint_sha256=np.asarray(sha256(a.model/'checkpoints'/bundle.checkpoint)),
        source_pair_roster_sha256=np.asarray(sha256(a.pairs)),source_start=np.asarray(a.start),source_stop=np.asarray(stop),benchmark_labels_accessed=np.asarray(False))
    print(json.dumps({'pairs':len(pair_ids),'features':features.shape[1],'context':a.context,'output':str(a.output)},sort_keys=True))
if __name__=='__main__':main()
