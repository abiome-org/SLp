"""Molecular development evaluation, including held genes and paired-cell generation."""
from __future__ import annotations
import argparse,json,time,struct,zipfile,shutil
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import cdist
from data import Corpus,arrays
from inference import WorldModel
from prepare import sha,write


def scores(predicted,truth,basal,observed,mean_response=None):
    observed=np.asarray(observed,bool);p=np.asarray(predicted,np.float64);t=np.asarray(truth,np.float64)
    error=(p-t)[observed];result={'mse':float(np.mean(error**2)),'unchanged_mse':float(np.mean((t-basal)[observed]**2)),
        'rows':len(t),'observed_coordinates':int(observed.sum())}
    if mean_response is not None:result['training_mean_mse']=float(np.mean((t-basal-mean_response)[observed]**2))
    # Remove each query's average perturbation effect across this evaluated
    # population. These centroids are used for scoring, never for predictions.
    n=observed.sum(0).clip(1);pd=p-basal;td=t-basal
    pd=pd-np.where(observed,pd,0).sum(0)/n;td=td-np.where(observed,td,0).sum(0)/n
    x=pd[observed];y=td[observed];den=np.linalg.norm(x)*np.linalg.norm(y)
    result['centered_landscape_correlation']=float(np.dot(x,y)/den) if den>1e-12 else None
    return result


def predict(world,source,targets,basal,ids,observed,query_indices=None,limit=0,batch=16):
    q=np.arange(len(source.query)) if query_indices is None else query_indices
    targets=targets[:,q];basal=basal[:,q];observed=observed[:,q]&source.support[q][None]
    if limit:targets=targets[:limit];basal=basal[:limit];ids=ids[:limit];observed=observed[:limit]
    eligible=np.flatnonzero(source.support[q]&(source.modality[q]==0));encoder=np.random.default_rng(1731).choice(eligible,min(192,len(eligible)),replace=False)
    scale=np.broadcast_to(source.scale,(len(source.query),))[q].astype(np.float32)
    output=[];control=getattr(source,'control',None)
    for start in range(0,len(targets),batch):
        end=min(start+batch,len(targets));base=basal[start:end]
        c=base if control is None else np.broadcast_to(control[q],base.shape)
        state=world.encode(base,base,source.query[q],modality=source.modality[q],scale=scale,
            assay=source.assay,taxon=source.taxon,mechanism=source.mechanism,control=c,observed_mask=observed[start:end],
            encoder_indices=encoder,normalized_descriptors=True)
        actions,mask=source.action(ids[start:end]);changed=world.intervene(state,actions,mask,normalized_descriptors=True)
        output.append(world.decode(changed)['values'])
    return np.concatenate(output),targets,basal,observed


def population_development(world,corpus,root,output,limit,queries):
    reports={};predictions={};excluded=corpus.excluded
    for name in ('k562','rpe1','gwps','hepg2'):
        source=next(s for s in corpus.sources if s.name==name)
        path=root/f'data/derived/slp11-joint-world-expanded-development-string-v1/{name}.npz';z=arrays(path)
        if not np.array_equal(z['query_ids'],source.query_ids):raise ValueError('development query panel differs')
        ids=z['gene_ids'].astype(str)[:,None]
        if not set(ids.reshape(-1))<=excluded:raise ValueError('held genes absent from global fitting exclusion')
        truth=z['truth'];basal=z.get('basal',z['control_prediction']);observed=z.get('observed',np.ones_like(truth,bool))
        qi=None if not queries else np.sort(np.random.default_rng(731).choice(np.flatnonzero(source.support),queries,replace=False))
        p,t,b,m=predict(world,source,truth,basal,ids,observed,qi,limit)
        mean=np.where(source.observed,source.target-source.basal,0).sum(0)/source.observed.sum(0).clip(1)
        mean=mean if qi is None else mean[qi]
        report=scores(p,t,b,m,mean);report['source_sha256']=sha(path);report['taxon']=9606
        # Multiple source views of a held gene are explicitly pooled, equally
        # weighting views here; the source-view metric above remains available.
        used_ids=ids[:len(p),0];genes=np.unique(used_ids)
        if len(genes)<len(p):
            pp=[];tt=[];bb=[];mm=[]
            for gene in genes:
                take=used_ids==gene;valid=m[take];counts=valid.sum(0);mm.append(counts>0)
                pp.append((p[take]*valid).sum(0)/counts.clip(1));tt.append((t[take]*valid).sum(0)/counts.clip(1));bb.append((b[take]*valid).sum(0)/counts.clip(1))
            report['unique_gene']=scores(np.array(pp),np.array(tt),np.array(bb),np.array(mm),mean)
        reports[name]=report
        predictions[name+'_prediction']=p;predictions[name+'_action_ids']=used_ids
        predictions[name+'_query_ids']=source.query_ids if qi is None else source.query_ids[qi]
        predictions[name+'_observed_mask']=m
        print(json.dumps({'event':'molecular_development','source':name,**report}),flush=True)
    np.savez_compressed(output/'human-predictions.npz',**predictions)
    return reports


def combination_development(world,corpus,root,limit,queries):
    reports={}
    for source in corpus.sources:
        if source.name not in ('norman','mcf10a_full_d0','mcf10a_full_d6','mcf10a_tgfb1_d6'):continue
        z=arrays(root/source.record['path']);rows=z['combination_rows'][z['combination_fold']==0]
        # Pair test genes must have fitting singles and statically described actions.
        roster=z['action_ids'];mask=z['action_mask'];ids=np.full(mask.shape,'',dtype=roster.dtype)
        if roster.ndim==2:ids=roster
        else:
            for i in range(len(mask)):ids[i,mask[i]]=roster[z['action_offsets'][i]:z['action_offsets'][i+1]]
        fitting=set(source.ids.reshape(-1));rows=np.array([r for r in rows if set(ids[r,mask[r]])<=fitting],np.int64)
        if limit:rows=rows[:limit]
        qi=np.arange(len(source.query)) if not queries else np.sort(np.random.default_rng(731).choice(np.flatnonzero(source.support),queries,replace=False))
        target=z['targets'][rows];basal=z['basal'][rows]
        observed=z.get('observed',z.get('target_observed',np.ones_like(z['targets'],bool)))[rows]
        direct,t,b,m=predict(world,source,target,basal,ids[rows],observed,qi)
        latent=[];reverse=[];additive=[]
        for start in range(0,len(rows),16):
            r=rows[start:start+16];bb=z['basal'][r][:,qi];qmask=source.support[qi]
            enc=np.random.default_rng(1731).choice(np.flatnonzero(qmask),min(192,qmask.sum()),False)
            state=world.encode(bb,bb,source.query[qi],modality=source.modality[qi],scale=source.scale,assay=source.assay,taxon=source.taxon,
                mechanism=source.mechanism,encoder_indices=enc,normalized_descriptors=True)
            actions,am=source.action(ids[r]);first=am.copy();first[:,1:]=False;second=am&~first
            a=world.intervene(state,actions,first,normalized_descriptors=True);bs=world.intervene(state,actions,second,normalized_descriptors=True)
            ab=world.intervene(a,actions,second,normalized_descriptors=True);ba=world.intervene(bs,actions,first,normalized_descriptors=True)
            latent.append(world.decode(ab)['values']);reverse.append(world.decode(ba)['values'])
            additive.append(world.decode(a)['values']+world.decode(bs)['values']-bb)
        ab=np.concatenate(latent);ba=np.concatenate(reverse);add=np.concatenate(additive)
        reports[source.name]={'direct':scores(direct,t,b,m),'latent_two_order_average':scores((ab+ba)*.5,t,b,m),
            'predicted_additive':scores(add,t,b,m),'latent_order_rms':float(np.sqrt(np.mean((ab-ba)[m]**2))),
            'scope':'known-gene, held-pair molecular development; simultaneous endpoints, not biological time courses'}
        print(json.dumps({'event':'combination_development','source':source.name,**reports[source.name]}),flush=True)
    return reports


def paired_gene_development(world,corpus,root,output,limit,queries):
    source=next(s for s in corpus.sources if s.name=='frangieh_cells')
    path=root/'data/derived/slp11-frangieh/paired-development-v1/development.npz';z=arrays(path)
    rows=z['split_validation'];known=np.array([str(z['action_ids'][r]) in source.features.human for r in rows]);missing=int((~known).sum());rows=rows[known]
    if not set(z['action_ids'][rows].astype(str))<=corpus.excluded:raise ValueError('paired validation genes entered fitting')
    rna=np.flatnonzero(source.support&(source.modality==0));protein=np.flatnonzero(source.modality==1)
    qi=np.concatenate((np.sort(np.random.default_rng(731).choice(rna,min(queries,len(rna)),False)),protein)) if queries else np.arange(len(source.query))
    reports={};payload={};fitting=source.pop
    for ctx,index in source.contexts.items():
        rr=rows[z['context_ids'][rows]==ctx]
        if limit:rr=rr[:limit]
        target=np.concatenate((z['rna_targets'][rr],z['protein_targets'][rr]),1)
        observed=np.concatenate((z['rna_observed'][rr],z['protein_observed'][rr]),1)
        basal=np.broadcast_to(source.basal[index],target.shape)
        prediction,t,b,m=predict(world,source,target,basal,z['action_ids'][rr,None],observed,qi)
        fitting_rows=np.flatnonzero(fitting['context_ids']==ctx)
        fitting_targets=np.concatenate((fitting['rna_targets'][fitting_rows],fitting['protein_targets'][fitting_rows]),1)
        mean=np.average(fitting_targets-source.basal[index],axis=0,weights=fitting['num_cells'][fitting_rows])[qi]
        reports[ctx]={}
        for modality,label in ((0,'rna'),(1,'protein')):
            take=source.modality[qi]==modality
            reports[ctx][label]=scores(prediction[:,take],t[:,take],b[:,take],m[:,take],mean[take])
        payload[ctx+'_prediction']=prediction;payload[ctx+'_action_ids']=z['action_ids'][rr];payload[ctx+'_query_ids']=source.query_ids[qi]
    np.savez_compressed(output/'paired-held-gene-predictions.npz',**payload)
    return {'contexts':reports,'missing_static_descriptor_populations':missing,'source_sha256':sha(path),
        'scope':'globally held intervention genes; deterministic prediction conditioned on each context control mean; not a Monte Carlo population estimator'}


def energy(x,y):
    return float(2*cdist(x,y).mean()-cdist(x,x).mean()-cdist(y,y).mean())


def stored_member(path,key):
    """Map only the requested uncompressed numeric member of a source NPZ."""
    with zipfile.ZipFile(path) as archive:
        info=archive.getinfo(key+'.npy')
        if info.compress_type!=zipfile.ZIP_STORED:raise ValueError('expected shard-streamable stored NPY member')
        with Path(path).open('rb') as f:
            f.seek(info.header_offset);header=f.read(30);name,extra=struct.unpack('<HH',header[26:30]);f.seek(name+extra,1)
            version=np.lib.format.read_magic(f)
            if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
            elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
            else:raise ValueError('unsupported source NPY format')
            offset=f.tell()
    return np.memmap(path,mode='r',dtype=dtype,shape=shape,offset=offset,order='F' if fortran else 'C')


def yeast_development(world,corpus,root,output,limit,queries):
    source=next(s for s in corpus.sources if s.name=='yeast');ref=source.ref
    static=root/'data/derived/slp11-yeast-shared-static/current-sgd-strict-query-full-raw-actions-esm8m-complete-shared-go-v2/yeast-static-esm8m-shared-go-mf-cc-features.npz'
    static_digest=sha(static)
    if static_digest!='81cda9469380c9efa000a40b2cd5e816a1d397ce777288fa53b0bcf26a55dc25':raise ValueError('changed yeast static descriptors')
    with np.load(static,allow_pickle=False) as z:
        if not (z['entity_taxon']==4932).all():raise ValueError('yeast static taxonomy mismatch')
        for gene,feature in zip(z['entity_id'].astype(str),z['feature_values']):
            if gene not in source.features.yeast:source.features.yeast[gene]=(np.pad(feature,(0,65))-source.features.mean)/source.features.scale
    path=root/'data/derived/slp11-yeast-atlas-counts/nadal-ribelles-raw-rna-development-v1'
    manifest=json.loads((path/'moments-manifest.json').read_text());grouped={};receipts={}
    qi=np.arange(len(source.query)) if not queries else np.sort(np.random.default_rng(731).choice(np.flatnonzero(source.support),queries,False))
    fitting=set(source.z['action_ids'].astype(str));batch_lookup={(str(ref['context_ids'][c]),str(b)):i for i,(b,c) in enumerate(zip(ref['batch_ids'],ref['batch_context_index']))}
    for rec in manifest['shards']:
        ctx=rec['context'];batchid=rec['batchId'];file=path/('control' if ctx=='Control' else 'nacl')/'moments'/(batchid+'.npz')
        digest=sha(file)
        if digest!=rec['sha256']:raise ValueError('changed yeast development shard')
        receipts[file.relative_to(root).as_posix()]=digest
        with np.load(file,allow_pickle=False) as z:
            ids=z['group_action_id'].astype(str);rows=np.flatnonzero((z['development_role']=='validation')&z['mean_observed']&(z['num_cells']>0))
            if limit:rows=rows[:limit]
            cells=z['num_cells'][rows];ids=ids[rows]
            if set(ids)&fitting:raise ValueError('yeast fitting/validation genes overlap')
            if not np.array_equal(z['query_ids'],source.query_ids):raise ValueError('yeast query mismatch')
        if not len(rows):continue
        sums=stored_member(file,'sum');truth=np.asarray(sums[rows],np.float64)/cells[:,None];del sums
        basal=np.broadcast_to(source.basal[batch_lookup[(ctx,batchid)]],truth.shape)
        p,t,b,m=predict(world,source,truth.astype(np.float32),basal,ids[:,None],np.ones_like(truth,bool),qi)
        table=grouped.setdefault(ctx,{})
        for gene,count,pp,tt,bb in zip(ids,cells,p,t,b):
            if gene not in table:table[gene]=[np.zeros(len(qi),np.float64) for _ in range(3)]+[0]
            acc=table[gene];acc[0]+=pp*count;acc[1]+=tt*count;acc[2]+=bb*count;acc[3]+=int(count)
        print(json.dumps({'event':'yeast_batch','context':ctx,'batch':batchid,'held_views':len(rows)}),flush=True)
    reports={};payload={}
    for ctx,table in grouped.items():
        ids=sorted(table);p=np.array([table[g][0]/table[g][3] for g in ids]);t=np.array([table[g][1]/table[g][3] for g in ids]);b=np.array([table[g][2]/table[g][3] for g in ids])
        mean=ref['fitting_residual_centroid'][list(ref['context_ids']).index(ctx),qi]
        reports[ctx]=scores(p,t,b,np.ones_like(p,bool),mean)
        reports[ctx].update(taxon=4932,aggregation='cell-weight batch views within native held gene; equal weight across genes',units='population mean of per-cell ln1p(CP10K)')
        payload[ctx+'_action_ids']=np.asarray(ids);payload[ctx+'_prediction']=p.astype(np.float32)
        payload[ctx+'_query_ids']=source.query_ids[qi]
    np.savez_compressed(output/'yeast-predictions.npz',**payload)
    return {'contexts':reports,'source_receipts':receipts,'static_features_sha256':static_digest,'scope':'within-yeast held-gene development; not a cross-species transfer proof'}


def paired_cell_development(world,corpus,output):
    source=next(s for s in corpus.sources if s.name=='frangieh_cells');rng=np.random.default_rng(1731)
    cols,ei,di=source.columns(rng,192,320);ei=ei[source.modality[cols[ei]]==0]
    values=[];actions=[];contexts=[];cell_ids=[];control_values={ctx:[] for ctx in source.contexts}
    # Read a fixed set of shards, retaining only held cells and control values
    # on this small panel. Reconstruction validation never entered fitting.
    for index in np.linspace(0,len(source.manifest['shards'])-1,min(12,len(source.manifest['shards'])),dtype=int):
        source.shard(rng,index);z=source.cache
        held=(z['reconstruction_split']=='validation')&np.array([g in source.allowed for g in z['action_ids']])
        rows=np.flatnonzero(held);values.append(source.values(rows,cols));actions.extend(z['action_ids'][rows]);contexts.extend(z['context_ids'][rows]);cell_ids.extend(z['cell_ids'][rows])
        for ctx,rows in source.controls.items():
            rows=rows[:64];control_values[ctx].append(source.values(rows,cols))
    values=np.concatenate(values);actions=np.asarray(actions);contexts=np.asarray(contexts);cell_ids=np.asarray(cell_ids)
    control_values={ctx:np.concatenate(v) for ctx,v in control_values.items()};groups=[]
    for ctx in source.contexts:
        candidates=[]
        for g in np.unique(actions[contexts==ctx]):
            rows=np.flatnonzero((actions==g)&(contexts==ctx))
            if len(rows)>=4 and len(control_values[ctx])>=4:candidates.append((ctx,g,rows))
        groups.extend(sorted(candidates,key=lambda x:(-len(x[2]),x[1]))[:2])
    if not groups:raise ValueError('no paired-cell validation groups available')
    reports=[];artifacts={}
    for group,(ctx,g,rows) in enumerate(groups):
        rows=rows[:16];controls=rng.choice(len(control_values[ctx]),len(rows),replace=False)
        target=values[rows];initial=control_values[ctx][controls]
        basal=np.broadcast_to(source.basal[source.contexts[ctx],cols],target.shape)
        kwargs=dict(modality=source.modality[cols],scale=source.scale[cols],assay=source.assay,taxon=source.taxon,
            mechanism=source.mechanism,encoder_indices=ei,normalized_descriptors=True)
        encoded=world.encode(target,basal,source.query[cols],**kwargs);reconstructed=world.reconstruct(encoded)
        initial_state=world.encode(initial,basal,source.query[cols],**kwargs)
        actions,mask=source.action(np.full((len(rows),1),g))
        if world.model.config.observation_likelihood:
            generated_result=world.generate(initial_state,actions,mask,seed=731+group,steps=16,normalized_descriptors=True)
            generated=generated_result['values'];mode='learned sparse RNA and continuous protein observation distributions'
        else:
            sample=world.sample(initial_state,actions,mask,seed=731+group,steps=16,normalized_descriptors=True)
            generated=world.decode(sample)['values'];mode='latent transport with observed control anchor; no sparse observation likelihood'
        rec={'context':ctx,'action_id':g,'cells':len(rows),'generation_mode':mode}
        for modality,label in ((0,'rna'),(1,'protein')):
            take=di[source.modality[cols[di]]==modality];x=target[:,take];y=generated[:,take];control=initial[:,take]
            rec[label]={'withheld_reconstruction_mse':float(np.mean((reconstructed[:,take]-x)**2)),
                'basal_reconstruction_mse':float(np.mean((basal[:,take]-x)**2)),
                'generated_mean_mse':float(np.mean((y.mean(0)-x.mean(0))**2)),
                'control_mean_mse':float(np.mean((control.mean(0)-x.mean(0))**2)),
                'generated_energy_distance':energy(y,x),'control_energy_distance':energy(control,x),
                'generated_variance':float(np.var(y,axis=0).mean()),'observed_variance':float(np.var(x,axis=0).mean())}
            if modality==0:
                rec[label]['generated_zero_fraction']=float(np.mean(y==0));rec[label]['observed_zero_fraction']=float(np.mean(x==0))
                if world.model.config.observation_likelihood:rec[label]['detection_brier']=float(np.mean((generated_result['rna_detection_probability'][:,take]-(x>0))**2))
        reports.append(rec);artifacts[f'group{group}_observed']=target;artifacts[f'group{group}_generated']=generated
        artifacts[f'group{group}_reconstructed']=reconstructed;artifacts[f'group{group}_query_ids']=source.query_ids[cols]
        artifacts[f'group{group}_cell_ids']=cell_ids[rows]
    np.savez_compressed(output/'paired-cell-generation.npz',**artifacts)
    return {'scope':'held cells within fitting intervention genes; RNA-only state encoding for RNA/protein reconstruction',
            'groups':reports,'shards':source.used_shards}


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--index',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--limit',type=int,default=0);p.add_argument('--queries',type=int,default=0);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);started=time.monotonic()
    captured=a.output/'source';captured.mkdir()
    for file in Path(__file__).parent.iterdir():
        if file.is_file():shutil.copy2(file,captured/file.name)
    corpus=Corpus(a.source_root,a.index);world=WorldModel(a.checkpoint,a.device)
    # A checkpoint may precede bundle export and therefore lack the normalizer;
    # evaluation supplies the exact normalized descriptors from the fitting index.
    with torch.inference_mode(),torch.autocast(a.device,dtype=torch.bfloat16,enabled=a.device=='cuda'):
        human=population_development(world,corpus,a.source_root,a.output,a.limit,a.queries)
        combinations=combination_development(world,corpus,a.source_root,a.limit,a.queries)
        paired_genes=paired_gene_development(world,corpus,a.source_root,a.output,a.limit,a.queries)
        yeast=yeast_development(world,corpus,a.source_root,a.output,a.limit,a.queries)
        cells=paired_cell_development(world,corpus,a.output)
    report={'schema':'slp.cell-world-development/v1','checkpoint_sha256':sha(a.checkpoint/'model.safetensors'),
        'training_index_sha256':sha(a.index/'manifest.json'),'human':human,'combinations':combinations,'paired_held_genes':paired_genes,'yeast':yeast,'paired_cells':cells,
        'seconds':time.monotonic()-started,'limit':a.limit,'query_limit':a.queries,'final_holdout_opened':False,'benchmark_labels_used':False,
        'source_hashes':{p.name:sha(p) for p in captured.iterdir()},'torch':torch.__version__,'numpy':np.__version__,'device':a.device,
        'prediction_files':{p.name:sha(p) for p in a.output.glob('*.npz')}}
    write(a.output/'report.json',report);write(a.output/'metrics.json',{name+'_mse':r['mse'] for name,r in human.items()})
    print(json.dumps({'event':'complete','output':str(a.output),'seconds':report['seconds']}),flush=True)


if __name__=='__main__':main()
