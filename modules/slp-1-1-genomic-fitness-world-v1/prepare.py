"""Species-native quantitative fitness corpus and frozen molecular signatures."""
import argparse,csv,hashlib,io,json,sys,time,zipfile
from pathlib import Path
import numpy as np
import torch

DTYPE=np.dtype([('a','<u2'),('b','<u2'),('single_a','<f4'),('single_b','<f4'),('double','<f4')])


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def dump(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


@torch.inference_mode()
def signatures(world,raw,query,basal):
    device=world.device;cols=np.sort(np.random.default_rng(731).choice(len(query),512,replace=False))
    descriptors=np.pad(query[cols],((0,0),(0,60)));b=basal[cols]
    state=world.encode(b[None],b[None],descriptors,modality=np.zeros(512,int),scale=.5,assay=7,taxon=4932,mechanism=2,encoder_indices=np.arange(192))
    rng=np.random.default_rng(731);lp=torch.tensor((rng.standard_normal((256,64))/16).astype('f4'),device=device)
    rng=np.random.default_rng(732);rp=torch.tensor((rng.standard_normal((512,32))/np.sqrt(512)).astype('f4'),device=device)
    model=world.model;one=(state.assay,state.taxon,state.mechanism)
    base,_=model.decode(state.latent,state.descriptors,state.modality,*one,control=state.control)
    actions=world.descriptors(raw);output=[]
    for i in range(0,len(raw),128):
        a=actions[i:i+128,None];n=len(a);initial=state.latent.expand(n,-1,-1);context=[v.expand(n) for v in one]
        latent=model.transition(initial,a,torch.ones((n,1),dtype=torch.bool,device=device),*context)
        predicted,_=model.decode(latent,state.descriptors,state.modality,*context,control=state.control.expand(n,-1))
        output.append(torch.cat(((latent-initial).mean(1)@lp,(predicted-base)@rp),1).cpu().numpy())
    return np.concatenate(output),state.latent.mean(1).cpu().numpy()[0,:128]


def main(a):
    start=time.time();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    with np.load(a.human_genes) as z:human_ids=z['gene_ids'].astype(str);human_raw=z['action_features']
    human_signatures=[]
    for name in ('k562','rpe1','hepg2'):
        with np.load(a.human_states/f'{name}-single-responses.npz') as z:
            assert np.array_equal(z['gene_ids'],human_ids);human_signatures.append(z['features'])
    human_signatures=np.concatenate(human_signatures,1)
    with np.load(a.world_features) as z:
        yeast_ids=z['yeast_query_ids'].astype(str);yeast_raw=z['yeast_query_features'];fm=z['feature_mean'];fs=z['feature_scale']
    lookup={g:i for i,g in enumerate(yeast_ids)};sgd={}
    with a.sgd.open() as f:
        for line in f:
            r=json.loads(line)
            if r['ncbiTaxon']!=4932:raise ValueError('mapping is not yeast-native')
            if r['canonicalSgdCurie'] in lookup:sgd[r['systematicName']]=lookup[r['canonicalSgdCurie']]
    if len(sgd)!=len(set(sgd.values())):raise ValueError('ambiguous systematic yeast mapping')
    sys.path.insert(0,str(a.world.resolve()));from inference import WorldModel
    world=WorldModel(a.world,'cuda')
    with np.load(a.yeast_reference) as z:
        assert np.array_equal(z['query_ids'],yeast_ids)
        controls=z['control_mean'];context_index=z['batch_context_index']
        basal=[controls[context_index==0][0],controls[context_index==0].mean(0),controls[context_index==1].mean(0)]
    yeast_signatures=[];yeast_context=None
    for b in basal:
        sig,ctx=signatures(world,yeast_raw,yeast_raw,b);yeast_signatures.append(sig)
        if yeast_context is None:yeast_context=ctx
    yeast_signatures=np.concatenate(yeast_signatures,1)
    np.savez_compressed(a.output/'genes.npz',human_ids=human_ids,human_raw=(human_raw-fm)/fs,human_signatures=human_signatures,
        yeast_ids=yeast_ids,yeast_raw=(yeast_raw-fm)/fs,yeast_signatures=yeast_signatures,feature_mean=fm,feature_scale=fs,yeast_context=yeast_context)
    meta=list(csv.DictReader(a.meta.open(encoding='utf-8-sig')));mapping={r['ensembl_gene_id'].split('.')[0]:i for i,r in enumerate(meta)}
    assert len(mapping)==len(meta);columns=np.array([mapping[g] for g in human_ids]);excluded=set(a.human_excluded.read_text().splitlines())
    with np.load(a.human_fitness) as z:
        targets=z['dependency'][:,columns].T.astype('f4');known=z['dependency_known'][:,columns].T
        allowed=z['train_gene'][columns]&np.array([g not in excluded for g in human_ids]);contexts=z['cell_state'];context_ids=z['model_ids'];cells=z['train_cell']
    held_human=np.array([hashlib.sha256(g.encode()).digest()[0]<51 for g in human_ids]);train_genes=allowed&~held_human;valid_genes=allowed&held_human
    held_cells=np.array([hashlib.sha256(str(g).encode()).digest()[0]<51 for g in context_ids]);train_cells=cells&~held_cells
    targets=np.where(known,targets,0.).astype('f4')
    for name,take in [('human-train',train_genes),('human-validation',valid_genes)]:
        gi=np.flatnonzero(take);ci=np.flatnonzero(train_cells)
        np.savez_compressed(a.output/(name+'.npz'),gene_indices=gi,targets=targets[gi][:,ci],known=known[gi][:,ci],contexts=contexts[ci],context_ids=context_ids[ci])
    np.savez_compressed(a.output/'human-contexts.npz',contexts=contexts[cells],context_ids=context_ids[cells])
    old_held=set()
    with np.load(a.yeast_held_ids) as z:
        for key in z.files:
            if key.endswith('_action_ids'):old_held.update(z[key].astype(str))
    held_yeast=np.array([g in old_held or hashlib.sha256(g.encode()).digest()[0]<26 for g in yeast_ids])
    dump(a.output/'yeast-partitions.json',{'held_gene_ids':yeast_ids[held_yeast].tolist(),'previous_molecular_held_ids':sorted(old_held)})
    counts={'source_rows':0,'sampled':0,'unmapped':0,'invalid':0,'mixed_held':0,'train':0,'validation':0};rng=np.random.default_rng(731)
    buffers={'train':[],'validation':[]};handles={k:(a.output/f'yeast-{k}.bin').open('wb') for k in buffers}
    def flush(role):
        if buffers[role]:np.asarray(buffers[role],dtype=DTYPE).tofile(handles[role]);buffers[role].clear()
    with zipfile.ZipFile(a.yeast_source) as archive:
        entry=next(n for n in archive.namelist() if n.endswith('/SGA_NxN.txt'))
        with io.TextIOWrapper(archive.open(entry),encoding='utf-8') as f:
            header=f.readline().strip().split('\t')
            assert header[7:10]==['Query single mutant fitness (SMF)','Array SMF','Double mutant fitness']
            random=rng.random(1000000)
            for number,line in enumerate(f):
                counts['source_rows']+=1
                if number and number%len(random)==0:random=rng.random(1000000)
                if random[number%len(random)]>=.25:continue
                counts['sampled']+=1;row=line.rstrip('\n').split('\t')
                left=sgd.get(row[0].split('_')[0]);right=sgd.get(row[2].split('_')[0])
                if left is None or right is None or left==right:counts['unmapped']+=1;continue
                if row[4]!='DMA30':counts['invalid']+=1;continue
                try:values=[float(row[k]) for k in (7,8,9)]
                except ValueError:counts['invalid']+=1;continue
                if not all(np.isfinite(v) and v>=0 for v in values):counts['invalid']+=1;continue
                if held_yeast[left]!=held_yeast[right]:counts['mixed_held']+=1;continue
                role='validation' if held_yeast[left] else 'train';counts[role]+=1
                buffers[role].append((left,right,*values))
                if len(buffers[role])==50000:flush(role)
                if counts['sampled']%500000==0:print(json.dumps({**counts,'seconds':time.time()-start}),flush=True)
    for role in buffers:flush(role);handles[role].close()
    manifest={'schema':'slp.species-native-fitness-corpus/v1','human_fit_genes':int(train_genes.sum()),'human_validation_genes':int(valid_genes.sum()),
        'human_fit_cells':int(train_cells.sum()),'yeast':counts,'yeast_record_dtype':DTYPE.descr,'seconds':time.time()-start,
        'source_scope':'human single CRISPR effects; yeast NxN deletion single/double relative fitness at30C. Published epsilon/P-value columns are unused.',
        'world_weights_sha256':sha(a.world/'model.safetensors'),'inputs':{str(p):sha(p) for p in (a.human_fitness,a.yeast_source,a.sgd,a.world_features,a.human_genes,a.human_excluded,a.yeast_held_ids)},
        'files':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()},'pair_sampling':'seed731 Bernoulli .25 by source row, independent of every outcome'}
    dump(a.output/'manifest.json',manifest);print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('human-genes','human-states','world-features','world','yeast-reference','meta','human-fitness','human-excluded','yeast-source','sgd','yeast-held-ids','output'):
        p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
