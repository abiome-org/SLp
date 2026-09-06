"""Bounded molecular source adapters. All quantitative fitting rows are indexed explicitly."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from prepare import sha


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


class Features:
    def __init__(self,path):
        z=arrays(path); self.mean=z['feature_mean'];self.scale=z['feature_scale']
        self.human={g:(f-self.mean)/self.scale for g,f in zip(z['human_ids'],z['human_features'])}
        self.yeast={g:(f-self.mean)/self.scale for g,f in zip(z['yeast_query_ids'],z['yeast_query_features'])}
        self.yeast.update({g:(f-self.mean)/self.scale for g,f in zip(z['yeast_action_ids'],z['yeast_action_features'])})

    def genes(self,ids,taxon=9606):
        table=self.human if taxon==9606 else self.yeast
        ids=np.asarray(ids).astype(str);flat=ids.reshape(-1)
        values=np.asarray([table.get(g,np.zeros(642,np.float32)) for g in flat],np.float32).reshape(*ids.shape,642)
        supported=np.asarray([g in table for g in flat]).reshape(ids.shape)
        return values,supported


class Source:
    def __init__(self,record,features):
        self.record=record;self.features=features;self.name=record['name']
        self.weight=record['weight'];self.assay=record['assay'];self.mechanism=record['mechanism'];self.taxon=record['taxon']
        self.cell=False;self.draws=0;self.parents={}

    def panel(self,ids):
        self.query_ids=np.asarray(ids).astype(str)
        feats,supported=self.features.genes(self.query_ids,self.taxon)
        self.query=np.pad(feats,((0,0),(0,60)));self.support=supported;self.modality=np.zeros(len(ids),np.int64)

    def action(self,ids):
        values,supported=self.features.genes(ids,self.taxon)
        mask=np.asarray(ids)!=''
        if np.any(mask&~supported):raise ValueError(f'{self.name}: missing action descriptors')
        return values,mask

    def columns(self,rng,enc,out):
        rna=np.flatnonzero(self.support&(self.modality==0));protein=np.flatnonzero(self.modality==1)
        chosen=rng.choice(rna,min(enc+out,len(rna)),replace=False)
        cols=np.concatenate((chosen,protein));ei=np.arange(min(enc,len(chosen)))
        # RNA-only observation regularly forces paired-protein imputation.
        if len(protein) and rng.random()<.25:ei=np.concatenate((ei,np.arange(len(chosen),len(cols))))
        di=np.concatenate((np.arange(min(enc,len(chosen)),len(chosen)),np.arange(len(chosen),len(cols))))
        return cols,ei,di

    def finish(self,cols,ei,di,target,initial,basal,actions,control=None,observed=None):
        a,m=self.action(actions);b=len(target);self.draws+=1
        scale=np.broadcast_to(self.scale,(len(self.query),))[cols].astype(np.float32)
        observed=np.ones_like(target,dtype=bool) if observed is None else observed
        observed=observed&self.support[cols][None]
        result=dict(target=np.asarray(target,np.float32)/scale,initial=np.asarray(initial,np.float32)/scale,
            basal=np.asarray(basal,np.float32)/scale,control=np.asarray(basal if control is None else control,np.float32),
            observed=observed,query=self.query[cols],modality=self.modality[cols],actions=a,action_mask=m,
            assay=np.full(b,self.assay,np.int64),taxon=np.full(b,0 if self.taxon==9606 else 1,np.int64),
            mechanism=np.full(b,self.mechanism,np.int64),encoder_indices=ei,decoder_indices=di,
            cell=self.cell,scale=scale,name=self.name,query_ids=self.query_ids[cols],action_ids=np.asarray(actions))
        return result


class Population(Source):
    def __init__(self,root,index,record,features):
        super().__init__(record,features);z=arrays(root/record['path']);ix=arrays(index/record['index'])
        rows=ix['rows'];self.ids=ix['action_ids'];self.panel(z['query_ids'])
        self.target=z['targets'][rows];self.basal=z['basal'][rows]
        self.observed=z.get('observed',z.get('target_observed',np.ones_like(z['targets'],dtype=bool)))[rows]
        residual=self.target-self.basal
        self.scale=float(max(np.sqrt(np.square(residual[self.observed]).mean()),.03))
        self.control=z.get('control_context_values',None)
        self.parent_rows=np.full((len(rows),2),-1,np.int64)
        if 'combination_rows' in z:
            mapping={int(r):i for i,r in enumerate(rows)}
            for r,parents in zip(z['combination_rows'],z['combination_single_rows']):
                if int(r) in mapping and all(int(p) in mapping for p in parents):
                    aligned=[]
                    for g in self.ids[mapping[int(r)]]:
                        matches=[mapping[int(p)] for p in parents if g in self.ids[mapping[int(p)]]]
                        if len(matches)!=1:raise ValueError('parent action alignment is ambiguous')
                        aligned.append(matches[0])
                    self.parent_rows[mapping[int(r)]]=aligned

    def draw(self,rng,batch,enc,out):
        rows=rng.integers(len(self.target),size=batch);cols,ei,di=self.columns(rng,enc,out)
        basal=self.basal[np.ix_(rows,cols)];control=basal if self.control is None else np.broadcast_to(self.control[cols],basal.shape)
        result=self.finish(cols,ei,di,self.target[np.ix_(rows,cols)],basal,basal,self.ids[rows],control,self.observed[np.ix_(rows,cols)])
        parent=self.parent_rows[rows];valid=(parent>=0).all(1)
        if valid.any():
            result['parent_target']=self.target[np.ix_(np.maximum(parent[:,0],0),cols)]/result['scale']
            result['other_parent_target']=self.target[np.ix_(np.maximum(parent[:,1],0),cols)]/result['scale']
            result['parent_mask']=valid
        return result


class RawCells(Source):
    def __init__(self,root,index,record,features):
        super().__init__(record,features);self.cell=True;self.z=arrays(index/record['index']);z=self.z
        self.panel(z['query_ids']);self.counts=np.memmap(root/record['path'],dtype=np.uint16,mode='r',shape=tuple(record['shape']))
        self.groups={g:np.flatnonzero((z['action_ids']==g)&~z['is_control']) for g in np.unique(z['action_ids'][~z['is_control']])}
        self.genes=sorted(self.groups);self.controls=np.flatnonzero(z['is_control']);self.scale=1.
        self.controls_by_gem={g:self.controls[z['gem_group'][self.controls]==g] for g in np.unique(z['gem_group'])}
        # This is the log of the mean CP10K control, not the mean cell log value.
        total=np.zeros(len(self.query),np.float64)
        for start in range(0,len(self.controls),256):
            r=self.controls[start:start+256];total+=(self.counts[z['rows'][r]].astype(np.float32)*(1e4/np.maximum(z['library_size'][r],1))[:,None]).sum(0)
        self.control=np.log1p(total/len(self.controls)).astype(np.float32)

    def values(self,rows,cols):
        return np.log1p(self.counts[np.ix_(self.z['rows'][rows],cols)].astype(np.float32)*(1e4/np.maximum(self.z['library_size'][rows],1))[:,None])

    def draw(self,rng,batch,enc,out):
        gene=self.genes[rng.integers(len(self.genes))];rows=rng.choice(self.groups[gene],batch)
        controls=np.array([rng.choice(self.controls_by_gem[g] if len(self.controls_by_gem[g]) else self.controls) for g in self.z['gem_group'][rows]])
        cols,ei,di=self.columns(rng,enc,out);basal=np.broadcast_to(self.control[cols],(batch,len(cols)))
        return self.finish(cols,ei,di,self.values(rows,cols),self.values(controls,cols),basal,np.full((batch,1),gene))


class Yeast(Source):
    def __init__(self,root,record,features):
        super().__init__(dict(record,name='yeast',assay=7,mechanism=2,taxon=4932),features)
        path=root/record['path'];self.ref=arrays(path/'reference.npz');self.z=arrays(path/'train-metadata.npz')
        self.target=np.load(path/'train-targets.npy',mmap_mode='r');self.panel(self.ref['query_ids'])
        # The source moments already sum per-cell log1p(CP10K); these arrays
        # are population means in log expression space, not CP10K rates.
        self.basal=self.ref['control_mean'];self.scale=.5

    def draw(self,rng,batch,enc,out):
        rows=rng.integers(len(self.target),size=batch);cols,ei,di=self.columns(rng,enc,out)
        basal=self.basal[np.ix_(self.z['batch_index'][rows],cols)]
        return self.finish(cols,ei,di,self.target[np.ix_(rows,cols)],basal,basal,self.z['action_ids'][rows,None])


class PairedCells(Source):
    def __init__(self,root,index,record,features):
        super().__init__(dict(record,name='frangieh_cells',assay=6,mechanism=2,taxon=9606),features)
        self.cell=True;self.path=root/record['path'];self.manifest=json.loads((self.path/'manifest.json').read_text())
        self.pop=arrays(index/record['population_file']);self.allowed=set(self.pop['action_ids'].astype(str));self.cache=None;self.used_shards={}
        p=self.pop;self.panel(p['rna_query_ids']);n=len(self.query);self.rna_count=n
        protein=np.zeros((len(p['protein_channel_ids']),702),np.float32)
        for i,barcode in enumerate(p['protein_channel_ids']):
            for j,base in enumerate(str(barcode)):protein[i,642+j*4+'ACGT'.index(base)]=1.
        self.query=np.concatenate((self.query,protein));self.query_ids=np.concatenate((self.query_ids,p['protein_channel_ids']))
        self.support=np.concatenate((self.support,np.ones(len(protein),bool)));self.modality=np.concatenate((self.modality,np.ones(len(protein),np.int64)))
        self.basal=np.concatenate((p['control_rna_targets'],p['control_protein_targets']),axis=1)
        self.contexts={g:i for i,g in enumerate(p['control_context_ids'])};self.scale=np.concatenate((np.ones(n),np.full(len(protein),.5))).astype(np.float32)

    def shard(self,rng,index=None):
        rec=self.manifest['shards'][rng.integers(len(self.manifest['shards'])) if index is None else index];path=self.path/rec['path']
        if rec['path'] not in self.used_shards:
            digest=sha(path)
            if digest!=rec['sha256']:raise ValueError('changed paired cell shard')
            self.used_shards[rec['path']]=digest
        z=arrays(path);self.rna=csr_matrix((z.pop('rna_data'),z.pop('rna_indices'),z.pop('rna_indptr')),shape=tuple(z.pop('rna_shape')))
        self.cache=z
        allowed=np.array([g in self.allowed for g in z['action_ids']])&(z['source_split']=='train')&(z['reconstruction_split']=='train')
        self.groups={}
        for ctx in np.unique(z['context_ids']):
            for g in np.unique(z['action_ids'][allowed&(z['context_ids']==ctx)]):
                self.groups[(ctx,g)]=np.flatnonzero(allowed&(z['context_ids']==ctx)&(z['action_ids']==g))
        self.group_keys=list(self.groups)
        self.controls={ctx:np.flatnonzero((z['source_split']=='control')&(z['reconstruction_split']=='train')&(z['context_ids']==ctx)) for ctx in self.contexts}

    def values(self,rows,cols):
        rna=cols<self.rna_count;result=np.empty((len(rows),len(cols)),np.float32)
        result[:,rna]=self.rna[rows][:,cols[rna]].toarray();result[:,~rna]=self.cache['protein_values'][np.ix_(rows,cols[~rna]-self.rna_count)]
        return result

    def draw(self,rng,batch,enc,out):
        if self.cache is None or self.draws%48==0:self.shard(rng)
        ctx,g=self.group_keys[rng.integers(len(self.group_keys))];rows=rng.choice(self.groups[(ctx,g)],batch)
        cols,ei,di=self.columns(rng,enc,out);basal=np.broadcast_to(self.basal[self.contexts[ctx],cols],(batch,len(cols)))
        controls=self.controls[ctx]
        initial=self.values(rng.choice(controls,batch),cols) if len(controls) else basal
        return self.finish(cols,ei,di,self.values(rows,cols),initial,basal,np.full((batch,1),g))


class Corpus:
    def __init__(self,root,index,verify=True):
        root=Path(root);index=Path(index);self.manifest=json.loads((index/'manifest.json').read_text())
        self.excluded=set((index/'excluded-human-action-ids.txt').read_text().splitlines())
        if verify:
            for filename,digest in self.manifest['files'].items():
                if sha(index/filename)!=digest:raise ValueError(f'changed index: {filename}')
            for filename,digest in self.manifest['sourceReceipts'].items():
                if sha(root/filename)!=digest:raise ValueError(f'changed source: {filename}')
        f=Features(index/'features.npz')
        self.sources=[Population(root,index,r,f) for r in self.manifest['populations']]
        self.sources += [RawCells(root,index,r,f) for r in self.manifest['raw_cells']]
        self.sources += [PairedCells(root,index,self.manifest['frangieh'],f),Yeast(root,self.manifest['yeast'],f)]
        self.weights=np.array([s.weight for s in self.sources]);self.weights/=self.weights.sum()

    def draw(self,rng,batch=16,enc=192,out=256):
        source=self.sources[rng.choice(len(self.sources),p=self.weights)]
        return source.draw(rng,batch,enc,out)

    def report(self):
        return [{'name':s.name,'weight':float(w),'draws':s.draws,'queries':len(s.query),'supported_queries':int(s.support.sum()),
                 'assay':s.assay,'mechanism':s.mechanism,'taxon':s.taxon,'scale_range':[float(np.min(s.scale)),float(np.max(s.scale))],
                 'used_shards':getattr(s,'used_shards',{})} for s,w in zip(self.sources,self.weights)]
