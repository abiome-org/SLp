"""Standalone world-simulation -> SL-decoder inference for arbitrary gene pairs."""
import argparse,csv,json,shutil
from pathlib import Path
import numpy as np
import torch
from bridge import Simulator,load_world,sha,write


class SLPredictor:
    def __init__(self,directory,device='cpu'):
        import lightgbm as lgb
        self.directory=Path(directory)
        manifest=json.loads((self.directory/'manifest.json').read_text())
        for name,digest in manifest['files'].items():
            path=(self.directory/name).resolve()
            if not path.is_relative_to(self.directory.resolve()) or sha(path)!=digest:raise ValueError('invalid SL predictor payload '+name)
        self.world=load_world(self.directory/'world',device)
        self.contexts=[Simulator(self.world,self.directory/'contexts'/f'{name}-context.npz') for name in ('k562','rpe1','hepg2')]
        self.decoders=[lgb.Booster(model_file=str(self.directory/name)) for name in manifest['decoders']]
        self.registry=None
        if (self.directory/'genes.npz').exists():
            with np.load(self.directory/'genes.npz') as z:
                self.registry=z['action_features'];self.identifiers=z['gene_ids'].astype(str);symbols=z['symbols'].astype(str)
            self.lookup={g:i for i,g in enumerate(self.identifiers)}
            for i,symbol in enumerate(symbols):
                if symbol:
                    if symbol in self.lookup and self.lookup[symbol]!=i:raise ValueError('ambiguous gene alias '+symbol)
                    self.lookup[symbol]=i

    def predict_pairs(self,gene_pairs,batch=128):
        """Score human stable-ID or exact-symbol pairs using bundled descriptors."""
        if self.registry is None:raise ValueError('this export has no descriptor registry; use predict with explicit descriptors')
        pairs=np.asarray(gene_pairs,dtype=str)
        if pairs.ndim!=2 or pairs.shape[1]!=2 or not len(pairs):raise ValueError('nonempty [pairs,2] gene names required')
        missing=sorted(set(pairs.reshape(-1))-self.lookup.keys())
        if missing:raise ValueError('unknown gene descriptors: '+', '.join(missing))
        indices=np.array([[self.lookup[g] for g in row] for row in pairs])
        if np.any(indices[:,0]==indices[:,1]):raise ValueError('synthetic lethality requires two distinct genes')
        unique,inverse=np.unique(indices,return_inverse=True)
        result=self.predict(self.registry[unique],inverse.reshape(-1,2),batch)
        result['gene_pairs']=self.identifiers[indices]
        return result

    def predict(self,gene_descriptors,pair_indices,batch=128):
        """Descriptors are the world's raw 642 biological coordinates, not IDs.

        Scores average the ten frozen fold decoders for research inference.
        Reported benchmark predictions use their own single held-out-fold decoder.
        """
        raw=np.asarray(gene_descriptors,np.float32);pairs=np.asarray(pair_indices)
        if raw.ndim!=2 or raw.shape[1]!=642 or not np.isfinite(raw).all():raise ValueError('finite [genes,642] descriptors required')
        if pairs.ndim!=2 or pairs.shape[1]!=2 or pairs.dtype.kind not in 'iu' or np.any(pairs<0) or np.any(pairs>=len(raw)):raise ValueError('invalid gene pair indices')
        if batch<1:raise ValueError('positive batch required')
        blocks=[];scores=[]
        for sim in self.contexts:
            single=sim.singles(raw,batch);x=[];z=[]
            for start in range(0,len(pairs),batch):
                features,similarity,_=sim.pairs(pairs[start:start+batch],*single);x.append(features);z.append(similarity)
            blocks.append(np.concatenate(x));scores.append(np.concatenate(z))
        features=np.concatenate(blocks,1)
        predictions=np.array([model.predict(features,num_threads=4) for model in self.decoders])
        return {'sl_score':predictions.mean(0),'fold_decoder_scores':predictions,'label_free_response_similarity':np.array(scores).mean(0),
                'features':features}


def export(a):
    fit=json.loads((a.fit/'fit-manifest.json').read_text())
    for name,digest in fit['files'].items():
        if sha(a.fit/name)!=digest:raise ValueError('changed trained decoder')
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'world').mkdir();(a.output/'contexts').mkdir();(a.output/'decoders').mkdir()
    world=json.loads((a.bundle/'manifest.json').read_text())
    for name,digest in world['files'].items():
        if sha(a.bundle/name)!=digest:raise ValueError('changed world model')
        dest=a.output/'world'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a.bundle/name,dest)
    shutil.copyfile(a.bundle/'manifest.json',a.output/'world/manifest.json')
    for name in ('k562','rpe1','hepg2'):shutil.copyfile(a.features/f'{name}-context.npz',a.output/'contexts'/f'{name}-context.npz')
    decoders=[]
    for seed in (42,432):
        for fold in range(5):
            name=f'seed{seed}-fold{fold}-world.txt';shutil.copyfile(a.fit/name,a.output/'decoders'/name);decoders.append('decoders/'+name)
    for name in ('predict.py','bridge.py','requirements.lock','CONTRACT.md'):shutil.copyfile(Path(__file__).parent/name,a.output/name)
    if a.roster is not None:
        meta={row['ensembl_gene_id'].split('.')[0]:row['symbol'] for row in csv.DictReader(a.gene_meta.open(encoding='utf-8-sig'))}
        with np.load(a.roster/'genes.npz') as z:
            ids=z['gene_ids'].astype(str)
            np.savez_compressed(a.output/'genes.npz',gene_ids=ids,action_features=z['action_features'],symbols=np.array([meta.get(g,'') for g in ids]),ncbi_taxon=9606)
    for src,name in ((a.fit/'fit.json','decoder-training.json'),(a.fit/'scores.json','benchmark-scores.json'),(a.features/'manifest.json','feature-manifest.json')):
        shutil.copyfile(src,a.output/name)
    write(a.output/'manifest.json',{'schema':'slp.cell-world-sl-predictor/v1','decoders':decoders,
        'files':{p.relative_to(a.output).as_posix():sha(p) for p in a.output.rglob('*') if p.is_file()},
        'world_weights_sha256':sha(a.bundle/'model.safetensors'),'decoder_lock_sha256':sha(a.fit/'fit-manifest.json'),
        'scope':'research SL decoder over frozen molecular world; ensemble for inference, fold-local scores for benchmark'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true')
    for name in ('fit','bundle','features','output','request','roster','gene-meta','pair-file'):p.add_argument('--'+name,type=Path)
    p.add_argument('--device',default='cpu');a=p.parse_args();torch.set_num_threads(4)
    if a.export:export(a)
    else:
        predictor=SLPredictor(a.bundle,a.device)
        if a.pair_file:
            pairs=[(row['gene_a'],row['gene_b']) for row in csv.DictReader(a.pair_file.open())]
            r=predictor.predict_pairs(pairs)
        else:
            with np.load(a.request) as z:r=predictor.predict(z['gene_descriptors'],z['pair_indices'])
        np.savez_compressed(a.output,**r)
