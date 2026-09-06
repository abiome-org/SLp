"""Usable cellular + genomic functional world with separate SL decoders."""
import argparse,csv,json,shutil
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from bridge import load_world,Simulator,sha,write
from functional_bridge import FunctionalSimulator


@dataclass(frozen=True)
class WorldState:
    molecular: object
    functional: object
    functional_control: object
    taxon: int


class SLpWorld:
    def __init__(self,bundle,device='cpu'):
        import lightgbm as lgb
        self.root=Path(bundle).resolve();self.manifest=json.loads((self.root/'manifest.json').read_text())
        for name,digest in self.manifest['files'].items():
            path=(self.root/name).resolve()
            if not path.is_relative_to(self.root) or sha(path)!=digest:raise ValueError('changed world bundle '+name)
        self.molecular=load_world(self.root/'molecular_model',device)
        self.functional_simulator=FunctionalSimulator(self.root/'functional_model',self.root/'genes.npz',self.root/'functional_contexts.npz',device)
        self.functional=self.functional_simulator.world
        self.decoders=[lgb.Booster(model_file=str(self.root/name)) for name in self.manifest['decoders']]
        self.ids=self.functional_simulator.ids;self.lookup={g:i for i,g in enumerate(self.ids)}
        aliases=json.loads((self.root/'gene-aliases.json').read_text())
        for name,g in aliases.items():
            if name in self.lookup and self.lookup[name]!=self.lookup[g]:raise ValueError('ambiguous alias')
            self.lookup[name]=self.lookup[g]

    def encode(self,observed,basal,query_descriptors,*,fitness_context,taxon=9606,**molecular_options):
        molecular=self.molecular.encode(observed,basal,query_descriptors,taxon=taxon,**molecular_options)
        functional=self.functional.encode(fitness_context,taxon)
        if len(molecular.latent)!=len(functional.latent):raise ValueError('molecular and functional contexts must describe the same batch')
        return WorldState(molecular,functional,functional,taxon)

    @torch.inference_mode()
    def intervene(self,state,raw_actions,action_mask,*,molecular_signatures=None):
        raw=np.asarray(raw_actions,np.float32);mask=np.asarray(action_mask,bool)
        if raw.ndim!=3 or raw.shape[-1]!=642 or raw.shape[:2]!=mask.shape or len(raw)!=len(state.functional.latent):raise ValueError('expected matching [batch,actions,642] descriptors and mask')
        if not mask.any():return state
        raw=np.where(mask[:,:,None],raw,0.)
        if molecular_signatures is None:
            if state.taxon!=9606:raise ValueError('supply species-native molecular signatures for nonhuman composite actions')
            actions=self.actions_from_descriptors(raw.reshape(-1,642))
        else:actions=self.functional.actions(raw.reshape(-1,642),np.asarray(molecular_signatures).reshape(-1,288))
        actions=actions.reshape(*mask.shape,-1)
        functional=self.functional.intervene(state.functional,actions,torch.as_tensor(mask,device=actions.device))
        molecular=self.molecular.intervene(state.molecular,raw,mask)
        return WorldState(molecular,functional,state.functional_control,state.taxon)

    def decode(self,state):
        return {'molecular':self.molecular.decode(state.molecular),
            'fitness_effect':self.functional.decode(state.functional,state.functional_control).cpu().numpy()}

    def predict_pairs(self,pairs):
        pairs=np.asarray(pairs,str)
        if pairs.ndim!=2 or pairs.shape[1]!=2 or not len(pairs):raise ValueError('nonempty pairs required')
        missing=set(pairs.ravel())-self.lookup.keys()
        if missing:raise ValueError('missing descriptors: '+', '.join(sorted(missing)))
        ix=np.array([[self.lookup[g] for g in row] for row in pairs])
        if np.any(ix[:,0]==ix[:,1]):raise ValueError('two distinct intervention genes required')
        score,features=self.functional_simulator.pairs(ix,features=True)
        folds=np.array([d.predict(features,num_threads=4) for d in self.decoders])
        return {'sl_score':folds.mean(0),'fold_decoder_scores':folds,'label_free_excess_fitness_loss':score,'features':features}

    @torch.inference_mode()
    def actions_from_descriptors(self,raw):
        """Encode novel genes through actual molecular simulations, without IDs."""
        signatures=[]
        for name in ('k562','rpe1','hepg2'):
            sim=Simulator(self.molecular,self.root/'molecular_contexts'/f'{name}-context.npz')
            signatures.append(sim.singles(np.asarray(raw,np.float32))[-1].numpy())
        return self.functional.actions(raw,np.concatenate(signatures,1))


def export(a):
    a.output.mkdir(parents=True,exist_ok=False)
    def copy(src,name):
        target=a.output/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
    molecular=json.loads((a.molecular/'manifest.json').read_text())
    for name,digest in molecular['files'].items():
        if sha(a.molecular/name)!=digest:raise ValueError('changed molecular world')
        copy(a.molecular/name,'molecular_model/'+name)
    copy(a.molecular/'manifest.json','molecular_model/manifest.json')
    for name in ('model.safetensors','normalizer.npz','training.json'):copy(a.functional/name,'functional_model/'+name)
    for name in ('model.py','capacity_model.py','fitness_inference.py','CONTRACT.md','requirements.lock'):copy(a.functional_source/name,'functional_model/'+name)
    for name in ('k562','rpe1','hepg2'):copy(a.molecular_features/f'{name}-context.npz','molecular_contexts/'+f'{name}-context.npz')
    copy(a.genes,'genes.npz');copy(a.contexts,'functional_contexts.npz')
    with np.load(a.genes) as z:ids=set(z['human_ids'].astype(str))
    aliases={}
    for row in csv.DictReader(a.meta.open(encoding='utf-8-sig')):
        g=row['ensembl_gene_id'].split('.')[0];symbol=row['symbol']
        if g in ids and symbol:
            if symbol in aliases and aliases[symbol]!=g:raise ValueError('ambiguous symbol')
            aliases[symbol]=g
    write(a.output/'gene-aliases.json',aliases)
    lock=json.loads((a.fit/'fit-manifest.json').read_text())
    for name,digest in lock['files'].items():
        if sha(a.fit/name)!=digest:raise ValueError('changed decoder lock')
    decoders=[]
    for seed in (42,432):
        for fold in range(5):
            name=f'seed{seed}-fold{fold}-world.txt';copy(a.fit/name,'decoders/'+name);decoders.append('decoders/'+name)
    for source,name in ((a.fit/'fit.json','decoder-training.json'),(a.fit/'scores.json','benchmark-scores.json'),
        (a.fit/'fit-manifest.json','decoder-lock.json')):copy(source,name)
    for name in ('functional_predict.py','functional_bridge.py','bridge.py','requirements.lock'):copy(Path(__file__).parent/name,name)
    write(a.output/'manifest.json',{'schema':'slp.cellular-genomic-world/v1','decoders':decoders,
        'files':{p.relative_to(a.output).as_posix():sha(p) for p in a.output.rglob('*') if p.is_file()},
        'molecular_weights_sha256':sha(a.molecular/'model.safetensors'),'functional_weights_sha256':sha(a.functional/'model.safetensors'),
        'scope':'research world with RNA/protein generation, species-native conditional fitness, fixed excess-loss SL scoring, and a separately supervised SL decoder'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true')
    for name in ('molecular','molecular-features','functional','functional-source','genes','contexts','fit','meta','output','bundle','pairs'):p.add_argument('--'+name,type=Path)
    p.add_argument('--device',default='cpu');a=p.parse_args();torch.set_num_threads(4)
    if a.export:export(a)
    else:
        model=SLpWorld(a.bundle,a.device);pairs=[(r['gene_a'],r['gene_b']) for r in csv.DictReader(a.pairs.open())]
        np.savez_compressed(a.output,**model.predict_pairs(pairs))
