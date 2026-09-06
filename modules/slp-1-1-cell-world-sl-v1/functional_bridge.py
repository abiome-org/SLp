"""SL application: excess conditional fitness loss in a frozen genomic world.

The sign is fixed by the endpoint, not selected from benchmark performance.
The world itself never sees this score or an SL pair label.
"""
import argparse,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
from bridge import sha,write


class FunctionalSimulator:
    def __init__(self,bundle,genes,contexts,device='cuda',random=False):
        spec=importlib.util.spec_from_file_location('_slp_functional_inference',Path(bundle)/'fitness_inference.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.world=module.FunctionalWorld(bundle,device,untrained=random);self.random=random
        with np.load(genes) as z:
            self.ids=z['human_ids'].astype(str);n=self.world.norm
            raw=z['human_raw']*n['feature_scale']+n['feature_mean']
            self.actions=self.world.actions(raw,z['human_signatures'])
        with np.load(contexts) as z:self.contexts=z['contexts']
        self.states=[self.world.encode(c[None]) for c in self.contexts]
        self.single=[];self.deltas=[]
        for state in self.states:
            base=self.world.expand(state,len(self.actions));after=self.world.intervene(base,self.actions)
            self.single.append(self.world.observe(base,self.actions))
            self.deltas.append(after.latent-base.latent)
        # World-state coordinates remain full width. No raw descriptor bypass.
        self.embedding=torch.stack(self.deltas).mean(0).cpu().numpy()

    @torch.inference_mode()
    def pairs(self,pairs,features=False,batch=2048):
        pairs=np.asarray(pairs,int)
        if pairs.ndim!=2 or pairs.shape[1]!=2 or np.any(pairs<0) or np.any(pairs>=len(self.ids)):raise ValueError('invalid pair indices')
        scores=[];blocks=[]
        for start in range(0,len(pairs),batch):
            p=pairs[start:start+batch];a,b=self.actions[p[:,0]],self.actions[p[:,1]];effects=[];details=[]
            for state,single in zip(self.states,self.single):
                s=self.world.expand(state,len(p));sa,sb=single[p[:,0]],single[p[:,1]]
                ab=self.world.observe(self.world.intervene(s,a),b);ba=self.world.observe(self.world.intervene(s,b),a)
                effect=(ab-sb+ba-sa)*.5;effects.append(effect)
                details.append(torch.stack((sa+sb,abs(sa-sb),ab+ba,abs(ab-ba),effect),1))
            scores.append((-torch.stack(effects).mean(0)).cpu().numpy())
            if features:
                l,r=self.embedding[p[:,0]],self.embedding[p[:,1]]
                blocks.append(np.concatenate((l+r,abs(l-r),l*r,torch.cat(details,1).cpu().numpy()),1))
        return (np.concatenate(scores),np.concatenate(blocks)) if features else np.concatenate(scores)


def contexts(a):
    a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.data/'human-train.npz') as z:c=z['contexts']
    # Observed fitting contexts: closest to mean and each extreme of first PC.
    center=c.mean(0);_,_,v=np.linalg.svd(c-center,full_matrices=False);projection=(c-center)@v[0]
    indices=np.array([np.argmin(((c-center)**2).sum(1)),np.argmin(projection),np.argmax(projection)])
    np.savez_compressed(a.output/'contexts.npz',contexts=c[indices],indices=indices)
    write(a.output/'manifest.json',{'rule':'mean-nearest and first-PC extreme observed fitting contexts; no outcome-based context choice',
        'source_sha256':sha(a.data/'human-train.npz'),'contexts_sha256':sha(a.output/'contexts.npz')})


def benchmark(a):
    from zero_shot import run,pair_scores
    started=time.time();torch.set_num_threads(4)
    sims={}
    def load(path):
        random=str(path)=='untrained'
        sim=FunctionalSimulator(a.functional,a.genes,a.contexts,a.device,random);sims[random]=sim
        return sim.ids,sim
    def score(embedding,pairs):return embedding.pairs(pairs) if isinstance(embedding,FunctionalSimulator) else pair_scores(embedding,pairs)
    a.world=Path('trained');a.random=Path('untrained')
    a.rule='negative mean symmetrized conditional excess human gene effect across three fixed observed contexts; no SL training, no sign selection'
    run(a,representation_loader=load,scorer=score,model_receipts={'functional_weights_sha256':sha(a.functional/'model.safetensors'),
        'functional_training_sha256':sha(a.functional/'training.json'),'contexts_sha256':sha(a.contexts),'genes_sha256':sha(a.genes)})
    print(json.dumps({'seconds':time.time()-started}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['contexts','benchmark'],required=True)
    for name in ('data','root','functional','genes','contexts','roster','bundle','index','output'):p.add_argument('--'+name,type=Path)
    p.add_argument('--device',default='cuda');a=p.parse_args();contexts(a) if a.phase=='contexts' else benchmark(a)
