"""SL application over an explicit, frozen molecular world-model artifact.

No benchmark labels enter simulation. All inputs to the learned SL decoder
come from molecular states and decoded consequences, not raw gene descriptors.
"""
import argparse, dataclasses, hashlib, importlib, json, math, shutil, sys, time
from pathlib import Path
import numpy as np
import torch


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def load_world(bundle, device, random=False):
    sys.path.insert(0, str(Path(bundle).resolve()))
    WorldModel=importlib.import_module('inference').WorldModel
    world=WorldModel(bundle,device)
    if random:
        module=importlib.import_module('model')
        torch.manual_seed(1731)
        world.model=module.CellWorld(world.model.config).to(device).eval()
        # The training initializer deliberately has zero dynamics. Enable its
        # random final projection to obtain a nonconstant random-feature control.
        torch.nn.init.kaiming_uniform_(world.model.delta.weight,a=math.sqrt(5))
        torch.nn.init.zeros_(world.model.delta.bias)
    return world


def contexts(root,index,output):
    """Construct application contexts from molecular FITTING controls only."""
    manifest=json.loads((index/'manifest.json').read_text())
    with np.load(index/'features.npz') as f:
        ids=f['human_ids'].astype(str);features=f['human_features'];lookup={g:i for i,g in enumerate(ids)}
    records=[]
    for rec in manifest['populations']:
        if rec['name'] not in ('k562','rpe1','hepg2'):continue
        with np.load(index/rec['index']) as ix:rows=ix['rows']
        with np.load(root/rec['path']) as z:
            qi=z['query_ids'].astype(str);available=np.array([i for i,g in enumerate(qi) if g in lookup])
            selected=np.sort(np.random.default_rng(731).choice(available,min(512,len(available)),replace=False))
            base=z['basal'][rows].mean(0);target=z['targets'][rows];basals=z['basal'][rows]
            observed=z['observed'][rows] if 'observed' in z else np.ones_like(target,bool)
            scale=max(float(np.sqrt(np.square((target-basals)[observed]).mean())),.03)
            control=z['control_context_values'] if 'control_context_values' in z else base
            path=output/(rec['name']+'-context.npz')
            np.savez_compressed(path,query_ids=qi[selected],query_descriptors=np.pad(features[[lookup[g] for g in qi[selected]]],((0,0),(0,60))),
                basal=base[selected].astype('f4'),control=control[selected].astype('f4'),scale=np.float32(scale),assay=rec['assay'],
                mechanism=rec['mechanism'],taxon=rec['taxon'],encoder_indices=np.arange(192))
        records.append({'name':rec['name'],'path':path.name,'sha256':sha(path),'source':rec['path'],'source_sha256':sha(root/rec['path']),
                        'index_sha256':sha(index/rec['index'])})
    return records


def expand(state, batch):
    return dataclasses.replace(state,**{field.name:getattr(state,field.name).expand(batch,*getattr(state,field.name).shape[1:])
        for field in dataclasses.fields(state) if field.name not in ('descriptors','modality','scale')})


def projection(rows,cols,seed,device):
    return torch.tensor((np.random.default_rng(seed).standard_normal((rows,cols))/np.sqrt(rows)).astype('f4'),device=device)


def cosine(a,b):
    return (a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(1e-10)


class Simulator:
    width=360
    def __init__(self,world,context):
        self.world=world;self.model=world.model
        with np.load(context) as z:
            self.state=world.encode(z['basal'][None],z['basal'][None],z['query_descriptors'],modality=np.zeros(len(z['basal']),int),
                scale=z['scale'],assay=int(z['assay']),taxon=int(z['taxon']),mechanism=int(z['mechanism']),control=z['control'][None],
                encoder_indices=z['encoder_indices'])
        self.lp=projection(self.model.config.width,64,731,world.device)
        self.rp=projection(len(self.state.descriptors),32,732,world.device)
        with torch.inference_mode():self.base=self.decode(self.state.latent)

    def decode(self,latent):
        state=expand(self.state,len(latent))
        mean,_=self.model.decode(latent,state.descriptors,state.modality,state.assay,state.taxon,state.mechanism,control=state.control)
        return mean

    def transition(self,latent,actions):
        state=expand(self.state,len(latent))
        return self.model.transition(latent,actions,torch.ones(actions.shape[:2],device=actions.device,dtype=torch.bool),
                                     state.assay,state.taxon,state.mechanism)

    @torch.inference_mode()
    def singles(self,raw,batch=128):
        actions=self.world.descriptors(raw);states=[];responses=[];features=[]
        for start in range(0,len(actions),batch):
            a=actions[start:start+batch,None];initial=expand(self.state,len(a)).latent
            latent=self.transition(initial,a);r=self.decode(latent)-self.base
            f=torch.cat(((latent-initial).mean(1)@self.lp,r@self.rp),1)
            states.append(latent.cpu());responses.append(r.cpu());features.append(f.cpu())
        return actions,torch.cat(states),torch.cat(responses),torch.cat(features)

    @torch.inference_mode()
    def pairs(self,indices,actions,states,responses,features):
        left,right=indices.T;device=self.world.device
        a=actions[left];b=actions[right];single_a=features[left].to(device);single_b=features[right].to(device)
        ra=responses[left].to(device);rb=responses[right].to(device)
        initial=expand(self.state,len(indices)).latent
        both=self.transition(initial,torch.stack((a,b),1));joint=self.decode(both)-self.base
        # Molecular nonadditivity has an explicit unchanged-state reference.
        interaction=joint-ra-rb
        latent_interaction=(both-states[left].to(device)-states[right].to(device)+initial).mean(1)
        stats=torch.stack((cosine(ra,rb),joint.norm(dim=1),interaction.norm(dim=1),
            ra.norm(dim=1)+rb.norm(dim=1),(ra.norm(dim=1)-rb.norm(dim=1)).abs(),
            joint.mean(1),interaction.mean(1),cosine(joint,ra+rb)),1)
        result=torch.cat((single_a+single_b,(single_a-single_b).abs(),single_a*single_b,
                          joint@self.rp,interaction@self.rp,stats),1)
        # The latent interaction is diagnostic; stored features contain the
        # actual decoded molecular interaction, not an SL-tuned latent score.
        assert result.shape[1]==self.width
        return result.cpu().numpy(),stats[:,0].cpu().numpy(),latent_interaction.square().mean(1).sqrt().cpu().numpy()


def extract(a):
    started=time.time();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(731)
    world=load_world(a.bundle,a.device,a.random)
    recs=contexts(a.root,a.index,a.output)
    with np.load(a.roster/'genes.npz') as z:genes=z['gene_ids'].astype(str);raw=z['action_features']
    with np.load(a.roster/'pairs.npz') as z:pairs=z['gene_indices'];pair_ids=z['pair_ids']
    if a.limit:pairs=pairs[:a.limit];pair_ids=pair_ids[:a.limit]
    result=np.lib.format.open_memmap(a.output/'features.npy',mode='w+',dtype='f4',shape=(len(pairs),len(recs)*Simulator.width))
    zeros=np.zeros((len(pairs),len(recs)),np.float32);latent_diag=np.zeros_like(zeros)
    for ci,rec in enumerate(recs):
        sim=Simulator(world,a.output/rec['path']);single=sim.singles(raw,a.batch)
        np.savez_compressed(a.output/(rec['name']+'-single-responses.npz'),gene_ids=genes,responses=single[2].numpy(),features=single[3].numpy())
        for start in range(0,len(pairs),a.batch):
            stop=min(start+a.batch,len(pairs));x,z,d=sim.pairs(pairs[start:stop],*single)
            result[start:stop,ci*Simulator.width:(ci+1)*Simulator.width]=x;zeros[start:stop,ci]=z;latent_diag[start:stop,ci]=d
            if start%(a.batch*200)==0:print(json.dumps({'context':rec['name'],'pairs':stop,'total':len(pairs),'seconds':time.time()-started}),flush=True)
        result.flush()
    np.savez_compressed(a.output/'pair-scores.npz',pair_ids=pair_ids,response_similarity=zeros.mean(1),context_similarity=zeros,latent_nonadditivity=latent_diag)
    files={p.name:sha(p) for p in a.output.iterdir() if p.is_file()}
    write(a.output/'manifest.json',{'schema':'slp.cell-world-sl-features/v1','random_control':a.random,'rows':len(pairs),'width':result.shape[1],
        'world_manifest_sha256':sha(a.bundle/'manifest.json'),'world_weights_sha256':sha(a.bundle/'model.safetensors'),
        'pair_roster_sha256':sha(a.roster/'pairs.npz'),'gene_roster_sha256':sha(a.roster/'genes.npz'),'training_index_sha256':sha(a.index/'manifest.json'),
        'contexts':recs,'files':files,'seconds':time.time()-started,'labels_read':False,'source_sha256':sha(__file__),
        'random_control_definition':'seed1731 untrained architecture with nonzero random dynamics output projection' if a.random else None,
        'zero_shot_rule':'mean positive cosine of predicted single-intervention molecular responses over the three fixed contexts'} )
    print(json.dumps({'event':'complete','seconds':time.time()-started,'output':str(a.output)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--index',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--roster',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--batch',type=int,default=128);p.add_argument('--random',action='store_true');p.add_argument('--limit',type=int)
    extract(p.parse_args())
