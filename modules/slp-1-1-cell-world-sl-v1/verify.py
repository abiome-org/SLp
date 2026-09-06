"""Real-artifact checks for molecular feature routing and pair symmetry."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from bridge import Simulator,load_world,sha,write
from decoder import inner


def main(a):
    torch.set_num_threads(4);world=load_world(a.bundle,a.device)
    with np.load(a.roster/'genes.npz') as z:raw=z['action_features']
    with np.load(a.roster/'pairs.npz') as z:indices=z['gene_indices'];ids=z['pair_ids']
    pick=np.array([0,13,500,15000,180000]);selected=indices[pick];unique=np.unique(selected)
    positions={int(g):i for i,g in enumerate(unique)};pairs=np.array([[positions[int(g)] for g in row] for row in selected])
    expected=np.load(a.features/'features.npy',mmap_mode='r')[pick];actual=[];symmetry=0.;signals=[]
    for name in ('k562','rpe1','hepg2'):
        sim=Simulator(world,a.features/f'{name}-context.npz');single=sim.singles(raw[unique],batch=3)
        x,_,_=sim.pairs(pairs,*single);reverse,_,_=sim.pairs(pairs[:,::-1].copy(),*single)
        symmetry=max(symmetry,float(abs(x-reverse).max()));actual.append(x)
        # Last 32 projected molecular coordinates before summaries are exactly
        # double - single A - single B, and must carry a nonzero numerical signal.
        signals.append(float(np.std(x[:,320:352])))
    error=float(abs(np.concatenate(actual,1)-expected).max())
    assert error<2e-4,error
    assert symmetry<2e-4,symmetry
    assert all(v>1e-7 for v in signals),signals
    for seed in (42,432):
        for fold in range(5):
            with np.load(a.roster/f'seed{seed}-fold{fold}.npz') as z:
                tr=z['pair_indices'][z['partition']==0];te=z['pair_indices'][z['partition']==1]
            assert not(set(ids[tr].reshape(-1))&set(ids[te].reshape(-1)))
            f,v=inner(ids[tr],seed,fold)
            assert f.any() and v.any() and not(set(ids[tr][f].reshape(-1))&set(ids[tr][v].reshape(-1)))
    write(a.output,{'passed':True,'feature_replay_max_error':error,'pair_order_max_error':symmetry,
         'molecular_interaction_std':signals,'outer_and_inner_gene_disjoint_splits':10,'world_weights_sha256':sha(a.bundle/'model.safetensors')})
    print(a.output.read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('bundle','features','roster','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda');main(p.parse_args())
