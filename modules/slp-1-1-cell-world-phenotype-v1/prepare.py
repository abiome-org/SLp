"""Align frozen molecular states to single-gene fitness, keeping fitting separate."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from phenotype import sha


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    with np.load(a.genes) as z:ids=z['gene_ids'].astype(str);raw=z['action_features']
    states=[]
    for name in ('k562','rpe1','hepg2'):
        with np.load(a.states/f'{name}-single-responses.npz') as z:
            if not np.array_equal(ids,z['gene_ids']):raise ValueError('molecular states not aligned')
            states.append(z['features'])
    sys.path.insert(0,str(a.world.resolve()));from inference import WorldModel
    world=WorldModel(a.world,'cuda')
    with torch.inference_mode():action=world.model.action(world.descriptors(raw)).cpu().numpy()
    states=np.concatenate([*states,action],1).astype('f4')
    meta=list(csv.DictReader(a.meta.open(encoding='utf-8-sig')));mapping={r['ensembl_gene_id'].split('.')[0]:i for i,r in enumerate(meta)}
    if not all(g in mapping for g in ids):raise ValueError('fitness source lacks gene identity mapping')
    columns=np.array([mapping[g] for g in ids]);excluded=set(a.excluded.read_text().splitlines())
    with np.load(a.fitness) as z:
        y=z['dependency'][:,columns].T.astype('f4');known=z['dependency_known'][:,columns].T
        context=z['cell_state'];context_ids=z['model_ids'];allowed=z['train_gene'][columns]&np.array([g not in excluded for g in ids])
        cells=z['train_cell'];held_cells=np.array([hashlib.sha256(str(g).encode()).digest()[0]<51 for g in context_ids])
        fit_cells=cells&~held_cells
    dev=np.array([hashlib.sha256(g.encode()).digest()[0]<51 for g in ids]);fit=allowed&~dev;valid=allowed&dev
    # This is an outcome panel of continuous measured single knockouts. No pair
    # label, pair membership or interaction score is constructed here.
    y=np.where(known,y,0.).astype('f4')
    for name,take in [('train',fit),('validation',valid)]:
        np.savez_compressed(a.output/(name+'.npz'),gene_ids=ids[take],states=states[take],targets=y[take][:,fit_cells],known=known[take][:,fit_cells],
            contexts=context[fit_cells],context_ids=context_ids[fit_cells])
    np.savez_compressed(a.output/'forecast-requests.npz',gene_ids=ids,states=states,contexts=context,context_ids=context_ids,reference_cells=cells)
    final=(~allowed)&(known[:,cells].mean(1)>.8)
    np.savez_compressed(a.output/'held-phenotypes.npz',gene_ids=ids[final],states=states[final],targets=y[final][:,cells&held_cells],known=known[final][:,cells&held_cells],
        contexts=context[cells&held_cells],context_ids=context_ids[cells&held_cells])
    manifest={'source':'DepMap24Q2 raw continuous CRISPR gene effects; species9606','fit_genes':int(fit.sum()),'development_genes':int(valid.sum()),
        'held_genes':int(final.sum()),'fitting_cells':int(fit_cells.sum()),'held_cells':int((cells&held_cells).sum()),
        'rights':'DepMap24Q2 public CC-BY4.0; downstream redistribution is separate','world_weights_sha256':sha(a.world/'model.safetensors'),
        'inputs':{str(p):sha(p) for p in (a.fitness,a.meta,a.genes,a.excluded,a.states/'manifest.json')},
        'files':{p.name:sha(p) for p in a.output.glob('*.npz')},'normalization':'raw gene effects; no second transform or use of held outcome normalization'}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('genes','states','world','meta','excluded','fitness','output'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
