"""Additional fixed-seed controls for quantitative-pretraining transfer."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from bridge import sha,write
from functional_bridge import FunctionalSimulator


def main(a):
    torch.set_num_threads(4);parts=[];truth=[];trained=[];static=[];cold=[]
    with np.load(a.predictions) as z:
        for seed in (42,432):
            for fold in range(5):
                k=f'CV3-{seed}-{fold}';parts.append(z[k+'-pairs']);truth.append(z[k+'-y']);trained.append(z[k+'-world_label_free']);static.append(z[k+'-static_cosine']);cold.append(z[k+'-molecular-held'])
    ids=np.concatenate(parts);y=np.concatenate(truth);p=np.concatenate(trained);static=np.concatenate(static);cold=np.concatenate(cold)
    sim=FunctionalSimulator(a.functional,a.genes,a.contexts,a.device,random=True);lookup={g:i for i,g in enumerate(sim.ids)}
    ix=np.array([[lookup[g] for g in row] for row in ids]);controls=[]
    with np.load(a.genes) as z:
        n=sim.world.norm;raw=z['human_raw']*n['feature_scale']+n['feature_mean'];signatures=z['human_signatures']
    seeds=list(range(1731,1736))
    for seed in seeds:
        torch.manual_seed(seed);sim.world.model=type(sim.world.model)().to(a.device).eval()
        sim.actions=sim.world.actions(raw,signatures);sim.states=[sim.world.encode(c[None]) for c in sim.contexts]
        sim.single=[sim.world.observe(sim.world.expand(s,len(sim.actions)),sim.actions) for s in sim.states]
        score=sim.pairs(ix);controls.append(score)
    controls=np.stack(controls);average=controls.mean(0);genes,inverse=np.unique(ids,return_inverse=True);gi=inverse.reshape(-1,2)
    rng=np.random.default_rng(731);draws=[]
    for _ in range(500):
        gw=rng.poisson(1.,len(genes));weight=gw[gi[:,0]]*gw[gi[:,1]]
        w=roc_auc_score(y,p,sample_weight=weight);r=roc_auc_score(y,average,sample_weight=weight)
        draws.append([w,r,w-r])
    draws=np.array(draws);point=[roc_auc_score(y,p),roc_auc_score(y,average)];point.append(point[0]-point[1])
    report={'seeds':seeds,'random_seed_pooled_auroc':[float(roc_auc_score(y,v)) for v in controls],
        'scores':{k:{'auroc':float(point[j]),'interval95':np.quantile(draws[:,j],[.025,.975]).tolist()} for j,k in enumerate(('trained','mean_random_score','trained_minus_mean_random'))},
        'both_molecular_held':{'n':int(cold.sum()),'trained_pooled_auroc':float(roc_auc_score(y[cold],p[cold])),
            'mean_random_pooled_auroc':float(roc_auc_score(y[cold],average[cold])),'static_cosine_pooled_auroc':float(roc_auc_score(y[cold],static[cold]))},
        'scope':'fixed model, five prespecified untrained functional initializations over the same frozen molecular signatures; 500 shared gene-weight bootstrap draws. Retrospective transfer, not prospective replication.',
        'predictions_sha256':sha(a.predictions),'functional_weights_sha256':sha(a.functional/'model.safetensors')}
    write(a.output,report);np.savez_compressed(a.output.with_suffix('.npz'),pair_ids=ids,random_scores=controls,trained_score=p,y=y)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('predictions','functional','genes','contexts','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda');main(p.parse_args())
