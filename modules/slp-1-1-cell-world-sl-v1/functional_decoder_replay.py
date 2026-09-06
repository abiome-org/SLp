"""Replay every frozen fold prediction without accessing SL labels."""
import argparse,json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import torch
from bridge import sha,write
from functional_bridge import FunctionalSimulator


def main(a):
    torch.set_num_threads(4);lock=json.loads((a.fit/'fit-manifest.json').read_text())
    for name,digest in lock['files'].items():
        if sha(a.fit/name)!=digest:raise ValueError('changed lock '+name)
    with np.load(a.roster/'pairs.npz') as z:ix=z['gene_indices'];ids=z['pair_ids']
    sims={arm:FunctionalSimulator(a.functional,a.genes,a.contexts,a.device,random=arm=='random') for arm in ('world','random')}
    rows=[]
    for seed in (42,432):
        for fold in range(5):
            prefix=f'seed{seed}-fold{fold}'
            with np.load(a.fit/f'{prefix}-predictions.npz') as z:
                pairs=ix[z['pair_indices']];assert np.array_equal(ids[z['pair_indices']],z['pair_ids'])
                for arm,sim in sims.items():
                    score,features=sim.pairs(pairs,features=True)
                    model=lgb.Booster(model_file=str(a.fit/f'{prefix}-{arm}.txt'));p=model.predict(features,num_threads=4)
                    rows.append({'seed':seed,'fold':fold,'arm':arm,'n':len(p),'decoder_error':float(abs(p-z[arm]).max()),
                        'label_free_error':float(abs(score-z['label_free_'+arm]).max())})
    passed=all(r['decoder_error']<=1e-7 and r['label_free_error']<=1e-6 for r in rows)
    report={'passed':passed,'rows':rows,'test_labels_accessed':False,'fit_manifest_sha256':sha(a.fit/'fit-manifest.json')}
    write(a.output,report);print(json.dumps({'passed':passed,'models':len(rows),'test_occurrences_per_arm':sum(r['n'] for r in rows if r['arm']=='world'),
        'decoder_max_error':max(r['decoder_error'] for r in rows),'label_free_max_error':max(r['label_free_error'] for r in rows)}))
    if not passed:raise AssertionError('saved fold prediction replay failed')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('fit','functional','genes','contexts','roster','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda');main(p.parse_args())
