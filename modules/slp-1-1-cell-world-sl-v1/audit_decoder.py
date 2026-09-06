"""Replay every saved fold decoder against its frozen SL predictions."""
import argparse,json
from pathlib import Path
import lightgbm as lgb
import numpy as np
from bridge import sha,write


def main(a):
    lock=json.loads((a.fit/'fit-manifest.json').read_text())
    for name,digest in lock['files'].items():
        if sha(a.fit/name)!=digest:raise ValueError('changed fit artifact')
    with np.load(a.roster/'genes.npz') as z:raw=z['action_features']
    with np.load(a.roster/'pairs.npz') as z:ix=z['gene_indices'];ids=z['pair_ids']
    world=np.load(a.world/'features.npy',mmap_mode='r');random=np.load(a.random/'features.npy',mmap_mode='r')
    errors={};count=0
    for seed in (42,432):
        for fold in range(5):
            prefix=f'seed{seed}-fold{fold}'
            with np.load(a.fit/f'{prefix}-predictions.npz') as z:
                rows=z['pair_indices'];assert np.array_equal(ids[rows],z['pair_ids']);count+=len(rows)
                for arm in ('world','random','static'):
                    if arm=='static':
                        left=raw[ix[rows,0]];right=raw[ix[rows,1]];x=np.concatenate((left+right,abs(left-right),left*right),1)
                    else:x=np.asarray((world if arm=='world' else random)[rows])
                    p=lgb.Booster(model_file=str(a.fit/f'{prefix}-{arm}.txt')).predict(x,num_threads=4)
                    error=float(abs(p-z[arm]).max());errors[f'{prefix}/{arm}']=error
                    assert error<=1e-12,error
    write(a.output,{'passed':True,'models':len(errors),'pair_occurrences':count,'max_error':max(errors.values()),'errors':errors,
                   'fit_manifest_sha256':sha(a.fit/'fit-manifest.json'),'test_labels_read':False})
    print(json.dumps({'models':len(errors),'pairs':count,'max_error':max(errors.values())}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('fit','roster','world','random','output'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
