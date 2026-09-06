"""Fold-local SL decoding of frozen functional-world states, without raw inputs."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from bridge import sha,write
from decoder import labels,inner,lgb,logloss,metrics
from functional_bridge import FunctionalSimulator


def fit(a):
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);start=time.time()
    with np.load(a.roster/'pairs.npz') as z:ids=z['pair_ids'].astype(str);ix=z['gene_indices']
    trained=FunctionalSimulator(a.functional,a.genes,a.contexts,a.device)
    random=FunctionalSimulator(a.functional,a.genes,a.contexts,a.device,random=True)
    assert np.array_equal(trained.ids[ix],ids)
    rows=[];receipts={str(p):sha(p) for p in (a.functional/'model.safetensors',a.functional/'training.json',a.genes,a.contexts,a.roster/'pairs.npz')}
    for seed in (42,432):
        trainfile=a.labels/f'train_labels_seed{seed}.pkl';receipts[str(trainfile)]=sha(trainfile)
        for fold in range(5):
            path=a.roster/f'seed{seed}-fold{fold}.npz';receipts[str(path)]=sha(path)
            with np.load(path) as z:
                take=z['partition']==0;tr=z['pair_indices'][take];te=z['pair_indices'][~take];source=z['source_row'][take];testsource=z['source_row'][~take]
            if set(ids[tr].ravel())&set(ids[te].ravel()):raise ValueError('held gene overlap')
            y=labels(trainfile,fold)[source];f,v=inner(ids[tr],seed,fold);pred={'pair_ids':ids[te],'pair_indices':te,'source_row':testsource}
            prefix=f'seed{seed}-fold{fold}';row={'seed':seed,'fold':fold,'train':len(tr),'test':len(te),'arms':{}}
            for arm,sim in [('world',trained),('random',random)]:
                _,X=sim.pairs(ix[tr],features=True);zero,T=sim.pairs(ix[te],features=True)
                model=lgb(X[f],y[f],(X[v],y[v]));trees=int(model.best_iteration_);loss=logloss(y[v],model.predict_proba(X[v])[:,1])
                model=lgb(X,y,trees=trees);pred[arm]=model.predict_proba(T)[:,1];pred['label_free_'+arm]=zero
                model.booster_.save_model(str(a.output/f'{prefix}-{arm}.txt'))
                row['arms'][arm]={'trees':trees,'inner_logloss':loss,'features':X.shape[1]}
                print(json.dumps({'seed':seed,'fold':fold,'arm':arm,'seconds':time.time()-start,**row['arms'][arm]}),flush=True)
            np.savez_compressed(a.output/f'{prefix}-predictions.npz',**pred);rows.append(row)
            write(a.output/'progress.json',{'rows':rows,'seconds':time.time()-start})
    write(a.output/'fit.json',{'rows':rows,'seconds':time.time()-start,'world_weights_frozen':True,
        'features':'768 symmetric coordinates of full 256-dimensional post-intervention state change +15 continuous conditional fitness coordinates; no raw descriptor bypass',
        'label_free_rule':'negative mean symmetrized conditional excess gene effect; three fixed observed contexts',
        'decoder':'same LightGBM31leaves .03LR training-gene-held inner selection as previous cellular-world decoder'})
    write(a.output/'fit-manifest.json',{'files':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()},
        'inputs':receipts,'test_labels_accessed':False,'source_sha256':sha(__file__)})


def score(a):
    manifest=json.loads((a.output/'fit-manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if sha(a.output/name)!=digest:raise ValueError('changed frozen outputs')
    if (a.output/'scores.json').exists():raise FileExistsError('already scored')
    rows=[];arms=('world','random','label_free_world','label_free_random')
    for seed in (42,432):
        for fold in range(5):
            with np.load(a.output/f'seed{seed}-fold{fold}-predictions.npz') as z:
                y=labels(a.labels/f'test_labels_seed{seed}.pkl',fold)[z['source_row']]
                for arm in arms:rows.append({'seed':seed,'fold':fold,'arm':arm,'n':len(y),**metrics(y,z[arm])})
    macro={arm:{k:float(np.mean([r[k] for r in rows if r['arm']==arm])) for k in ('auroc','ap','pr_auc')} for arm in arms}
    write(a.output/'scores.json',{'rows':rows,'macro':macro,'fit_manifest_sha256':sha(a.output/'fit-manifest.json'),
        'test_labels':{str(a.labels/f'test_labels_seed{s}.pkl'):sha(a.labels/f'test_labels_seed{s}.pkl') for s in (42,432)},
        'scope':'retrospective official MuSL CV3; two SL-label-cold genes. Functional pretraining uses human single and species-native yeast double fitness.'})
    print(json.dumps(macro,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','score'],required=True)
    for name in ('functional','genes','contexts','roster','labels','output'):p.add_argument('--'+name,type=Path)
    p.add_argument('--device',default='cuda');a=p.parse_args();fit(a) if a.phase=='fit' else score(a)
