"""Fold-local SL decoders and a separately scored label-free world readout."""
import argparse,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
from bridge import sha,write


def labels(path,fold):
    with path.open('rb') as f:folds=pickle.load(f)
    if len(folds)!=5:raise ValueError('five official folds required')
    y=np.asarray(folds[fold]).reshape(-1)
    if not np.isin(y,[0,1]).all():raise ValueError('invalid binary targets')
    return y.astype('u1')


def inner(ids,seed,fold):
    genes=np.unique(ids);held={g for g in genes if hashlib.sha256(f'{seed}:{fold}:{g}'.encode()).digest()[0]<51}
    a=np.isin(ids[:,0],list(held));b=np.isin(ids[:,1],list(held))
    return ~(a|b),a&b


def logloss(y,p):
    p=np.clip(p,1e-7,1-1e-7)
    return float(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean())


def lgb(X,y,validation=None,trees=1000):
    import lightgbm as l
    m=l.LGBMClassifier(objective='binary',n_estimators=trees,num_leaves=31,learning_rate=.03,
        feature_fraction=.7,lambda_l2=1.,min_child_samples=30,deterministic=True,force_col_wise=True,seed=123,num_threads=4,verbosity=-1)
    kwargs={} if validation is None else {'eval_set':[validation],'eval_metric':'binary_logloss','callbacks':[l.early_stopping(50,verbose=False)]}
    m.fit(X,y,**kwargs)
    return m


def verify(directory):
    m=json.loads((directory/'manifest.json').read_text())
    for file,digest in m['files'].items():
        if sha(directory/file)!=digest:raise ValueError('changed feature payload '+file)
    return m


def fit(a):
    started=time.time();a.output.mkdir(parents=True,exist_ok=False)
    wm=verify(a.world);rm=verify(a.random)
    with np.load(a.roster/'pairs.npz') as z:ids=z['pair_ids'].astype(str);ix=z['gene_indices']
    with np.load(a.roster/'genes.npz') as z:raw=z['action_features'];assert np.array_equal(z['gene_ids'][ix],ids)
    for m in (wm,rm):
        if m['pair_roster_sha256']!=sha(a.roster/'pairs.npz') or m['rows']!=len(ids):raise ValueError('features differ from canonical roster')
    if wm['random_control'] or not rm['random_control']:raise ValueError('world/random arms reversed')
    xw=np.load(a.world/'features.npy',mmap_mode='r');xr=np.load(a.random/'features.npy',mmap_mode='r')
    with np.load(a.world/'pair-scores.npz') as z:
        assert np.array_equal(ids,z['pair_ids']);zero=z['response_similarity']
    with np.load(a.random/'pair-scores.npz') as z:
        assert np.array_equal(ids,z['pair_ids']);random_zero=z['response_similarity']
    inputs={str(p):sha(p) for p in (a.roster/'pairs.npz',a.roster/'genes.npz',a.world/'manifest.json',a.random/'manifest.json',Path(__file__))}
    rows=[]
    for seed in (42,432):
        trainfile=a.labels/f'train_labels_seed{seed}.pkl';inputs[str(trainfile)]=sha(trainfile)
        for fold in range(5):
            foldpath=a.roster/f'seed{seed}-fold{fold}.npz';inputs[str(foldpath)]=sha(foldpath)
            with np.load(foldpath) as z:
                train=z['partition']==0;test=~train;tr=z['pair_indices'][train];te=z['pair_indices'][test]
                source=z['source_row'][train];test_source=z['source_row'][test]
            if set(ids[tr].reshape(-1))&set(ids[te].reshape(-1)):raise ValueError('SL train/test genes overlap')
            y=labels(trainfile,fold)[source];f,v=inner(ids[tr],seed,fold)
            prefix=f'seed{seed}-fold{fold}';pred={'pair_ids':ids[te],'pair_indices':te,'source_row':test_source,
                'label_free_world':zero[te],'label_free_random':random_zero[te]}
            row={'seed':seed,'fold':fold,'train':len(tr),'test':len(te),'inner_fit':int(f.sum()),'inner_validation':int(v.sum()),'arms':{}}
            for arm in ('world','random','static'):
                if arm=='static':
                    left=raw[ix[tr,0]];right=raw[ix[tr,1]];X=np.concatenate((left+right,abs(left-right),left*right),1)
                    left=raw[ix[te,0]];right=raw[ix[te,1]];T=np.concatenate((left+right,abs(left-right),left*right),1)
                else:X=np.asarray((xw if arm=='world' else xr)[tr]);T=np.asarray((xw if arm=='world' else xr)[te])
                model=lgb(X[f],y[f],(X[v],y[v]));trees=int(model.best_iteration_)
                loss=logloss(y[v],model.predict_proba(X[v])[:,1])
                model=lgb(X,y,trees=trees);pred[arm]=model.predict_proba(T)[:,1]
                model.booster_.save_model(str(a.output/f'{prefix}-{arm}.txt'))
                row['arms'][arm]={'inner_logloss':loss,'trees':trees,'features':X.shape[1]}
                print(json.dumps({'event':'decoder_fit','seed':seed,'fold':fold,'arm':arm,'inner_logloss':loss,'trees':trees,'seconds':time.time()-started}),flush=True)
            np.savez_compressed(a.output/f'{prefix}-predictions.npz',**pred);rows.append(row)
            write(a.output/'progress.json',{'rows':rows,'seconds':time.time()-started})
    write(a.output/'fit.json',{'rows':rows,'seconds':time.time()-started,'primary_arm':'world','world_weights_frozen':True,
         'decoder':'LightGBM31leaves; learning_rate.03; training-only gene-held early stopping; no raw descriptors in world arm',
         'label_free_rule':'positive response cosine, equal mean over K562/RPE1/HepG2; no labels, sign selection or calibration'})
    files={p.name:sha(p) for p in a.output.iterdir() if p.is_file()}
    write(a.output/'fit-manifest.json',{'files':files,'inputs':inputs,'test_labels_accessed':False,'source_sha256':sha(__file__)})


def metrics(y,p):
    from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,auc
    pr,re,_=precision_recall_curve(y,p)
    return {'auroc':float(roc_auc_score(y,p)),'ap':float(average_precision_score(y,p)),'pr_auc':float(auc(re,pr))}


def score(a):
    verify_fit=json.loads((a.output/'fit-manifest.json').read_text())
    for file,digest in verify_fit['files'].items():
        if sha(a.output/file)!=digest:raise ValueError('locked prediction/model changed')
    if (a.output/'scores.json').exists():raise FileExistsError('scores already exist')
    rows=[];arms=('world','random','static','label_free_world','label_free_random')
    for seed in (42,432):
        for fold in range(5):
            with np.load(a.output/f'seed{seed}-fold{fold}-predictions.npz') as z:
                y=labels(a.labels/f'test_labels_seed{seed}.pkl',fold)[z['source_row']]
                for arm in arms:rows.append({'seed':seed,'fold':fold,'arm':arm,'n':len(y),'positives':int(y.sum()),**metrics(y,z[arm])})
    macro={arm:{k:float(np.mean([r[k] for r in rows if r['arm']==arm])) for k in ('auroc','ap','pr_auc')} for arm in arms}
    # Paired fold differences are descriptive: official folds overlap across
    # seeds and are not ten independent biological replicates.
    differences={}
    for base in ('random','static'):
        differences[base]={k:[r[k]-s[k] for r,s in zip([r for r in rows if r['arm']=='world'],[r for r in rows if r['arm']==base])] for k in ('auroc','ap')}
    write(a.output/'scores.json',{'benchmark':'official MuSL CV3, both SL-label-cold genes, seeds42/432, five folds each',
        'rows':rows,'macro':macro,'world_minus_control_per_fold':differences,'fit_manifest_sha256':sha(a.output/'fit-manifest.json'),
        'test_labels':{str(a.labels/f'test_labels_seed{s}.pkl'):sha(a.labels/f'test_labels_seed{s}.pkl') for s in (42,432)},
        'scope':'retrospective benchmark; molecular pretraining may include single-intervention outcomes of SL-held genes; SL decoder training never includes their SL labels'})
    print(json.dumps(macro,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','score'],required=True)
    for name in ('roster','labels','world','random','output'):p.add_argument('--'+name,type=Path,required=name in ('labels','output'))
    a=p.parse_args();fit(a) if a.phase=='fit' else score(a)
