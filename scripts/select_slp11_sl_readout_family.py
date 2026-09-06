"""Lock a readout family on fitting validation, then score frozen predictions."""
from __future__ import annotations
import argparse, hashlib, json, pickle, shutil
from pathlib import Path
import numpy as np

SEEDS=(42,432); FOLDS=range(5); COUNT_KEYS=('outer_train','outer_test','inner_fit','inner_validation','inner_held_genes')

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def load_json(path):return json.loads(Path(path).read_text())

def verify_manifest(root):
    manifest=load_json(root/'fit-manifest.json')
    for name,digest in manifest['artifacts'].items():
        if sha(root/name)!=digest:raise ValueError(f'{root.name}: artifact hash mismatch: {name}')
    return manifest

def run_map(params):
    result={(int(x['seed']),int(x['fold'])):x for x in params['runs']}
    if len(params['runs'])!=10 or set(result)!={(s,f) for s in SEEDS for f in FOLDS}:raise ValueError('params must contain exactly ten seed/fold runs')
    return result

def select_family(v1_loss,v2_loss):
    if not np.isfinite([v1_loss,v2_loss]).all():raise ValueError('inner losses must be finite')
    return 'v1' if float(v1_loss)<=float(v2_loss) else 'v2'

def validate_alignment(left,right,roster_fold,global_pair_ids):
    expected=np.asarray(roster_fold['pair_indices'])[np.asarray(roster_fold['partition'])==1]
    expected_rows=np.asarray(roster_fold['source_row'])[np.asarray(roster_fold['partition'])==1]
    for values in (left,right):
        if not np.array_equal(values['pair_indices'],expected) or not np.array_equal(values['source_row'],expected_rows):raise ValueError('prediction rows differ from canonical roster test partition')
        if not np.array_equal(values['pair_ids'].astype(str),global_pair_ids[expected].astype(str)):raise ValueError('prediction pair identities differ from canonical roster')
    for key in ('pair_indices','source_row','pair_ids'):
        if not np.array_equal(left[key],right[key]):raise ValueError(f'readout families disagree on {key}')

def lock(v1,v2,roster,output):
    if output.exists():raise FileExistsError(output)
    manifests=[verify_manifest(x) for x in (v1,v2)]; params=[load_json(x/'params.json') for x in (v1,v2)]; runs=[run_map(x) for x in params]
    output.mkdir(parents=True); pair_ids=np.load(roster/'pairs.npz',allow_pickle=False)['pair_ids']; selections=[]; artifacts={}
    for label,root in (('v1',v1),('v2',v2)):
        for source,suffix in (('params.json','params.json'),('fit-manifest.json','fit-manifest.json')):
            name=f'{label}-{suffix}';shutil.copyfile(root/source,output/name);artifacts[name]=sha(output/name)
    for seed in SEEDS:
        for fold in FOLDS:
            key=(seed,fold); a,b=runs[0][key],runs[1][key]
            if any(int(a[k])!=int(b[k]) for k in COUNT_KEYS):raise ValueError(f'run split counts differ for {key}')
            paths=[root/f'seed{seed}-fold{fold}-predictions.npz' for root in (v1,v2)]
            with np.load(paths[0],allow_pickle=False) as x, np.load(paths[1],allow_pickle=False) as y, np.load(roster/f'seed{seed}-fold{fold}.npz',allow_pickle=False) as rf:
                left={k:np.asarray(x[k]) for k in x.files};right={k:np.asarray(y[k]) for k in y.files};fold_values={k:np.asarray(rf[k]) for k in rf.files}
            validate_alignment(left,right,fold_values,pair_ids)
            family=select_family(a['inner_logloss'],b['inner_logloss'])
            chosen=left if family=='v1' else right; destination=output/f'seed{seed}-fold{fold}-selected.npz'
            np.savez_compressed(destination,pair_indices=chosen['pair_indices'],source_row=chosen['source_row'],pair_ids=chosen['pair_ids'],prediction=chosen['blend'],selected_family=np.asarray(family))
            artifacts[destination.name]=sha(destination);selections.append({'seed':seed,'fold':fold,'selectedFamily':family,'v1InnerLogloss':float(a['inner_logloss']),'v2InnerLogloss':float(b['inner_logloss']),'counts':{k:int(a[k]) for k in COUNT_KEYS},'predictionSha256':artifacts[destination.name]})
    receipt={'schema':'slp.sl-readout-family-lock/v1','phase':'locked-before-test-label-access','testLabelsAccessed':False,'rule':'lower inner_logloss per seed/fold; exact tie selects v1','families':{'v1':{'path':str(v1),'fitManifestSha256':sha(v1/'fit-manifest.json'),'paramsSha256':sha(v1/'params.json')},'v2':{'path':str(v2),'fitManifestSha256':sha(v2/'fit-manifest.json'),'paramsSha256':sha(v2/'params.json')}},'rosterManifestSha256':sha(roster/'manifest.json'),'selections':selections,'artifacts':artifacts}
    receipt.update(retrospective=True, selectorSha256=sha(__file__), priorV1ScoresPreviouslyInspected=True)
    (output/'lock.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps({'output':str(output),'selected':{x:sum(s['selectedFamily']==x for s in selections) for x in ('v1','v2')},'lockSha256':sha(output/'lock.json')}))

def score(output,labels_root):
    from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,auc,f1_score
    if (output/'score.json').exists():raise FileExistsError(output/'score.json')
    receipt=load_json(output/'lock.json')
    if receipt.get('phase')!='locked-before-test-label-access' or receipt.get('testLabelsAccessed') is not False:raise ValueError('invalid pre-score lock')
    for name,digest in receipt['artifacts'].items():
        if sha(output/name)!=digest:raise ValueError(f'locked prediction hash mismatch: {name}')
    for family,record in receipt['families'].items():
        root=Path(record['path'])
        if sha(root/'fit-manifest.json')!=record['fitManifestSha256'] or sha(root/'params.json')!=record['paramsSha256']:raise ValueError(f'{family} source lock changed')
    rows=[]
    for seed in SEEDS:
        with (labels_root/f'test_labels_seed{seed}.pkl').open('rb') as f: labels=pickle.load(f)
        for fold in FOLDS:
            with np.load(output/f'seed{seed}-fold{fold}-selected.npz',allow_pickle=False) as a:
                source_row=np.asarray(a['source_row'],np.int64);full_labels=np.asarray(labels[fold]).reshape(-1)
                if not np.isin(full_labels,[0,1]).all():raise ValueError('official labels must be binary')
                if np.any(source_row<0) or np.any(source_row>=len(full_labels)):raise ValueError('locked source_row is outside official fold labels')
                y=full_labels[source_row];p=np.asarray(a['prediction'],np.float64);family=str(a['selected_family'])
            if y.shape!=p.shape or not np.isfinite(p).all():raise ValueError('labels/predictions do not align')
            precision,recall,_=precision_recall_curve(y,p)
            rows.append({'seed':seed,'fold':fold,'selectedFamily':family,'pairs':len(y),'positives':int(y.sum()),'auroc':float(roc_auc_score(y,p)),'averagePrecision':float(average_precision_score(y,p)),'trapezoidalPrAuc':float(auc(recall,precision)),'f1AtPoint5':float(f1_score(y,p>=.5))})
    metrics=('auroc','averagePrecision','trapezoidalPrAuc','f1AtPoint5');report={'schema':'slp.sl-readout-family-score/v1','lockSha256':sha(output/'lock.json'),'rows':rows,'macro':{k:float(np.mean([r[k] for r in rows])) for k in metrics}}
    report['retrospective']=True
    report['testLabelReceipts']={str(labels_root/f'test_labels_seed{s}.pkl'):sha(labels_root/f'test_labels_seed{s}.pkl') for s in SEEDS}
    (output/'score.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report['macro']))

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='phase',required=True);q=sub.add_parser('lock');q.add_argument('--v1',type=Path,required=True);q.add_argument('--v2',type=Path,required=True);q.add_argument('--roster',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q=sub.add_parser('score');q.add_argument('--output',type=Path,required=True);q.add_argument('--labels-root',type=Path,required=True);a=p.parse_args()
    lock(a.v1,a.v2,a.roster,a.output) if a.phase=='lock' else score(a.output,a.labels_root)
if __name__=='__main__':main()
