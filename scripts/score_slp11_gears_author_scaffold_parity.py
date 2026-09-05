#!/usr/bin/env python3
"""Score already-frozen GEARS means through SLIM's unchanged cell scaffold/eval."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, shutil, sys, tempfile
from pathlib import Path
import anndata as ad
import numpy as np
import h5py
from sklearn.decomposition import PCA

ROOT=Path(__file__).resolve().parents[1]
UP=ROOT/'data/tooling/slim-5a7e9ade'; MODELS=ROOT/'results/slp11-transition/gears-response-models-v4-corrected-receipts'
DATA=ROOT/'data/derived/slp11-gears-canonical-v1'; SOURCE=ROOT/'data/sources/slim-canonical-gears-v1/extracted'
NAMES=('replogle_k562_essential','replogle_rpe1_essential'); ARMS=('slimPublished','slimFitCv','reducedRankConcat')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
RUN=module('parity_gears_runner',ROOT/'scripts/run_slp11_gears_response_models.py'); PREP=module('parity_gears_prep',ROOT/'scripts/prepare_slp11_gears_benchmark.py'); RRR=RUN.RRR
SCAFFOLD=module('parity_slim_scaffold',UP/'src/slim/scaffold.py');EVAL=module('parity_slim_eval',UP/'src/slim/eval.py');CONDS=module('parity_slim_conditions',UP/'src/slim/condition_utils.py')
build_result_h5ad=SCAFFOLD.build_result_h5ad;compute_perturbation_metrics=EVAL.compute_perturbation_metrics;normalize_condition=CONDS.normalize_condition
def predict(z,x):
 if str(z['family'])=='slim':return (z['query_basis']@z['weight']@(((x-z['feature_mean'])/z['feature_scale']).T)+z['bias']).T
 m=RRR.ReducedRankResponse(z['feature_mean'],z['feature_scale'],z['design_mean'],z['state_projection'],z['query_loading'],z['intercept'],float(z['regularization']));return m.predict(x)
def exact_published_means(train,train_genes,test_genes,*,solver='full',random_state=None):
 cond=train['conditions'].astype(str);Y=train['mean_expression'][cond!='ctrl'].T;b=Y.mean(1,keepdims=True);centered=Y-b;G=PCA(n_components=10,svd_solver=solver,random_state=random_state).fit_transform(centered)
 with h5py.File(UP/'data/gene_string_embeddings.v0.3.h5','r') as h:
  P=np.stack([np.asarray(h[g]) for g in train_genes]);Pn=np.stack([np.asarray(h[g]) for g in test_genes])
 Z=np.linalg.solve(G.T@G+.1*np.eye(10),G.T@centered);W=np.linalg.solve(P.T@P+.1*np.eye(P.shape[1]),(Z@P).T).T
 return (G@W@Pn.T+b).T
def main(out):
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);receipt=json.loads((MODELS/'MODELS-FROZEN-BEFORE-CANONICAL-DEVELOPMENT.json').read_text())
 files=[UP/'src/slim/scaffold.py',UP/'src/slim/eval.py',UP/'src/slim/model.py',UP/'src/slim/datasets/gears_adapter.py',UP/'scripts/eval/run_eval.py']
 protocol={'schema':'slp.gears-author-scaffold-parity/v1','retrospective':True,'testOutcomesPreviouslyAccessed':True,'selectionAfterTest':False,'trainingOnlyAlgebraicValidationAfterPriorTestAccess':True,'validationPredictionsUsedForScoring':False,'arms':list(ARMS),'seed':1,'populationMethod':'rescale','authorCommit':'5a7e9ade5d0a6b6331e6dbc81181450605047bcc','authorFiles':{str(p.relative_to(ROOT)):sha(p) for p in files},'models':receipt['models'],'sources':{}}
 (out/'PROTOCOL-BEFORE-SCORING.json').write_text(json.dumps(protocol,indent=2,sort_keys=True)+'\n')
 report={'schema':'slp.gears-author-scaffold-parity-results/v1','protocolSha256':sha(out/'PROTOCOL-BEFORE-SCORING.json'),'sources':{}}
 for name in NAMES:
  adata=ad.read_h5ad(SOURCE/name/'perturb_processed.h5ad');train=load(DATA/name/'training.npz');sealed=json.loads((DATA/'sealed-test'/f'{name}.json').read_text());conds=np.asarray(sealed['conditions']).astype(str)
  test=[c for c in conds if 'ctrl' in c and c!='ctrl']; genes=[next(g for g in c.split('+') if g!='ctrl') for c in test]
  train_conditions=train['conditions'].astype(str);train_genes=[next(g for g in c.split('+') if g!='ctrl') for c in train_conditions if c!='ctrl']
  # Exact author fit validation for the published arm; no selection and no saved refit.
  exact_means=exact_published_means(train,train_genes,genes,solver='full')
  _,string,concat,present=RUN.feats(name,np.asarray(test));frozen_pub=load(MODELS/receipt['models'][name]['slimPublished']['path']);frozen_means=frozen_pub['control_mean']+predict(frozen_pub,string)
  drift=np.abs(exact_means-frozen_means)
  if drift.max()>1e-8:raise ValueError(f'{name}: frozen published means differ from exact author fit: {drift.max()}')
  control=adata[adata.obs['condition'].astype(str)=='ctrl'].copy();val=set(load(DATA/name/'development.npz')['conditions'].astype(str));held=set(test)|val
  train_cells=adata[~adata.obs['condition'].astype(str).isin(held)].copy(); counts={g:int(np.sum(adata.obs['condition'].astype(str)==c)) for g,c in zip(genes,test)}
  auto=[]
  for seed in (1,2,3):
   candidate=exact_published_means(train,train_genes,genes,solver='randomized',random_state=seed);delta=np.abs(candidate-frozen_means);auto.append({'seed':seed,'maxAbsVsFrozen':float(delta.max()),'meanAbsVsFrozen':float(delta.mean())})
  src={'conditions':test,'conditionRosterSha256':hashlib.sha256(('\n'.join(test)+'\n').encode()).hexdigest(),'queryIds':train['query_ids'].astype(str).tolist(),'queryIdsSha256':hashlib.sha256(('\n'.join(train['query_ids'].astype(str))+'\n').encode()).hexdigest(),'testCellCounts':counts,'publishedMeanValidation':{'implementation':'deterministic full SVD corresponding to frozen model receipt','maxAbs':float(drift.max()),'meanAbs':float(drift.mean()),'authorDefaultRandomizedPcaSensitivity':auto},'arms':{}}
  for arm in ARMS:
   z=load(MODELS/receipt['models'][name][arm]['path']);x=concat if arm=='reducedRankConcat' else string;means=z['control_mean']+predict(z,x);predictions={g:v for g,v in zip(genes,means)}
   with tempfile.TemporaryDirectory(dir='/tmp',prefix='slim-parity-') as td:
    result=build_result_h5ad(predictions,train_cells,counts,td,condition_col='condition',control_tag='ctrl',seed=1,method='rescale')
    result.obs['condition']=result.obs['condition'].astype(str).map(normalize_condition);real=adata.copy();real.obs['condition']=real.obs['condition'].astype(str).map(normalize_condition);real=real[real.obs['condition'].isin(set(test)|{'ctrl'})].copy()
    combined=ad.concat([real[real.obs['condition']=='ctrl'],result],label='source',keys=['ctrl','pred'])
    if hasattr(combined.X,'data'):combined.X.data=np.clip(combined.X.data,0,14.99)
    else:combined.X=np.clip(combined.X,0,14.99)
    metrics=compute_perturbation_metrics(real,combined)
    rows=[];post=[]
    for g,c in zip(genes,test):
     mask=result.obs['condition'].astype(str)==normalize_condition(g);post.append(np.asarray(result[mask].X.mean(0)).ravel());row=metrics[metrics['perturbation']==c]
     if len(row)!=1:raise ValueError(f'{name}/{arm}: score row mismatch {c}')
     rows.append(row.iloc[0])
    post=np.stack(post);np.savez_compressed(out/f'{name}-{arm}-postscaffold-means.npz',conditions=np.asarray(test),gene_ids=np.asarray(genes),query_ids=train['query_ids'],cell_counts=np.asarray([counts[g] for g in genes]),mean_expression=post)
    table=__import__('pandas').DataFrame(rows);table.to_csv(out/f'{name}-{arm}-per-condition.csv',index=False)
    src['arms'][arm]={'modelSha256':sha(MODELS/receipt['models'][name][arm]['path']),'postscaffoldSha256':sha(out/f'{name}-{arm}-postscaffold-means.npz'),'perConditionSha256':sha(out/f'{name}-{arm}-per-condition.csv'),'meanPearsonDelta':float(table.pearson_delta.mean()),'medianPearsonDelta':float(table.pearson_delta.median()),'meanMse':float(table.mse.mean()),'meanMae':float(table.mae.mean()),'meanMmd':float(table.mmd.mean())}
  report['sources'][name]=src;del adata,train_cells,control
 for name,src in report['sources'].items():
  candidate=__import__('pandas').read_csv(out/f'{name}-reducedRankConcat-per-condition.csv');src['pairedBootstrapConcatVs']={}
  for arm in ('slimPublished','slimFitCv'):
   base=__import__('pandas').read_csv(out/f'{name}-{arm}-per-condition.csv')
   if list(candidate.perturbation)!=list(base.perturbation):raise ValueError('paired bootstrap roster mismatch')
   rng=np.random.default_rng(731);ix=rng.integers(0,len(candidate),(10000,len(candidate)));gain=(candidate.pearson_delta.to_numpy()-base.pearson_delta.to_numpy())[ix].mean(1);reduction=(base.mse.to_numpy()-candidate.mse.to_numpy())[ix].mean(1);quant=lambda x:[float(v) for v in np.quantile(x,[.025,.5,.975])]
   src['pairedBootstrapConcatVs'][arm]={'replicates':10000,'seed':731,'pearsonDeltaGain95':quant(gain),'mseReduction95':quant(reduction)}
 (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({n:{a:r['arms'][a]['meanPearsonDelta'] for a in ARMS} for n,r in report['sources'].items()},sort_keys=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
