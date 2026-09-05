#!/usr/bin/env python3
"""Prepare fitting and development snapshots for the OMF2 rank-response experiment."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CONTEXTS={
 'k562':{
  'moments':'data/derived/slp11-human-k562-essential-fitting-action-moments-v1/fitting-action-moments.npz',
  'static':'data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1/k562-essential-count-static577.npz',
  'baseline':'results/slp11-transition/k562-essential-count-anchored-static-ridge-seed731-v1/model.npz',
  'metadata':'results/slp11-transition/human-essential-count-shared-context-seed731-v1/development-forecasts-k562.npz',
  'truth':'results/slp11-transition/human-essential-count-shared-context-development-evaluation-v2/development-truth-k562.npz'},
 'rpe1':{
  'moments':'data/derived/slp11-human-rpe1-essential-raw-cells-v1/fitting-action-moments.npz',
  'static':'data/derived/slp11-human-rpe1-essential-count-static/ensembl116-esm8m-shared-go-v1/rpe1-essential-count-static577.npz',
  'baseline':'results/slp11-transition/rpe1-essential-count-anchored-static-ridge-seed731-v1/model.npz',
  'metadata':'results/slp11-transition/human-essential-count-shared-context-seed731-v1/development-forecasts-rpe1.npz',
  'truth':'results/slp11-transition/human-essential-count-shared-context-development-evaluation-v2/development-truth-rpe1.npz'}}

def load(path):
 with np.load(ROOT/path,allow_pickle=False) as a:return {k:np.asarray(a[k]) for k in a.files}
def file_receipt(path):
 path=Path(path).resolve(); digest=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
 try: display=path.relative_to(ROOT.resolve())
 except ValueError: display=path
 return {'path':str(display).replace('\\','/'),'bytes':path.stat().st_size,'sha256':digest.hexdigest()}
def validate_partition(training_ids,development_ids,training_queries,development_queries):
 train=np.asarray(training_ids).astype(str); dev=np.asarray(development_ids).astype(str)
 tq=np.asarray(training_queries).astype(str); dq=np.asarray(development_queries).astype(str)
 if train.ndim!=1 or dev.ndim!=1 or len(set(train))!=len(train) or len(set(dev))!=len(dev): raise ValueError('intervention IDs must be unique vectors')
 overlap=set(train).intersection(dev)
 if overlap: raise ValueError(f'training/development intervention overlap: {sorted(overlap)[:3]}')
 if tq.ndim!=1 or dq.ndim!=1 or not np.array_equal(tq,dq): raise ValueError('training/development query axes differ')
def features(static,ids):
 lookup={x:i for i,x in enumerate(static['entity_id'].astype(str))}
 return np.asarray(static['feature_values'][[lookup[x] for x in ids.astype(str)]],np.float32)
def anchor(rate,weights):
 w=np.asarray(weights,np.float64); return np.log1p((w/w.sum(1)[:,None])@np.asarray(rate,np.float64))
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=ROOT/'data/derived/slp11-omf2-response-v1'); a=p.parse_args()
 train=a.output/'training'; dev=a.output/'development'; train.mkdir(parents=True,exist_ok=True); dev.mkdir(parents=True,exist_ok=True)
 train_manifest={'schema':'slp.omf2-response-training-snapshot/v1','protectedTestOpened':False,'contexts':{}}
 dev_manifest={'schema':'slp.omf2-response-development-snapshot/v1','protectedTestOpened':False,'contexts':{}}
 for source,paths in CONTEXTS.items():
  moments,static,base,meta,truth=map(load,(paths['moments'],paths['static'],paths['baseline'],paths['metadata'],paths['truth']))
  if not np.array_equal(moments['query_ids'].astype(str),base['query_ids'].astype(str)):raise ValueError('fitting query drift')
  for key in ('gene_ids','query_ids','cell_count','gem_cell_count'):
   if not np.array_equal(meta[key],truth[key]):raise ValueError(f'{source} development identity drift: {key}')
  validate_partition(moments['action_ids'],meta['gene_ids'],moments['query_ids'],meta['query_ids'])
  training_features=features(static,moments['action_ids'])
  development_features=features(static,meta['gene_ids'])
  y=np.log1p(moments['cp10k_sum']/moments['cell_count'][:,None])-anchor(base['basal_rate'],moments['gem_cell_count'])
  np.savez_compressed(train/f'{source}.npz',schema=np.asarray('slp.omf2-response-training/v1'),source_id=np.asarray(source),action_ids=moments['action_ids'],query_ids=moments['query_ids'],features=training_features,residual_targets=y)
  np.savez_compressed(dev/f'{source}.npz',schema=np.asarray('slp.omf2-response-development/v1'),source_id=np.asarray(source),gene_ids=meta['gene_ids'],query_ids=meta['query_ids'],features=development_features,control_prediction=meta['control_prediction'],static_ridge_prediction=meta['static_ridge_prediction'],truth=truth['truth_log1p_mean_cp10k'])
  sources={name:file_receipt(ROOT/path) for name,path in paths.items()}
  train_manifest['contexts'][source]={'fittingGenes':len(moments['action_ids']),'queries':len(moments['query_ids']),'sources':{k:sources[k] for k in ('moments','static','baseline')},'panel':file_receipt(train/f'{source}.npz')}
  dev_manifest['contexts'][source]={'developmentGenes':len(meta['gene_ids']),'queries':len(meta['query_ids']),'sources':{k:sources[k] for k in ('static','metadata','truth')},'panel':file_receipt(dev/f'{source}.npz')}
 (train/'manifest.json').write_text(json.dumps(train_manifest,indent=2,sort_keys=True)+'\n')
 (dev/'manifest.json').write_text(json.dumps(dev_manifest,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
