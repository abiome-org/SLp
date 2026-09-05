#!/usr/bin/env python3
"""Create an immutable joint-world corpus variant with STRING64 features."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_BASE=ROOT/'data/derived/slp11-joint-world-populations-v1'
DEFAULT_STRING=ROOT/'data/derived/slp11-string-embedding-v03'
SOURCES=('k562','rpe1','norman')

def load(path):
 with np.load(path,allow_pickle=False) as a:return {k:np.asarray(a[k]) for k in a.files}
def receipt(path):
 path=Path(path).resolve(); h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return {'path':str(path).replace('\\','/'),'bytes':path.stat().st_size,'sha256':h.hexdigest()}
def string_lookup(payload):
 ids=payload['entity_id'].astype(str); taxon=np.asarray(payload['entity_taxon']); values=np.asarray(payload['feature_values'],np.float32); present=np.asarray(payload['feature_present'])
 if len(set(ids))!=len(ids) or values.shape!=(len(ids),64) or present.shape!=(len(ids),) or np.any(taxon!=9606):raise ValueError('invalid human STRING64 entity payload')
 return {identifier:(values[i] if present[i] else np.zeros(64,np.float32),bool(present[i])) for i,identifier in enumerate(ids)}
def augment_features(base,ids,lookup):
 x=np.asarray(base,np.float32); names=np.asarray(ids).astype(str)
 if x.shape[:-1]!=names.shape or x.shape[-1]!=577:raise ValueError('feature and stable-ID axes differ')
 extra=np.zeros(names.shape+(65,),np.float32)
 for index,name in np.ndenumerate(names):
  vector,present=lookup.get(name,(None,False))
  if present: extra[index+(slice(0,64),)]=vector; extra[index+(64,)]=1.
 return np.concatenate((x,extra),axis=-1)
def row_action_ids(data):
 offsets=np.asarray(data['action_offsets'],np.int64); flat=data['action_ids'].astype(str); mask=np.asarray(data['action_mask'])
 if offsets.shape!=(len(mask)+1,) or offsets[0]!=0 or offsets[-1]!=len(flat):raise ValueError('invalid action offsets')
 result=np.full(mask.shape,'',dtype=flat.dtype)
 for row in range(len(mask)):
  active=np.flatnonzero(mask[row]); values=flat[offsets[row]:offsets[row+1]]
  if len(active)!=len(values):raise ValueError('action offsets and mask disagree')
  result[row,active]=values
 return result
def main():
 p=argparse.ArgumentParser(); p.add_argument('--base',type=Path,default=DEFAULT_BASE); p.add_argument('--string',type=Path,default=DEFAULT_STRING); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 if a.output.exists():raise FileExistsError('output must be new')
 loaded={}; augmented={}; fitting=[]
 for source in SOURCES:
  base_path=a.base/f'{source}.npz'; string_path=a.string/f'{source}-string64.npz'; data=load(base_path); lookup=string_lookup(load(string_path))
  if int(data['ncbi_taxon'])!=9606 or int(data['feature_dim'])!=577:raise ValueError(f'{source}: expected human static577 corpus')
  out=dict(data); out['action_features']=augment_features(data['action_features'],row_action_ids(data),lookup); out['action_features'][~data['action_mask']]=0
  out['action_roster_features']=augment_features(data['action_roster_features'],data['action_roster_ids'],lookup)
  out['query_features']=augment_features(data['query_features'],data['query_ids'],lookup); out['feature_dim']=np.asarray(642,np.int64)
  loaded[source]=(base_path,string_path); augmented[source]=out; fitting.append(out['action_roster_features'])
 all_fitting=np.concatenate(fitting,axis=0).astype(np.float64); mean=all_fitting.mean(0); sd=all_fitting.std(0); scale=np.where(sd>1e-5,sd,1.)
 a.output.mkdir(parents=True)
 manifest={'schema':'slp.joint-world-populations-string642/v1','feature_dim':642,'normalization':'all source fitting action rosters only','redistributionClaim':False,'sources':{}}
 for source in SOURCES:
  out=augmented[source]; out['feature_mean']=mean.astype(np.float32); out['feature_scale']=scale.astype(np.float32); out['schema']=np.asarray('slp.joint-world-population-string642/v1')
  target=a.output/f'{source}.npz'; np.savez_compressed(target,**out)
  base_path,string_path=loaded[source]; manifest['sources'][source]={'base':receipt(base_path),'string64':receipt(string_path),'output':receipt(target),'rows':len(out['targets']),'queries':len(out['query_ids'])}
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
if __name__=='__main__':main()
