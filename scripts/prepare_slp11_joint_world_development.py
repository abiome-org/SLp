#!/usr/bin/env python3
"""Prepare immutable STRING642 development inputs for joint-world evaluation."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, shutil, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RESPONSE=ROOT/'data/derived/slp11-omf2-response-v1/development'
STRING=ROOT/'data/derived/slp11-string-embedding-v03'
NORMAN=ROOT/'data/derived/slp11-joint-world-populations-string-v1/norman.npz'
AUGMENTER=ROOT/'scripts/augment_slp11_joint_world_features.py'

def load(path):
 with np.load(path,allow_pickle=False) as a:return {k:np.asarray(a[k]) for k in a.files}
def receipt(path):
 path=Path(path); h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return {'path':str(path.resolve()).replace('\\','/'),'bytes':path.stat().st_size,'sha256':h.hexdigest()}
def helper():
 spec=importlib.util.spec_from_file_location('slp11_string_augmenter',AUGMENTER); module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'data/derived/slp11-joint-world-development-string-v1');a=p.parse_args()
 if a.output.exists():raise FileExistsError('output must be new')
 aug=helper(); staged={}; records={}
 for source in ('k562','rpe1'):
  original=load(RESPONSE/f'{source}.npz'); strings=load(STRING/f'{source}-string64.npz'); lookup=aug.string_lookup(strings)
  out=dict(original);out['features']=aug.augment_features(original['features'],original['gene_ids'],lookup)
  if out['features'].shape!=(len(original['gene_ids']),642):raise ValueError(f'{source}: invalid augmented feature shape')
  for key,value in original.items():
   if key!='features' and not np.array_equal(value,out[key]):raise ValueError(f'{source}: changed development array {key}')
  staged[source]=out; records[source]={'base':receipt(RESPONSE/f'{source}.npz'),'string64':receipt(STRING/f'{source}-string64.npz')}
 a.output.mkdir(parents=True)
 for source,out in staged.items():
  target=a.output/f'{source}.npz';np.savez_compressed(target,**out);records[source]['output']=receipt(target)
 norman_target=a.output/'norman.npz';shutil.copyfile(NORMAN,norman_target);records['norman']={'base':receipt(NORMAN),'output':receipt(norman_target),'byteIdentical':receipt(NORMAN)['sha256']==receipt(norman_target)['sha256']}
 manifest={'schema':'slp.joint-world-development-string642/v1','feature_dim':642,'protectedTestOpened':False,'sources':records}
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
if __name__=='__main__':main()
