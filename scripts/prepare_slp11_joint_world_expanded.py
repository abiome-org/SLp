#!/usr/bin/env python3
"""Build fitting-only GWPS/HepG2 shards beside the STRING642 joint corpus."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,re,shutil,sys
from pathlib import Path
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1]; FOUR=ROOT/'data/derived/slp11-human-four-context-v2/development.npz'; STATIC=ROOT/'data/derived/slp11-human-shared-static/ensembl116-source3-esm8m-shared-go-complete-v2/human-static-esm8m-shared-go-mf-cc-features.npz'; EMB=ROOT/'data/tooling/slim-5a7e9ade/data/gene_string_embeddings.v0.3.h5'; GTF=ROOT/'data/sources/replogle-perturbseq-gi-code/data_sharing/cellranger-GRCh38-1.2.0_only_genes.gtf'; BASE=ROOT/'data/derived/slp11-joint-world-populations-string-v1'
DEVELOPMENT=ROOT/'data/derived/slp11-joint-world-development-string-v1'
def load(p):
 with np.load(p,allow_pickle=False) as a:return{k:np.asarray(a[k]) for k in a.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def names():
 out={};pattern=re.compile(r'gene_id "([^"]+)".*gene_name "([^"]+)"')
 with GTF.open(encoding='utf-8') as f:
  for line in f:
   m=pattern.search(line)
   if m:out.setdefault(m.group(1).split('.')[0],m.group(2))
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'data/derived/slp11-joint-world-expanded-string-v1');a=p.parse_args()
 if a.output.exists():raise FileExistsError('output must be new')
 four=load(FOUR);static=load(STATIC); lookup={x:i for i,x in enumerate(static['entity_id'].astype(str))}; symbols=names(); shards={}
 with h5py.File(EMB,'r') as h5:
  def features(ids):
   ids=np.asarray(ids).astype(str);base=np.asarray(static['feature_values'][[lookup[x] for x in ids]],np.float32);extra=np.zeros((len(ids),65),np.float32)
   for i,gene in enumerate(ids):
    symbol=symbols.get(gene,'')
    if symbol in h5:extra[i,:64]=h5[symbol][:];extra[i,64]=1
   return np.concatenate((base,extra),1)
  for label,index in (('gwps',2),('hepg2',3)):
   rows=np.flatnonzero((four['context_index']==index)&(four['split_role']=='train'));ids=four['action_ids'][rows].astype(str);qids=four['query_ids'].astype(str);af=features(ids);qf=features(qids);mask=np.ones((len(rows),1),bool);basal=np.zeros_like(four['targets'][rows])
   validation_ids=set(four['action_ids'][four['split_role']=='validation'].astype(str))
   if set(ids).intersection(validation_ids):raise ValueError(f'{label}: held intervention leaked into fitting shard')
   assay_id=2 if label=='gwps' else 3
   shards[label]={'schema':np.asarray('slp.joint-world-expanded-population/v1'),'source_id':np.asarray(label),'ncbi_taxon':np.asarray(9606),'mode_id':np.asarray(0),'assay_id':np.asarray(assay_id),'intervention_mode':np.asarray('crispri'),'assay':np.asarray(str(four['target_value_space_by_context'][index])),'target_units':np.asarray(str(four['target_value_space_by_context'][index])),'context_value_space':four['context_value_space'],'control_context_values':four['context_basal_expression'][index],'control_context_observed':four['context_basal_observed'][index],'action_ids':ids,'action_features':af[:,None,:],'action_mask':mask,'action_roster_ids':np.unique(ids),'query_ids':qids,'query_features':qf,'targets':four['targets'][rows],'basal':basal,'observed':four['observed'][rows],'record_ids':four['record_ids'][rows],'feature_dim':np.asarray(642),'source_rows':rows}
   roster=shards[label]['action_roster_ids'];shards[label]['action_roster_features']=features(roster)
 existing={s:load(BASE/f'{s}.npz') for s in ('k562','rpe1','norman')};fit=[x['action_roster_features'] for x in existing.values()]+[x['action_roster_features'] for x in shards.values()];matrix=np.concatenate(fit).astype(np.float64);mean=matrix.mean(0);sd=matrix.std(0);scale=np.where(sd>1e-5,sd,1.)
 development_ids=set()
 development_receipts={}
 for context in ('k562','rpe1'):
  path=DEVELOPMENT/f'{context}.npz'; payload=load(path); development_ids.update(payload['gene_ids'].astype(str));development_receipts[context]={'sha256':sha(path),'genes':len(payload['gene_ids'])}
 fitting_ids=set().union(*(set(x['action_ids'].astype(str)) for x in shards.values()),*(set(x['action_roster_ids'].astype(str)) for x in existing.values()))
 overlap=fitting_ids.intersection(development_ids)
 if overlap:raise ValueError(f'development intervention leaked into expanded fitting corpus: {sorted(overlap)[:3]}')
 a.output.mkdir(parents=True);roster=a.output/'excluded-development-action-ids.txt';roster.write_text(''.join(f'{x}\n' for x in sorted(development_ids)),encoding='ascii')
 manifest={'schema':'slp.joint-world-expanded-string642/v1','protectedTestOpened':False,'feature_dim':642,'inputs':{'fourContext':sha(FOUR),'static577':sha(STATIC),'string64':sha(EMB)},'developmentExclusion':{'roster':roster.name,'sha256':sha(roster),'genes':len(development_ids),'overlap':0,'sources':development_receipts},'sources':{}}
 for name,data in {**existing,**shards}.items():
  data['feature_mean']=mean.astype(np.float32);data['feature_scale']=scale.astype(np.float32);path=a.output/f'{name}.npz';np.savez_compressed(path,**data);manifest['sources'][name]={'sha256':sha(path),'rows':len(data['targets']),'queries':len(data['query_ids']),'units':str(data.get('target_units',data.get('assay','')))}
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
