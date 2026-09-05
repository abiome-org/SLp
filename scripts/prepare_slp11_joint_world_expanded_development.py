#!/usr/bin/env python3
"""Prepare held-gene GWPS/HepG2 development shards in native value spaces."""
from __future__ import annotations
import argparse,hashlib,json,re,shutil
from pathlib import Path
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1];FOUR=ROOT/'data/derived/slp11-human-four-context-v2/development.npz';STATIC=ROOT/'data/derived/slp11-human-shared-static/ensembl116-source3-esm8m-shared-go-complete-v2/human-static-esm8m-shared-go-mf-cc-features.npz';EMB=ROOT/'data/tooling/slim-5a7e9ade/data/gene_string_embeddings.v0.3.h5';GTF=ROOT/'data/sources/replogle-perturbseq-gi-code/data_sharing/cellranger-GRCh38-1.2.0_only_genes.gtf';TRAIN=ROOT/'data/derived/slp11-joint-world-expanded-omf-fold0-v1';ORIGINAL=ROOT/'data/derived/slp11-joint-world-development-string-v1'
def load(p):
 with np.load(p,allow_pickle=False) as a:return{k:np.asarray(a[k]) for k in a.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def symbol_map(path):
 out={};pattern=re.compile(r'gene_id "([^"]+)".*gene_name "([^"]+)"')
 with Path(path).open(encoding='utf-8') as f:
  for line in f:
   match=pattern.search(line)
   if match:out.setdefault(match.group(1).split('.')[0],match.group(2))
 return out
def held_rows(action_ids,roles,context_index,index,training_ids):
 rows=np.flatnonzero((context_index==index)&(roles=='validation')); keep=np.asarray([x not in training_ids for x in action_ids[rows].astype(str)])
 return rows,rows[keep]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'data/derived/slp11-joint-world-expanded-development-string-v1');a=p.parse_args()
 if a.output.exists():raise FileExistsError('output must be new')
 four=load(FOUR);static=load(STATIC);lookup={x:i for i,x in enumerate(static['entity_id'].astype(str))};symbols=symbol_map(GTF);training=set()
 for source in ('k562','rpe1','norman','gwps','hepg2'):training.update(load(TRAIN/f'{source}.npz')['action_ids'].astype(str))
 a.output.mkdir(parents=True);records={}
 for source in ('k562','rpe1','norman'):shutil.copyfile(ORIGINAL/f'{source}.npz',a.output/f'{source}.npz');records[source]={'copiedByteIdentical':True,'sha256':sha(a.output/f'{source}.npz')}
 with h5py.File(EMB,'r') as h5:
  def features(ids):
   ids=np.asarray(ids).astype(str);base=np.asarray(static['feature_values'][[lookup[x] for x in ids]],np.float32);extra=np.zeros((len(ids),65),np.float32)
   for row,gene in enumerate(ids):
    symbol=symbols.get(gene,'')
    if symbol in h5:extra[row,:64]=h5[symbol][:];extra[row,64]=1
   return np.concatenate((base,extra),1)
  for source,index,assay in (('gwps',2,2),('hepg2',3,3)):
   selected,kept=held_rows(four['action_ids'],four['split_role'],four['context_index'],index,training);ids=four['action_ids'][kept].astype(str);qids=four['query_ids'].astype(str);target=four['targets'][kept];basal=np.zeros_like(target)
   out={'schema':np.asarray('slp.joint-world-expanded-development/v1'),'source_id':np.asarray(source),'ncbi_taxon':np.asarray(9606),'mode_id':np.asarray(0),'assay_id':np.asarray(assay),'target_units':np.asarray(str(four['target_value_space_by_context'][index])),'gene_ids':ids,'features':features(ids),'query_ids':qids,'query_features':features(qids),'truth':target,'targets':target,'basal':basal,'control_prediction':basal,'observed':four['observed'][kept],'record_ids':four['record_ids'][kept],'source_rows':kept,'control_context_values':four['context_basal_expression'][index],'control_context_mask':four['context_basal_observed'][index]}
   path=a.output/f'{source}.npz';np.savez_compressed(path,**out);records[source]={'validationRowsBeforeExclusion':len(selected),'rows':len(kept),'uniqueGenes':len(set(ids)),'duplicatePopulationViews':len(kept)-len(set(ids)),'lostRowsTrainingOverlap':len(selected)-len(kept),'queries':len(qids),'fullyObservedQueries':int(out['observed'].all(0).sum()),'sha256':sha(path)}
 manifest={'schema':'slp.joint-world-expanded-development-string642/v1','protectedTestOpened':False,'trainingActionGenes':len(training),'inputs':{'fourContext':sha(FOUR),'static577':sha(STATIC),'string64':sha(EMB),'trainingManifest':sha(TRAIN/'manifest.json')},'sources':records}
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
