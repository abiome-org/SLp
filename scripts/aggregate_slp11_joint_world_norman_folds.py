#!/usr/bin/env python3
"""Aggregate disjoint Norman held-combination folds from saved evaluations."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ROUTES={"directTwoActions":"priorOnly","autonomousAverage":"predictedAdditive","observedParentAverage":"observedParentPrior"}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def row_corr(x,y):
 x=x-x.mean(1,keepdims=True);y=y-y.mean(1,keepdims=True);d=np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1);out=np.full(len(x),np.nan);ok=d>1e-12;out[ok]=(x[ok]*y[ok]).sum(1)/d[ok];return out
def summaries(truth,values,common):
 y=truth[:,common];add=values['observedAdditive'][:,common];centered_y=(y-add)-(y-add).mean(0,keepdims=True)
 out={};per={}
 for name,p in values.items():
  p=p[:,common];mse=np.square(y-p).mean(1);centered=(p-add)-(p-add).mean(0,keepdims=True);corr=row_corr(centered,centered_y)
  out[name]={"mse":float(mse.mean()),"centeredNonadditivePearson":float(np.nanmean(corr)) if np.isfinite(corr).any() else None,"conditions":len(y),"finiteCenteredConditions":int(np.isfinite(corr).sum())};per[name]=(mse,corr)
 return out,per
def bootstrap(per,seed=731,reps=10000):
 rng=np.random.default_rng(seed);n=len(next(iter(per.values()))[0]);ix=rng.integers(0,n,(reps,n));out={}
 for route,base in ROUTES.items():
  dm=(per[base][0]-per[route][0])[ix].mean(1);dc=np.nanmean((per[route][1]-per[base][1])[ix],axis=1);q=lambda x:[float(v) for v in np.quantile(x,[.025,.5,.975])]
  out[f'{route}Vs{base}']={"replicates":reps,"seed":seed,"mseReduction95":q(dm),"centeredNonadditivePearsonGain95":q(dc)}
 return out
def main(inputs,out):
 if out.exists():raise FileExistsError(out)
 reports=[];folds=[];query=None;seen=set();alltruth=[];allvalues={};common=None
 for directory in inputs:
  report=json.loads((directory/'report.json').read_text());fold=int(report['sources']['norman']['fold'])
  with np.load(directory/'predictions.npz',allow_pickle=False) as z:
   q=z['norman_query_ids'].astype(str);rows=z['norman_rows'].astype(int);mask=z['norman_common'].astype(bool)
   if query is not None and not np.array_equal(query,q):raise ValueError('query IDs differ across folds')
   if common is not None and not np.array_equal(common,mask):raise ValueError('common query masks differ across folds')
   ids=[f'norman-row:{r}' for r in rows]
   if seen.intersection(ids):raise ValueError('held combinations overlap across folds')
   seen.update(ids);query=q;common=mask;truth=np.asarray(z['norman_truth'],np.float64);values={k:np.asarray(z['norman_'+k],np.float64) for k in set(ROUTES)|set(ROUTES.values())|{'observedAdditive'}};values['zeroResponse']=np.zeros_like(truth)
  metrics,per=summaries(truth,values,mask);folds.append({'fold':fold,'conditions':len(rows),'pairIds':ids,'rowIndices':rows.tolist(),'metrics':metrics,'comparisons':{r:{'baseline':b,'mseReduction':metrics[b]['mse']-metrics[r]['mse'],'centeredNonadditivePearsonGain':(metrics[r]['centeredNonadditivePearson']-metrics[b]['centeredNonadditivePearson']) if metrics[b]['centeredNonadditivePearson'] is not None else None} for r,b in ROUTES.items()},'source':{'report':str(directory/'report.json'),'reportSha256':sha(directory/'report.json'),'predictions':str(directory/'predictions.npz'),'predictionsSha256':sha(directory/'predictions.npz')}});alltruth.append(truth)
  for k,v in values.items():allvalues.setdefault(k,[]).append(v)
 if sorted(x['fold'] for x in folds)!=[0,1,2] or len(seen)!=59:raise ValueError(f'expected disjoint folds 0/1/2 totaling 59 pairs, got {sorted(x["fold"] for x in folds)}/{len(seen)}')
 truth=np.concatenate(alltruth);values={k:np.concatenate(v) for k,v in allvalues.items()};metrics,per=summaries(truth,values,common)
 payload={'schema':'slp.joint-world-norman-three-fold-aggregate/v2','adaptiveDevelopment':True,'independentConfirmation':False,'zeroResponseDefinition':'exact all-zero prediction in Norman control-z target units','folds':sorted(folds,key=lambda x:x['fold']),'pooled':{'conditions':59,'uniquePairIds':len(seen),'commonQueries':int(common.sum()),'metrics':metrics,'comparisons':{r:{'baseline':b,'mseReduction':metrics[b]['mse']-metrics[r]['mse'],'centeredNonadditivePearsonGain':metrics[r]['centeredNonadditivePearson']-metrics[b]['centeredNonadditivePearson']} for r,b in ROUTES.items()},'pairedBootstrap':bootstrap(per)}}
 out.mkdir(parents=True);(out/'report.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps(payload['pooled'],sort_keys=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.input,a.output)
