#!/usr/bin/env python3
"""Compare SLIM and reduced-rank response across fitting-only feature choices."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data/derived/slp11-omf2-response-v1"; STRING=ROOT/"data/derived/slp11-string-embedding-v03"; OUT=ROOT/"results/slp11-transition/feature-comparators-v1"
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);import sys;sys.modules[name]=m;s.loader.exec_module(m);return m
SLIM=module("slim_feature_compare",ROOT/"modules/slp-1-1-slim-baseline-v1/slim_native.py")
RRR=module("rrr_feature_compare",ROOT/"modules/slp-1-1-reduced-rank-response-v1/response_model.py")
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}
def fold(ids):return np.asarray([int.from_bytes(hashlib.sha256(("slp-feature-cv-v1:"+x).encode()).digest()[:8],"little")%3 for x in ids.astype(str)])
def standardizer(x):
 mu=x.mean(0);sd=x.std(0);sd[sd<1e-8]=1.;return mu,sd
def features(src,ids,static):
 p=load(STRING/f"{src}-string64.npz"); lookup={x:i for i,x in enumerate(p["entity_id"].astype(str))}; ix=np.asarray([lookup[x] for x in ids.astype(str)]); string=p["feature_values"][ix].astype(float);present=p["feature_present"][ix,None].astype(float)
 return {"string64":string,"static577_string64_present":np.concatenate([static.astype(float),string,present],axis=1)}
def metric(y,p):return float(np.square(y-p).mean())
def score(truth,pred,anchor):
 x=truth-anchor;z=pred-anchor;x-=x.mean(0);z-=z.mean(0);x-=x.mean(1)[:,None];z-=z.mean(1)[:,None];d=np.linalg.norm(x,axis=1)*np.linalg.norm(z,axis=1);ok=d>1e-12
 return {"geneProfileMse":metric(truth,pred),"independentlyQueryCenteredResidualPearson":float(np.mean(np.sum(x[ok]*z[ok],1)/d[ok])),"finiteCorrelationGenes":int(ok.sum())}
def main(out):
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);start=time.perf_counter();report={"schema":"slp.feature-comparators/v1","protocol":{"selection":"three-fold intervention-identity fitting-only CV by residual MSE","slimGrid":{"rank":[10,32,64],"lambda":[.01,.1,1,10,100,1000]},"reducedRankGrid":{"rank":[16,32,64],"alpha":[10,100,1000]},"developmentOpenedAfterFreeze":True,"protectedTestOpened":False},"selection":{},"development":{}}; frozen={}
 for src in ("k562","rpe1"):
  tr=load(DATA/f"training/{src}.npz");fs=features(src,tr["action_ids"],tr["features"]);f=fold(tr["action_ids"]);report["selection"][src]={};frozen[src]={}
  for fname,x in fs.items():
   candidates=[]
   for family,grid in (("slim",[(r,a) for r in (10,32,64) for a in (.01,.1,1,10,100,1000)]),("reducedRank",[(r,a) for r in (16,32,64) for a in (10,100,1000)])):
    rows=[]
    for rank,reg in grid:
     losses=[]
     for h in range(3):
      a=f!=h;b=~a
      if family=="slim":mu,sd=standardizer(x[a]);model=SLIM.fit((x[a]-mu)/sd,tr["residual_targets"][a],rank=rank,lambda_reg=reg);pred=model.predict_residual((x[b]-mu)/sd)
      else:model=RRR.fit(x[a],tr["residual_targets"][a],rank=rank,alpha=reg);pred=model.predict(x[b])
      losses.append(metric(tr["residual_targets"][b],pred))
     rows.append({"rank":rank,"regularization":reg,"foldMse":losses,"meanMse":float(np.mean(losses))})
    best=min(rows,key=lambda q:(q["meanMse"],q["rank"],q["regularization"]));candidates.append((best,family,rows))
    if family=="slim":mu,sd=standardizer(x);model=SLIM.fit((x-mu)/sd,tr["residual_targets"],rank=best["rank"],lambda_reg=best["regularization"]); payload={"query_basis":model.query_basis,"weight":model.weight,"bias":model.bias,"feature_mean":mu,"feature_scale":sd}
    else:model=RRR.fit(x,tr["residual_targets"],rank=best["rank"],alpha=best["regularization"]);payload={k:getattr(model,k) for k in ("feature_mean","feature_scale","design_mean","state_projection","query_loading","intercept")}
    path=out/f"model-{src}-{fname}-{family}.npz";np.savez_compressed(path,schema=np.asarray("slp.feature-comparator-model/v1"),family=np.asarray(family),featureSet=np.asarray(fname),rank=best["rank"],regularization=best["regularization"],**payload);frozen[src][f"{fname}:{family}"]={"path":path.name,"selected":best}
    report["selection"][src][f"{fname}:{family}"]={"selected":best,"candidates":rows}
 (out/"MODELS-FROZEN-BEFORE-DEVELOPMENT.json").write_text(json.dumps({"models":frozen,"developmentOpened":False,"protectedTestOpened":False},indent=2)+"\n")
 for src in ("k562","rpe1"):
  dv=load(DATA/f"development/{src}.npz");fs=features(src,dv["gene_ids"],dv["features"]);report["development"][src]={}
  for key,rec in frozen[src].items():
   fname,family=key.split(":");z=load(out/rec["path"]);x=fs[fname]
   if family=="slim":pred=(z["query_basis"]@z["weight"]@(((x-z["feature_mean"])/z["feature_scale"]).T)+z["bias"]).T
   else:
    model=RRR.ReducedRankResponse(feature_mean=z["feature_mean"],feature_scale=z["feature_scale"],design_mean=z["design_mean"],state_projection=z["state_projection"],query_loading=z["query_loading"],intercept=z["intercept"],alpha=float(z["regularization"]));pred=model.predict(x)
   report["development"][src][key]=score(dv["truth"],np.maximum(dv["control_prediction"]+pred,0),dv["control_prediction"])
 report["seconds"]=time.perf_counter()-start;(out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps({"selection":{s:{k:v["selected"] for k,v in q.items()} for s,q in report["selection"].items()},"development":report["development"]},sort_keys=True))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=OUT)
 with threadpool_limits(2):main(p.parse_args().output)
