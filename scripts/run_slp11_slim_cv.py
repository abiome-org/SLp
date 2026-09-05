#!/usr/bin/env python3
"""Fitting-only CV selection for standardized-feature SLIM."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/derived/slp11-omf2-response-v1"; OUT=ROOT/"results/slp11-transition/slim-cv-v1"
MODULE=ROOT/"modules/slp-1-1-slim-baseline-v1/slim_native.py"
RANKS=(10,32,64); LAMBDAS=(.01,.1,1.,10.,100.,1000.)
def loadmod():
 s=importlib.util.spec_from_file_location("slim_native_cv",MODULE); m=importlib.util.module_from_spec(s); import sys;sys.modules[s.name]=m;s.loader.exec_module(m);return m
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}
def folds(ids):
 return np.asarray([int.from_bytes(hashlib.sha256(("slp-slim-cv-v1:"+x).encode()).digest()[:8],"little")%3 for x in ids.astype(str)])
def normalize_fit(x):
 mean=x.mean(0); scale=x.std(0); scale[scale<1e-8]=1.; return mean,scale
def score_mse(y,p): return float(np.square(y-p).mean())
def molecular_score(truth,pred,anchor):
 mse=float(np.square(truth-pred).mean()); x=truth-anchor;z=pred-anchor;x-=x.mean(0);z-=z.mean(0);x-=x.mean(1)[:,None];z-=z.mean(1)[:,None]
 d=np.linalg.norm(x,axis=1)*np.linalg.norm(z,axis=1); ok=d>1e-12
 return {"geneProfileMse":mse,"independentlyQueryCenteredResidualPearson":float(np.mean(np.sum(x[ok]*z[ok],axis=1)/d[ok])),"finiteCorrelationGenes":int(ok.sum())}
def main(out):
 if out.exists(): raise FileExistsError(out)
 out.mkdir(parents=True); m=loadmod(); started=time.perf_counter(); report={"schema":"slp.slim-fitting-cv-report/v1","selection":{},"development":{},"grid":{"ranks":RANKS,"lambdaReg":LAMBDAS,"folds":3},"featureTransform":"per-feature mean/SD fitted independently inside each fitting CV fold; final transform fitted on all fitting genes","protectedTestOpened":False}
 models={}
 for src in ("k562","rpe1"):
  tr=load(DATA/f"training/{src}.npz"); f=folds(tr["action_ids"]); rows=[]
  for rank in RANKS:
   for lam in LAMBDAS:
    losses=[]
    for hold in range(3):
     a=f!=hold;b=~a;mu,sd=normalize_fit(tr["features"][a].astype(float)); model=m.fit((tr["features"][a]-mu)/sd,tr["residual_targets"][a],rank=rank,lambda_reg=lam); losses.append(score_mse(tr["residual_targets"][b],model.predict_residual((tr["features"][b]-mu)/sd)))
    rows.append({"rank":rank,"lambdaReg":lam,"foldMse":losses,"meanMse":float(np.mean(losses))})
  best=min(rows,key=lambda x:(x["meanMse"],x["rank"],x["lambdaReg"])); mu,sd=normalize_fit(tr["features"].astype(float)); model=m.fit((tr["features"]-mu)/sd,tr["residual_targets"],rank=best["rank"],lambda_reg=best["lambdaReg"]);models[src]=(model,mu,sd)
  np.savez_compressed(out/f"model-{src}.npz",schema=np.asarray("slp.slim-standardized-cv-model/v1"),query_basis=model.query_basis,weight=model.weight,bias=model.bias,feature_mean=mu,feature_scale=sd,rank=model.rank,lambda_reg=model.lambda_reg)
  report["selection"][src]={"selected":best,"candidates":rows}
 (out/"MODELS-FROZEN-BEFORE-DEVELOPMENT.json").write_text(json.dumps({"selection":{s:report["selection"][s]["selected"] for s in models},"developmentOpened":False,"protectedTestOpened":False},indent=2)+"\n")
 for src,(model,mu,sd) in models.items():
  dv=load(DATA/f"development/{src}.npz"); pred=np.maximum(dv["control_prediction"]+model.predict_residual((dv["features"]-mu)/sd),0.);report["development"][src]=molecular_score(dv["truth"],pred,dv["control_prediction"])
 report["seconds"]=time.perf_counter()-started;(out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps({"selected":{s:report["selection"][s]["selected"] for s in models},"development":report["development"]},sort_keys=True))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=OUT)
 with threadpool_limits(2):main(p.parse_args().output)
