#!/usr/bin/env python3
"""Fit canonical GEARS train/development response comparators; never load test outcomes."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, time
from pathlib import Path
import h5py
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/"data/derived/slp11-gears-canonical-v1";OUT=ROOT/"results/slp11-transition/gears-response-models-v2";STRING=ROOT/"data/tooling/slim-5a7e9ade/data/gene_string_embeddings.v0.3.h5"
FROZEN_SELECTIONS={"replogle_k562_essential":{"slimPublished":(10,.1),"slimFitCv":(64,10),"reducedRankString":(16,100),"reducedRankConcat":(32,1000)},"replogle_rpe1_essential":{"slimPublished":(10,.1),"slimFitCv":(32,10),"reducedRankString":(32,1000),"reducedRankConcat":(16,1000)}}
STATIC={"replogle_k562_essential":ROOT/"data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1/k562-essential-count-static577.npz","replogle_rpe1_essential":ROOT/"data/derived/slp11-human-rpe1-essential-count-static/ensembl116-esm8m-shared-go-v1/rpe1-essential-count-static577.npz"}
SYMBOL={"replogle_k562_essential":ROOT/"data/derived/slp11-string-embedding-v03/k562-string64.npz","replogle_rpe1_essential":ROOT/"data/derived/slp11-string-embedding-v03/rpe1-string64.npz"}
def mod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);import sys;sys.modules[name]=m;s.loader.exec_module(m);return m
SLIM=mod("gears_slim",ROOT/"modules/slp-1-1-slim-baseline-v1/slim_native.py");RRR=mod("gears_rrr",ROOT/"modules/slp-1-1-reduced-rank-response-v1/response_model.py")
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}
def digest(p):
 with open(p,"rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def action(c):
 g=[x for x in str(c).split('+') if x!='ctrl'];
 if len(g)!=1:raise ValueError(f"not a single intervention: {c}")
 return g[0]
def feats(name,conditions):
 genes=np.asarray([action(x) for x in conditions]);sp=load(SYMBOL[name]);st=load(STATIC[name])
 # The static and STRING-pack arrays currently share an axis, but join static
 # features explicitly by stable identifier so row order can never substitute
 # for identity provenance.
 spids=sp["entity_id"].astype(str);stids=st["entity_id"].astype(str)
 if len(set(spids))!=len(spids) or len(set(stids))!=len(stids):raise ValueError("duplicate stable entity ID")
 stable_row={v:i for i,v in enumerate(stids)};symbol_entity={}
 for symbol,entity in zip(sp["gene_symbol"].astype(str),spids):
  if symbol and symbol in symbol_entity and symbol_entity[symbol]!=entity:raise ValueError(f"ambiguous symbol-to-stable-ID mapping: {symbol}")
  if symbol:symbol_entity[symbol]=entity
 static=np.zeros((len(genes),577),float);static_present=np.zeros(len(genes),bool)
 for i,gene in enumerate(genes):
  entity=symbol_entity.get(gene);row=stable_row.get(entity) if entity is not None else None
  if row is not None:static[i]=st["feature_values"][row];static_present[i]=True
 # Published SLIM consumes the official HDF5 by gene symbol, independently of
 # whether that gene occurs in the SLp stable-ID/static-feature roster.
 string=np.zeros((len(genes),64),float);string_present=np.zeros(len(genes),bool)
 with h5py.File(STRING,"r") as h5:
  for i,gene in enumerate(genes):
   if gene in h5:string[i]=np.asarray(h5[gene],float);string_present[i]=True
 return genes,string,np.concatenate([static,string,string_present[:,None]],axis=1),string_present
def folds(genes):return np.asarray([int.from_bytes(hashlib.sha256(("gears-fitcv-v1:"+g).encode()).digest()[:8],"little")%3 for g in genes])
def norm(x):
 m=x.mean(0);s=x.std(0);s[s<1e-8]=1.;return m,s
def mse(y,p):return float(np.square(y-p).mean())
def metrics(y,p,control):
 yd=y-control;pd=p-control;yc=yd-yd.mean(1)[:,None];pc=pd-pd.mean(1)[:,None];den=np.linalg.norm(yc,axis=1)*np.linalg.norm(pc,axis=1);ok=den>1e-12;pear=np.full(len(y),np.nan);pear[ok]=np.sum(yc[ok]*pc[ok],1)/den[ok]
 x=yd-yd.mean(0);z=pd-pd.mean(0);x-=x.mean(1)[:,None];z-=z.mean(1)[:,None];d=np.linalg.norm(x,axis=1)*np.linalg.norm(z,axis=1);good=d>1e-12
 return {"meanPearsonDelta":float(np.nanmean(pear)),"medianPearsonDelta":float(np.nanmedian(pear)),"meanProfileMse":mse(y,p),"independentlyQueryCenteredResidualPearson":float(np.mean(np.sum(x[good]*z[good],1)/d[good])),"conditions":len(y),"finitePearsonConditions":int(ok.sum())}
def main(out,fixed_correction=False):
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);start=time.perf_counter();gtf=ROOT/"data/sources/replogle-perturbseq-gi-code/data_sharing/cellranger-GRCh38-1.2.0_only_genes.gtf";inputs={"schema":"slp.gears-feature-input-receipt/v1","slimCommit":"5a7e9ade5d0a6b6331e6dbc81181450605047bcc","officialStringH5":{"path":str(STRING.relative_to(ROOT)),"sha256":digest(STRING)},"mappingGtf":{"path":str(gtf.relative_to(ROOT)),"sha256":digest(gtf)},"sources":{name:{"symbolEntityPack":{"path":str(SYMBOL[name].relative_to(ROOT)),"sha256":digest(SYMBOL[name])},"static577":{"path":str(STATIC[name].relative_to(ROOT)),"sha256":digest(STATIC[name])}} for name in STATIC}};(out/"FEATURE-INPUT-RECEIPT.json").write_text(json.dumps(inputs,indent=2,sort_keys=True)+"\n");protocol={"datasets":["replogle_k562_essential","replogle_rpe1_essential"],"featureInputReceiptSha256":digest(out/"FEATURE-INPUT-RECEIPT.json"),"target":"canonical processed-X perturbation mean minus canonical train control mean","prediction":"control plus predicted response, clipped [0,14.99]","selection":"previously frozen v2 selections; no retuning" if fixed_correction else "deterministic 3-fold fitting-intervention CV by residual MSE","featureCorrection":"official STRING HDF5 direct symbol lookup; static577 explicit stable-ID join; concat presence bit is STRING coverage" if fixed_correction else None,"retrospective":True,"protectedTestOutcomesOpened":bool(fixed_correction)};(out/"protocol.json").write_text(json.dumps(protocol,indent=2,sort_keys=True)+"\n");report={"schema":"slp.gears-response-models/v1","protocol":protocol,"selection":{},"development":{},"coverage":{}};frozen={}
 for name in report["protocol"]["datasets"]:
  tr=load(DATA/name/"training.npz");ctrl=tr["mean_expression"][np.flatnonzero(tr["conditions"].astype(str)=="ctrl")[0]];keep=tr["conditions"].astype(str)!="ctrl";conds=tr["conditions"][keep];y=tr["mean_expression"][keep]-ctrl;genes,string,concat,present=feats(name,conds);f=folds(genes);report["selection"][name]={};frozen[name]={};report["coverage"][name]={"train":float(present.mean()),"trainCovered":int(present.sum()),"trainTotal":len(present)}
  arms=[("slimPublished",string,present,"slim",[(10,.1)]),("slimFitCv",string,present,"slim",[(r,a) for r in (10,32,64) for a in (.01,.1,1,10,100,1000)]),("reducedRankString",string,np.ones(len(y),bool),"rrr",[(r,a) for r in (16,32,64) for a in (10,100,1000)]),("reducedRankConcat",concat,np.ones(len(y),bool),"rrr",[(r,a) for r in (16,32,64) for a in (10,100,1000)])]
  for arm,x,eligible,family,grid in arms:
   if fixed_correction:grid=[FROZEN_SELECTIONS[name][arm]]
   rows=[]
   for rank,reg in grid:
    losses=[]
    for h in range(3):
     a=(f!=h)&eligible;b=(f==h)
     if family=="slim":model=SLIM.fit(x[a],y[a],rank=rank,lambda_reg=reg);pred=model.predict_residual(x[b])
     else:model=RRR.fit(x[a],y[a],rank=rank,alpha=reg);pred=model.predict(x[b])
     losses.append(mse(y[b],pred))
    rows.append({"rank":rank,"regularization":reg,"foldMse":losses,"meanMse":float(np.mean(losses))})
   best=min(rows,key=lambda q:(q["meanMse"],q["rank"],q["regularization"]));mu,sd=norm(x[eligible])
   if family=="slim":model=SLIM.fit(x[eligible],y[eligible],rank=best["rank"],lambda_reg=best["regularization"]);payload={"query_basis":model.query_basis,"weight":model.weight,"bias":model.bias,"feature_mean":np.zeros(x.shape[1]),"feature_scale":np.ones(x.shape[1])}
   else:model=RRR.fit(x,y,rank=best["rank"],alpha=best["regularization"]);payload={k:getattr(model,k) for k in ("feature_mean","feature_scale","design_mean","state_projection","query_loading","intercept")}
   path=out/f"model-{name}-{arm}.npz";np.savez_compressed(path,schema=np.asarray("slp.gears-response-model/v1"),family=np.asarray(family),arm=np.asarray(arm),rank=best["rank"],regularization=best["regularization"],query_ids=tr["query_ids"],control_mean=ctrl,**payload);report["selection"][name][arm]={"selected":best,"candidates":rows};frozen[name][arm]=path.name
 receipts={s:{a:{"path":p,"sha256":hashlib.sha256((out/p).read_bytes()).hexdigest()} for a,p in q.items()} for s,q in frozen.items()};(out/"MODELS-FROZEN-BEFORE-CANONICAL-DEVELOPMENT.json").write_text(json.dumps({"protocolSha256":hashlib.sha256((out/"protocol.json").read_bytes()).hexdigest(),"models":receipts,"developmentOpened":bool(fixed_correction),"testOutcomesOpened":bool(fixed_correction)},indent=2)+"\n")
 for name in report["protocol"]["datasets"]:
  dv=load(DATA/name/"development.npz");genes,string,concat,present=feats(name,dv["conditions"]);report["coverage"][name].update({"development":float(present.mean()),"developmentCovered":int(present.sum()),"developmentTotal":len(present)});report["development"][name]={}
  for arm,path in frozen[name].items():
   z=load(out/path);x=concat if arm=="reducedRankConcat" else string
   if str(z["family"])=="slim":pred=(z["query_basis"]@z["weight"]@(((x-z["feature_mean"])/z["feature_scale"]).T)+z["bias"]).T
   else:model=RRR.ReducedRankResponse(z["feature_mean"],z["feature_scale"],z["design_mean"],z["state_projection"],z["query_loading"],z["intercept"],float(z["regularization"]));pred=model.predict(x)
   control=z["control_mean"];report["development"][name][arm]=metrics(dv["mean_expression"],np.clip(control+pred,0,14.99),control)
 report["seconds"]=time.perf_counter()-start;(out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps({"selection":{s:{a:r["selected"] for a,r in q.items()} for s,q in report["selection"].items()},"development":report["development"],"coverage":report["coverage"]},sort_keys=True))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=OUT);p.add_argument("--fixed-correction",action="store_true");a=p.parse_args()
 with threadpool_limits(2):main(a.output,a.fixed_correction)
