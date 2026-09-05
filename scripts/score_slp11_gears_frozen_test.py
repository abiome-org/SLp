#!/usr/bin/env python3
"""One-time retrospective scoring of already-frozen canonical GEARS models."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];MODELS=ROOT/"results/slp11-transition/gears-response-models-v2";DATA=ROOT/"data/derived/slp11-gears-canonical-v1";SOURCE=ROOT/"data/sources/slim-canonical-gears-v1/extracted";OUT=ROOT/"results/slp11-transition/gears-frozen-test-v1";NAMES=("replogle_k562_essential","replogle_rpe1_essential")
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);import sys;sys.modules[name]=m;s.loader.exec_module(m);return m
PREP=module("gears_test_prep",ROOT/"scripts/prepare_slp11_gears_benchmark.py");RUN=module("gears_frozen_runner",ROOT/"scripts/run_slp11_gears_response_models.py");RRR=RUN.RRR
def digest(p):
 with open(p,"rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}
def predict(z,x):
 if str(z["family"])=="slim":return (z["query_basis"]@z["weight"]@(((x-z["feature_mean"])/z["feature_scale"]).T)+z["bias"]).T
 m=RRR.ReducedRankResponse(z["feature_mean"],z["feature_scale"],z["design_mean"],z["state_projection"],z["query_loading"],z["intercept"],float(z["regularization"]));return m.predict(x)
def per_condition(y,p,control):
 yd=y-control;pd=p-control;yc=yd-yd.mean(1)[:,None];pc=pd-pd.mean(1)[:,None];d=np.linalg.norm(yc,axis=1)*np.linalg.norm(pc,axis=1);r=np.full(len(y),np.nan);ok=d>1e-12;r[ok]=np.sum(yc[ok]*pc[ok],1)/d[ok]
 return r,np.square(y-p).mean(1)
def centered(y,p,c):
 x=y-c;z=p-c;x-=x.mean(0);z-=z.mean(0);x-=x.mean(1)[:,None];z-=z.mean(1)[:,None];d=np.linalg.norm(x,axis=1)*np.linalg.norm(z,axis=1);ok=d>1e-12;return float(np.mean(np.sum(x[ok]*z[ok],1)/d[ok]))
def bootstrap(candidate_r,candidate_m,base_r,base_m,seed=731,reps=10000):
 rng=np.random.default_rng(seed);n=len(candidate_r);dr=[];dm=[]
 for _ in range(reps):
  ix=rng.integers(0,n,n);dr.append(np.nanmean(candidate_r[ix]-base_r[ix]));dm.append(np.mean(base_m[ix]-candidate_m[ix]))
 def q(x):return [float(v) for v in np.quantile(x,[.025,.5,.975])]
 return {"replicates":reps,"seed":seed,"pearsonDeltaGain95":q(dr),"mseReduction95":q(dm)}
def main(out,models=MODELS,correction=False):
 global MODELS
 MODELS=models
 if out.exists():raise FileExistsError(out)
 receipt=json.loads((MODELS/"MODELS-FROZEN-BEFORE-CANONICAL-DEVELOPMENT.json").read_text());verified={}
 for source,arms in receipt["models"].items():
  verified[source]={}
  for arm,record in arms.items():
   actual=digest(MODELS/record["path"])
   if actual!=record["sha256"]:raise ValueError(f"frozen model hash mismatch: {source}/{arm}")
   verified[source][arm]={"path":record["path"],"sha256":actual}
 if digest(MODELS/"protocol.json")!=receipt["protocolSha256"]:raise ValueError("frozen protocol mismatch")
 out.mkdir(parents=True);protocol={"schema":"slp.gears-frozen-retrospective-test-protocol/v1","models":verified,"modelFreezeReceiptSha256":digest(MODELS/"MODELS-FROZEN-BEFORE-CANONICAL-DEVELOPMENT.json"),"metrics":["mean and median per-condition Pearson of processed-X delta from train control","mean per-condition all-gene profile MSE","independently-query-centered residual Pearson"],"prediction":"frozen control plus response, clipped [0,14.99]","uncertainty":"paired condition bootstrap, 10000 replicates, seed731","selectionAfterTest":False,"featureCorrection":"direct official STRING HDF5 lookup; previously frozen hyperparameters" if correction else None,"retrospective":True,"limitation":"The underlying source biology participated in earlier SLp development; this is comparator parity, not independent prospective SOTA proof.","testOutcomesOpenedBeforeProtocol":bool(correction)};(out/"SCORING-PROTOCOL-BEFORE-TEST.json").write_text(json.dumps(protocol,indent=2,sort_keys=True)+"\n")
 report={"schema":"slp.gears-frozen-retrospective-test/v1","protocolSha256":digest(out/"SCORING-PROTOCOL-BEFORE-TEST.json"),"sources":{},"retrospective":True}
 for name in NAMES:
  sealed=json.loads((DATA/"sealed-test"/f"{name}.json").read_text());conds=np.asarray(sealed["conditions"])
  import h5py
  with h5py.File(SOURCE/name/"perturb_processed.h5ad","r") as h5:
   allcond=PREP.categorical(h5["obs"],"condition");query=PREP.categorical(h5["var"],"gene_name") if "gene_name" in h5["var"] else PREP.strings(h5["var/_index"][:]);truth,count=PREP.mean_rows(h5,allcond,conds)
  _,string,concat,present=RUN.feats(name,conds);arms={};per={}
  for arm,record in verified[name].items():
   z=load(MODELS/record["path"])
   if not np.array_equal(z["query_ids"].astype(str),query.astype(str)):raise ValueError("query roster mismatch")
   x=concat if arm=="reducedRankConcat" else string;pred=np.clip(z["control_mean"]+predict(z,x),0,14.99);r,m=per_condition(truth,pred,z["control_mean"]);per[arm]=(r,m);arms[arm]={"meanPearsonDelta":float(np.nanmean(r)),"medianPearsonDelta":float(np.nanmedian(r)),"meanProfileMse":float(np.mean(m)),"independentlyQueryCenteredResidualPearson":centered(truth,pred,z["control_mean"]),"finitePearsonConditions":int(np.isfinite(r).sum())}
  uncertainty={base:bootstrap(per["reducedRankConcat"][0],per["reducedRankConcat"][1],per[base][0],per[base][1]) for base in ("slimPublished","slimFitCv")}
  with open(out/f"{name}-per-condition.csv","w",newline="",encoding="utf-8") as f:
   w=csv.writer(f);w.writerow(["condition","cell_count"]+[v for arm in verified[name] for v in (arm+"_pearson_delta",arm+"_mse")]);
   for i,c in enumerate(conds):w.writerow([c,int(count[i])]+[v for arm in verified[name] for v in (per[arm][0][i],per[arm][1][i])])
  report["sources"][name]={"conditions":len(conds),"stringCovered":int(present.sum()),"metrics":arms,"pairedBootstrapConcatVs":uncertainty,"conditionRosterSha256":hashlib.sha256(("\n".join(conds)+"\n").encode()).hexdigest(),"perConditionSha256":digest(out/f"{name}-per-condition.csv")}
 (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report["sources"],sort_keys=True))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=OUT);p.add_argument("--models",type=Path,default=MODELS);p.add_argument("--correction",action="store_true");a=p.parse_args();main(a.output,a.models,a.correction)
