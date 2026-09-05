#!/usr/bin/env python3
"""Prepare canonical GEARS train/validation pseudobulk; never read test X rows."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"data/sources/slim-canonical-gears-v1/extracted"
OUT=ROOT/"data/derived/slp11-gears-canonical-v1"
NAMES=("replogle_k562_essential","replogle_rpe1_essential","norman")

def strings(x): return np.asarray([v.decode() if isinstance(v,bytes) else str(v) for v in x])
def categorical(group,key):
    node=group[key]
    if isinstance(node,h5py.Group): return strings(node["categories"][:])[node["codes"][:]]
    ref=node.attrs.get("categories")
    if ref is not None: return strings(group.file[ref][:])[node[:]]
    return strings(node[:])
def unique_order(x):
    seen=set();return [v for v in x if not (v in seen or seen.add(v))]
def genes(perts):return np.unique([g for p in np.unique(perts) for g in p.split('+') if g!='ctrl'])
def perts_from_genes(gs,perts,kind):
    candidates=[p for p in perts if (('ctrl' in p) if kind=='single' else ('ctrl' not in p)) and p!='ctrl']; wanted=set(gs)
    return [p for p in candidates if wanted.intersection(p.split('+'))]
def simulation(perts,fraction,combo_fraction,seed=1):
    allgenes=genes(perts);np.random.seed(seed);train_genes=np.random.choice(allgenes,int(len(allgenes)*fraction),replace=False);ood=np.setdiff1d(allgenes,train_genes)
    singles=perts_from_genes(train_genes,perts,'single'); combos=perts_from_genes(train_genes,perts,'combo'); seen1=[p for p in combos if sum(g in train_genes for g in p.split('+'))==1];remaining=np.setdiff1d(combos,seen1)
    np.random.seed(seed); combo_train=np.random.choice(remaining,int(len(remaining)*combo_fraction),replace=False);seen2=np.setdiff1d(remaining,combo_train).tolist();unseen=perts_from_genes(ood,perts,'single');oodcombo=perts_from_genes(ood,perts,'combo');seen0=[p for p in oodcombo if sum(g in train_genes for g in p.split('+'))==0]
    train=singles+list(combo_train);test=seen1+seen2+unseen+seen0
    assert len(train)+len(test)==len(perts)
    return train,test,{"combo_seen0":seen0,"combo_seen1":seen1,"combo_seen2":seen2,"unseen_single":unseen}
def split(conditions):
    perts=[p for p in unique_order(conditions) if p!='ctrl'];train,test,testgroups=simulation(perts,.75,.75);train,val,valg=simulation(train,.9,.9);train.append('ctrl');return train,val,test,testgroups,valg
def mean_rows(h5,conditions,allowed):
    x=h5["X"]; shape=tuple(x.attrs["shape"]); names=list(allowed);lookup={x:i for i,x in enumerate(names)}; sums=np.zeros((len(names),shape[1]),np.float64);counts=np.zeros(len(names),np.int64);indptr=x["indptr"]
    groups=np.asarray([lookup.get(v,-1) for v in conditions]);allowed_rows=groups>=0; edges=np.diff(np.r_[False,allowed_rows,False].astype(np.int8));starts=np.flatnonzero(edges==1);stops=np.flatnonzero(edges==-1)
    for lo,hi in zip(starts,stops):
        local=groups[lo:hi];counts+=np.bincount(local,minlength=len(names));a=int(indptr[lo]);b=int(indptr[hi]);ptr=indptr[lo:hi+1]-a;row=np.repeat(local,np.diff(ptr));idx=x["indices"][a:b];data=x["data"][a:b];flat=row*shape[1]+idx;sums+=np.bincount(flat,weights=data,minlength=sums.size).reshape(sums.shape)
    if np.any(counts==0):raise ValueError("empty requested condition")
    return sums/counts[:,None],counts
def main(out):
    out.mkdir(parents=True,exist_ok=False);report={"schema":"slp.gears-canonical-preparation/v1","gearsRevision":"df09d7ae34e90f5ef25afa389daf7c5c589e710d","split":"simulation seed1 0.75 then train simulation seed1 0.9","testOutcomesRead":False,"sources":{}}
    (out/"sealed-test").mkdir()
    for name in NAMES:
        path=SOURCE/name/"perturb_processed.h5ad"
        with h5py.File(path,"r") as h5:
            cond=categorical(h5["obs"],"condition");query=categorical(h5["var"],"gene_name") if "gene_name" in h5["var"] else strings(h5["var/_index"][:]);train,val,test,tg,vg=split(cond); allowed=train+list(val);means,counts=mean_rows(h5,cond,allowed)
        base=out/name;base.mkdir();ntrain=len(train)
        np.savez_compressed(base/"training.npz",schema=np.asarray("slp.gears-canonical-pseudobulk/v1"),conditions=np.asarray(train),query_ids=query,mean_expression=means[:ntrain],cell_count=counts[:ntrain])
        np.savez_compressed(base/"development.npz",schema=np.asarray("slp.gears-canonical-pseudobulk/v1"),conditions=np.asarray(val),query_ids=query,mean_expression=means[ntrain:],cell_count=counts[ntrain:])
        sealed={"conditions":test,"subgroups":tg,"outcomesMaterialized":False};(out/"sealed-test"/f"{name}.json").write_text(json.dumps(sealed,indent=2,sort_keys=True)+"\n")
        report["sources"][name]={"cells":len(cond),"queries":len(query),"trainConditions":len(train),"developmentConditions":len(val),"sealedTestConditions":len(test),"validationSubgroups":{k:len(v) for k,v in vg.items()},"testSubgroups":{k:len(v) for k,v in tg.items()}}
    (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report["sources"],sort_keys=True))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=OUT);main(p.parse_args().output)
