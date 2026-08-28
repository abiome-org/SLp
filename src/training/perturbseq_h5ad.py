import argparse, hashlib, json
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse


def values(x):
    if isinstance(x,h5py.Group):
        cat=np.asarray(x["categories"]).astype(str); code=np.asarray(x["codes"]); return cat[code]
    return np.asarray(x).astype(str)


def endpoints(condition,cardinality,delimiter):
    if cardinality==0:return "",""
    parts=[x.upper() for x in condition.split(delimiter) if x.upper()!="GFP"]
    return parts[0],parts[1] if cardinality==2 else ""


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def build(source,output,source_id,context,mode,delimiter,raw_counts=False,duration_hours=np.nan,dose=None):
    source=Path(source); output=Path(output); h=sha(source)
    with h5py.File(source,"r") as f:
        obs=f["obs"]; condition=values(obs["perturbation"]); card=np.asarray(obs["nperts"],"int8"); barcode=values(obs[obs.attrs["_index"]]); keep=card<=2; condition=condition[keep]; card=card[keep]; barcode=barcode[keep]
        bins=np.asarray([int(hashlib.sha256(x.lower().encode()).hexdigest(),16)%4 for x in barcode],"int8"); keys=sorted(set(zip(condition,bins))); key_id={x:i for i,x in enumerate(keys)}; group=np.asarray([key_id[x] for x in zip(condition,bins)]); counts=np.bincount(group,minlength=len(keys)); shape=tuple(f["X"].attrs["shape"]); sums=np.zeros((len(keys),shape[1]),"float64"); indptr=np.asarray(f["X/indptr"]); full_group=np.full(shape[0],-1); full_group[keep]=group
        for lo in range(0,shape[0],1024):
            hi=min(shape[0],lo+1024); start,end=indptr[lo],indptr[hi]; x=sparse.csr_matrix((np.asarray(f["X/data"][start:end]),np.asarray(f["X/indices"][start:end]),indptr[lo:hi+1]-start),shape=(hi-lo,shape[1]),dtype="float32"); take=keep[lo:hi]; x=x[take]
            if raw_counts:
                scale=1e4/np.maximum(1,np.asarray(x.sum(1)).ravel()); x=x.multiply(scale[:,None]).tocsr(); x.data=np.log1p(x.data)
            g=full_group[lo:hi][take]; assign=sparse.csr_matrix((np.ones(len(g)),(g,np.arange(len(g)))),shape=(len(keys),len(g))); sums+=(assign@x).toarray()
        future=(sums/np.maximum(1,counts[:,None])).astype("float32"); feature=values(f["var"][f["var"].attrs["_index"]]); control=future[[i for i,(c,_) in enumerate(keys) if c.lower()=="control"]].mean(0) if any(c.lower()=="control" for c,_ in keys) else future.mean(0)
    a,b=zip(*(endpoints(c,int(n),delimiter) for (c,_),n in zip(keys,[card[np.flatnonzero(group==i)[0]] for i in range(len(keys))]))); cards=np.asarray([card[np.flatnonzero(group==i)[0]] for i in range(len(keys))],"int8"); valid=counts>=2
    arrays={"source_id":source_id,"context_id":context,"mode":mode,"duration_hours":np.float32(duration_hours),"condition":np.asarray([x[0] for x in keys])[valid],"endpoint_a":np.asarray(a)[valid],"endpoint_b":np.asarray(b)[valid],"cardinality":cards[valid],"role":np.full(valid.sum(),"train"),"pseudoreplicate":np.asarray([x[1] for x in keys],"int16")[valid],"cell_count":counts[valid].astype("int32"),"feature_name":feature,"future_state":future[valid],"control_mean":control}
    if dose is not None:arrays["dose"]=np.asarray(dose)
    output.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(output,**arrays); audit={"schema":"sl-predict-h5ad-pseudobulk-v1","source_sha256":h,"source_id":source_id,"context_id":context,"mode":mode,"duration_hours":None if not np.isfinite(duration_hours) else duration_hours,"dose":dose,"cells":int(keep.sum()),"features":len(feature),"conditions":len(set(condition)),"pseudobulks":int(valid.sum()),"cardinalities":{str(n):int((cards[valid]==n).sum()) for n in np.unique(cards[valid])},"normalization":"per-cell log1p(CP10K)" if raw_counts else "source log-expression values","sl_labels_used":False}; output.with_suffix(output.suffix+".json").write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("source"); p.add_argument("output"); p.add_argument("--source-id",required=True); p.add_argument("--context",required=True); p.add_argument("--mode",required=True); p.add_argument("--delimiter",required=True); p.add_argument("--raw-counts",action="store_true"); p.add_argument("--duration-hours",type=float,default=np.nan); p.add_argument("--dose"); a=p.parse_args(); build(a.source,a.output,a.source_id,a.context,a.mode,a.delimiter,a.raw_counts,a.duration_hours,a.dose)
