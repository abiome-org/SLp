import argparse, csv, hashlib, json, re, sys
from pathlib import Path

import numpy as np


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(8<<20),b""):h.update(block)
    return h.hexdigest()


def pair_table(orientations,meta):
    import pandas as pd
    d=pd.read_parquet(orientations); base=list(csv.DictReader(open(meta))); ids={r["symbol"].upper():i for i,r in enumerate(base)}
    keep=d.query_gene.str.upper().isin(ids)&d.library_gene.str.upper().isin(ids); d=d[keep].copy(); pairs=d.drop_duplicates("pair_id")[["pair_id","query_gene","library_gene"]]
    p=np.asarray([(ids[a.upper()],ids[b.upper()]) for a,b in pairs[["query_gene","library_gene"]].itertuples(index=False)],"int16"); return d,pairs,p


def prepare(args):
    d,pairs,p=pair_table(args.orientations,args.meta); z=np.load(args.perturb); seen=set(z["pairs"][z["role"]==0].ravel()); seen.discard(-1); cold=np.asarray([(a not in seen and b not in seen) for a,b in p])
    np.savez_compressed(args.output,pairs=p,cold=cold,gins=np.asarray(np.sort(d.gin_id.astype(str).unique()),dtype="U16")); audit={"schema":"sl-predict-hap1-score-pack-v1","orientation_sha256":sha(args.orientations),"pairs":len(p),"orientations":len(d),"cold_pairs":int(cold.sum()),"cold_orientations":int(d.pair_id.isin(set(pairs.pair_id[cold])).sum())}; Path(args.output).with_suffix(".json").write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2))


def score(args):
    import torch
    sys.path.insert(0,str(Path(__file__).parents[1]/"training")); from world_model import SLPredict,ResidualInteraction,encode_genes,interaction_head,load_residual_endpoint,residual_interaction_inputs
    torch.set_grad_enabled(False); device="cuda" if torch.cuda.is_available() else "cpu"; pack=np.load(args.pack); state=np.load(args.features)["state"].astype("float32"); sd=torch.load(args.model,map_location="cpu",weights_only=True); context_dim=sd["context_proj.weight"].shape[1] if "context_proj.weight" in sd else 0; model=SLPredict(args.d,args.latent,args.layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],context_dim).to(device); model.load_state_dict(sd); model.eval(); ds=torch.load(args.decoder,map_location="cpu",weights_only=True); decoder=torch.nn.Linear(args.latent,ds["weight"].shape[0]).to(device); decoder.load_state_dict(ds); decoder.eval(); genes=torch.as_tensor(encode_genes(model,state,device),device=device); context_state=None
    if args.context_pack:
        z=np.load(args.context_pack); at=np.flatnonzero(z["model_ids"].astype(str)==args.context_model); context_state=torch.as_tensor(z["cell_state"][at[0]],device=device) if len(at)==1 else (_ for _ in ()).throw(ValueError(f"context model {args.context_model} not uniquely resolved"))
    interaction=None
    if args.interaction_head:
        interaction=interaction_head(args.latent).to(device); interaction.load_state_dict(torch.load(args.interaction_head,map_location="cpu",weights_only=True)); interaction.eval()
    ensemble=None; residual=None
    if getattr(args,"ensemble_model",None):
        es=torch.load(args.ensemble_model,map_location="cpu",weights_only=True); em=SLPredict(args.d,args.latent,args.layers,es["cell.weight"].shape[0],es["outcome.weight"].shape[0],state.shape[1],es["context_proj.weight"].shape[1]).to(device); em.load_state_dict(es); em.eval(); eg=torch.as_tensor(encode_genes(em,state,device),device=device); eh=interaction_head(args.latent).to(device); eh.load_state_dict(torch.load(args.ensemble_interaction_head,map_location="cpu",weights_only=True)); eh.eval(); ensemble=(em,eh,eg)
    if getattr(args,"residual_model",None):
        endpoint=load_residual_endpoint(args.residual_model,state.shape[1],device,args.d,args.latent,args.layers); rg=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); rh=ResidualInteraction(interaction_head(args.latent).to(device)).to(device); rh.load_state_dict(torch.load(args.residual_interaction_head,map_location="cpu",weights_only=True)); rh.eval(); residual=(endpoint,rh,rg)
    out=[]
    for at in range(0,len(pack["pairs"]),args.batch):
        p=torch.as_tensor(pack["pairs"][at:at+args.batch].astype("int64"),device=device); a,b=genes[p[:,0]],genes[p[:,1]]; cs=context_state.expand(len(p),-1) if context_state is not None else None; z=model.transition(a,b,context_state=cs)[0]
        if interaction is not None:
            value=-interaction(z)[:,0]
            if ensemble is not None:
                em,eh,eg=ensemble; second=-eh(em.transition(eg[p[:,0]],eg[p[:,1]],context_state=cs)[0])[:,0]; value=(1-args.ensemble_weight)*value+args.ensemble_weight*second
            if residual is not None:
                endpoint,rh,rg=residual; rz,rr=residual_interaction_inputs(endpoint,rg,p,cs); value=-rh(rz,rr)[:,0]
        else:
            joint=decoder(z); la=model.transition(a,context_state=cs)[0]; lb=model.transition(b,context_state=cs)[0]; ab=decoder(model.transition(b,state=la,context_state=cs)[0]); ba=decoder(model.transition(a,state=lb,context_state=cs)[0]); value=torch.linalg.vector_norm(joint-(ab+ba)/2,dim=1)
        out.append(value.cpu().numpy()); print(json.dumps({"scored":min(at+args.batch,len(pack["pairs"])),"total":len(pack["pairs"])}),flush=True) if at%(args.batch*20)==0 else None
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(args.output,score=np.concatenate(out).astype("float32")); print(json.dumps({"pairs":len(pack["pairs"]),"device":device,"context_model":args.context_model if context_state is not None else None,"output":args.output}))


def outcomes(path,gins):
    import pandas as pd
    header=pd.read_excel(path,sheet_name="qGI_scores",nrows=0).columns.astype(str).tolist(); available={f"GIN{m.group(1)}":x for x in header[1:] if (m:=re.search(r"_(\d{3})_(?:min|rich)$",x))}; selected=sorted(gins); use=["Gene.symbol"]+[available[x] for x in selected]; q=pd.read_excel(path,sheet_name="qGI_scores",usecols=use); f=pd.read_excel(path,sheet_name="qGI_FDR",usecols=use)
    if q.columns.tolist()!=f.columns.tolist() or not q["Gene.symbol"].equals(f["Gene.symbol"]):raise ValueError("qGI/FDR matrices disagree")
    n=len(selected); return pd.DataFrame({"source_library_gene":np.repeat(q["Gene.symbol"].astype(str).to_numpy(),n),"gin_id":np.tile(selected,len(q)),"qgi":q.iloc[:,1:].to_numpy().reshape(-1),"fdr":f.iloc[:,1:].to_numpy().reshape(-1)})


def extract(args):
    data=outcomes(args.outcome,np.load(args.pack)["gins"]); genes=np.asarray(data.source_library_gene.unique(),dtype="U32"); gins=np.asarray(data.gin_id.unique(),dtype="U16"); np.savez_compressed(args.output,genes=genes,gins=gins,qgi=data.qgi.to_numpy("float32").reshape(len(genes),len(gins)),fdr=data.fdr.to_numpy("float32").reshape(len(genes),len(gins))); Path(args.output).with_suffix(".json").write_text(json.dumps({"schema":"hap1-qgi-fdr-cache-v1","source_sha256":sha(args.outcome),"genes":len(genes),"queries":len(gins)},indent=2)); print(json.dumps({"genes":len(genes),"queries":len(gins)}))


def auxiliary(args):
    orient,pairs,p=pair_table(args.orientations,args.meta); raw=cached_outcomes(args.outcome_cache); data=orient.merge(raw,on=["gin_id","source_library_gene"],validate="one_to_one"); data=data[np.isfinite(data.qgi)&np.isfinite(data.fdr)].copy(); data["negative"]=(data.qgi<-.3)&(data.fdr<.1); data["positive"]=(data.qgi>.3)&(data.fdr<.1)
    calls=data.groupby("pair_id").agg(observations=("negative","size"),negative=("negative","sum"),positive=("positive","sum")); calls=calls.reindex(pairs.pair_id); y=((calls.negative==calls.observations)&(calls.negative>0)&(calls.positive==0)).to_numpy(); background=((calls.negative==0)&(calls.positive==0)).to_numpy(); rng=np.random.default_rng(args.seed); neg=np.flatnonzero(background); neg=rng.choice(neg,min(len(neg),args.background*int(y.sum())),replace=False); take=np.r_[np.flatnonzero(y),neg]; rng.shuffle(take); np.savez_compressed(args.output,pairs=p[take],label=y[take].astype("int8"))
    audit={"schema":"hap1-fold-local-auxiliary-v1","orientation_sha256":sha(args.orientations),"outcome_cache_sha256":sha(args.outcome_cache),"rule":"positive iff every measured orientation has qGI < -0.3 and FDR < 0.1; exclude any positive-GI call; fixed random no-call background","seed":args.seed,"background_ratio":args.background,"pairs":len(take),"positives":int(y.sum()),"background":len(neg)}; Path(args.output).with_suffix(".json").write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2))


def quantitative(args):
    orient,pairs,p=pair_table(args.orientations,args.meta); raw=cached_outcomes(args.outcome_cache); data=orient.merge(raw,on=["gin_id","source_library_gene"],validate="one_to_one"); finite=np.isfinite(data.qgi); grouped=data[finite].groupby("pair_id").qgi.agg(["mean","size"]).reindex(pairs.pair_id); keep=grouped["mean"].notna().to_numpy(); target=grouped["mean"].to_numpy("float32")[keep]; count=grouped["size"].fillna(0).to_numpy("int8")[keep]; out=Path(args.output); np.savez_compressed(out,pairs=p[keep],target=target,observations=count); audit={"schema":"hap1-pair-qgi-v1","orientation_sha256":sha(args.orientations),"outcome_cache_sha256":sha(args.outcome_cache),"orientations":len(data),"finite_orientations":int(finite.sum()),"pairs":len(target),"repeated_pairs":int((count>1).sum()),"target_mean":float(target.mean()),"target_sd":float(target.std()),"binary_sl_labels_used":False}; out.with_suffix(".json").write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2))


def cached_outcomes(path):
    import pandas as pd
    z=np.load(path); return pd.DataFrame({"source_library_gene":np.repeat(z["genes"],len(z["gins"])),"gin_id":np.tile(z["gins"],len(z["genes"])),"qgi":z["qgi"].reshape(-1),"fdr":z["fdr"].reshape(-1)})


def task(data,name):
    p=data.negative.to_numpy(); g=data.positive.to_numpy(); sn=data.strict_negative.to_numpy(); sg=data.strict_positive.to_numpy()
    if name=="negative_vs_no_call":keep=~g; y=p[keep]
    elif name=="negative_vs_all":keep=np.ones(len(data),bool); y=p
    elif name=="negative_vs_positive":keep=p|g; y=p[keep]
    elif name=="strict_negative_vs_positive":keep=sn|sg; y=sn[keep]
    else:raise ValueError(name)
    return data.loc[keep].copy(),y.astype("int8")


def summarize(data,name):
    import pandas as pd
    from sklearn.metrics import average_precision_score,roc_auc_score
    d,y=task(data,name); d["label"]=y; rows=[]
    for gin,x in d.groupby("gin_id",sort=True):
        z=x.label.to_numpy(); supported=0<z.sum()<len(z); rows.append({"gin_id":gin,"query_media":x.query_media.iloc[0],"rows":len(x),"positives":int(z.sum()),"prevalence":float(z.mean()),"supported":supported,"ap":float(average_precision_score(z,x.score)) if supported else np.nan,"auroc":float(roc_auc_score(z,x.score)) if supported else np.nan})
    perq=pd.DataFrame(rows); s=perq[perq.supported]; w=1/d.orientation_count.to_numpy(); return {"rows":len(d),"positives":int(y.sum()),"queries":len(perq),"supported_queries":len(s),"macro_prevalence":float(s.prevalence.mean()),"macro_ap":float(s.ap.mean()),"macro_ap_minus_prevalence":float((s.ap-s.prevalence).mean()),"macro_auroc":float(s.auroc.mean()),"canonical_weighted_pooled_ap":float(average_precision_score(y,d.score,sample_weight=w)),"canonical_weighted_pooled_auroc":float(roc_auc_score(y,d.score,sample_weight=w))},perq


def inference(data,perq):
    from sklearn.metrics import average_precision_score
    s=perq[perq.supported].reset_index(drop=True); rng=np.random.default_rng(20260824); strata={m:np.flatnonzero(s.query_media.to_numpy()==m) for m in sorted(s.query_media.unique())}; boot=[]
    for _ in range(10000):
        ix=np.concatenate([rng.choice(v,len(v),replace=True) for v in strata.values()]); x=s.iloc[ix]; boot.append([x.ap.mean(),(x.ap-x.prevalence).mean(),x.auroc.mean()])
    d,y=task(data,"negative_vs_no_call"); d["label"]=y; rng=np.random.default_rng(20260825); null=np.zeros(1999); observed=[]; queries=0
    for _,x in d.groupby("gin_id",sort=True):
        z=x.label.to_numpy("int8"); m=int(z.sum()); n=len(z)
        if not 0<m<n:continue
        values=np.sort(x.score.to_numpy())[::-1]; observed.append(average_precision_score(z,x.score)); groups=np.cumsum(np.r_[True,values[1:]!=values[:-1]])-1; ends=np.flatnonzero(np.r_[values[1:]!=values[:-1],True])+1
        for r in range(1999):
            chosen=rng.choice(n,m,replace=False,shuffle=False); g,c=np.unique(groups[chosen],return_counts=True); null[r]+=np.sum((np.cumsum(c)/ends[g])*(c/m))
        queries+=1
    null/=queries; obs=float(np.mean(observed)); return {"bootstrap":{"repetitions":10000,"seed":20260824,"strata":{k:len(v) for k,v in strata.items()},"macro_ap_95ci":np.quantile(np.asarray(boot)[:,0],[.025,.975]).tolist(),"macro_ap_minus_prevalence_95ci":np.quantile(np.asarray(boot)[:,1],[.025,.975]).tolist(),"macro_auroc_95ci":np.quantile(np.asarray(boot)[:,2],[.025,.975]).tolist()},"permutation":{"repetitions":1999,"seed":20260825,"observed_macro_ap":obs,"null_mean":float(null.mean()),"one_sided_p":float((1+(null>=obs).sum())/2000)}}


def evaluate(args):
    import pandas as pd
    orient,pairs,p=pair_table(args.orientations,args.meta); pack=np.load(args.pack); scored=np.load(args.scores)["score"]; assert np.array_equal(p,pack["pairs"]) and len(scored)==len(p); cold=pack["cold"].copy()
    if args.interaction_training:
        z=np.load(args.interaction_training,allow_pickle=True); q=z["pairs"].astype("int64"); held=np.arange(q.max()+1)%5==0; rows=z["context_known"][z["context"]]&(~held[q]).all(1); seen=set(q[rows].ravel()); cold&=np.asarray([(int(a) not in seen and int(b) not in seen) for a,b in p])
    pairs["score"]=scored; pairs["cold"]=cold; orient=orient.merge(pairs[["pair_id","score","cold"]],on="pair_id",validate="many_to_one"); raw=cached_outcomes(args.outcome_cache); data=orient.merge(raw,on=["gin_id","source_library_gene"],validate="one_to_one"); finite=np.isfinite(data.qgi)&np.isfinite(data.fdr); excluded=int((~finite).sum()); data=data[finite].copy(); data["negative"]=(data.qgi<-.3)&(data.fdr<.1); data["positive"]=(data.qgi>.3)&(data.fdr<.1); data["strict_negative"]=(data.qgi<-.6)&(data.fdr<.01); data["strict_positive"]=(data.qgi>.6)&(data.fdr<.01); data["orientation_count"]=data.groupby("pair_id").pair_id.transform("size")
    result={"protocol":json.loads(Path(args.protocol).read_text()),"nonfinite_excluded":excluded,"subsets":{}}; tables={}
    for subset,frame in (("intervention_cold",data[data.cold]),("all_mapped",data)):
        result["subsets"][subset]={"orientations":len(frame),"pairs":frame.pair_id.nunique(),"queries":frame.gin_id.nunique(),"tasks":{}}
        for name in ("negative_vs_no_call","negative_vs_all","negative_vs_positive","strict_negative_vs_positive"):
            result["subsets"][subset]["tasks"][name],tables[f"{subset}_{name}"]=summarize(frame,name)
        if subset=="intervention_cold":result["inference"]=inference(frame,tables[f"{subset}_negative_vs_no_call"])
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)); pd.concat([x.assign(table=k) for k,x in tables.items()]).to_parquet(out.with_suffix(".per_query.parquet"),index=False); print(json.dumps(result["subsets"]["intervention_cold"]|{"inference":result["inference"]},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("prepare"); a.add_argument("--orientations",required=True); a.add_argument("--meta",default="data/feng2024/data/preprocessed_data/meta_table_9845.csv"); a.add_argument("--perturb",default="results/sl_predict/perturbseq_world_v2.npz"); a.add_argument("--output",default="results/sl_predict/hap1_score_pack.npz")
    a=sub.add_parser("score"); a.add_argument("--pack",default="results/sl_predict/hap1_score_pack.npz"); a.add_argument("--model",required=True); a.add_argument("--decoder",required=True); a.add_argument("--interaction-head"); a.add_argument("--ensemble-model"); a.add_argument("--ensemble-interaction-head"); a.add_argument("--ensemble-weight",type=float,default=.75); a.add_argument("--residual-model"); a.add_argument("--residual-interaction-head"); a.add_argument("--features",default="results/sl_predict/features_spectral_safe.npz"); a.add_argument("--output",default="results/sl_predict/hap1_pair_scores.npz"); a.add_argument("--context-pack"); a.add_argument("--context-model",default="ACH-002475"); a.add_argument("--batch",type=int,default=8192); a.add_argument("--d",type=int,default=384); a.add_argument("--latent",type=int,default=128); a.add_argument("--layers",type=int,default=6)
    a=sub.add_parser("extract"); a.add_argument("--outcome",required=True); a.add_argument("--pack",default="results/sl_predict/hap1_score_pack.npz"); a.add_argument("--output",default="results/sl_predict/hap1_outcomes.npz")
    a=sub.add_parser("auxiliary"); a.add_argument("--orientations",required=True); a.add_argument("--outcome-cache",default="results/sl_predict/hap1_outcomes.npz"); a.add_argument("--meta",default="data/feng2024/data/preprocessed_data/meta_table_9845.csv"); a.add_argument("--output",default="results/sl_predict/hap1_auxiliary.npz"); a.add_argument("--background",type=int,default=4); a.add_argument("--seed",type=int,default=20260826)
    a=sub.add_parser("quantitative"); a.add_argument("--orientations",required=True); a.add_argument("--outcome-cache",default="results/sl_predict/hap1_outcomes.npz"); a.add_argument("--meta",default="data/feng2024/data/preprocessed_data/meta_table_9845.csv"); a.add_argument("--output",default="results/sl_predict/hap1_quantitative.npz")
    a=sub.add_parser("evaluate"); a.add_argument("--orientations",required=True); a.add_argument("--outcome-cache",default="results/sl_predict/hap1_outcomes.npz"); a.add_argument("--scores",default="results/sl_predict/hap1_pair_scores.npz"); a.add_argument("--pack",default="results/sl_predict/hap1_score_pack.npz"); a.add_argument("--interaction-training"); a.add_argument("--meta",default="data/feng2024/data/preprocessed_data/meta_table_9845.csv"); a.add_argument("--protocol",default="results/sl_predict/hap1_v2_protocol.json"); a.add_argument("--output",default="results/sl_predict/hap1_v2_result.json")
    args=p.parse_args(); {"prepare":prepare,"score":score,"extract":extract,"auxiliary":auxiliary,"quantitative":quantitative,"evaluate":evaluate}[args.command](args)
