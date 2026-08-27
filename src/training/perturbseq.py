from pathlib import Path
import argparse, hashlib, json
import numpy as np, pandas as pd
from sklearn.decomposition import PCA

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def build(source,dimensions=32,fitness=False,additional=(),name=None,nested_prefix=0):
    source=Path(source); paths=sorted(source.glob("*.npz"))+[Path(x) for x in additional]; meta=pd.read_csv(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"); ids={str(s).upper():int(i) for s,i in zip(meta.symbol,meta.unified_id)}; ens={str(e).split(".")[0]:str(s).upper() for e,s in zip(meta.ensembl_gene_id,meta.symbol) if pd.notna(e)}; held=set(json.loads((OUT/"slkb_outcomes_intervention_external.json").read_text())["symbols"])
    canon=lambda x:ens.get(str(x).split(".")[0],str(x).upper()); loaded=[]; common=None
    for path in paths:
        z=np.load(path,allow_pickle=False); names=np.asarray([canon(x) for x in z["feature_name"]]); present=set(names); common=present if common is None else common&present; loaded.append((path,z,names))
    common=sorted(common); normalized=[]; rows=[]; audits=[]
    for sid,(path,z,names) in enumerate(loaded):
        fmap={x:i for i,x in enumerate(names)}; cols=np.asarray([fmap[x] for x in common]); delta=z["future_state"][:,cols]-z["control_mean"][cols]; a=np.char.upper(z["endpoint_a"].astype(str)); b=np.char.upper(z["endpoint_b"].astype(str)); card=z["cardinality"]
        mapped=np.asarray([(c in (1,2)) and (x in ids) and (c<2 or y in ids) for x,y,c in zip(a,b,card)]); cold=np.asarray([(x not in held) and (c<2 or y not in held) for x,y,c in zip(a,b,card)]); val=lambda x:int.from_bytes(hashlib.sha256(x.encode()).digest()[:4],"big")%5==0
        role=np.full(len(card),-1,"int8")
        for i,(x,y,c) in enumerate(zip(a,b,card)):
            if c<1 or not mapped[i] or not cold[i]: continue
            v=(val(x),)+(val(y),) if c==2 else (val(x),); role[i]=1 if all(v) else (0 if not any(v) else -1)
        fit=role==0; center=delta[fit].mean(0); scale=delta[fit].std(0).clip(.05); x=((delta-center)/scale).astype("float32"); normalized.append(x[fit][np.linspace(0,fit.sum()-1,min(1000,fit.sum())).astype(int)])
        condition=z["condition"].astype(str); unique,inverse=np.unique(condition,return_inverse=True); abundance=np.bincount(inverse,weights=z["cell_count"]); log_abundance=np.log1p(abundance[inverse]); center_fit=np.median(log_abundance[fit]); scale_fit=max(.1,np.median(np.abs(log_abundance[fit]-center_fit))); fitness_target=((log_abundance-center_fit)/scale_fit).clip(-8,8).astype("float32")
        keep=role>=0; pair=np.full((keep.sum(),2),-1,"int16"); pair[:,0]=[ids[x] for x in a[keep]]; double=card[keep]==2; pair[double,1]=[ids[x] for x in b[keep][double]]; rows.append((pair,np.full(keep.sum(),sid,"int8"),card[keep].astype("int8"),role[keep],x[keep],fitness_target[keep]))
        audits.append({"source":path.stem,"input_sha256":sha(path),"train_rows":int(fit.sum()),"validation_rows":int((role==1).sum()),"train_double_rows":int((fit&(card==2)).sum()),"validation_double_rows":int(((role==1)&(card==2)).sum())})
    pool=np.concatenate(normalized)
    if nested_prefix:
        prefix=PCA(nested_prefix,svd_solver="randomized",random_state=731).fit(pool); residual=lambda x:x-(x@prefix.components_.T)@prefix.components_; tail=PCA(dimensions-nested_prefix,svd_solver="randomized",random_state=731).fit(residual(pool)); project=lambda x:np.column_stack((x@prefix.components_.T,residual(x)@tail.components_.T)); reconstruction=lambda x:(x@prefix.components_.T)@prefix.components_+((residual(x)@tail.components_.T)@tail.components_); explained=1-((pool-reconstruction(pool))**2).sum()/((pool-pool.mean(0))**2).sum(); blocks=np.asarray([nested_prefix,dimensions],"int16")
    else:
        pca=PCA(dimensions,svd_solver="randomized",random_state=731).fit(pool); project=lambda x:x@pca.components_.T; explained=float(pca.explained_variance_ratio_.sum()); blocks=np.asarray([dimensions],"int16")
    raw=np.concatenate([project(x[4]) for x in rows]); train=np.concatenate([x[3]==0 for x in rows]); latent_scale=raw[train].std(0).clip(.05); target=(raw/latent_scale).astype("float32"); target=np.column_stack((target,np.concatenate([x[5] for x in rows]))) if fitness else target
    arrays={"pairs":np.concatenate([x[0] for x in rows]),"source":np.concatenate([x[1] for x in rows]),"cardinality":np.concatenate([x[2] for x in rows]),"role":np.concatenate([x[3] for x in rows]),"target":target.astype("float32"),"sources":np.asarray([x[0].stem for x in loaded]),"state_dimensions":np.int16(dimensions),"state_blocks":blocks,"state_block_weights":np.full(len(blocks),1/len(blocks),"float32"),"fitness_index":np.int16(dimensions if fitness else -1)}; name=name or ("perturbseq_world_fitness" if fitness else "perturbseq_world"); np.savez_compressed(OUT/f"{name}.npz",**arrays)
    audit={"schema":"sl-predict-perturbseq-world-v1","sources":audits,"rows":len(target),"train_rows":int(train.sum()),"validation_rows":int((~train).sum()),"common_expression_genes":len(common),"state_dimensions":dimensions,"state_blocks":blocks.tolist(),"nested_residual_endpoint":bool(nested_prefix),"signed_fitness_target":"within-source robust z-score of log recovered cells by condition" if fitness else None,"held_scenario3_genes":len(held),"benchmark_membership_used_for_exclusion_only":True,"sl_labels_used":False,"explained_variance":float(explained)}; (OUT/f"{name}.json").write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2))


def build_source_landmark(source,dimensions=32,name="perturbseq_source_landmark"):
    source=Path(source); meta=pd.read_csv(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"); ids={str(s).upper():int(i) for s,i in zip(meta.symbol,meta.unified_id)}; ens={str(e).split(".")[0]:str(s).upper() for e,s in zip(meta.ensembl_gene_id,meta.symbol) if pd.notna(e)}; held=set(json.loads((OUT/"slkb_outcomes_intervention_external.json").read_text())["symbols"]); lincs=pd.read_csv(ROOT/"data/raw/norman2019_perturbseq/LINCS2020_geneinfo_beta.txt",sep="\t"); landmark=set(lincs.loc[lincs.feature_space.eq("landmark"),"gene_symbol"].astype(str).str.upper()); val=lambda x:int.from_bytes(hashlib.sha256(x.encode()).digest()[:4],"big")%5==0; rows=[]; audits=[]
    for sid,path in enumerate(sorted(source.glob("*.npz"))):
        z=np.load(path); names=np.asarray([ens.get(str(x).split(".")[0],str(x).upper()) for x in z["feature_name"]]); cols=np.asarray([i for i,x in enumerate(names) if x in landmark]); a=np.char.upper(z["endpoint_a"].astype(str)); b=np.char.upper(z["endpoint_b"].astype(str)); card=z["cardinality"]; mapped=np.asarray([(x in ids) and (c<2 or y in ids) for x,y,c in zip(a,b,card)]); cold=np.asarray([(x not in held) and (c<2 or y not in held) for x,y,c in zip(a,b,card)]); role=np.full(len(card),-1,"int8")
        for i,(x,y,c) in enumerate(zip(a,b,card)):
            if c<1 or not mapped[i] or not cold[i]:continue
            v=(val(x),)+(val(y),) if c==2 else (val(x),); role[i]=1 if all(v) else (0 if not any(v) else -1)
        fit=role==0; delta=z["future_state"][:,cols]-z["control_mean"][cols]; center=delta[fit].mean(0); scale=delta[fit].std(0).clip(.05); x=(delta-center)/scale; pca=PCA(dimensions,svd_solver="randomized",random_state=731).fit(x[fit]); raw=pca.transform(x); latent_scale=raw[fit].std(0).clip(.05); target=(raw/latent_scale).astype("float32"); keep=role>=0; pair=np.full((keep.sum(),2),-1,"int16"); pair[:,0]=[ids[x] for x in a[keep]]; double=card[keep]==2; pair[double,1]=[ids[x] for x in b[keep][double]]; rows.append((pair,np.full(keep.sum(),sid,"int8"),card[keep].astype("int8"),role[keep],target[keep])); audits.append({"source":path.stem,"input_sha256":sha(path),"landmark_genes":len(cols),"train_rows":int(fit.sum()),"validation_rows":int((role==1).sum()),"train_double_rows":int((fit&(card==2)).sum()),"validation_double_rows":int(((role==1)&(card==2)).sum()),"explained_variance":float(pca.explained_variance_ratio_.sum())})
    basal=np.load(OUT/"basal_context.npz"); arrays={"pairs":np.concatenate([x[0] for x in rows]),"source":np.concatenate([x[1] for x in rows]),"cardinality":np.concatenate([x[2] for x in rows]),"role":np.concatenate([x[3] for x in rows]),"target":np.concatenate([x[4] for x in rows]),"sources":np.asarray([x["source"] for x in audits]),"context_state":basal["source_state"][:len(rows)],"state_dimensions":np.int16(dimensions)}; np.savez_compressed(OUT/f"{name}.npz",**arrays); audit={"schema":"sl-predict-source-landmark-state-v1","fixed_panel":"LINCS 2020 landmark genes","sources":audits,"rows":len(arrays["target"]),"train_rows":int((arrays["role"]==0).sum()),"validation_rows":int((arrays["role"]==1).sum()),"held_scenario3_genes":len(held),"benchmark_membership_used_for_exclusion_only":True,"sl_labels_used":False}; (OUT/f"{name}.json").write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("source",nargs="?",default=ROOT/"data/processed/perturbseq_world_v1"); p.add_argument("--dimensions",type=int,default=32); p.add_argument("--nested-prefix",type=int,default=0); p.add_argument("--fitness",action="store_true"); p.add_argument("--source-landmark",action="store_true"); p.add_argument("--additional",nargs="*",default=()); p.add_argument("--name"); a=p.parse_args(); build_source_landmark(a.source,a.dimensions,a.name or "perturbseq_source_landmark") if a.source_landmark else build(a.source,a.dimensions,a.fitness,a.additional,a.name,a.nested_prefix)
