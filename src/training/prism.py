from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parents[2]

def corr(a,b):
    keep=np.isfinite(a)&np.isfinite(b); return float(np.corrcoef(a[keep],b[keep])[0,1]) if keep.sum()>2 else np.nan

def zscore(x):
    return (x-np.nanmean(x,0))/np.nanstd(x,0).clip(.1)

def build(output=ROOT/"results/sl_predict/prism_single_target.npz"):
    meta=pd.read_csv(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"); gene={x.upper():i for i,x in enumerate(meta.symbol.astype(str))}; treatment=pd.read_csv(ROOT/"data/prism/primary-screen-replicate-collapsed-treatment-info.csv"); matrix=pd.read_csv(ROOT/"data/prism/primary-screen-replicate-collapsed-logfold-change.csv",index_col=0); basal=np.load(ROOT/"results/sl_predict/basal_context.npz"); row={x:i for i,x in enumerate(matrix.index.astype(str))}; models=np.asarray([x for x in basal["model_ids"].astype(str) if x in row]); matrix=matrix.loc[models]; dep_row={x:i for i,x in enumerate(basal["model_ids"].astype(str))}; dependency=basal["dependency"][[dep_row[x] for x in models]].astype("float32"); dependency_known=basal["dependency_known"][[dep_row[x] for x in models]]; treatment=treatment[treatment.column_name.isin(matrix.columns)].copy(); treatment["symbol"]=treatment.target.fillna("").str.strip().str.upper(); treatment=treatment[treatment.symbol.isin(gene)].copy(); values=matrix[treatment.column_name].to_numpy("float32"); values=zscore(values); halves=np.asarray([hashlib.sha256(x.encode()).digest()[0]&1 for x in treatment.broad_id.astype(str)]); targets=[]; profiles=[]; counts=[]; half_corr=[]; half_x=[]; half_y=[]; dep_x=[]; dep_y=[]
    for symbol,ix in treatment.groupby("symbol",sort=True).indices.items():
        ix=np.asarray(ix); left=ix[halves[ix]==0]; right=ix[halves[ix]==1]
        if treatment.iloc[left].broad_id.nunique()<2 or treatment.iloc[right].broad_id.nunique()<2:continue
        a=np.nanmean(values[:,left],1); b=np.nanmean(values[:,right],1); shared=np.isfinite(a)&np.isfinite(b)
        if shared.sum()<100:continue
        half_corr.append(corr(a,b)); half_x.append(a[shared]); half_y.append(b[shared]); consensus=np.nanmean(values[:,ix],1); d=dependency[:,gene[symbol]]; valid=np.isfinite(consensus)&dependency_known[:,gene[symbol]]
        if valid.sum()>=100:
            dep_x.append(zscore(consensus[valid,None]).ravel()); dep_y.append(zscore(d[valid,None]).ravel())
        targets.append(gene[symbol]); profiles.append(consensus); counts.append(len(ix))
    pooled_half=corr(np.concatenate(half_x),np.concatenate(half_y)); pooled_dep=corr(np.concatenate(dep_x),np.concatenate(dep_y)); median=float(np.nanmedian(half_corr)); admitted=bool(pooled_half>=.15 and median>=.10 and pooled_dep>=.10); out=Path(output); np.savez_compressed(out,model_ids=models,target_gene=np.asarray(targets,"int16"),profile=np.asarray(profiles,"float16"),compound_count=np.asarray(counts,"int16")); audit={"schema":"sl-predict-prism-single-target-v1","models":len(models),"annotated_single_target_conditions":len(treatment),"eligible_replicated_targets":len(targets),"median_compounds_per_target":float(np.median(counts)),"pooled_half_profile_pearson":pooled_half,"median_target_half_pearson":median,"pooled_consensus_crispr_pearson":pooled_dep,"admitted":admitted,"double_perturbation_data_used":False,"sl_labels_used":False}; out.with_suffix(".json").write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2)); return audit

if __name__=="__main__":build()
