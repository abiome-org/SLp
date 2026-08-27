from pathlib import Path
import csv, hashlib, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]

def sha(path,algorithm="sha256"):
    h=hashlib.new(algorithm)
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def aligned(path,symbols):
    header=pd.read_csv(path,nrows=0).columns; named={x.rsplit(" (",1)[0].upper():x for x in header[1:]}; use=[header[0]]+[named[x] for x in symbols if x in named]; frame=pd.read_csv(path,usecols=use); out=np.full((len(frame),len(symbols)),np.nan,"float32"); pos={x:i for i,x in enumerate(symbols)}
    for col in use[1:]:out[:,pos[col.rsplit(" (",1)[0].upper()]]=frame[col].to_numpy("float32")
    return frame.iloc[:,0].astype(str).to_numpy(),out,len(use)-1

def residualize(effect,lineage,subset):
    out=effect.copy(); global_mean=np.nanmean(out[subset],0)
    for group in np.unique(lineage[subset]):
        rows=subset&(lineage==group); mean=np.nanmean(out[rows],0); mean=np.where(np.isfinite(mean),mean,global_mean); out[rows]-=mean
    return out

def estimate(loss,known,effect,lineage,subset,min_loss,min_wt):
    state=residualize(effect,lineage,subset); genes=np.flatnonzero((loss&subset[:,None]).sum(0)>=min_loss); delta=np.full((len(genes),effect.shape[1]),np.nan,"float32")
    for row,gene in enumerate(genes):
        low=subset&known[:,gene]&loss[:,gene]; wt=subset&known[:,gene]&~loss[:,gene]
        if low.sum()<min_loss or wt.sum()<min_wt:continue
        delta[row]=np.nanmean(state[low],0)-np.nanmean(state[wt],0)
    return genes,delta

def ridge_residual(value,known,x,subset,fold,alpha=100.):
    out=np.zeros_like(value,"float32")
    for f in range(5):
        train=subset&(fold!=f); test=subset&(fold==f); mu=np.nanmean(np.where(known[train],value[train],np.nan),0); mu=np.nan_to_num(mu); center=x[train].mean(0); scale=x[train].std(0).clip(.1); a=(x[train]-center)/scale; b=np.linalg.solve(a.T@a+alpha*np.eye(a.shape[1]),a.T@(np.where(known[train],value[train],mu)-mu)); pred=mu+(x[test]-center)/scale@b; out[test]=np.where(known[test],value[test]-pred,0)
    return out

def estimate_adjusted(loss,known,effect,effect_known,x,subset,fold,min_loss,min_wt):
    genes=np.flatnonzero(((loss&known)&subset[:,None]).sum(0)>=min_loss)
    genes=genes[((~loss&known)&subset[:,None]).sum(0)[genes]>=min_wt]; tr=ridge_residual(loss[:,genes].astype("float32"),known[:,genes],x,subset,fold); yr=ridge_residual(effect,effect_known,x,subset,fold); denominator=(tr*tr).T@effect_known.astype("float32"); delta=(tr.T@yr)/np.maximum(denominator,1e-8); delta[denominator<1e-6]=np.nan
    return genes,delta.astype("float32")

def build(cn_path,mutation_path,output):
    meta=pd.read_csv(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"); symbols=meta.symbol.astype(str).str.upper().to_numpy(); basal=np.load(ROOT/"results/sl_predict/basal_context.npz"); model=basal["model_ids"].astype(str); effect=basal["dependency"].astype("float32"); effect[~basal["dependency_known"]]=np.nan; cn_id,cn,cn_genes=aligned(cn_path,symbols); mut_id,mut,mut_genes=aligned(mutation_path,symbols); cn_row={x:i for i,x in enumerate(cn_id)}; mut_row={x:i for i,x in enumerate(mut_id)}; keep=np.asarray([x in cn_row and x in mut_row for x in model]); model=model[keep]; effect=effect[keep]; cn=cn[[cn_row[x] for x in model]]; mut=mut[[mut_row[x] for x in model]]; known=np.isfinite(cn)&np.isfinite(mut); loss=(cn<.5)|(mut>.5)
    lineage_map={r["ModelID"]:r["OncotreeLineage"] or "unknown" for r in csv.DictReader(open(ROOT/"data/depmap24q2/Model.csv"))}; lineage=np.asarray([lineage_map.get(x,"unknown") for x in model]); half=np.asarray([hashlib.sha256(x.encode()).digest()[0]&1 for x in model]); full_gene,full=estimate(loss,known,effect,lineage,np.ones(len(model),bool),5,50); h0_gene,h0=estimate(loss,known,effect,lineage,half==0,3,20); h1_gene,h1=estimate(loss,known,effect,lineage,half==1,3,20); common=np.intersect1d(h0_gene,h1_gene); p0={g:i for i,g in enumerate(h0_gene)}; p1={g:i for i,g in enumerate(h1_gene)}; a=np.concatenate([h0[p0[g]] for g in common]); b=np.concatenate([h1[p1[g]] for g in common]); valid=np.isfinite(a)&np.isfinite(b); diagonal=np.zeros_like(valid)
    for i,g in enumerate(common):diagonal[i*len(symbols)+g]=True
    valid&=~diagonal; x=a[valid]; y=b[valid]; pearson=float(np.corrcoef(x,y)[0,1]); spearman=float(spearmanr(x,y).statistic); sign=float(np.mean(np.signbit(x)==np.signbit(y))); tolerance=np.nanmean(effect,0).astype("float32"); out=Path(output); out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,model_ids=model,loss_gene=full_gene.astype("int16"),delta=full.astype("float16"),mean_effect=tolerance)
    audit={"schema":"sl-predict-natural-loss-v1","input_sha256":{"copy_number":sha(cn_path),"damaging_mutation":sha(mutation_path)},"manifest_md5_verified":{"copy_number":sha(cn_path,"md5")=="f784a67fa640d7e353a052cb3694668b","damaging_mutation":sha(mutation_path,"md5")=="02f3568b71af0ca3e8d10e681eefac86"},"aligned_models":len(model),"universe_genes":len(symbols),"copy_number_genes":cn_genes,"mutation_genes":mut_genes,"full_supported_loss_genes":len(full_gene),"half0_supported_loss_genes":len(h0_gene),"half1_supported_loss_genes":len(h1_gene),"shared_half_loss_genes":len(common),"shared_supported_directions":int(valid.sum()),"split_half_pearson":pearson,"split_half_spearman":spearman,"split_half_sign_agreement":sign,"admitted":bool(pearson>=.15 and sign>=.55),"double_perturbation_data_used":False,"sl_labels_used":False}; out.with_suffix(".json").write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2)); return audit

def build_adjusted(cn_path,mutation_path,output):
    meta=pd.read_csv(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"); symbols=meta.symbol.astype(str).str.upper().to_numpy(); basal=np.load(ROOT/"results/sl_predict/basal_context.npz"); all_model=basal["model_ids"].astype(str); effect=basal["dependency"].astype("float32"); effect_known=basal["dependency_known"]; cn_id,cn,_=aligned(cn_path,symbols); mut_id,mut,_=aligned(mutation_path,symbols); cr={x:i for i,x in enumerate(cn_id)}; mr={x:i for i,x in enumerate(mut_id)}; keep=np.asarray([x in cr and x in mr for x in all_model]); model=all_model[keep]; effect=effect[keep]; effect_known=effect_known[keep]; cn=cn[[cr[x] for x in model]]; mut=mut[[mr[x] for x in model]]; known=np.isfinite(cn)&np.isfinite(mut); loss=(cn<.5)|(mut>.5); rows=np.flatnonzero(keep); state=basal["cell_state"][rows,:32]
    lineage_map={r["ModelID"]:r["OncotreeLineage"] or "unknown" for r in csv.DictReader(open(ROOT/"data/depmap24q2/Model.csv"))}; lineage=np.asarray([lineage_map.get(x,"unknown") for x in model]); levels=sorted(set(lineage)); onehot=np.stack([lineage==x for x in levels[1:]],1).astype("float32"); x=np.concatenate((state,onehot),1); digest=np.asarray([hashlib.sha256(v.encode()).digest() for v in model]); half=np.asarray([v[0]&1 for v in digest]); fold=np.asarray([v[1]%5 for v in digest]); full_gene,full=estimate_adjusted(loss,known,effect,effect_known,x,np.ones(len(model),bool),fold,5,50); h0_gene,h0=estimate_adjusted(loss,known,effect,effect_known,x,half==0,fold,3,20); h1_gene,h1=estimate_adjusted(loss,known,effect,effect_known,x,half==1,fold,3,20); common=np.intersect1d(h0_gene,h1_gene); p0={g:i for i,g in enumerate(h0_gene)}; p1={g:i for i,g in enumerate(h1_gene)}; a=np.concatenate([h0[p0[g]] for g in common]); b=np.concatenate([h1[p1[g]] for g in common]); valid=np.isfinite(a)&np.isfinite(b)
    for i,g in enumerate(common):valid[i*len(symbols)+g]=False
    xv=a[valid]; yv=b[valid]; pearson=float(np.corrcoef(xv,yv)[0,1]); spearman=float(spearmanr(xv,yv).statistic); sign=float(np.mean(np.signbit(xv)==np.signbit(yv))); out=Path(output); np.savez_compressed(out,model_ids=model,loss_gene=full_gene.astype("int16"),delta=full.astype("float16"),mean_effect=np.nanmean(effect,0).astype("float32")); audit={"schema":"sl-predict-natural-loss-adjusted-v1","aligned_models":len(model),"confounder_dimensions":x.shape[1],"basal_dimensions":32,"lineages":len(levels),"ridge_alpha":100.,"crossfit_folds":5,"full_supported_loss_genes":len(full_gene),"shared_half_loss_genes":len(common),"shared_supported_directions":int(valid.sum()),"split_half_pearson":pearson,"split_half_spearman":spearman,"split_half_sign_agreement":sign,"admitted":bool(pearson>=.15 and sign>=.55),"double_perturbation_data_used":False,"sl_labels_used":False}; out.with_suffix(".json").write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2)); return audit

if __name__=="__main__":
    import sys
    adjusted="--adjusted" in sys.argv; build_adjusted(ROOT/"data/depmap24q2/OmicsAbsoluteCNGene.csv",ROOT/"data/depmap24q2/OmicsSomaticMutationsMatrixDamaging.csv",ROOT/"results/sl_predict/natural_loss_adjusted.npz") if adjusted else build(ROOT/"data/depmap24q2/OmicsAbsoluteCNGene.csv",ROOT/"data/depmap24q2/OmicsSomaticMutationsMatrixDamaging.csv",ROOT/"results/sl_predict/natural_loss.npz")
