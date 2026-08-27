from pathlib import Path
import csv,json
import h5py,numpy as np

ROOT=Path(__file__).resolve().parents[2]

def build(output=ROOT/"results/sl_predict/rnai_dependency.npz"):
    rows=list(csv.DictReader(open(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"))); gene={x["symbol"].upper():i for i,x in enumerate(rows)}; basal=np.load(ROOT/"results/sl_predict/basal_context.npz"); br={x:i for i,x in enumerate(basal["model_ids"].astype(str))}
    with h5py.File(ROOT/"data/rnai/RNAiTarget.hdf5") as f:
        source_model=np.asarray([x.decode() for x in f["dim_0"]]); source_gene=np.asarray([x.decode().rsplit(" (",1)[0].upper() for x in f["dim_1"]]); model=np.asarray([x for x in source_model if x in br]); sr={x:i for i,x in enumerate(source_model)}; gi=np.asarray([i for i,x in enumerate(source_gene) if x in gene]); target=np.asarray([gene[source_gene[i]] for i in gi],"int16"); rnai=np.asarray(f["data"][[sr[x] for x in model]][:,gi],"float32")
    dep=basal["dependency"][[br[x] for x in model]].astype("float32"); known=basal["dependency_known"][[br[x] for x in model]]; correlations=[]; products=0.; observations=0; eligible=0
    for j,g in enumerate(target):
        keep=np.isfinite(rnai[:,j])&known[:,g]
        if keep.sum()<100:continue
        a=rnai[keep,j]; b=dep[keep,g]; a=(a-a.mean())/a.std().clip(.1); b=(b-b.mean())/b.std().clip(.1); r=float(np.mean(a*b)); correlations.append(r); products+=float(a@b); observations+=len(a); eligible+=1
    pooled=products/observations; median=float(np.median(correlations)); admitted=bool(pooled>=.15 and median>=.10); out=Path(output); np.savez_compressed(out,model_ids=model,gene_index=target,rnai=rnai.astype("float16"),known=np.isfinite(rnai)); audit={"schema":"sl-predict-rnai-dependency-v1","source_models":len(source_model),"source_genes":len(source_gene),"aligned_models":len(model),"aligned_genes":len(target),"genes_with_100_paired_models":eligible,"paired_observations":observations,"pooled_crispr_rnai_pearson":pooled,"median_gene_pearson":median,"admitted":admitted,"double_perturbation_data_used":False,"sl_labels_used":False}; out.with_suffix(".json").write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2)); return audit

if __name__=="__main__":build()
