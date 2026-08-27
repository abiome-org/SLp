import csv, hashlib, json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"results/sl_predict/tahoe_prism_validation.npz"
SOURCE_META=ROOT/"results/sl_predict/tahoe_prism_validation.json"
FEATURES=ROOT/"results/sl_predict/features_spectral_safe.npz"
META=ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"
OUT=ROOT/"results/sl_predict/tahoe_gene_projection.json"

def order(x): return hashlib.sha256(str(x).encode()).hexdigest()

def main():
    source=np.load(SOURCE,allow_pickle=True); info=json.loads(SOURCE_META.read_text())
    state=np.load(FEATURES)["state"].astype("float32"); genes=[r["symbol"] for r in csv.DictReader(open(META))]
    index={g:i for i,g in enumerate(genes)}; targets=[str(x).split(", ") for x in source["targets"]]
    supported=np.array([all(g in index for g in t) for t in targets]); x=np.stack([state[[index[g] for g in t]].mean(0) for t,k in zip(targets,supported) if k])
    response=source["delta"].astype("float32").mean(1)[supported]
    names=source["drugs"].astype(str)[supported]; targets=[t for t,k in zip(targets,supported) if k]
    train=np.array([int(order(n),16)%2==0 for n in names]); pca=PCA(n_components=min(16,train.sum()-1),random_state=731).fit(response[train])
    y=pca.transform(response); scaler=StandardScaler().fit(x[train]); model=Ridge(alpha=10).fit(scaler.transform(x[train]),y[train])
    pred=model.predict(scaler.transform(x[~train])); truth=y[~train]
    cosine=np.sum(pred*truth,1)/(np.linalg.norm(pred,axis=1)*np.linalg.norm(truth,axis=1)+1e-8)
    train_targets=set(sum([t for t,k in zip(targets,train) if k],[])); isolated=np.array([not bool(set(t)&train_targets) for t,k in zip(targets,~train) if k])
    def metrics(a,b):
        c=np.sum(a*b,1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-8)
        return {"drugs":len(a),"mean_cosine":float(c.mean()),"pearson":float(pearsonr(a.ravel(),b.ravel()).statistic),"spearman":float(spearmanr(a.ravel(),b.ravel()).statistic)}
    result={"source_drugs":len(source["drugs"]),"supported_drugs":len(names),"train_drugs":int(train.sum()),"held_drugs":int((~train).sum()),
            "components":pca.n_components_,"held":metrics(pred,truth),"target_isolated_held":metrics(pred[isolated],truth[isolated]) if isolated.any() else {"drugs":0}}
    result["admitted"]=bool(result["held"]["mean_cosine"]>=.15 and result["held"]["spearman"]>=.15 and result["target_isolated_held"]["drugs"]>=5 and result["target_isolated_held"]["mean_cosine"]>0)
    OUT.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    if result["admitted"]:
        profile=model.predict(scaler.transform(state)).astype("float16")
        np.savez_compressed(OUT.with_suffix(".npz"),genes=np.asarray(genes,dtype="U"),profile=profile,reliability=np.float32(result["held"]["mean_cosine"]))

if __name__=="__main__": main()
