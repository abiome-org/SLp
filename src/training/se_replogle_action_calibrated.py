import hashlib, json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/"results/sl_predict/se_replogle_state.npz"; OUT=ROOT/"results/sl_predict/se_replogle_action_calibrated.json"

def huber(x):
    a=np.abs(x); return float(np.where(a<1,.5*a*a,a-.5).mean())
def fold(x): return int.from_bytes(hashlib.sha256(str(x).encode()).digest()[:4],"big")%5

def main():
    z=np.load(DATA); x=z["features"].astype("float32"); response=z["response"].astype("float32"); rows=[]; models=[]
    for sid,name in enumerate(("replogle2022_k562","replogle2022_rpe1")):
        source=z["source"]==sid; train=source&(z["role"]=="train"); confirm=source&(z["role"]=="intrinsic_validation")
        pca=PCA(n_components=32,svd_solver="randomized",random_state=731).fit(response[train]); y=pca.transform(response); oof=np.zeros((train.sum(),response.shape[1]),"float32")
        tr=np.flatnonzero(train); gene=z["gene"][train].astype(str); folds=np.asarray([fold(g) for g in gene])
        for f in range(5):
            fit=tr[folds!=f]; valid=tr[folds==f]; scaler=StandardScaler().fit(x[fit]); ridge=Ridge(alpha=10).fit(scaler.transform(x[fit]),y[fit]); oof[folds==f]=pca.inverse_transform(ridge.predict(scaler.transform(x[valid])))
        truth=response[train]; shrink=float(np.clip(np.sum(oof*truth)/(np.sum(oof*oof)+1e-12),0,1))
        scaler=StandardScaler().fit(x[train]); ridge=Ridge(alpha=10).fit(scaler.transform(x[train]),y[train]); pred=shrink*pca.inverse_transform(ridge.predict(scaler.transform(x[confirm]))); truth=response[confirm]
        cosine=np.sum(pred*truth,1)/(np.linalg.norm(pred,axis=1)*np.linalg.norm(truth,axis=1)+1e-8); zero=huber(truth); loss=huber(pred-truth)
        rows.append({"source":name,"train_genes":int(train.sum()),"confirmation_genes":int(confirm.sum()),"cross_fitted_shrinkage":shrink,
          "mean_cosine":float(cosine.mean()),"median_cosine":float(np.median(cosine)),"zero_huber":zero,"huber":loss,"huber_improvement":1-loss/zero,
          "effect_magnitude_spearman":float(spearmanr(np.linalg.norm(pred,axis=1),np.linalg.norm(truth,axis=1)).statistic)})
        models.append((pca,scaler,ridge,shrink))
    improvement=float(np.mean([r["huber_improvement"] for r in rows])); result={"sources":rows,"source_macro_cosine":float(np.mean([r["mean_cosine"] for r in rows])),
      "source_macro_huber_improvement":improvement,"admitted":bool(all(r["mean_cosine"]>=.1 and r["huber_improvement"]>0 for r in rows) and improvement>=.05),
      "double_perturbation_data_used":False,"sl_labels_used":False}
    OUT.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    if result["admitted"]:
        np.savez_compressed(OUT.with_suffix(".npz"),pca_components=np.stack([m[0].components_ for m in models]),pca_mean=np.stack([m[0].mean_ for m in models]),
          scaler_mean=np.stack([m[1].mean_ for m in models]),scaler_scale=np.stack([m[1].scale_ for m in models]),ridge_coef=np.stack([m[2].coef_ for m in models]),
          ridge_intercept=np.stack([m[2].intercept_ for m in models]),shrinkage=np.asarray([m[3] for m in models],"float32"))

if __name__=="__main__": main()
