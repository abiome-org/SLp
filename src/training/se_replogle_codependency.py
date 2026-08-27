import csv, json, math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from scipy.stats import pearsonr, spearmanr
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; MODEL=OUT/"se_replogle_action_calibrated.npz"

def main():
    rows=list(csv.DictReader(open(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"))); genes=np.asarray([r["symbol"].upper() for r in rows]);
    ens={r["ensembl_gene_id"].split(".")[0]:r["symbol"].upper() for r in rows if r["ensembl_gene_id"]}; state=np.load(OUT/"se_replogle_state.npz"); trained=set(state["gene"][state["role"]=="train"].astype(str)); model=np.load(MODEL)
    vocab=list(torch.load(ROOT/"data/models/weights/SE-600M/protein_embeddings.pt",map_location="cpu",weights_only=True)); vid={g:i for i,g in enumerate(vocab)}; known=np.asarray([g in vid for g in genes]); token=np.zeros((len(genes),2048),"float32")
    with safe_open(str(ROOT/"data/models/weights/SE-600M/model.safetensors"),framework="pt",device="cpu") as f:
        pe=f.get_tensor("pe_embedding.weight"); w=f.get_tensor("encoder.0.weight").cuda().to(torch.bfloat16); b=f.get_tensor("encoder.0.bias").cuda().to(torch.bfloat16)
        lw=f.get_tensor("encoder.1.weight").cuda().to(torch.bfloat16); lb=f.get_tensor("encoder.1.bias").cuda().to(torch.bfloat16); ids=np.flatnonzero(known)
        for at in range(0,len(ids),256):
            ix=ids[at:at+256]; x=F.normalize(pe[torch.as_tensor([vid[genes[i]] for i in ix])].cuda().to(torch.bfloat16),dim=1)
            x=F.silu(F.layer_norm(F.linear(x,w,b),(2048,),lw,lb))*math.sqrt(2048); token[ix]=x.float().cpu().numpy()
    profiles=[]; coordinates=[]
    for sid,name in enumerate(("replogle2022_k562","replogle2022_rpe1")):
        z=np.load(ROOT/f"data/perturbseq_sources/base/{name}.npz"); symbols=[ens.get(str(x).split(".")[0],str(x).upper()) for x in z["feature_name"]]; fmap={g:i for i,g in enumerate(symbols)}
        expression=np.asarray([z["control_mean"][fmap[g]] if g in fmap else 0 for g in genes],"float32")[:,None]; x=np.column_stack((token,expression))
        standardized=(x-model["scaler_mean"][sid])/model["scaler_scale"][sid]; coords=standardized@model["ridge_coef"][sid].T+model["ridge_intercept"][sid]
        coords=model["shrinkage"][sid]*coords; coordinates.append(coords.astype("float32")); response=coords@model["pca_components"][sid]+model["shrinkage"][sid]*model["pca_mean"][sid]; response/=np.linalg.norm(response,axis=1,keepdims=True).clip(1e-8); profiles.append(response.astype("float32"))
    np.savez_compressed(OUT/"se_replogle_gene_features.npz",genes=genes,features=np.column_stack(coordinates).astype("float16"),known=known)
    dep=np.load(OUT/"depmap_codependency.npz"); dep_genes=dep["genes"].astype("int64"); held_pos=np.asarray([i for i,g in enumerate(dep_genes) if known[g] and genes[g] not in trained]); n=len(held_pos); total=n*(n-1)//2; count=min(2_000_000,total); k=np.random.default_rng(731).choice(total,count,replace=False); q=2*n-1
    i=np.floor((q-np.sqrt(q*q-8*k))/2).astype("int64"); before=i*(2*n-i-1)//2; j=k-before+i+1; pi,pj=held_pos[i],held_pos[j]; gi,gj=dep_genes[pi],dep_genes[pj]
    source_scores=[]
    for profile in profiles:
        score=np.empty(count,"float32")
        for at in range(0,count,100000): score[at:at+100000]=np.sum(profile[gi[at:at+100000]]*profile[gj[at:at+100000]],1)
        source_scores.append(score)
    score=np.mean(source_scores,axis=0); metrics=[]
    for half in ("half0","half1"):
        target=dep[half][pi,pj].astype("float32"); metrics.append({"half":half,"pearson":float(pearsonr(score,target).statistic),"spearman":float(spearmanr(score,target).statistic)})
    result={"eligible_genes":len(dep_genes),"genes_absent_from_replogle_fitting":n,"sample_pairs":count,"source_score_pearson":float(pearsonr(*source_scores).statistic),
      "metrics":metrics,"admitted":bool(all(m["pearson"]>=.15 and m["spearman"]>=.15 for m in metrics)),"double_perturbation_data_used":False,"sl_labels_used":False}
    path=OUT/"se_replogle_codependency.json"; path.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    if result["admitted"]: np.savez_compressed(path.with_suffix(".npz"),genes=np.flatnonzero(known).astype("int16"),k562=profiles[0][known].astype("float16"),rpe1=profiles[1][known].astype("float16"))

if __name__=="__main__": main()
