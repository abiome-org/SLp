import ast, hashlib, json, math, re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from tahoe_transition_source import ROOT, SE, ST, StateEncoder, Transition, load_se, load_transition, action_map

PRISM = ROOT / "data/prism"
EXPR = ROOT / "data/depmap24q2/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
OUT = ROOT / "results/sl_predict/tahoe_prism_validation.json"


def norm(x): return re.sub("[^a-z0-9]", "", str(x).lower()).replace("hydrochloride", "")
def order(x): return hashlib.sha256(str(x).encode()).hexdigest()


def inputs(frame, vocab):
    symbols = [x.rsplit(" (", 1)[0] for x in frame.columns]; index = {g:i for i,g in enumerate(vocab)}
    keep = [i for i,g in enumerate(symbols) if g in index]
    values = frame.iloc[:, keep].to_numpy("float32") * np.float32(math.log(2))
    ids = np.asarray([index[symbols[i]] for i in keep]); top = np.argsort(-values, 1, kind="stable")[:, :2047]
    tokens = np.concatenate((np.full((len(frame),1), index[symbols[keep[3]]]), ids[top]), 1)
    weights = values / values.sum(1, keepdims=True)
    counts = np.concatenate((weights[:,3:4], np.take_along_axis(weights, top, 1)), 1) * 100
    return tokens, counts, len(keep)


def main():
    torch.manual_seed(731); torch.set_grad_enabled(False); device=torch.device("cuda"); dtype=torch.bfloat16
    amap=action_map(); parsed={}
    for key in amap:
        name,dose,_=ast.literal_eval(str(key))[0]; parsed.setdefault(norm(name),[]).append((abs(float(dose)-2.5),str(key),key))
    treatment=pd.read_csv(PRISM/"primary-screen-replicate-collapsed-treatment-info.csv")
    treatment=treatment[treatment.target.notna()].copy(); treatment["normalized"]=treatment.name.map(norm)
    treatment=treatment[treatment.normalized.isin(parsed)]
    treatment["distance"]=(treatment.dose-2.5).abs()
    treatment=treatment.sort_values(["distance","column_name"]).drop_duplicates("normalized")
    treatment=treatment.iloc[np.argsort(treatment.normalized.map(order))[:64]].reset_index(drop=True)
    cells=pd.read_csv(PRISM/"primary-screen-cell-line-info.csv").depmap_id.dropna().astype(str)
    expression=pd.read_csv(EXPR,index_col=0); cells=sorted(set(cells)&set(expression.index),key=order)[:48]
    treatment["action_key"]=[min(parsed[x])[2] for x in treatment.normalized]
    lfc=pd.read_csv(PRISM/"primary-screen-replicate-collapsed-logfold-change.csv",
                    usecols=["Unnamed: 0"]+treatment.column_name.tolist()).set_index("Unnamed: 0").loc[cells]
    frame=expression.loc[cells]; del expression
    vocab=list(torch.load(SE/"protein_embeddings.pt",map_location="cpu",weights_only=True))
    tokens,counts,overlap=inputs(frame,vocab); del frame,vocab
    encoder=StateEncoder(); load_se(encoder); encoder.to(device,dtype=dtype).eval(); basal=[]
    for at in range(0,len(cells),2):
        basal.append(encoder(torch.as_tensor(tokens[at:at+2],device=device),
            torch.as_tensor(counts[at:at+2],device=device,dtype=dtype)).float().cpu())
    basal=torch.cat(basal); del encoder; torch.cuda.empty_cache()
    transition=Transition(); load_transition(transition); transition.to(device,dtype=dtype).eval()
    control=next(k for k in amap if "DMSO_TF', 0.0" in str(k)); actions=[control]+treatment.action_key.tolist()
    pairs=[(a,c) for a in range(len(actions)) for c in range(len(cells))]; predictions=[]
    for at in range(0,len(pairs),16):
        chunk=pairs[at:at+16]; b=torch.stack([basal[c] for _,c in chunk]).to(device,dtype=dtype)
        a=torch.stack([amap[actions[i]] for i,_ in chunk]).to(device,dtype=dtype)
        predictions.append(transition(b[:,None].expand(-1,256,-1),a[:,None].expand(-1,256,-1)).mean(1).float().cpu())
    predictions=torch.cat(predictions).reshape(len(actions),len(cells),-1)
    delta=(predictions[1:]-predictions[:1]).numpy(); del transition,predictions
    y=lfc[treatment.column_name].to_numpy().T.astype("float32")
    train_drug=np.array([int(order(x),16)%2==0 for x in treatment.normalized]); train_cell=np.array([int(order(x),16)%2==0 for x in cells])
    train=np.outer(train_drug,train_cell)&np.isfinite(y); test=np.outer(~train_drug,~train_cell)&np.isfinite(y)
    xtr,ytr=delta[train],y[train]; xte,yte=delta[test],y[test]
    pca=PCA(n_components=min(32,len(ytr)-1),svd_solver="randomized",random_state=731).fit(xtr)
    scaler=StandardScaler().fit(pca.transform(xtr)); model=Ridge(alpha=10).fit(scaler.transform(pca.transform(xtr)),ytr)
    pred=model.predict(scaler.transform(pca.transform(xte))); threshold=float(np.quantile(ytr,.25))
    norm_score=np.linalg.norm(xte,axis=1)
    result={"cells":len(cells),"drugs":len(treatment),"gene_overlap":overlap,
      "train_cells":int(train_cell.sum()),"test_cells":int((~train_cell).sum()),
      "train_drugs":int(train_drug.sum()),"test_drugs":int((~train_drug).sum()),
      "train_rows":len(ytr),"test_rows":len(yte),"pca_components":pca.n_components_,
      "pearson":float(pearsonr(pred,yte).statistic),"spearman":float(spearmanr(pred,yte).statistic),
      "bottom_quartile_threshold":threshold,
      "bottom_quartile_auroc":float(roc_auc_score(yte<threshold,-pred)),
      "decoder_free_effect_norm_pearson_with_sensitivity":float(pearsonr(norm_score,-yte).statistic),
      "decoder_free_effect_norm_spearman_with_sensitivity":float(spearmanr(norm_score,-yte).statistic)}
    result["admitted"]=bool(result["pearson"]>=.1 and result["spearman"]>=.1 and result["bottom_quartile_auroc"]>=.6)
    result["drugs_with_targets"]=[{"name":r["name"],"target":r["target"],"tahoe_action":str(r["action_key"])} for _,r in treatment.iterrows()]
    OUT.write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="drugs_with_targets"},indent=2))
    if result["admitted"]: np.savez_compressed(OUT.with_suffix(".npz"),cells=np.asarray(cells,dtype="U"),drugs=treatment.name.astype(str).to_numpy(dtype="U"),targets=treatment.target.astype(str).to_numpy(dtype="U"),delta=delta.astype("float16"),viability=y)


if __name__=="__main__": main()
