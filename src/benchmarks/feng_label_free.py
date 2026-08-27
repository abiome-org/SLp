import json, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"


def metric(y,s):
    precision,recall,threshold=precision_recall_curve(y,s); f1=2*precision*recall/(precision+recall+1e-12); k=int(y.sum()); pred=np.zeros(len(y),"int8"); pred[np.argsort(s)[-k:]]=1
    return {"auroc":float(roc_auc_score(y,s)),"average_precision":float(average_precision_score(y,s)),"pr_auc":float(auc(recall,precision)),"max_f1":float(f1.max()),"f1_at_prevalence":float(2*(pred*y).sum()/(pred.sum()+y.sum()))}


@torch.no_grad()
def main():
    sys.path.insert(0,str(ROOT/"src/training")); from world_model import SLPredict, encode_genes
    run=OUT/"native_spectral_safe_scaled_d768_z256_l8_p12_single_only_d3_t10_r3"; state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); split=np.load(OUT/"cv3_benchmarks.npz"); device="cuda" if torch.cuda.is_available() else "cpu"; sd=torch.load(run/"world_model.pt",map_location="cpu",weights_only=True); model=SLPredict(768,256,8,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],sd["context_proj.weight"].shape[1]).to(device); model.load_state_dict(sd); model.eval(); genes=torch.as_tensor(encode_genes(model,state,device),device=device); rows=[]
    for fold in range(5):
        pos=split[f"full_random_1_pos_test_{fold}"].astype("int64"); neg=split[f"full_random_1_neg_test_{fold}"].astype("int64"); pair=np.row_stack((pos,neg)); y=np.r_[np.ones(len(pos),"int8"),np.zeros(len(neg),"int8")]; scores=[]
        for lo in range(0,len(pair),8192):
            p=torch.as_tensor(pair[lo:lo+8192],device=device); scores.append(model.outcome(model.transition(genes[p[:,0]],genes[p[:,1]])[0])[:,1].cpu())
        row={"benchmark":"Feng-2024-CV3-full-random-1","fold":fold,"pairs":len(pair),"positives":len(pos),**metric(y,torch.cat(scores).numpy())}; rows.append(row); print(json.dumps(row),flush=True)
    keys=("auroc","average_precision","pr_auc","max_f1","f1_at_prevalence"); result={"schema":"sl-predict-scaled-single-only-feng-cv3-v1","protocol":"Locked direct_strength score of the frozen 59.7M single-only world on all five official Feng full-data balanced-random CV3 test folds; no SL fitting, calibration, sign/output/context selection, fusion or post-result variant", "model_parameters":sum(p.numel() for p in model.parameters()),"rows":rows,"mean":{k:float(np.mean([r[k] for r in rows])) for k in keys},"standard_deviation":{k:float(np.std([r[k] for r in rows])) for k in keys},"double_perturbation_data_used":False,"sl_labels_used_for_fitting_or_selection":False}; (run/"feng_cv3_direct_strength.json").write_text(json.dumps(result,indent=2)); print(json.dumps({"mean":result["mean"],"standard_deviation":result["standard_deviation"]},indent=2))


if __name__=="__main__": main()
