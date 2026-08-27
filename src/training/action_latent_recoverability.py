import json, sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; sys.path.insert(0,str(Path(__file__).parent)); from world_model import SLPredict, encode_genes, load_residual_endpoint

def latent(path,state,device,d,latent,layers,residual=False):
    if residual: model=load_residual_endpoint(path,state.shape[1],device,d,latent,layers).world
    else:
        sd=torch.load(path,map_location="cpu",weights_only=True); model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],sd["context_proj.weight"].shape[1]).to(device); model.load_state_dict(sd); model.eval()
    return encode_genes(model,state,device)

def evaluate(name,x,target,train,valid):
    sx,sy=StandardScaler().fit(x[train]),StandardScaler().fit(target[train]); ytr=sy.transform(target[train]); pred=Ridge(alpha=10).fit(sx.transform(x[train]),ytr).predict(sx.transform(x[valid])).astype("float32"); truth=sy.transform(target[valid]).astype("float32"); zero=float(torch.nn.functional.huber_loss(torch.zeros_like(torch.from_numpy(truth)),torch.from_numpy(truth))); huber=float(torch.nn.functional.huber_loss(torch.from_numpy(pred),torch.from_numpy(truth))); return {"model":name,"latent_dimensions":x.shape[1],"training_genes":len(train),"validation_genes":len(valid),"cosine":float(torch.nn.functional.cosine_similarity(torch.from_numpy(pred),torch.from_numpy(truth)).mean()),"zero_huber":zero,"huber":huber,"huber_improvement":1-huber/zero,"pearson":float(np.corrcoef(pred.ravel(),truth.ravel())[0,1]),"spearman":float(spearmanr(pred.ravel(),truth.ravel()).statistic)}

def main():
    device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); action=np.load(OUT/"se_replogle_gene_features.npz"); known=action["known"]; ids=np.arange(len(known)); train=np.flatnonzero(known&(ids%5!=0)); valid=np.flatnonzero(known&(ids%5==0)); target=action["features"].astype("float32"); compact=OUT/"native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3/world_model.pt"; scaled=OUT/"native_spectral_safe_scaled_d768_z256_l8_p12_single_only_d3_t10_r3/world_model.pt"; rows=[evaluate("compact",latent(compact,state,device,384,128,6,True),target,train,valid),evaluate("scaled",latent(scaled,state,device,768,256,8),target,train,valid)]; admitted=any(r["cosine"]>=.5 and r["huber_improvement"]>=.25 and r["pearson"]>=.5 and r["spearman"]>=.5 for r in rows); preferred="scaled" if rows[1]["cosine"]>=rows[0]["cosine"]+.05 else "compact"; result={"schema":"sl-predict-action-latent-recoverability-v1","models":rows,"preferred":preferred,"admitted":bool(admitted),"double_perturbation_data_used":False,"sl_labels_used":False}; (OUT/"action_latent_recoverability.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
