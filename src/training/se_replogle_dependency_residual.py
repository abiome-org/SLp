import json, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; RUN=OUT/"native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"
sys.path.insert(0,str(Path(__file__).parent)); from world_model import batches, dependency_landscape_head, encode_genes, load_residual_endpoint

def metric(pred,truth):
    return {"genes":len(pred),"huber":float(torch.nn.functional.huber_loss(torch.from_numpy(pred),torch.from_numpy(truth))),"cosine":float(torch.nn.functional.cosine_similarity(torch.from_numpy(pred),torch.from_numpy(truth)).mean()),"correlation":float(np.corrcoef(pred.ravel(),truth.ravel())[0,1])}

def main():
    device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); dep=np.load(OUT/"dependency_landscape.npz"); action=np.load(OUT/"se_replogle_gene_features.npz")
    endpoint=load_residual_endpoint(RUN/"world_model.pt",state.shape[1],device); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); decoder=dependency_landscape_head(128,16).to(device); decoder.load_state_dict(torch.load(RUN/"dependency_core_endpoint.pt",map_location="cpu",weights_only=True)); decoder.eval()
    pred=[]
    with torch.no_grad():
        for ix in batches(len(genes),2048,False): pred.append(decoder(endpoint.world.transition(genes[ix])[0]).cpu())
    baseline=torch.cat(pred).numpy(); target=dep["target"][:,:16].astype("float32"); train,valid,excluded=map(np.flatnonzero,(dep["train"],dep["valid"],dep["excluded"])); base={"generic":metric(baseline[valid],target[valid]),"intervention_isolated":metric(baseline[excluded],target[excluded])}; reference=json.loads((RUN/"dependency_core_endpoint_metrics.json").read_text())
    assert abs(base["generic"]["huber"]-reference["selected"]["huber"])<1e-6 and abs(base["intervention_isolated"]["huber"]-reference["intervention_isolated"]["huber"])<1e-6
    features=np.column_stack((action["features"].astype("float32"),action["known"].astype("float32"))); scaler=StandardScaler().fit(features[train]); ridge=Ridge(alpha=10).fit(scaler.transform(features[train]),target[train]-baseline[train]); candidate=baseline+ridge.predict(scaler.transform(features)).astype("float32"); cand={"generic":metric(candidate[valid],target[valid]),"intervention_isolated":metric(candidate[excluded],target[excluded])}
    improved={name:{"huber_fraction":cand[name]["huber"]/base[name]["huber"],"cosine_delta":cand[name]["cosine"]-base[name]["cosine"]} for name in base}; admitted=all(x["huber_fraction"]<=.99 and x["cosine_delta"]>=0 for x in improved.values())
    result={"training_genes":len(train),"features":features.shape[1],"known_tokens":int(action["known"].sum()),"baseline":base,"candidate":cand,"comparison":improved,"admitted":bool(admitted),"sl_labels_used":False,"double_perturbation_data_used":False}; (OUT/"se_replogle_dependency_residual.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    if admitted: np.savez_compressed(OUT/"se_replogle_dependency_residual.npz",scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,coef=ridge.coef_,intercept=ridge.intercept_)

if __name__=="__main__": main()
