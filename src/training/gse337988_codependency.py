import json,sys
from pathlib import Path
import numpy as np,torch
from scipy.stats import pearsonr,spearmanr
from torch import nn

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; RUN=OUT/"native_spectral_safe_scaled_d768_z256_l8_p12_single_only_d3_t10_r3"; sys.path.insert(0,str(Path(__file__).parent))
from world_model import SLPredict,batches,encode_genes

def main():
    device="cuda" if torch.cuda.is_available() else "cpu"; data=np.load(OUT/"gse337988_moi_state.npz",allow_pickle=True); dep=np.load(OUT/"depmap_codependency.npz"); state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); basal=np.load(OUT/"basal_context.npz"); sd=torch.load(RUN/"world_model.pt",map_location="cpu",weights_only=True)
    world=SLPredict(768,256,8,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],sd["context_proj.weight"].shape[1]).to(device); world.load_state_dict(sd); world.eval().requires_grad_(False); head=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256),nn.GELU(),nn.Linear(256,64)).to(device); head.load_state_dict(torch.load(RUN/"gse337988_single_endpoint.pt",map_location="cpu",weights_only=True)); head.eval().requires_grad_(False)
    held=np.unique(data["gene"][data["role"]==1]).astype("int64"); pos={int(g):i for i,g in enumerate(dep["genes"])}; genes=np.asarray([g for g in held if g in pos],"int64"); dep_pos=np.asarray([pos[int(g)] for g in genes]); observed=np.stack([data["target_med"][(data["role"]==1)&(data["gene"]==g)].mean(0) for g in genes]); observed/=np.linalg.norm(observed,axis=1,keepdims=True).clip(1e-8)
    encoded=torch.as_tensor(encode_genes(world,state,device),device=device); context=torch.as_tensor(basal["cell_state"][np.flatnonzero(basal["model_ids"].astype(str)=="ACH-001061")[0]],device=device); predicted=[]
    with torch.no_grad():
        for at in batches(len(state),2048,False):
            ix=at.to(device); z=world.transition(encoded[ix],context_state=context[None].expand(len(ix),-1))[0]; predicted.append(nn.functional.normalize(head(z),dim=1).cpu())
    predicted=torch.cat(predicted).numpy(); n=len(genes); i,j=np.triu_indices(n,1); scores={"observed":np.sum(observed[i]*observed[j],1),"predicted":np.sum(predicted[genes[i]]*predicted[genes[j]],1)}; metrics=[]
    for name,score in scores.items():
        for half in ("half0","half1"):
            target=dep[half][dep_pos[i],dep_pos[j]].astype("float32"); metrics.append({"score":name,"half":half,"pearson":float(pearsonr(score,target).statistic),"spearman":float(spearmanr(score,target).statistic)})
    admitted=all(x[m]>=.15 for x in metrics for m in ("pearson","spearman")); result={"schema":"sl-predict-gse337988-codependency-v1","eligible_held_genes":n,"pairs":len(i),"metrics":metrics,"admitted":bool(admitted),"double_perturbation_data_used":False,"sl_labels_used":False}; (OUT/"gse337988_codependency.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    if admitted:np.savez_compressed(OUT/"gse337988_codependency.npz",genes=np.arange(len(state),dtype="int16"),profile=predicted.astype("float16"),held=genes.astype("int16"))

if __name__=="__main__":main()
