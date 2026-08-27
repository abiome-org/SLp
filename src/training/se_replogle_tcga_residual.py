import json, sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; RUN=OUT/"native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; sys.path.insert(0,str(Path(__file__).parent))
from tcga_relation_decoder import Head
from world_model import encode_genes, load_residual_endpoint

class Residual(torch.nn.Module):
    def __init__(self,dim=65):
        super().__init__(); self.net=torch.nn.Sequential(torch.nn.LayerNorm(2*dim),torch.nn.Linear(2*dim,64),torch.nn.GELU(),torch.nn.Linear(64,1)); torch.nn.init.zeros_(self.net[-1].weight); torch.nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,a,b): return self.net(torch.cat((x[a]*x[b],(x[a]-x[b]).abs()),1)).squeeze(1)

def main(epochs=12,epoch_pairs=1_000_000,batch=8192):
    torch.manual_seed(787); np.random.seed(787); device="cuda" if torch.cuda.is_available() else "cpu"; data=np.load(OUT/"tcga_mutual_exclusivity.npz"); saved=torch.load(RUN/"tcga_relation_head.pt",map_location="cpu",weights_only=True); supported=data["genes"].astype("int64"); i,j=np.triu_indices(len(supported),1); i=i.astype("uint16"); j=j.astype("uint16"); held=supported%5==0; train=np.flatnonzero(~held[i]&~held[j]); valid=np.flatnonzero(held[i]&held[j]); target0=(data["half0"]-saved["target_mean"])/saved["target_scale"]; target1=(data["half1"]-saved["target_mean"])/saved["target_scale"]
    state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); endpoint=load_residual_endpoint(RUN/"world_model.pt",state.shape[1],device); actions=torch.as_tensor(encode_genes(endpoint.world,state,device)[supported],device=device); baseline=Head(actions.shape[1]).to(device); baseline.load_state_dict(saved["state_dict"]); baseline.eval().requires_grad_(False)
    action=np.load(OUT/"se_replogle_gene_features.npz"); extra=torch.as_tensor(np.column_stack((action["features"].astype("float32"),action["known"].astype("float32")))[supported],device=device); residual=Residual(extra.shape[1]).to(device); opt=torch.optim.AdamW(residual.parameters(),3e-4,weight_decay=1e-3); history=[]; states=[]
    @torch.no_grad()
    def evaluate(epoch):
        base=[]; pred=[]; residual.eval()
        for lo in range(0,len(valid),batch):
            ix=valid[lo:lo+batch]; a=torch.as_tensor(i[ix].astype("int64"),device=device); b=torch.as_tensor(j[ix].astype("int64"),device=device); z=baseline(actions,a,b); base.append(z.cpu()); pred.append((z+residual(extra,a,b)).cpu())
        truth=target1[valid]; base=torch.cat(base).numpy(); pred=torch.cat(pred).numpy()
        def metric(x): return {"huber":float(torch.nn.functional.huber_loss(torch.from_numpy(x),torch.from_numpy(truth))),"pearson":float(np.corrcoef(truth,x)[0,1]) if x.std()>0 else 0.,"spearman":float(spearmanr(truth,x).statistic) if x.std()>0 else 0.}
        row={"epoch":epoch,"validation_pairs":len(valid),"baseline":metric(base),"candidate":metric(pred)}; history.append(row); states.append({k:v.detach().cpu().clone() for k,v in residual.state_dict().items()}); print(json.dumps(row),flush=True)
    evaluate(0)
    for epoch in range(epochs):
        chosen=np.random.default_rng(787+epoch).choice(train,epoch_pairs); residual.train()
        for lo in range(0,len(chosen),batch):
            ix=chosen[lo:lo+batch]; a=torch.as_tensor(i[ix].astype("int64"),device=device); b=torch.as_tensor(j[ix].astype("int64"),device=device); y=torch.as_tensor(target0[ix],device=device)
            with torch.no_grad(): base=baseline(actions,a,b)
            loss=torch.nn.functional.huber_loss(base+residual(extra,a,b),y); opt.zero_grad(); loss.backward(); opt.step()
        evaluate(epoch+1)
    selected=max(range(len(history)),key=lambda q:(history[q]["candidate"]["pearson"],history[q]["candidate"]["spearman"])); row=history[selected]; base=row["baseline"]; candidate=row["candidate"]; residual.load_state_dict(states[selected]); reconstructed=history[0]["baseline"]; assert abs(reconstructed["pearson"]-.17604423537715636)<1e-6 and abs(reconstructed["spearman"]-.17417520342635268)<1e-6
    admitted=candidate["pearson"]>=base["pearson"]+.01 and candidate["spearman"]>=base["spearman"]+.01 and candidate["huber"]<=base["huber"]; artifact={"state_dict":residual.state_dict(),"supported_genes":torch.as_tensor(supported),"target_mean":saved["target_mean"],"target_scale":saved["target_scale"]}; torch.save(artifact,RUN/"se_replogle_tcga_residual_head.pt"); result={"schema":"sl-predict-se-replogle-tcga-residual-v1","parameters":sum(p.numel() for p in residual.parameters()),"training_pairs":len(train),"known_supported_genes":int(action["known"][supported].sum()),"selected":row,"pearson_gain":candidate["pearson"]-base["pearson"],"spearman_gain":candidate["spearman"]-base["spearman"],"admitted":bool(admitted),"double_perturbation_data_used":False,"sl_labels_used":False,"history":history}; (RUN/"se_replogle_tcga_residual_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2))

if __name__=="__main__": main()
