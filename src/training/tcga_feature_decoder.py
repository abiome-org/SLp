from pathlib import Path
import json,sys
import numpy as np,torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; sys.path.insert(0,str(Path(__file__).parent))
from world_model import encode_genes,load_residual_endpoint

class Head(torch.nn.Module):
    def __init__(self,dim=128):
        super().__init__(); self.net=torch.nn.Sequential(torch.nn.LayerNorm(4*dim),torch.nn.Linear(4*dim,128),torch.nn.GELU(),torch.nn.Linear(128,1)); torch.nn.init.zeros_(self.net[-1].weight); torch.nn.init.zeros_(self.net[-1].bias)
    def forward(self,actions,state,a,b):return self.net(torch.cat((actions[a]*actions[b],(actions[a]-actions[b]).abs(),state[a]*state[b],(state[a]-state[b]).abs()),1)).squeeze(1)

def fit(epochs=12,epoch_pairs=1000000,batch=8192):
    torch.manual_seed(759); np.random.seed(759); device="cuda" if torch.cuda.is_available() else "cpu"; data=np.load(OUT/"tcga_mutual_exclusivity.npz"); supported=data["genes"].astype("int64"); i,j=np.triu_indices(len(supported),1); i=i.astype("uint16"); j=j.astype("uint16"); held=supported%5==0; train=np.flatnonzero(~held[i]&~held[j]); valid=np.flatnonzero(held[i]&held[j]); mean=float(data["half0"][train].mean()); scale=float(data["half0"][train].std()); target0=(data["half0"]-mean)/scale; target1=(data["half1"]-mean)/scale; raw=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); pca=PCA(128,whiten=True,svd_solver="randomized",random_state=759).fit(raw[supported[~held]]); feature=torch.as_tensor(pca.transform(raw).astype("float32"),device=device); model_dir=OUT/"native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; endpoint=load_residual_endpoint(model_dir/"world_model.pt",raw.shape[1],device); actions=torch.as_tensor(encode_genes(endpoint.world,raw,device),device=device); head=Head(actions.shape[1]).to(device); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3); history=[]; saved=[]
    @torch.no_grad()
    def evaluate(epoch):
        pred=[]; head.eval()
        for lo in range(0,len(valid),batch):
            ix=valid[lo:lo+batch]; a=torch.as_tensor(supported[i[ix]].astype("int64"),device=device); b=torch.as_tensor(supported[j[ix]].astype("int64"),device=device); pred.append(head(actions,feature,a,b).cpu())
        pred=torch.cat(pred).numpy(); truth=target1[valid]; row={"epoch":epoch,"validation_pairs":len(valid),"huber":float(torch.nn.functional.huber_loss(torch.from_numpy(pred),torch.from_numpy(truth))),"pearson":float(np.corrcoef(truth,pred)[0,1]) if pred.std()>0 else 0.,"spearman":float(spearmanr(truth,pred).statistic) if pred.std()>0 else 0.}; history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()}); print(json.dumps(row),flush=True)
    evaluate(0)
    for epoch in range(epochs):
        chosen=np.random.default_rng(759+epoch).choice(train,epoch_pairs); head.train()
        for lo in range(0,len(chosen),batch):
            ix=chosen[lo:lo+batch]; a=torch.as_tensor(supported[i[ix]].astype("int64"),device=device); b=torch.as_tensor(supported[j[ix]].astype("int64"),device=device); y=torch.as_tensor(target0[ix],device=device); loss=torch.nn.functional.huber_loss(head(actions,feature,a,b),y); opt.zero_grad(); loss.backward(); opt.step()
        evaluate(epoch+1)
    selected=max(range(len(history)),key=lambda q:(history[q]["pearson"],history[q]["spearman"])); metric=history[selected]; head.load_state_dict(saved[selected]); advanced=metric["pearson"]>=.19604423537715636 and metric["spearman"]>=.19417520342635268; torch.save({"state_dict":head.state_dict(),"pca_state":feature.cpu(),"target_mean":mean,"target_scale":scale},model_dir/"tcga_feature_relation_head.pt"); result={"schema":"sl-predict-tcga-feature-decoder-v1","parameters":sum(p.numel() for p in head.parameters()),"pca_dimensions":128,"pca_fit_genes":int((~held).sum()),"pca_explained_variance":float(pca.explained_variance_ratio_.sum()),"training_pairs":len(train),"selected":metric,"advanced":bool(advanced),"world_parameters_changed":0,"double_perturbation_data_used":False,"sl_labels_used":False,"history":history}; (model_dir/"tcga_feature_relation_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2)); return result

if __name__=="__main__":fit()
