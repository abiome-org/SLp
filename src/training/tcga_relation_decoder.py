from pathlib import Path
import json,sys
import numpy as np,torch
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; sys.path.insert(0,str(Path(__file__).parent))
from world_model import SLPredict,encode_genes,load_residual_endpoint

class Head(torch.nn.Module):
    def __init__(self,dim=128):
        super().__init__(); self.net=torch.nn.Sequential(torch.nn.LayerNorm(2*dim),torch.nn.Linear(2*dim,128),torch.nn.GELU(),torch.nn.Linear(128,1)); torch.nn.init.zeros_(self.net[-1].weight); torch.nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,a,b): return self.net(torch.cat((x[a]*x[b],(x[a]-x[b]).abs()),1)).squeeze(1)

def fit(epochs=12,epoch_pairs=1000000,batch=8192):
    torch.manual_seed(743); np.random.seed(743); device="cuda" if torch.cuda.is_available() else "cpu"; data=np.load(OUT/"tcga_mutual_exclusivity.npz"); supported=data["genes"].astype("int64"); i,j=np.triu_indices(len(supported),1); i=i.astype("uint16"); j=j.astype("uint16"); held=supported%5==0; train=np.flatnonzero(~held[i]&~held[j]); valid=np.flatnonzero(held[i]&held[j]); mean=float(data["half0"][train].mean()); scale=float(data["half0"][train].std()); target0=(data["half0"]-mean)/scale; target1=(data["half1"]-mean)/scale; state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); model_dir=OUT/"native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; endpoint=load_residual_endpoint(model_dir/"world_model.pt",state.shape[1],device); actions=torch.as_tensor(encode_genes(endpoint.world,state,device)[supported],device=device); head=Head(actions.shape[1]).to(device); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3); history=[]; saved=[]
    @torch.no_grad()
    def evaluate(epoch):
        pred=[]; head.eval()
        for lo in range(0,len(valid),batch):
            ix=valid[lo:lo+batch]; pred.append(head(actions,torch.as_tensor(i[ix].astype("int64"),device=device),torch.as_tensor(j[ix].astype("int64"),device=device)).cpu())
        pred=torch.cat(pred).numpy(); truth=target1[valid]; row={"epoch":epoch,"validation_pairs":len(valid),"huber":float(torch.nn.functional.huber_loss(torch.from_numpy(pred),torch.from_numpy(truth))),"pearson":float(np.corrcoef(truth,pred)[0,1]) if pred.std()>0 else 0.,"spearman":float(spearmanr(truth,pred).statistic) if pred.std()>0 else 0.}; history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()}); print(json.dumps(row),flush=True)
    evaluate(0)
    for epoch in range(epochs):
        rng=np.random.default_rng(743+epoch); chosen=rng.choice(train,epoch_pairs); head.train()
        for lo in range(0,len(chosen),batch):
            ix=chosen[lo:lo+batch]; a=torch.as_tensor(i[ix].astype("int64"),device=device); b=torch.as_tensor(j[ix].astype("int64"),device=device); y=torch.as_tensor(target0[ix],device=device); loss=torch.nn.functional.huber_loss(head(actions,a,b),y); opt.zero_grad(); loss.backward(); opt.step()
        evaluate(epoch+1)
    selected=max(range(len(history)),key=lambda q:(history[q]["pearson"],history[q]["spearman"])); metric=history[selected]; head.load_state_dict(saved[selected]); advanced=metric["pearson"]>=.10 and metric["spearman"]>=.10; torch.save({"state_dict":head.state_dict(),"target_mean":mean,"target_scale":scale,"supported_genes":torch.as_tensor(supported)},model_dir/"tcga_relation_head.pt"); result={"schema":"sl-predict-tcga-relation-decoder-v1","parameters":sum(p.numel() for p in head.parameters()),"training_pairs":len(train),"selected":metric,"advanced":bool(advanced),"double_perturbation_data_used":False,"sl_labels_used":False,"history":history}; (model_dir/"tcga_relation_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2)); return result

def fit_world(model_path,out_dir,epochs=12,epoch_pairs=1000000,batch=8192,d=768,latent=256,layers=8):
    torch.manual_seed(743); np.random.seed(743); device="cuda" if torch.cuda.is_available() else "cpu"; data=np.load(OUT/"tcga_mutual_exclusivity.npz"); supported=data["genes"].astype("int64"); i,j=np.triu_indices(len(supported),1); i=i.astype("uint16"); j=j.astype("uint16"); held=supported%5==0; train=np.flatnonzero(~held[i]&~held[j]); valid=np.flatnonzero(held[i]&held[j]); mean=float(data["half0"][train].mean()); scale=float(data["half0"][train].std()); target0=(data["half0"]-mean)/scale; target1=(data["half1"]-mean)/scale; state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); sd=torch.load(model_path,map_location="cpu",weights_only=True); world=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],sd["context_proj.weight"].shape[1]).to(device); world.load_state_dict(sd); world.eval().requires_grad_(False); actions=torch.as_tensor(encode_genes(world,state,device)[supported],device=device); head=Head(latent).to(device); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3); history=[]; saved=[]
    @torch.no_grad()
    def evaluate(epoch):
        pred=[]; head.eval()
        for lo in range(0,len(valid),batch):
            ix=valid[lo:lo+batch]; pred.append(head(actions,torch.as_tensor(i[ix].astype("int64"),device=device),torch.as_tensor(j[ix].astype("int64"),device=device)).cpu())
        pred=torch.cat(pred).numpy(); truth=target1[valid]; row={"epoch":epoch,"validation_pairs":len(valid),"huber":float(torch.nn.functional.huber_loss(torch.from_numpy(pred),torch.from_numpy(truth))),"pearson":float(np.corrcoef(truth,pred)[0,1]) if pred.std()>0 else 0.,"spearman":float(spearmanr(truth,pred).statistic) if pred.std()>0 else 0.}; history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()}); print(json.dumps(row),flush=True)
    evaluate(0)
    for epoch in range(epochs):
        chosen=np.random.default_rng(743+epoch).choice(train,epoch_pairs); head.train()
        for lo in range(0,len(chosen),batch):
            ix=chosen[lo:lo+batch]; a=torch.as_tensor(i[ix].astype("int64"),device=device); b=torch.as_tensor(j[ix].astype("int64"),device=device); y=torch.as_tensor(target0[ix],device=device); loss=torch.nn.functional.huber_loss(head(actions,a,b),y); opt.zero_grad(); loss.backward(); opt.step()
        evaluate(epoch+1)
    selected=max(range(len(history)),key=lambda q:(history[q]["pearson"],history[q]["spearman"])); metric=history[selected]; head.load_state_dict(saved[selected]); advanced=metric["pearson"]>=.18604423537715637 and metric["spearman"]>=.18417520342635269; out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":head.state_dict(),"target_mean":mean,"target_scale":scale,"supported_genes":torch.as_tensor(supported)},out/"tcga_relation_head.pt"); result={"schema":"sl-predict-scaled-tcga-relation-decoder-v1","parameters":sum(p.numel() for p in head.parameters()),"training_pairs":len(train),"selected":metric,"advanced":bool(advanced),"double_perturbation_data_used":False,"sl_labels_used":False,"history":history}; (out/"tcga_relation_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2)); return result


def fit_pooled(model_path,out_dir,kind="residual",epochs=12,epoch_pairs=1000000,batch=8192,d=768,latent=256,layers=8):
    torch.manual_seed(773); np.random.seed(773); device="cuda" if torch.cuda.is_available() else "cpu"; data=np.load(OUT/"tcga_mutual_exclusivity.npz"); supported=data["genes"].astype("int64"); i,j=np.triu_indices(len(supported),1); i=i.astype("uint16"); j=j.astype("uint16"); held=supported%5==0; train=np.flatnonzero(~held[i]&~held[j]); valid=np.flatnonzero(held[i]&held[j]); pooled=(data["half0"]+data["half1"])/2; mean=float(pooled[train].mean()); scale=float(pooled[train].std()); target=(pooled-mean)/scale; truth={"half0":(data["half0"][valid]-mean)/scale,"half1":(data["half1"][valid]-mean)/scale,"pooled":target[valid]}; state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32")
    if kind=="residual": world=load_residual_endpoint(model_path,state.shape[1],device).world
    else:
        sd=torch.load(model_path,map_location="cpu",weights_only=True); world=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],sd["context_proj.weight"].shape[1]).to(device); world.load_state_dict(sd)
    world.eval().requires_grad_(False); actions=torch.as_tensor(encode_genes(world,state,device)[supported],device=device); head=Head(actions.shape[1]).to(device); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3); history=[]; saved=[]
    @torch.no_grad()
    def evaluate(epoch):
        pred=[]; head.eval()
        for lo in range(0,len(valid),batch):
            ix=valid[lo:lo+batch]; pred.append(head(actions,torch.as_tensor(i[ix].astype("int64"),device=device),torch.as_tensor(j[ix].astype("int64"),device=device)).cpu())
        pred=torch.cat(pred).numpy(); row={"epoch":epoch,"validation_pairs":len(valid)}
        for name,y in truth.items(): row[name]={"huber":float(torch.nn.functional.huber_loss(torch.from_numpy(pred),torch.from_numpy(y))),"pearson":float(np.corrcoef(y,pred)[0,1]) if pred.std()>0 else 0.,"spearman":float(spearmanr(y,pred).statistic) if pred.std()>0 else 0.}
        history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()}); print(json.dumps(row),flush=True)
    evaluate(0)
    for epoch in range(epochs):
        chosen=np.random.default_rng(773+epoch).choice(train,epoch_pairs); head.train()
        for lo in range(0,len(chosen),batch):
            ix=chosen[lo:lo+batch]; a=torch.as_tensor(i[ix].astype("int64"),device=device); b=torch.as_tensor(j[ix].astype("int64"),device=device); y=torch.as_tensor(target[ix],device=device); loss=torch.nn.functional.huber_loss(head(actions,a,b),y); opt.zero_grad(); loss.backward(); opt.step()
        evaluate(epoch+1)
    selected=max(range(len(history)),key=lambda q:(history[q]["pooled"]["pearson"],history[q]["pooled"]["spearman"])); metric=history[selected]; head.load_state_dict(saved[selected]); minimum={"half0":{"pearson":.19822505512334415,"spearman":.20049470929592858},"half1":{"pearson":.18604423542581874,"spearman":.18417520328874357},"pooled":{"pearson":.24961534602680263,"spearman":.24534202386976672}}; advanced=all(metric[name][key]>=minimum[name][key] for name in minimum for key in minimum[name]); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":head.state_dict(),"target_mean":mean,"target_scale":scale,"supported_genes":torch.as_tensor(supported)},out/f"tcga_relation_pooled_{kind}_head.pt"); result={"schema":"sl-predict-tcga-pooled-relation-decoder-v1","representation":kind,"parameters":sum(p.numel() for p in head.parameters()),"training_pairs":len(train),"selected":metric,"minimum":minimum,"advanced":bool(advanced),"double_perturbation_data_used":False,"sl_labels_used":False,"history":history}; (out/f"tcga_relation_pooled_{kind}_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2)); return result


if __name__=="__main__": fit_world(*sys.argv[2:]) if len(sys.argv)>1 and sys.argv[1]=="fit_world" else fit()
