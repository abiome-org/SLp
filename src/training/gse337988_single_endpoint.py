import json,sys
from pathlib import Path
import numpy as np,torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; sys.path.insert(0,str(Path(__file__).parent))
from world_model import SLPredict,batches,encode_genes

def grouped(data,role):
    rows=np.flatnonzero(data["role"]==role); genes=np.unique(data["gene"][rows]); target=np.stack([data["target_med"][rows[data["gene"][rows]==g]].mean(0) for g in genes]).astype("float32"); return genes.astype("int64"),target

def run(epochs=16):
    torch.manual_seed(104); np.random.seed(104); device="cuda" if torch.cuda.is_available() else "cpu"; data=np.load(OUT/"gse337988_moi_state.npz",allow_pickle=True); train_gene,train_target=grouped(data,0); valid_gene,valid_target=grouped(data,1); assert not set(train_gene)&set(valid_gene)
    state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); basal=np.load(OUT/"basal_context.npz"); row=np.flatnonzero(basal["model_ids"].astype(str)=="ACH-001061"); assert len(row)==1
    model_dir=OUT/"native_spectral_safe_scaled_d768_z256_l8_p12_single_only_d3_t10_r3"; sd=torch.load(model_dir/"world_model.pt",map_location="cpu",weights_only=True); world=SLPredict(768,256,8,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],sd["context_proj.weight"].shape[1]).to(device); world.load_state_dict(sd); world.eval().requires_grad_(False)
    genes=torch.as_tensor(encode_genes(world,state,device),device=device); context=torch.as_tensor(basal["cell_state"][row[0]],device=device); head=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256),nn.GELU(),nn.Linear(256,64)).to(device); nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3); history=[]; saved=[]
    @torch.no_grad()
    def latent(index,exact):
        out=[]
        for at in batches(len(index),2048,False):
            cs=context[None].expand(len(at),-1) if exact else None; out.append(world.transition(genes[torch.as_tensor(index[at],device=device)],context_state=cs)[0].cpu())
        return torch.cat(out)
    train_z=(latent(train_gene,False),latent(train_gene,True)); valid_z=(latent(valid_gene,False),latent(valid_gene,True)); truth=torch.as_tensor(valid_target)
    @torch.no_grad()
    def evaluate(epoch):
        row={"epoch":epoch}
        for name,z in zip(("unknown","exact"),valid_z):
            pred=head(z.to(device)).cpu(); huber=float(nn.functional.huber_loss(pred,truth)); zero=float(nn.functional.huber_loss(torch.zeros_like(truth),truth)); cosine=float(nn.functional.cosine_similarity(pred,truth).mean()); row[name]={"huber":huber,"zero_huber":zero,"improvement":1-huber/zero,"cosine":cosine}
        row["selection_loss"]=row["unknown"]["huber"]+row["exact"]["huber"]; history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()}); print(json.dumps(row),flush=True)
    evaluate(0); target=torch.as_tensor(train_target)
    for epoch in range(epochs):
        order=np.random.default_rng(104+epoch).permutation(len(train_gene)); head.train()
        for at in batches(len(order),512,False):
            ix=order[at]; exact=torch.rand(len(ix))<.5; z=torch.where(exact[:,None],train_z[1][ix],train_z[0][ix]).to(device); loss=nn.functional.huber_loss(head(z),target[ix].to(device)); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(head.parameters(),1.); opt.step()
        head.eval(); evaluate(epoch+1)
    selected=min(range(len(history)),key=lambda i:history[i]["selection_loss"]); chosen=history[selected]; head.load_state_dict(saved[selected]); advanced=selected>0 and all(chosen[c]["improvement"]>=.01 and chosen[c]["cosine"]>=.15 for c in ("unknown","exact")); torch.save(head.state_dict(),model_dir/"gse337988_single_endpoint.pt"); result={"schema":"sl-predict-gse337988-single-endpoint-v1","parameters":sum(p.numel() for p in head.parameters()),"fitting_genes":len(train_gene),"held_genes":len(valid_gene),"selected":chosen,"advanced":bool(advanced),"world_parameters_changed":0,"double_perturbation_data_used":False,"sl_labels_used":False,"history":history}; (model_dir/"gse337988_single_endpoint_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2))

if __name__=="__main__":run()
