from pathlib import Path
import hashlib,json,sys
import numpy as np,torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(Path(__file__).parent))
from world_model import encode_genes,load_residual_endpoint,tolerance_head

@torch.no_grad()
def evaluate(world,head,genes,data,cells,gene_ids,device):
    c=np.repeat(cells,len(gene_ids)); g=np.tile(gene_ids,len(cells)); keep=data["dependency_known"][c,g]; c,g=c[keep],g[keep]; pred=[]
    for at in range(0,len(c),4096):
        ci=torch.as_tensor(c[at:at+4096],device=device); gi=torch.as_tensor(g[at:at+4096],device=device); cs=torch.as_tensor(data["cell_state"][c[at:at+4096]],device=device); pred.append(head(world.transition(genes[gi],context_state=cs)[0]).squeeze(1).cpu())
    pred=torch.cat(pred).numpy(); target=data["dependency"][c,g].astype("float32"); huber=float(torch.nn.functional.huber_loss(torch.from_numpy(pred),torch.from_numpy(target))); zero=float(torch.nn.functional.huber_loss(torch.zeros(len(target)),torch.from_numpy(target))); return {"observations":len(target),"zero_huber":zero,"huber":huber,"improvement":1-huber/zero,"pearson":float(np.corrcoef(target,pred)[0,1]),"spearman":float(spearmanr(target,pred).statistic),"dependency_auroc":float(roc_auc_score(target<-.5,-pred))}

def fit(epochs=8,epoch_pairs=500000):
    torch.manual_seed(731); np.random.seed(731); device="cuda" if torch.cuda.is_available() else "cpu"; data=np.load(ROOT/"results/sl_predict/depmap_tolerance.npz"); state=np.load(ROOT/"results/sl_predict/features_spectral_safe.npz")["state"].astype("float32"); model_dir=ROOT/"results/sl_predict/native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; endpoint=load_residual_endpoint(model_dir/"world_model.pt",state.shape[1],device); world=endpoint.world; genes=torch.as_tensor(encode_genes(world,state,device),device=device); cell_held=np.asarray([hashlib.sha256(x.encode()).digest()[0]%5==0 for x in data["model_ids"].astype(str)]); gene_held=np.arange(len(state))%5==0; train_cells=np.flatnonzero(data["train_cell"]&~cell_held); valid_cells=np.flatnonzero(data["train_cell"]&cell_held); train_genes=np.flatnonzero(data["train_gene"]&~gene_held); valid_genes=np.flatnonzero(data["train_gene"]&gene_held); head=tolerance_head(genes.shape[1]).to(device); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3); best=None; best_metrics=None; history=[]
    for epoch in range(epochs):
        rng=np.random.default_rng(731+epoch); head.train(); total=seen=0
        while seen<epoch_pairs:
            c=rng.choice(train_cells,4096); g=rng.choice(train_genes,4096); keep=data["dependency_known"][c,g]; c,g=c[keep],g[keep]
            if not len(c):continue
            ci=torch.as_tensor(c,device=device); gi=torch.as_tensor(g,device=device); target=torch.as_tensor(data["dependency"][c,g].astype("float32"),device=device)
            with torch.no_grad():z=world.transition(genes[gi],context_state=torch.as_tensor(data["cell_state"][c],device=device))[0]
            raw=torch.nn.functional.huber_loss(head(z).squeeze(1),target,reduction="none"); weight=1+2*(target<-.5); loss=(raw*weight).sum()/weight.sum(); opt.zero_grad(); loss.backward(); opt.step(); total+=raw.sum().item(); seen+=len(c)
        head.eval(); metrics=evaluate(world,head,genes,data,valid_cells,valid_genes,device); row={"epoch":epoch+1,"train_huber":total/seen,**metrics}; history.append(row); print(json.dumps(row),flush=True)
        if best_metrics is None or metrics["huber"]<best_metrics["huber"]:best={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}; best_metrics=row
    head.load_state_dict(best); advanced=best_metrics["improvement"]>=.05 and best_metrics["pearson"]>=.30 and best_metrics["spearman"]>=.30 and best_metrics["dependency_auroc"]>=.80; result={"schema":"sl-predict-conditional-viability-v1","parameters":sum(p.numel() for p in head.parameters()),"frozen_world_parameters":sum(p.numel() for p in endpoint.parameters()),"train_cells":len(train_cells),"train_genes":len(train_genes),"validation_cells":len(valid_cells),"validation_genes":len(valid_genes),"best":best_metrics,"advanced":bool(advanced),"double_perturbation_data_used":False,"sl_labels_used":False,"history":history}; torch.save(head.state_dict(),model_dir/"cold_tolerance_head.pt"); (model_dir/"cold_tolerance_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2)); return result

if __name__=="__main__":fit()
