import json, sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; RUN=OUT/"native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; sys.path.insert(0,str(Path(__file__).parent)); from context_transition import Shift; from world_model import encode_genes, load_residual_endpoint

class Residual(torch.nn.Module):
    def __init__(self): super().__init__(); self.net=torch.nn.Sequential(torch.nn.LayerNorm(32),torch.nn.Linear(32,64),torch.nn.GELU(),torch.nn.Linear(64,128)); torch.nn.init.zeros_(self.net[-1].weight); torch.nn.init.zeros_(self.net[-1].bias)
    def forward(self,base,x,known): return torch.nn.functional.normalize(base+known[:,None]*self.net(x),dim=1)

def main(epochs=60):
    torch.manual_seed(821); np.random.seed(821); device="cuda" if torch.cuda.is_available() else "cpu"; data=np.load(OUT/"context_transition.npz"); feature=np.load(OUT/"se_replogle_gene_features.npz"); state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); endpoint=load_residual_endpoint(RUN/"world_model.pt",state.shape[1],device); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); shift=Shift().to(device); shift.load_state_dict(torch.load(RUN/"extended_unit_context_transition_head.pt",map_location="cpu",weights_only=True)); shift.eval().requires_grad_(False)
    gene=data["gene"].astype("int64"); source=data["source"].astype("int64"); cid=(source>=4).astype("int64"); raw=np.stack([feature["features"][g,32*c:32*(c+1)] for g,c in zip(gene,cid)]).astype("float32"); known=feature["known"][gene].astype("float32"); train=np.flatnonzero(data["role"]==0); valid=np.flatnonzero(data["role"]==1); mean=np.stack([raw[train[cid[train]==c]].mean(0) for c in range(2)]); scale=np.stack([raw[train[cid[train]==c]].std(0).clip(1e-6) for c in range(2)]); x=torch.as_tensor(((raw-mean[cid])/scale[cid])*known[:,None],device=device); known_t=torch.as_tensor(known,device=device); context=torch.as_tensor(data["context_state"],device=device); target=torch.nn.functional.normalize(torch.as_tensor(data["target"],device=device),dim=1)
    with torch.no_grad(): base=torch.nn.functional.normalize(shift(genes[torch.as_tensor(gene,device=device)],context[torch.as_tensor(source,device=device)]),dim=1)
    head=Residual().to(device); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3); counts=np.bincount(source[train],minlength=5); history=[]; saved=[]
    @torch.no_grad()
    def evaluate(epoch):
        pred=head(base[valid],x[valid],known_t[valid]); rows=[]
        for sid,name in enumerate(data["sources"].astype(str)):
            loc=np.flatnonzero(source[valid]==sid); at=torch.as_tensor(loc,device=device); y=target[torch.as_tensor(valid[loc],device=device)]; rows.append({"source":name,"rows":len(loc),"known_rows":int(known[valid[loc]].sum()),"baseline_cosine":float(torch.nn.functional.cosine_similarity(base[valid][at],y).mean()),"cosine":float(torch.nn.functional.cosine_similarity(pred[at],y).mean())})
        row={"epoch":epoch,"source_macro_cosine":float(np.mean([r["cosine"] for r in rows])),"sources":rows}; history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()}); return row
    print(json.dumps(evaluate(0)),flush=True)
    for epoch in range(epochs):
        order=np.random.default_rng(821+epoch).permutation(train); head.train()
        for lo in range(0,len(order),512):
            take=order[lo:lo+512]; ix=torch.as_tensor(take,device=device); raw_loss=1-torch.nn.functional.cosine_similarity(head(base[ix],x[ix],known_t[ix]),target[ix]); weight=torch.as_tensor(1/counts[source[take]],dtype=torch.float32,device=device); loss=(raw_loss*weight).sum()/weight.sum(); opt.zero_grad(); loss.backward(); opt.step()
        head.eval(); row=evaluate(epoch+1); print(json.dumps({"epoch":row["epoch"],"source_macro_cosine":row["source_macro_cosine"]}),flush=True)
    best=max(range(len(history)),key=lambda i:history[i]["source_macro_cosine"]); selected=history[best]; head.load_state_dict(saved[best]); advanced=selected["source_macro_cosine"]>=.390586 and all(r["cosine"]>=.10 and r["cosine"]>=r["baseline_cosine"]-.02 for r in selected["sources"]); torch.save({"state_dict":head.state_dict(),"mean":torch.as_tensor(mean),"scale":torch.as_tensor(scale)},RUN/"se_action_context_transition_head.pt"); result={"schema":"sl-predict-se-action-context-transition-v1","parameters":sum(p.numel() for p in head.parameters()),"selected":selected,"advanced":bool(advanced),"double_perturbation_data_used":False,"sl_labels_used":False,"history":history}; (RUN/"se_action_context_transition_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2))

if __name__=="__main__": main()
