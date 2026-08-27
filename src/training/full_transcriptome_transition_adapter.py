import json,sys
from pathlib import Path
import numpy as np,torch

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; RUN=OUT/"native_spectral_safe_scaled_d768_z256_l8_p12_single_only_d3_t10_r3"; sys.path.insert(0,str(Path(__file__).parent))
from world_model import SLPredict,SourceEndpoint,batches,encode_genes

class Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.net=torch.nn.Sequential(torch.nn.LayerNorm(256),torch.nn.Linear(256,64,bias=False),torch.nn.GELU(),torch.nn.Linear(64,256,bias=False)); torch.nn.init.zeros_(self.net[-1].weight)
    def forward(self,z): return z+self.net(z)

def main(epochs=12):
    torch.manual_seed(1009); np.random.seed(1009); device="cuda" if torch.cuda.is_available() else "cpu"; data=np.load(OUT/"full_transcriptome_single_endpoint32.npz"); state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); sd=torch.load(RUN/"world_model.pt",map_location="cpu",weights_only=True); world=SLPredict(768,256,8,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],sd["context_proj.weight"].shape[1]).to(device); world.load_state_dict(sd); world.eval().requires_grad_(False); head=SourceEndpoint(len(data["sources"]),256,32).to(device); head.load_state_dict(torch.load(RUN/"full_transcriptome_single_endpoint32.pt",map_location="cpu",weights_only=True)); head.eval().requires_grad_(False); adapter=Adapter().to(device); genes=torch.as_tensor(encode_genes(world,state,device),device=device); gene=torch.as_tensor(data["gene"].astype("int64"),device=device); source=torch.as_tensor(data["source"].astype("int64"),device=device); context=torch.as_tensor(data["context_state"].astype("float32"),device=device); target=torch.as_tensor(data["target"].astype("float32"),device=device); train=np.flatnonzero(data["role"]==0); valid=np.flatnonzero(data["role"]==1); counts=np.bincount(data["source"][train]); weight=1/counts[data["source"][train]]; weight/=weight.sum(); opt=torch.optim.AdamW(adapter.parameters(),3e-4,weight_decay=1e-3); history=[]; saved=[]
    @torch.no_grad()
    def evaluate(epoch):
        row={"epoch":epoch}
        for name,exact in (("unknown",False),("exact",True)):
            pred=[]
            for at in batches(len(valid),2048,False):
                ix=valid[at]; cs=context[source[ix]] if exact else None; z=world.transition(genes[gene[ix]],context_state=cs)[0]; pred.append(head(adapter(z),source[ix]).cpu())
            pred=torch.cat(pred).numpy(); truth=data["target"][valid].astype("float32"); sources=[]
            for s,label in enumerate(data["sources"].astype(str)):
                k=data["source"][valid]==s; sources.append({"source":label,"rows":int(k.sum()),"huber":float(torch.nn.functional.huber_loss(torch.from_numpy(pred[k]),torch.from_numpy(truth[k]))),"cosine":float(np.mean(np.sum(pred[k]*truth[k],1)/(np.linalg.norm(pred[k],axis=1)*np.linalg.norm(truth[k],axis=1)+1e-8)))})
            row[name]={"source_macro_huber":float(np.mean([x["huber"] for x in sources])),"source_macro_cosine":float(np.mean([x["cosine"] for x in sources])),"sources":sources}
        row["selection_loss"]=row["unknown"]["source_macro_huber"]+row["exact"]["source_macro_huber"]; history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}); print(json.dumps(row),flush=True)
    evaluate(0)
    for epoch in range(epochs):
        chosen=np.random.default_rng(1009+epoch).choice(train,len(train),replace=True,p=weight); adapter.train()
        for at in batches(len(chosen),512,False):
            ix=chosen[at]; cs=context[source[ix]]; cs=torch.where((torch.rand(len(ix),device=device)<.5)[:,None],torch.zeros_like(cs),cs)
            with torch.no_grad(): z=world.transition(genes[gene[ix]],context_state=cs)[0]
            loss=torch.nn.functional.huber_loss(head(adapter(z),source[ix]),target[ix]); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.); opt.step()
        adapter.eval(); evaluate(epoch+1)
    selected=min(range(len(history)),key=lambda i:history[i]["selection_loss"]); chosen=history[selected]; adapter.load_state_dict(saved[selected]); base=history[0]; exact0={x["source"]:x for x in base["exact"]["sources"]}; exact={x["source"]:x for x in chosen["exact"]["sources"]}; advanced=all(chosen[c]["source_macro_huber"]<=.98*base[c]["source_macro_huber"] and chosen[c]["source_macro_cosine"]>=base[c]["source_macro_cosine"] for c in ("unknown","exact")) and all(x["cosine"]>0 and x["cosine"]>=exact0[x["source"]]["cosine"]-.02 for x in exact.values()); result={"schema":"sl-predict-full-transcriptome-transition-adapter-v1","parameters":sum(p.numel() for p in adapter.parameters()),"selected":chosen,"baseline":base,"advanced":bool(advanced),"world_parameters_changed":0,"endpoint_parameters_changed":0,"double_perturbation_data_used":False,"sl_labels_used":False,"history":history}; torch.save(adapter.state_dict(),RUN/"full_transcriptome_transition_adapter.pt"); (RUN/"full_transcriptome_transition_adapter_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2))

if __name__=="__main__": main()
