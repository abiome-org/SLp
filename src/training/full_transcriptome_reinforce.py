import json,sys
from pathlib import Path
import numpy as np,torch

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; RUN=OUT/"native_spectral_safe_scaled_d768_z256_l8_p12_single_only_d3_t10_r3"; sys.path.insert(0,str(Path(__file__).parent))
from world_model import SLPredict,SourceEndpoint,batches,dependency_loss,encode_genes,perturbseq_loss,relation_loss

@torch.no_grad()
def endpoint_metrics(world,head,genes,data,device):
    valid=np.flatnonzero(data["role"]==1); gene=torch.as_tensor(data["gene"].astype("int64"),device=device); source=torch.as_tensor(data["source"].astype("int64"),device=device); context=torch.as_tensor(data["context_state"].astype("float32"),device=device); result={}
    for name,exact in (("unknown",False),("exact",True)):
        pred=[]
        for at in batches(len(valid),2048,False):
            ix=valid[at]; cs=context[source[ix]] if exact else None; pred.append(head(world.transition(genes[gene[ix]],context_state=cs)[0],source[ix]).cpu())
        pred=torch.cat(pred).numpy(); truth=data["target"][valid].astype("float32"); rows=[]
        for s,label in enumerate(data["sources"].astype(str)):
            k=data["source"][valid]==s; rows.append({"source":label,"rows":int(k.sum()),"huber":float(torch.nn.functional.huber_loss(torch.from_numpy(pred[k]),torch.from_numpy(truth[k]))),"cosine":float(np.mean(np.sum(pred[k]*truth[k],1)/(np.linalg.norm(pred[k],axis=1)*np.linalg.norm(truth[k],axis=1)+1e-8)))})
        result[name]={"source_macro_huber":float(np.mean([x["huber"] for x in rows])),"source_macro_cosine":float(np.mean([x["cosine"] for x in rows])),"sources":rows}
    result["selection_loss"]=result["unknown"]["source_macro_huber"]+result["exact"]["source_macro_huber"]; return result

def preservation(world,genes,state,device):
    pack=np.load(OUT/"features_spectral_safe.npz"); pairs=torch.as_tensor(pack["pairs"].astype("int64")); valid=torch.nonzero((pairs[:,0]*1000003+pairs[:,1])%20==0).squeeze(1); relation=relation_loss(world,torch.as_tensor(state),pairs,torch.as_tensor(pack["relations"],dtype=torch.float32),valid,device); dep=np.load(OUT/"basal_context.npz"); cells=np.flatnonzero(dep["train_cell"]&(np.arange(len(dep["train_cell"]))%10==0)); gene_ids=np.flatnonzero(dep["train_gene"]); dependency=dependency_loss(world,genes,cells,gene_ids,torch.as_tensor(dep["cell_state"],device=device),torch.as_tensor(dep["dependency"].astype("float32"),device=device),dep["dependency_known"],device); perturb=np.load(OUT/"perturbseq_world_v3.npz"); saved=torch.load(RUN/"perturb_decoder.pt",map_location="cpu",weights_only=True); decoder=torch.nn.Linear(saved["weight"].shape[1],saved["weight"].shape[0]).to(device); decoder.load_state_dict(saved); decoder.eval(); unknown=perturbseq_loss(world,decoder,genes,perturb,device,single_only=True)[0]; exact=perturbseq_loss(world,decoder,genes,perturb,device,source_context=True,single_only=True)[0]; return {"relation_huber":relation,"dependency_huber":dependency,"prior_single_state":unknown+exact}

def main(epochs=3,group=4):
    torch.manual_seed(1031); np.random.seed(1031); device="cuda" if torch.cuda.is_available() else "cpu"; data=np.load(OUT/"full_transcriptome_single_endpoint32.npz"); state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); sd=torch.load(RUN/"world_model.pt",map_location="cpu",weights_only=True); world=SLPredict(768,256,8,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],sd["context_proj.weight"].shape[1]).to(device); world.load_state_dict(sd); world.eval().requires_grad_(False); head=SourceEndpoint(5,256,32).to(device); head.load_state_dict(torch.load(RUN/"full_transcriptome_single_endpoint32.pt",map_location="cpu",weights_only=True)); head.eval().requires_grad_(False); genes=torch.as_tensor(encode_genes(world,state,device),device=device)
    modules=(world.state_up,world.dist,world.context_proj)
    for module in modules: module.requires_grad_(True)
    for parameter in (world.basal,world.context,world.time): parameter.requires_grad_(True)
    trainable=[p for p in world.parameters() if p.requires_grad]; train=np.flatnonzero(data["role"]==0); counts=np.bincount(data["source"][train]); weight=1/counts[data["source"][train]]; weight/=weight.sum(); gene=torch.as_tensor(data["gene"].astype("int64"),device=device); source=torch.as_tensor(data["source"].astype("int64"),device=device); context=torch.as_tensor(data["context_state"].astype("float32"),device=device); target=torch.as_tensor(data["target"].astype("float32"),device=device); opt=torch.optim.AdamW(trainable,2e-5,weight_decay=1e-3); history=[]; best=None
    base=endpoint_metrics(world,head,genes,data,device); base["epoch"]=0; history.append(base); best={n:p.detach().cpu().clone() for n,p in world.named_parameters() if p.requires_grad}; print(json.dumps(base),flush=True)
    for epoch in range(epochs):
        chosen=np.random.default_rng(1031+epoch).choice(train,len(train),replace=True,p=weight); total=0.
        for at in batches(len(chosen),256,False):
            ix=chosen[at]; cs=context[source[ix]]; cs=torch.where((torch.rand(len(ix),device=device)<.5)[:,None],torch.zeros_like(cs),cs); mu,logsd=world.transition(genes[gene[ix]],context_state=cs); sdv=logsd.exp(); sample=(mu[:,None]+sdv[:,None]*torch.randn(len(ix),group,mu.shape[1],device=device)).detach(); src=source[ix,None].expand(-1,group).reshape(-1); pred=head(sample.reshape(-1,mu.shape[1]),src).view(len(ix),group,-1); reward=-(pred-target[ix,None]).square().mean(2); advantage=(reward-reward.mean(1,keepdim=True))/(reward.std(1,keepdim=True)+1e-5); logp=torch.distributions.Normal(mu[:,None],sdv[:,None]).log_prob(sample).mean(2); loss=-(advantage.detach()*logp).mean()-.001*torch.distributions.Normal(mu,sdv).entropy().mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,1.); opt.step(); total+=loss.item()*len(ix)
        row=endpoint_metrics(world,head,genes,data,device); row.update({"epoch":epoch+1,"reinforce_loss":total/len(chosen)}); history.append(row); print(json.dumps(row),flush=True)
        if row["selection_loss"]<min(x["selection_loss"] for x in history[:-1]): best={n:p.detach().cpu().clone() for n,p in world.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for name,p in world.named_parameters():
            if name in best:p.copy_(best[name])
    selected=min(history,key=lambda x:x["selection_loss"]); baseline_world=SLPredict(768,256,8,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],sd["context_proj.weight"].shape[1]).to(device); baseline_world.load_state_dict(sd); baseline_world.eval(); baseline=preservation(baseline_world,genes,state,device); candidate=preservation(world,genes,state,device); endpoint_ok=selected["epoch"]>0 and all(selected[c]["source_macro_huber"]<=.99*base[c]["source_macro_huber"] and selected[c]["source_macro_cosine"]>=base[c]["source_macro_cosine"] for c in ("unknown","exact")); preserved=all(candidate[k]<=1.01*baseline[k] for k in baseline); advanced=endpoint_ok and preserved; torch.save(world.state_dict(),RUN/"world_model_full_transcriptome_rl.pt"); result={"schema":"sl-predict-full-transcriptome-reinforce-v1","trainable_parameters":sum(p.numel() for p in trainable),"selected":selected,"baseline_endpoint":base,"baseline_state":baseline,"candidate_state":candidate,"endpoint_advanced":bool(endpoint_ok),"preserved":bool(preserved),"advanced":bool(advanced),"double_perturbation_data_used":False,"sl_labels_used":False,"history":history}; (RUN/"full_transcriptome_reinforce_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2))

if __name__=="__main__": main()
