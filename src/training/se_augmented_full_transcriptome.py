from pathlib import Path
import hashlib,json,sys
import numpy as np,torch

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; RUN=OUT/"native_spectral_safe_scaled_d768_z256_l8_p12_single_only_d3_t10_r3"; sys.path.insert(0,str(Path(__file__).parent))
from world_model import SLPredict,SourceEndpoint,batches,encode_genes

def split(data,action):
    se=np.load(OUT/"se_replogle_state.npz"); fitted=set(se["gene"][se["role"]=="train"].astype(str)); held=np.unique(data["gene"][data["role"]==1]); strict=np.asarray([g for g in held if action["known"][g] and action["genes"][g] not in fitted]); parity={g:hashlib.sha256(str(action["genes"][g]).encode()).digest()[0]&1 for g in strict}; return np.flatnonzero((data["role"]==1)&np.isin(data["gene"],[g for g,q in parity.items() if q==0])),np.flatnonzero((data["role"]==1)&np.isin(data["gene"],[g for g,q in parity.items() if q==1]))

@torch.no_grad()
def score(head,z,aux,source,target,indices,names):
    pred=[]
    for at in batches(len(indices),2048,False):
        ix=indices[at]; pred.append(head(torch.cat((z[ix],aux[ix]),1),source[ix]).cpu())
    pred=torch.cat(pred).numpy(); truth=target[indices].cpu().numpy(); rows=[]
    for sid,name in enumerate(names):
        keep=source[indices].cpu().numpy()==sid
        if keep.any():
            p,y=pred[keep],truth[keep]; rows.append({"source":str(name),"rows":int(keep.sum()),"huber":float(torch.nn.functional.huber_loss(torch.from_numpy(p),torch.from_numpy(y))),"cosine":float(np.mean(np.sum(p*y,1)/(np.linalg.norm(p,axis=1)*np.linalg.norm(y,axis=1)+1e-8)))})
    return {"source_macro_huber":float(np.mean([r["huber"] for r in rows])),"source_macro_cosine":float(np.mean([r["cosine"] for r in rows])),"sources":rows}

def main(epochs=20):
    torch.manual_seed(119); np.random.seed(119); device="cuda" if torch.cuda.is_available() else "cpu"; data=np.load(OUT/"full_transcriptome_single_endpoint32.npz"); action=np.load(OUT/"se_replogle_gene_features.npz"); state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); sd=torch.load(RUN/"world_model.pt",map_location="cpu",weights_only=True); world=SLPredict(768,256,8,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],sd["context_proj.weight"].shape[1]).to(device); world.load_state_dict(sd); world.eval().requires_grad_(False); genes=torch.as_tensor(encode_genes(world,state,device),device=device); gene=torch.as_tensor(data["gene"].astype("int64"),device=device); source=torch.as_tensor(data["source"].astype("int64"),device=device); target=torch.as_tensor(data["target"].astype("float32"),device=device); context=torch.as_tensor(data["context_state"].astype("float32"),device=device); train=np.flatnonzero((data["role"]==0)&action["known"][data["gene"]]); select,confirm=split(data,action); train_genes=np.unique(data["gene"][train]); raw=action["features"].astype("float32"); mean=raw[train_genes].mean(0); scale=raw[train_genes].std(0).clip(1e-6); auxiliary=torch.as_tensor((raw[data["gene"]]-mean)/scale,device=device); zero=torch.zeros_like(auxiliary)
    with torch.no_grad():
        unknown=torch.empty((len(gene),256),device=device); exact=torch.empty_like(unknown)
        for at in batches(len(gene),2048,False):
            unknown[at]=world.transition(genes[gene[at]])[0]; exact[at]=world.transition(genes[gene[at]],context_state=context[source[at]])[0]
    heads={}; opts={}; history={k:[] for k in ("baseline","candidate")}; saved={k:[] for k in history}
    for name in history:
        torch.manual_seed(120); heads[name]=SourceEndpoint(len(data["sources"]),320,32).to(device); opts[name]=torch.optim.AdamW(heads[name].parameters(),3e-4,weight_decay=1e-3)
    counts=np.bincount(data["source"][train],minlength=len(data["sources"])); weight=1/counts[data["source"][train]]; weight/=weight.sum()
    def evaluate(epoch):
        for name,aux in (("baseline",zero),("candidate",auxiliary)):
            heads[name].eval(); row={"epoch":epoch,"unknown":score(heads[name],unknown,aux,source,target,select,data["sources"]),"exact":score(heads[name],exact,aux,source,target,select,data["sources"])}; row["selection_loss"]=row["unknown"]["source_macro_huber"]+row["exact"]["source_macro_huber"]; history[name].append(row); saved[name].append({k:v.detach().cpu().clone() for k,v in heads[name].state_dict().items()})
        print(json.dumps({k:history[k][-1] for k in history}),flush=True)
    evaluate(0)
    for epoch in range(epochs):
        chosen=np.random.default_rng(119+epoch).choice(train,len(train),replace=True,p=weight); exact_context=np.random.default_rng(219+epoch).random(len(chosen))>=.5
        for name,aux in (("baseline",zero),("candidate",auxiliary)):
            heads[name].train()
            for at in batches(len(chosen),512,False):
                ix=chosen[at]; z=torch.where(torch.as_tensor(exact_context[at],device=device)[:,None],exact[ix],unknown[ix]); pred=heads[name](torch.cat((z,aux[ix]),1),source[ix]); loss=torch.nn.functional.huber_loss(pred,target[ix]); opts[name].zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(heads[name].parameters(),1.); opts[name].step()
        evaluate(epoch+1)
    selected={name:min(range(len(history[name])),key=lambda i:history[name][i]["selection_loss"]) for name in history}
    for name in heads:heads[name].load_state_dict(saved[name][selected[name]]); heads[name].eval()
    confirmation={name:{ctx:score(heads[name],z,zero if name=="baseline" else auxiliary,source,target,confirm,data["sources"]) for ctx,z in (("unknown",unknown),("exact",exact))} for name in heads}; gains={}
    for ctx in ("unknown","exact"):
        b,c=confirmation["baseline"][ctx],confirmation["candidate"][ctx]; by_base={r["source"]:r for r in b["sources"]}; gains[ctx]={"huber_improvement":1-c["source_macro_huber"]/b["source_macro_huber"],"cosine_gain":c["source_macro_cosine"]-b["source_macro_cosine"],"maximum_source_cosine_regression":max((by_base[r["source"]]["cosine"]-r["cosine"] for r in c["sources"] if r["rows"]>=8),default=0.)}
    source_rows=np.bincount(data["source"][confirm],minlength=len(data["sources"])); confirm_genes=len(np.unique(data["gene"][confirm])); advanced=confirm_genes>=20 and source_rows.min()>=4 and all(gains[c]["huber_improvement"]>=.02 and gains[c]["cosine_gain"]>=.02 and gains[c]["maximum_source_cosine_regression"]<=.02 for c in gains); torch.save({"baseline":heads["baseline"].state_dict(),"candidate":heads["candidate"].state_dict(),"action_mean":torch.as_tensor(mean),"action_scale":torch.as_tensor(scale)},RUN/"se_augmented_full_transcriptome_heads.pt"); result={"schema":"sl-predict-se-augmented-full-transcriptome-v1","parameters_per_head":sum(p.numel() for p in heads["baseline"].parameters()),"training_rows":len(train),"training_genes":len(train_genes),"selection_rows":len(select),"selection_genes":len(np.unique(data["gene"][select])),"confirmation_rows":len(confirm),"confirmation_genes":confirm_genes,"confirmation_source_rows":{str(n):int(source_rows[i]) for i,n in enumerate(data["sources"])},"selected_epochs":selected,"selected_metrics":{n:history[n][selected[n]] for n in history},"confirmation":confirmation,"gains":gains,"advanced":bool(advanced),"world_parameters_changed":0,"double_perturbation_data_used":False,"sl_labels_used":False,"history":history}; (RUN/"se_augmented_full_transcriptome_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2)); return result

if __name__=="__main__":main()
