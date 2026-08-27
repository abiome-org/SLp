import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.decomposition import PCA

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; sys.path.insert(0,str(Path(__file__).parent))
from world_model import SLPredict,SourceEndpoint,batches,encode_genes

def build():
    meta=pd.read_csv(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"); symbols=meta.symbol.astype(str).str.upper().to_numpy(); ids={g:i for i,g in enumerate(symbols)}; held=set(json.loads((OUT/"slkb_outcomes_intervention_external.json").read_text())["symbols"]); val=lambda g:int.from_bytes(hashlib.sha256(g.encode()).digest()[:4],"big")%5==0; paths=sorted((ROOT/"data/perturbseq_sources/base").glob("*.npz")); rows=[]; audits=[]
    for sid,path in enumerate(paths):
        z=np.load(path); a=np.char.upper(z["endpoint_a"].astype(str)); card=z["cardinality"]; role=np.full(len(a),-1,"int8")
        for i,g in enumerate(a):
            if card[i]==1 and g in ids and g not in held: role[i]=1 if val(g) else 0
        fit=role==0; delta=z["future_state"].astype("float32")-z["control_mean"].astype("float32"); center=delta[fit].mean(0); scale=delta[fit].std(0).clip(.05); x=(delta-center)/scale; pca=PCA(32,svd_solver="randomized",random_state=967).fit(x[fit]); target=pca.transform(x).astype("float32"); latent_scale=target[fit].std(0).clip(.05); target/=latent_scale; keep=role>=0; rows.append((np.asarray([ids[g] for g in a[keep]],"int16"),np.full(keep.sum(),sid,"int8"),role[keep],target[keep])); audits.append({"source":path.stem,"response_genes":delta.shape[1],"fitting_rows":int(fit.sum()),"held_rows":int((role==1).sum()),"pca_explained_variance":float(pca.explained_variance_ratio_.sum())})
    basal=np.load(OUT/"basal_context.npz"); arrays={"gene":np.concatenate([x[0] for x in rows]),"source":np.concatenate([x[1] for x in rows]),"role":np.concatenate([x[2] for x in rows]),"target":np.concatenate([x[3] for x in rows]),"context_state":basal["source_state"][:len(rows)].astype("float32"),"sources":np.asarray([p.stem for p in paths])}; np.savez_compressed(OUT/"full_transcriptome_single_endpoint32.npz",**arrays); result={"schema":"sl-predict-full-transcriptome-single-state32-v1","sources":audits,"rows":len(arrays["gene"]),"fitting_rows":int((arrays["role"]==0).sum()),"held_rows":int((arrays["role"]==1).sum()),"state_dimensions":32,"double_perturbation_data_used":False,"sl_labels_used":False}; (OUT/"full_transcriptome_single_endpoint32.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

def fit(epochs=16):
    torch.manual_seed(967); np.random.seed(967); device="cuda" if torch.cuda.is_available() else "cpu"; data=np.load(OUT/"full_transcriptome_single_endpoint32.npz"); state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); model_dir=OUT/"native_spectral_safe_scaled_d768_z256_l8_p12_single_only_d3_t10_r3"; sd=torch.load(model_dir/"world_model.pt",map_location="cpu",weights_only=True); world=SLPredict(768,256,8,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],sd["context_proj.weight"].shape[1]).to(device); world.load_state_dict(sd); world.eval().requires_grad_(False); genes=torch.as_tensor(encode_genes(world,state,device),device=device); gene=torch.as_tensor(data["gene"].astype("int64"),device=device); source=torch.as_tensor(data["source"].astype("int64"),device=device); context=torch.as_tensor(data["context_state"].astype("float32"),device=device); target=torch.as_tensor(data["target"].astype("float32"),device=device); train=np.flatnonzero(data["role"]==0); valid=np.flatnonzero(data["role"]==1); head=SourceEndpoint(len(data["sources"]),256,32).to(device); counts=np.bincount(data["source"][train]); weight=1/counts[data["source"][train]]; weight/=weight.sum(); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3); history=[]; saved=[]
    @torch.no_grad()
    def evaluate(epoch):
        result={"epoch":epoch}; zero={}
        for name,exact in (("unknown",False),("exact",True)):
            pred=[]
            for at in batches(len(valid),2048,False):
                ix=valid[at]; cs=context[source[ix]] if exact else None; z=world.transition(genes[gene[ix]],context_state=cs)[0]; pred.append(head(z,source[ix]).cpu())
            pred=torch.cat(pred).numpy(); truth=data["target"][valid].astype("float32"); rows=[]
            for s,label in enumerate(data["sources"].astype(str)):
                k=data["source"][valid]==s
                if k.any(): rows.append({"source":label,"rows":int(k.sum()),"huber":float(torch.nn.functional.huber_loss(torch.from_numpy(pred[k]),torch.from_numpy(truth[k]))),"zero_huber":float(torch.nn.functional.huber_loss(torch.zeros_like(torch.from_numpy(truth[k])),torch.from_numpy(truth[k]))),"cosine":float(np.mean(np.sum(pred[k]*truth[k],1)/(np.linalg.norm(pred[k],axis=1)*np.linalg.norm(truth[k],axis=1)+1e-8)))})
            macro=float(np.mean([r["huber"] for r in rows])); base=float(np.mean([r["zero_huber"] for r in rows])); result[name]={"source_macro_huber":macro,"source_macro_zero":base,"improvement":1-macro/base,"source_macro_cosine":float(np.mean([r["cosine"] for r in rows])),"sources":rows}; zero[name]=base
        result["selection_loss"]=result["unknown"]["source_macro_huber"]+result["exact"]["source_macro_huber"]; history.append(result); saved.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()}); print(json.dumps(result),flush=True)
    evaluate(0)
    for epoch in range(epochs):
        chosen=np.random.default_rng(967+epoch).choice(train,len(train),replace=True,p=weight); head.train()
        for at in batches(len(chosen),512,False):
            ix=chosen[at]; cs=context[source[ix]]; cs=torch.where((torch.rand(len(ix),device=device)<.5)[:,None],torch.zeros_like(cs),cs)
            with torch.no_grad(): z=world.transition(genes[gene[ix]],context_state=cs)[0]
            loss=torch.nn.functional.huber_loss(head(z,source[ix]),target[ix]); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),1.); opt.step()
        head.eval(); evaluate(epoch+1)
    selected=min(range(len(history)),key=lambda i:history[i]["selection_loss"]); chosen=history[selected]; head.load_state_dict(saved[selected]); exact={r["source"]:r for r in chosen["exact"]["sources"]}; positive=all(r["cosine"]>0 for c in ("unknown","exact") for r in chosen[c]["sources"] if r["rows"]>=10); advanced=chosen["unknown"]["source_macro_cosine"]>=.13 and chosen["exact"]["source_macro_cosine"]>=.17 and chosen["unknown"]["improvement"]>=.01 and chosen["exact"]["improvement"]>=.01 and positive and exact["replogle2022_k562"]["cosine"]>=.15 and exact["replogle2022_rpe1"]["cosine"]>=.15; torch.save(head.state_dict(),model_dir/"full_transcriptome_single_endpoint32.pt"); result={"schema":"sl-predict-full-transcriptome-single-endpoint32-v1","parameters":sum(p.numel() for p in head.parameters()),"selected":chosen,"advanced":bool(advanced),"world_parameters_changed":0,"double_perturbation_data_used":False,"sl_labels_used":False,"history":history}; (model_dir/"full_transcriptome_single_endpoint32_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2))

if __name__=="__main__": {"build":build,"fit":fit}[sys.argv[1] if len(sys.argv)>1 else "build"]()
