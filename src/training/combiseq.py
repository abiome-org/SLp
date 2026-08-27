from pathlib import Path
import hashlib, json
import numpy as np, pandas as pd
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def build(matrix,metadata,gse_state,output):
    meta=pd.read_csv(metadata); raw=pd.read_csv(matrix); sample=meta.iloc[:,0].astype(str).to_numpy(); assert raw.columns[1:].tolist()==sample.tolist()
    genes=raw.iloc[:,0].astype(str).str.split(".").str[0].to_numpy(); counts=raw.iloc[:,1:].to_numpy("float32"); library=counts.sum(0).clip(1)
    mapping=pd.read_csv(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"); ens={str(e).split(".")[0]:str(s) for e,s in zip(mapping.ensembl_gene_id,mapping.symbol) if pd.notna(e)}
    gse=np.load(gse_state,allow_pickle=True); feature=gse["feature_name"].astype(str); position={x:i for i,x in enumerate(feature)}; rows=[]; cols=[]
    for i,g in enumerate(genes):
        symbol=ens.get(g)
        if symbol in position:rows.append(i); cols.append(position[symbol])
    effect=np.zeros((len(sample),len(feature)),"float32"); effect[:,cols]=np.log1p(counts[rows].T*(1e4/library[:,None])); control=(meta.Drug1.eq("DMSO")&meta.Drug2.eq("DMSO")).to_numpy(); effect-=effect[control].mean(0)
    state=(effect@gse["pca_components"].T/gse["target_scale"]).astype("float32"); keys=list(zip(meta.Drug1.astype(str),meta.Drug2.astype(str))); groups={}
    for i,k in enumerate(keys):groups.setdefault(k,[]).append(i)
    means={k:state[v].mean(0) for k,v in groups.items()}; singles={a:means[a,"DMSO"] for a in set(meta.Drug1)-{"DMSO"}}|{b:means["DMSO",b] for b in set(meta.Drug2)-{"DMSO"}}
    held=lambda x:int.from_bytes(hashlib.sha256(x.upper().encode()).digest()[:4],"big")%5==0; combo=[k for k in sorted(groups) if "DMSO" not in k]; a=np.asarray([x[0] for x in combo]); b=np.asarray([x[1] for x in combo]); role=np.asarray([1 if held(x)&held(y) else (0 if not held(x) and not held(y) else -1) for x,y in combo],"int8")
    sa=np.stack([singles[x] for x in a]); sb=np.stack([singles[x] for x in b]); target=np.stack([means[x] for x in combo]); residual=target-sa-sb
    run={r:np.stack([state[groups[k][[meta.Run.iloc[i] for i in groups[k]].index(r)]] for k in sorted(groups)]) for r in ("rep1","rep2","rep3")}; replicate=float(np.mean([np.corrcoef(run[x].ravel(),run[y].ravel())[0,1] for x,y in (("rep1","rep2"),("rep1","rep3"),("rep2","rep3"))])); noise=float(np.sqrt(np.mean(np.concatenate([state[v]-means[k] for k,v in groups.items()])**2))); contrast=float(np.sqrt(np.mean(residual**2)))
    out=Path(output); out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,drug_a=a,drug_b=b,role=role,single_a=sa,single_b=sb,target=target,residual=residual,decoder_offset=(gse["pca_mean"]@gse["pca_components"].T/gse["target_scale"]).astype("float32"))
    audit={"schema":"sl-predict-combiseq-state-v1","input_sha256":{"matrix":sha(matrix),"metadata":sha(metadata),"gse_state":sha(gse_state)},"samples":len(sample),"conditions":len(groups),"drug_combinations":len(combo),"matched_single_drugs":len(singles),"biological_replicates":3,"fixed_response_panel_genes":len(feature),"mapped_response_genes":len(rows),"training_combinations":int((role==0).sum()),"both_drugs_new_validation_combinations":int((role==1).sum()),"excluded_mixed_combinations":int((role<0).sum()),"mean_replicate_correlation":replicate,"replicate_rms_noise":noise,"nonadditive_rms_contrast":contrast,"contrast_to_noise":contrast/noise,"admitted":bool(replicate>=.2 and contrast/noise>=1),"genetic_double_perturbations_used":False,"sl_labels_used":False}; out.with_suffix(".json").write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2)); return audit

class StateComposition(nn.Module):
    def __init__(self,d=64):
        super().__init__(); self.net=nn.Sequential(nn.LayerNorm(3*d),nn.Linear(3*d,d),nn.GELU(),nn.Linear(d,d)); nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,a,b):return self.net(torch.cat((a+b,(a-b).abs(),a*b),1))

def fit(state_path,output,epochs=30):
    torch.manual_seed(731); np.random.seed(731); device="cuda" if torch.cuda.is_available() else "cpu"; z=np.load(state_path); role=z["role"]; train=np.flatnonzero(role==0); valid=np.flatnonzero(role==1); a=torch.as_tensor(z["single_a"],device=device); b=torch.as_tensor(z["single_b"],device=device); target=torch.as_tensor(z["target"],device=device); residual=torch.as_tensor(z["residual"],device=device); model=StateComposition(a.shape[1]).to(device); opt=torch.optim.AdamW(model.parameters(),1e-3,weight_decay=1e-2); history=[]; saved=[]
    def evaluate(epoch):
        model.eval()
        with torch.no_grad(): correction=model(a[valid],b[valid]); row={"epoch":epoch,"residual_huber":float(nn.functional.huber_loss(correction,residual[valid])),"full_state_huber":float(nn.functional.huber_loss(a[valid]+b[valid]+correction,target[valid]))}
        history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()}); print(json.dumps(row),flush=True)
    evaluate(0)
    for epoch in range(epochs):
        model.train(); order=np.random.default_rng(731+epoch).permutation(train)
        for start in range(0,len(order),64):
            ix=order[start:start+64]; loss=nn.functional.huber_loss(model(a[ix],b[ix]),residual[ix]); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        evaluate(epoch+1)
    selected=min(range(len(history)),key=lambda i:history[i]["residual_huber"]); model.load_state_dict(saved[selected]); baseline=history[0]; chosen=history[selected]; improved=selected>0 and chosen["residual_huber"]<=.98*baseline["residual_huber"]; preserved=chosen["full_state_huber"]<=1.01*baseline["full_state_huber"]; out=Path(output); out.mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),out/"combiseq_state_composer.pt"); result={"parameters":sum(p.numel() for p in model.parameters()),"training_combinations":len(train),"validation_combinations":len(valid),"baseline":baseline,"selected":chosen,"relative_residual_improvement":(baseline["residual_huber"]-chosen["residual_huber"])/baseline["residual_huber"],"full_state_preserved":bool(preserved),"improved":bool(improved),"advanced":bool(improved and preserved),"genetic_double_perturbations_used":False,"sl_labels_used":False,"history":history}; (out/"metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2)); return result

if __name__=="__main__":
    state=ROOT/"results/sl_predict/combiseq_state.npz"; build(ROOT/"data/perturbseq_sources/combosciplex/GSE174695_large_scale_matrix.csv.gz",ROOT/"data/perturbseq_sources/combosciplex/GSE174695_large_scale_meta.csv.gz",ROOT/"results/sl_predict/gse337988_scaled_med_state.npz",state); fit(state,ROOT/"results/sl_predict/combiseq_state_composition")
