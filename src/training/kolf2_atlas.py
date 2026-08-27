from pathlib import Path
import csv,gzip,json,sys
import numpy as np,torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; DATA=ROOT/"data/kolf2"; sys.path.insert(0,str(Path(__file__).parent))
from world_model import SLPredict,SourceEndpoint,batches,encode_genes

def signatures():
    rows={}
    for line in gzip.open(DATA/"cm4ai.gmt.gz","rt"):
        p=line.rstrip().split("\t"); term=p[0]; rows[(term.split("::")[0].upper(),term.rsplit(" ",1)[-1])]=[g.upper() for g in p[2:]]
    return rows

def build(dim=64):
    symbols=[r["symbol"].upper() for r in csv.DictReader(open(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"))]; ids={g:i for i,g in enumerate(symbols)}; rows=signatures(); targets=sorted(t for t in ids if (t,"down") in rows and t in rows[(t,"down")]); ri=[]; ci=[]; value=[]
    for r,t in enumerate(targets):
        mapped=[(ids[g],s) for d,s in (("up",1.),("down",-1.)) for g in rows[(t,d)] if g in ids]; scale=len(mapped)**-.5
        for c,s in mapped:ri.append(r); ci.append(c); value.append(s*scale)
    x=sparse.csr_matrix((value,(ri,ci)),shape=(len(targets),len(symbols)),dtype="float32"); gene=np.asarray([ids[t] for t in targets],"int16"); role=(gene%5==0).astype("int8"); fit=role==0; svd=TruncatedSVD(dim,random_state=1013).fit(x[fit]); target=svd.transform(x).astype("float32"); mean=target[fit].mean(0); scale=target[fit].std(0).clip(.05); target=(target-mean)/scale; np.savez_compressed(OUT/"kolf2_single_endpoint.npz",gene=gene,role=role,target=target,components=svd.components_.astype("float32"),mean=mean,scale=scale)
    all_targets=sorted({k[0] for k in rows}); outputs={g for v in rows.values() for g in v}; overlaps=[len(set(rows[(t,"up")])&set(rows[(t,"down")])) for t in all_targets]; audit={"schema":"sl-predict-kolf2-source-audit-v1","signatures":len(rows),"targets":len(all_targets),"complete_target_pairs":sum((t,"up") in rows and (t,"down") in rows for t in all_targets),"mapped_targets":sum(t in ids for t in all_targets),"mapped_self_down_targets":len(targets),"mapped_response_genes":len(outputs&set(ids)),"duplicate_entries":sum(len(v)-len(set(v)) for v in rows.values()),"maximum_up_down_overlap":max(overlaps),"fitting_targets":int(fit.sum()),"held_targets":int((~fit).sum()),"dimensions":dim,"svd_explained_variance":float(svd.explained_variance_ratio_.sum()),"double_perturbation_data_used":False,"sl_labels_used":False}; (OUT/"kolf2_single_endpoint.json").write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2)); return audit

def fit(epochs=20):
    torch.manual_seed(1013); np.random.seed(1013); device="cuda" if torch.cuda.is_available() else "cpu"; data=np.load(OUT/"kolf2_single_endpoint.npz"); state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); model_dir=OUT/"native_spectral_safe_scaled_d768_z256_l8_p12_single_only_d3_t10_r3"; sd=torch.load(model_dir/"world_model.pt",map_location="cpu",weights_only=True); world=SLPredict(768,256,8,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],sd["context_proj.weight"].shape[1]).to(device); world.load_state_dict(sd); world.eval().requires_grad_(False); genes=torch.as_tensor(encode_genes(world,state,device),device=device); gene=torch.as_tensor(data["gene"].astype("int64"),device=device); target=torch.as_tensor(data["target"].astype("float32"),device=device); train=np.flatnonzero(data["role"]==0); valid=np.flatnonzero(data["role"]==1); source=torch.zeros(len(data["gene"]),dtype=torch.long,device=device); head=SourceEndpoint(1,256,64).to(device); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3); history=[]; saved=[]
    @torch.no_grad()
    def evaluate(epoch):
        head.eval(); pred=[]
        for at in batches(len(valid),2048,False):
            ix=valid[at]; pred.append(head(world.transition(genes[gene[ix]])[0],source[ix]).cpu())
        pred=torch.cat(pred); truth=target[valid].cpu(); huber=float(torch.nn.functional.huber_loss(pred,truth)); zero=float(torch.nn.functional.huber_loss(torch.zeros_like(truth),truth)); cosine=float((pred*truth).sum(1).div(pred.norm(dim=1)*truth.norm(dim=1)+1e-8).mean()); row={"epoch":epoch,"held_targets":len(valid),"huber":huber,"zero_huber":zero,"improvement":1-huber/zero,"cosine":cosine}; history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()}); print(json.dumps(row),flush=True)
    evaluate(0)
    for epoch in range(epochs):
        head.train()
        for at in batches(len(train),512):
            ix=train[at]; z=world.transition(genes[gene[ix]])[0].detach(); loss=torch.nn.functional.huber_loss(head(z,source[ix]),target[ix]); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),1.); opt.step()
        evaluate(epoch+1)
    selected=min(range(len(history)),key=lambda i:history[i]["huber"]); metric=history[selected]; head.load_state_dict(saved[selected]); advanced=metric["held_targets"]>=600 and metric["cosine"]>=.15 and metric["improvement"]>=.02; torch.save(head.state_dict(),model_dir/"kolf2_single_endpoint.pt"); result={"schema":"sl-predict-kolf2-single-endpoint-v1","parameters":sum(p.numel() for p in head.parameters()),"selected":metric,"advanced":bool(advanced),"world_parameters_changed":0,"double_perturbation_data_used":False,"sl_labels_used":False,"history":history}; (model_dir/"kolf2_single_endpoint_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2)); return result

if __name__=="__main__": {"build":build,"fit":fit}[sys.argv[1] if len(sys.argv)>1 else "build"]()
