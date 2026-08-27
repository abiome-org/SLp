from pathlib import Path
import csv,gzip,hashlib,json,sys
import numpy as np,pandas as pd,torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; DATA=ROOT/"data/l1000_xpr"; sys.path.insert(0,str(Path(__file__).parent))
from world_model import SLPredict,SourceEndpoint,batches,encode_genes

def signatures():
    rows={}
    for line in gzip.open(DATA/"lincs-l1000-xpr.gmt.gz","rt"):
        p=line.rstrip().split("\t"); term=p[0]; direction=term.rsplit(" ",1)[-1]; base=term.rsplit(" ",1)[0].split("::"); rows[(base[0].upper(),base[1],direction)]=[g.upper() for g in p[2:]]
    return rows

def build(dim=64):
    symbols=[r["symbol"].upper() for r in csv.DictReader(open(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"))]; ids={g:i for i,g in enumerate(symbols)}; cell_meta=pd.read_parquet(DATA/"lincs-l1000-xpr-cell_line_metadata.parquet"); dep=np.load(OUT/"depmap_world.npz"); state={m:s for m,s in zip(dep["model_ids"].astype(str),dep["cell_state"])}; cells={r.cell_name:(i,state[r.Cell_ID_DepMap]) for i,r in cell_meta.iterrows()}; rows=signatures(); keys=sorted({(t,c) for t,c,d in rows if t in ids and c in cells and (t,c,"up") in rows and (t,c,"down") in rows}); ri=[]; ci=[]; value=[]
    for r,(t,c) in enumerate(keys):
        mapped=[(ids[g],s) for d,s in (("up",1.),("down",-1.)) for g in rows[(t,c,d)] if g in ids]; scale=len(mapped)**-.5
        for col,s in mapped:ri.append(r); ci.append(col); value.append(s*scale)
    x=sparse.csr_matrix((value,(ri,ci)),shape=(len(keys),len(symbols)),dtype="float32"); gene=np.asarray([ids[t] for t,c in keys],"int16"); cell=np.asarray([cells[c][0] for t,c in keys],"int8"); context=np.asarray([cells[c][1] for t,c in keys],"float32"); role=(gene%5==0).astype("int8"); fit=role==0; svd=TruncatedSVD(dim,random_state=1031).fit(x[fit]); target=svd.transform(x).astype("float32"); mean=target[fit].mean(0); scale=target[fit].std(0).clip(.05); target=(target-mean)/scale; np.savez_compressed(OUT/"l1000_xpr_endpoint.npz",gene=gene,cell=cell,role=role,target=target,context_state=context,components=svd.components_.astype("float32"),mean=mean,scale=scale)
    cos=[]
    for g in np.unique(gene):
        q=np.flatnonzero(gene==g)
        if len(q)<4:continue
        q=sorted(q,key=lambda z:hashlib.sha256(f"{int(gene[z])}:{int(cell[z])}".encode()).digest()); a=x[q[::2]].mean(0); b=x[q[1::2]].mean(0); cos.append(float((a@b.T/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8)).item()))
    all_pairs={(t,c) for t,c,d in rows}; all_targets={t for t,c,d in rows}; outputs={g for v in rows.values() for g in v}; overlaps=[len(set(rows[(t,c,"up")])&set(rows[(t,c,"down")])) for t,c in all_pairs]; audit={"schema":"sl-predict-l1000-xpr-source-audit-v1","signatures":len(rows),"complete_pairs":sum((t,c,"up") in rows and (t,c,"down") in rows for t,c in all_pairs),"targets":len(all_targets),"mapped_targets":len({t for t,c in keys}),"mapped_target_cell_states":len(keys),"mapped_response_genes":len(outputs&set(ids)),"cell_lines":len({c for t,c in keys}),"duplicate_entries":sum(len(v)-len(set(v)) for v in rows.values()),"maximum_up_down_overlap":max(overlaps),"cross_context_targets":len(cos),"cross_context_median_cosine":float(np.median(cos)),"cross_context_positive_fraction":float(np.mean(np.asarray(cos)>0)),"fitting_rows":int(fit.sum()),"held_rows":int((~fit).sum()),"held_genes":int(len(np.unique(gene[~fit]))),"dimensions":dim,"svd_explained_variance":float(svd.explained_variance_ratio_.sum()),"double_perturbation_data_used":False,"sl_labels_used":False}; audit["admitted"]=bool(audit["complete_pairs"]>=30000 and audit["mapped_targets"]>=2500 and len(keys)>=10000 and audit["mapped_response_genes"]>=9000 and audit["duplicate_entries"]==0 and audit["maximum_up_down_overlap"]==0 and len(cos)>=1000 and audit["cross_context_median_cosine"]>=.05 and audit["cross_context_positive_fraction"]>=.6); (OUT/"l1000_xpr_endpoint.json").write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2)); return audit

def fit(epochs=20):
    torch.manual_seed(1031); np.random.seed(1031); device="cuda" if torch.cuda.is_available() else "cpu"; data=np.load(OUT/"l1000_xpr_endpoint.npz"); assert json.loads((OUT/"l1000_xpr_endpoint.json").read_text())["admitted"]; state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); model_dir=OUT/"native_spectral_safe_scaled_d768_z256_l8_p12_single_only_d3_t10_r3"; sd=torch.load(model_dir/"world_model.pt",map_location="cpu",weights_only=True); world=SLPredict(768,256,8,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],sd["context_proj.weight"].shape[1]).to(device); world.load_state_dict(sd); world.eval().requires_grad_(False); genes=torch.as_tensor(encode_genes(world,state,device),device=device); gene=torch.as_tensor(data["gene"].astype("int64"),device=device); context=torch.as_tensor(data["context_state"].astype("float32"),device=device); target=torch.as_tensor(data["target"].astype("float32"),device=device); train=np.flatnonzero(data["role"]==0); valid=np.flatnonzero(data["role"]==1); source=torch.zeros(len(gene),dtype=torch.long,device=device); count=np.bincount(data["gene"][train],minlength=len(state)); weight=1/count[data["gene"][train]]; weight/=weight.sum(); head=SourceEndpoint(1,256,64).to(device); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3); history=[]; saved=[]
    @torch.no_grad()
    def evaluate(epoch):
        result={"epoch":epoch,"held_rows":len(valid),"held_genes":int(len(np.unique(data["gene"][valid])))}; head.eval()
        for name,exact in (("unknown",False),("exact",True)):
            pred=[]
            for at in batches(len(valid),2048,False):
                ix=valid[at]; pred.append(head(world.transition(genes[gene[ix]],context_state=context[ix] if exact else None)[0],source[ix]).cpu())
            pred=torch.cat(pred); truth=target[valid].cpu(); row_loss=torch.nn.functional.huber_loss(pred,truth,reduction="none").mean(1).numpy(); zero_loss=torch.nn.functional.huber_loss(torch.zeros_like(truth),truth,reduction="none").mean(1).numpy(); row_cos=(pred*truth).sum(1).div(pred.norm(dim=1)*truth.norm(dim=1)+1e-8).numpy(); groups=data["gene"][valid]; macro=lambda x:float(np.mean([x[groups==g].mean() for g in np.unique(groups)])); h,z=macro(row_loss),macro(zero_loss); result[name]={"gene_macro_huber":h,"gene_macro_zero":z,"improvement":1-h/z,"gene_macro_cosine":macro(row_cos)}
        result["selection_loss"]=result["unknown"]["gene_macro_huber"]+result["exact"]["gene_macro_huber"]; history.append(result); saved.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()}); print(json.dumps(result),flush=True)
    evaluate(0)
    for epoch in range(epochs):
        chosen=np.random.default_rng(1031+epoch).choice(train,len(train),replace=True,p=weight); head.train()
        for at in batches(len(chosen),512,False):
            ix=chosen[at]; cs=context[ix]; cs=torch.where((torch.rand(len(ix),device=device)<.5)[:,None],torch.zeros_like(cs),cs); z=world.transition(genes[gene[ix]],context_state=cs)[0].detach(); loss=torch.nn.functional.huber_loss(head(z,source[ix]),target[ix]); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),1.); opt.step()
        evaluate(epoch+1)
    selected=min(range(len(history)),key=lambda i:history[i]["selection_loss"]); metric=history[selected]; head.load_state_dict(saved[selected]); advanced=metric["held_genes"]>=500 and metric["unknown"]["improvement"]>=.02 and metric["exact"]["improvement"]>=.02 and metric["unknown"]["gene_macro_cosine"]>=.10 and metric["exact"]["gene_macro_cosine"]>=.15; torch.save(head.state_dict(),model_dir/"l1000_xpr_endpoint.pt"); result={"schema":"sl-predict-l1000-xpr-endpoint-v1","parameters":sum(p.numel() for p in head.parameters()),"selected":metric,"advanced":bool(advanced),"world_parameters_changed":0,"double_perturbation_data_used":False,"sl_labels_used":False,"history":history}; (model_dir/"l1000_xpr_endpoint_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2)); return result

if __name__=="__main__": {"build":build,"fit":fit}[sys.argv[1] if len(sys.argv)>1 else "build"]()
