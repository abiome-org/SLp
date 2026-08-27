from pathlib import Path
import csv,hashlib,json,sys
import h5py,numpy as np,pandas as pd,torch
from sklearn.decomposition import PCA

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; DATA=ROOT/"data/l1000_xpr"; sys.path.insert(0,str(Path(__file__).parent))
from world_model import SLPredict,SourceEndpoint,batches,encode_genes

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest().upper()

def build(dim=64):
    path=DATA/"xpr_coeff_mat.gctx"; meta=pd.read_csv(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"); ids={str(s).upper():int(i) for s,i in zip(meta.symbol,meta.unified_id)}; by_ens={str(e).split(".")[0]:int(i) for e,i in zip(meta.ensembl_gene_id,meta.unified_id) if pd.notna(e)}; by_entrez={str(int(e)):int(i) for e,i in zip(meta.entrez_id,meta.unified_id) if pd.notna(e)}; target_meta=pd.read_parquet(DATA/"lincs-l1000-xpr-gene_target_metadata.parquet"); target_ids={}
    for r in target_meta.itertuples():
        uid=ids.get(str(r.gene_symbol).upper(),by_ens.get(str(r.ensembl_id).split(".")[0],by_entrez.get(str(r.ncbi_gene_id))))
        if uid is not None:target_ids[str(r.gene_symbol).upper()]=uid
    cell_meta=pd.read_parquet(DATA/"lincs-l1000-xpr-cell_line_metadata.parquet"); dep=np.load(OUT/"depmap_world.npz"); state={m:s for m,s in zip(dep["model_ids"].astype(str),dep["cell_state"])}; cells={r.cell_name:(i,state[r.Cell_ID_DepMap]) for i,r in cell_meta.iterrows()}; info=pd.read_csv(DATA/"geneinfo_beta.txt",sep="\t"); landmark_info=info.loc[info.feature_space.eq("landmark")]; landmark_fixed=set(landmark_info.loc[[str(r.gene_symbol).upper() in ids or str(r.ensembl_id).split(".")[0] in by_ens or str(r.gene_id) in by_entrez for r in landmark_info.itertuples()],"gene_symbol"].astype(str).str.upper())
    with h5py.File(path,"r") as f:
        d=f["0/DATA/0/matrix"]; columns=np.asarray([x.decode() for x in f["0/META/COL/id"][:]]); rows=np.asarray([x.decode().upper() for x in f["0/META/ROW/id"][:]]); response=np.asarray([i for i,g in enumerate(rows) if g in landmark_fixed]); parts=[x.split("_",4) for x in columns]; parsed=[(p[4].upper(),p[1].split(".")[0]) if len(p)==5 and "_" not in p[4] else ("","") for p in parts]; mapped=np.asarray([t in target_ids and c in cells for t,c in parsed]); mapped_at=np.flatnonzero(mapped); keys=sorted(set(parsed[i] for i in mapped_at)); key_id={k:i for i,k in enumerate(keys)}; pair=np.full(len(columns),-1,"int32")
        for i in mapped_at:pair[i]=key_id[parsed[i]]
        sums=np.zeros((len(keys),len(response)),"float64"); count=np.zeros(len(keys),"int32"); nonfinite=0
        for start in range(0,len(columns),2048):
            stop=min(start+2048,len(columns)); keep=pair[start:stop]>=0
            if not keep.any():continue
            x=d[start:stop,response][keep]; nonfinite+=int((~np.isfinite(x)).sum()); p=pair[start:stop][keep]; order=np.argsort(p); ps=p[order]; xs=x[order]; at=np.r_[0,np.flatnonzero(ps[1:]!=ps[:-1])+1]; u=ps[at]; sums[u]+=np.add.reduceat(xs,at,axis=0); count[u]+=np.add.reduceat(np.ones(len(ps),"int32"),at)
    x=(sums/count[:,None]).astype("float32"); gene=np.asarray([target_ids[t] for t,c in keys],"int16"); cell=np.asarray([cells[c][0] for t,c in keys],"int8"); context=np.asarray([cells[c][1] for t,c in keys],"float32"); role=(gene%5==0).astype("int8"); fit=role==0; center=x[fit].mean(0); feature_scale=x[fit].std(0).clip(1e-6); standardized=(x-center)/feature_scale; pca=PCA(dim,svd_solver="randomized",random_state=1031).fit(standardized[fit]); target=pca.transform(standardized).astype("float32"); component_scale=target[fit].std(0).clip(1e-6); target/=component_scale
    cos=[]
    for g in np.unique(gene):
        q=np.flatnonzero(gene==g)
        if len(q)<4:continue
        q=np.asarray(sorted(q,key=lambda z:hashlib.sha256(f"{int(gene[z])}:{int(cell[z])}".encode()).digest())); a=x[q[::2]].mean(0); b=x[q[1::2]].mean(0); cos.append(float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8)))
    np.savez_compressed(OUT/"l1000_xpr_full_endpoint.npz",gene=gene,cell=cell,role=role,target=target,context_state=context,response_gene=rows[response],components=pca.components_.astype("float32"),center=center,feature_scale=feature_scale,component_scale=component_scale)
    audit={"schema":"sl-predict-l1000-xpr-full-source-audit-v1","input_sha256":sha(path),"input_bytes":path.stat().st_size,"geneinfo_sha256":sha(DATA/"geneinfo_beta.txt"),"matrix_shape":[len(columns),len(rows)],"identity_resolution":"exact symbol, then stable Ensembl or NCBI identifier","single_knockout_signatures":int(mapped.sum()),"excluded_non_single_or_unmapped_signatures":int((~mapped).sum()),"fixed_universe_targets":int(len(np.unique(gene))),"mapped_target_cell_states":len(keys),"mapped_cell_lines":int(len(np.unique(cell))),"fixed_universe_measured_landmarks":len(response),"nonfinite_coefficients":nonfinite,"cross_context_targets":len(cos),"cross_context_median_cosine":float(np.median(cos)),"cross_context_positive_fraction":float(np.mean(np.asarray(cos)>0)),"fitting_rows":int(fit.sum()),"held_rows":int((~fit).sum()),"held_genes":int(len(np.unique(gene[~fit]))),"dimensions":dim,"pca_explained_variance":float(pca.explained_variance_ratio_.sum()),"double_perturbation_data_used":False,"sl_labels_used":False}; audit["admitted"]=bool(audit["input_bytes"]==7016682296 and audit["single_knockout_signatures"]>=30000 and audit["fixed_universe_targets"]>=2500 and audit["mapped_target_cell_states"]>=10000 and audit["mapped_cell_lines"]==19 and audit["fixed_universe_measured_landmarks"]>=800 and nonfinite==0 and len(cos)>=1000 and audit["cross_context_median_cosine"]>=.05 and audit["cross_context_positive_fraction"]>=.6); (OUT/"l1000_xpr_full_endpoint.json").write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2)); return audit

def fit(epochs=20):
    torch.manual_seed(1031); np.random.seed(1031); device="cuda" if torch.cuda.is_available() else "cpu"; data=np.load(OUT/"l1000_xpr_full_endpoint.npz"); assert json.loads((OUT/"l1000_xpr_full_endpoint.json").read_text())["admitted"]; state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); model_dir=OUT/"native_spectral_safe_scaled_d768_z256_l8_p12_single_only_d3_t10_r3"; sd=torch.load(model_dir/"world_model.pt",map_location="cpu",weights_only=True); world=SLPredict(768,256,8,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],sd["context_proj.weight"].shape[1]).to(device); world.load_state_dict(sd); world.eval().requires_grad_(False); genes=torch.as_tensor(encode_genes(world,state,device),device=device); gene=torch.as_tensor(data["gene"].astype("int64"),device=device); context=torch.as_tensor(data["context_state"].astype("float32"),device=device); target=torch.as_tensor(data["target"].astype("float32"),device=device); train=np.flatnonzero(data["role"]==0); valid=np.flatnonzero(data["role"]==1); source=torch.zeros(len(gene),dtype=torch.long,device=device); count=np.bincount(data["gene"][train],minlength=len(state)); weight=1/count[data["gene"][train]]; weight/=weight.sum(); head=SourceEndpoint(1,256,64).to(device); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3); history=[]; saved=[]
    @torch.no_grad()
    def evaluate(epoch):
        result={"epoch":epoch,"held_rows":len(valid),"held_genes":int(len(np.unique(data["gene"][valid])))}; head.eval()
        for name,exact in (("unknown",False),("exact",True)):
            pred=[]
            for at in batches(len(valid),2048,False):
                ix=valid[at]; pred.append(head(world.transition(genes[gene[ix]],context_state=context[ix] if exact else None)[0],source[ix]).cpu())
            pred=torch.cat(pred); truth=target[valid].cpu(); row_loss=torch.nn.functional.huber_loss(pred,truth,reduction="none").mean(1).numpy(); zero_loss=torch.nn.functional.huber_loss(torch.zeros_like(truth),truth,reduction="none").mean(1).numpy(); row_cos=(pred*truth).sum(1).div(pred.norm(dim=1)*truth.norm(dim=1)+1e-8).numpy(); groups=data["gene"][valid]; macro=lambda a:float(np.mean([a[groups==g].mean() for g in np.unique(groups)])); h,z=macro(row_loss),macro(zero_loss); result[name]={"gene_macro_huber":h,"gene_macro_zero":z,"improvement":1-h/z,"gene_macro_cosine":macro(row_cos)}
        result["selection_loss"]=result["unknown"]["gene_macro_huber"]+result["exact"]["gene_macro_huber"]; history.append(result); saved.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()}); print(json.dumps(result),flush=True)
    evaluate(0)
    for epoch in range(epochs):
        chosen=np.random.default_rng(1031+epoch).choice(train,len(train),replace=True,p=weight); head.train()
        for at in batches(len(chosen),512,False):
            ix=chosen[at]; cs=context[ix]; cs=torch.where((torch.rand(len(ix),device=device)<.5)[:,None],torch.zeros_like(cs),cs); z=world.transition(genes[gene[ix]],context_state=cs)[0].detach(); loss=torch.nn.functional.huber_loss(head(z,source[ix]),target[ix]); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),1.); opt.step()
        evaluate(epoch+1)
    selected=min(range(len(history)),key=lambda i:history[i]["selection_loss"]); metric=history[selected]; head.load_state_dict(saved[selected]); advanced=metric["held_genes"]>=500 and metric["unknown"]["improvement"]>=.02 and metric["exact"]["improvement"]>=.02 and metric["unknown"]["gene_macro_cosine"]>=.10 and metric["exact"]["gene_macro_cosine"]>=.15; torch.save(head.state_dict(),model_dir/"l1000_xpr_full_endpoint.pt"); result={"schema":"sl-predict-l1000-xpr-full-endpoint-v1","parameters":sum(p.numel() for p in head.parameters()),"selected":metric,"advanced":bool(advanced),"world_parameters_changed":0,"double_perturbation_data_used":False,"sl_labels_used":False,"history":history}; (model_dir/"l1000_xpr_full_endpoint_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2)); return result

if __name__=="__main__": {"build":build,"fit":fit}[sys.argv[1] if len(sys.argv)>1 else "build"]()
