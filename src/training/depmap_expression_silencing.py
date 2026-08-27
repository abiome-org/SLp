from pathlib import Path
import csv,hashlib,json,sys
import numpy as np,torch
from scipy.stats import spearmanr
from depmap_crossmodal_relation import ROOT,OUT,DATA,expression,pairs,md5
from tcga_relation_decoder import Head
from world_model import SLPredict,encode_genes,load_residual_endpoint


def residual(value,known,lineage):
    out=np.where(known,value,np.nan).astype("float32"); global_mean=np.nanmean(out,0)
    for group in np.unique(lineage):
        rows=lineage==group; mean=np.nanmean(out[rows],0) if rows.sum()>=5 else global_mean; out[rows]-=np.where(np.isfinite(mean),mean,global_mean)
    return out


def relation(ex,dependency,known,lineage,device,chunk=256):
    xr=residual(ex,np.isfinite(ex),lineage); yr=residual(dependency,known,lineage); yr/=np.nanstd(yr,0).clip(.1); q0,q1=np.nanquantile(xr,(.25,.75),axis=0); yt=torch.as_tensor(np.nan_to_num(yr),device=device); kt=torch.as_tensor(np.isfinite(yr).astype("float32"),device=device); n=xr.shape[1]; directed=np.empty((n,n),"float16")
    for at in range(0,n,chunk):
        lo=torch.as_tensor((xr[:,at:at+chunk]<=q0[None,at:at+chunk]).astype("float32"),device=device); hi=torch.as_tensor((xr[:,at:at+chunk]>=q1[None,at:at+chunk]).astype("float32"),device=device); hc=(hi.T@kt).clamp(1); lc=(lo.T@kt).clamp(1); directed[at:at+chunk]=((hi.T@yt)/hc-(lo.T@yt)/lc).half().cpu().numpy()
    symmetric=np.empty_like(directed)
    for at in range(0,n,chunk): symmetric[at:at+chunk]=((directed[at:at+chunk].astype("float32")+directed[:,at:at+chunk].T.astype("float32"))/2).astype("float16")
    np.fill_diagonal(symmetric,0); return symmetric


def build(sample_pairs=2000000):
    source=np.load(OUT/"basal_context.npz"); ids=source["model_ids"].astype(str); ex,path=expression(ids); dependency=source["dependency"].astype("float32"); known=source["dependency_known"]; lineage_map={r["ModelID"]:(r["OncotreeLineage"] or "unknown") for r in csv.DictReader((DATA/"Model.csv").open())}; lineage=np.asarray([lineage_map.get(x,"unknown") for x in ids]); parity=np.asarray([hashlib.sha256(x.encode()).digest()[0]&1 for x in ids]); halves=[source["train_cell"]&(parity==q) for q in (0,1)]; eligible=np.ones(ex.shape[1],bool)
    for h in halves:
        eligible&=np.isfinite(ex[h]).mean(0)>=.8; eligible&=known[h].mean(0)>=.8; eligible&=np.nanstd(residual(ex[h],np.isfinite(ex[h]),lineage[h]),0)>1e-4
    genes=np.flatnonzero(eligible); device="cuda" if torch.cuda.is_available() else "cpu"; matrices=[relation(ex[h][:,genes],dependency[h][:,genes],known[h][:,genes],lineage[h],device) for h in halves]; rng=np.random.default_rng(787); i,j=pairs(rng,len(genes),min(sample_pairs,len(genes)*(len(genes)-1)//2)); a=matrices[0][i,j].astype("float32"); b=matrices[1][i,j].astype("float32"); qa,qb=np.quantile(a,.99),np.quantile(b,.99); pearson=float(np.corrcoef(a,b)[0,1]); spearman=float(spearmanr(a,b).statistic); enrichment=float(np.mean((a>=qa)&(b>=qb))/.0001); admitted=pearson>=.15 and spearman>=.15 and enrichment>=3; np.savez_compressed(OUT/"depmap_expression_silencing.npz",genes=genes.astype("int16"),half0=matrices[0],half1=matrices[1]); result={"schema":"sl-predict-depmap-expression-silencing-v1","expression_file":str(path.relative_to(ROOT)).replace('\\','/'),"expression_md5":md5(path),"cells_per_half":[int(h.sum()) for h in halves],"eligible_genes":len(genes),"eligible_pairs":len(genes)*(len(genes)-1)//2,"validation_sample_pairs":len(i),"split_half_pearson":pearson,"split_half_spearman":spearman,"top_one_percent_overlap_enrichment":enrichment,"admitted":bool(admitted),"double_perturbation_data_used":False,"sl_labels_used":False}; (OUT/"depmap_expression_silencing.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); return result


def fit(model_path,out_dir,kind,d,latent,layers,epochs=12,epoch_pairs=1000000,batch=8192):
    torch.manual_seed(789); np.random.seed(789); device="cuda" if torch.cuda.is_available() else "cpu"; assert json.loads((OUT/"depmap_expression_silencing.json").read_text())["admitted"]; data=np.load(OUT/"depmap_expression_silencing.npz"); supported=data["genes"].astype("int64"); half0=data["half0"]; half1=data["half1"]; train=np.flatnonzero(supported%5!=0); valid=np.flatnonzero(supported%5==0); rng=np.random.default_rng(789); si,sj=pairs(rng,len(train),min(2000000,len(train)*(len(train)-1)//2)); stats=half0[train[si],train[sj]].astype("float32"); mean=float(stats.mean()); scale=float(stats.std()); vi,vj=np.triu_indices(len(valid),1); state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32")
    if kind=="residual": world=load_residual_endpoint(model_path,state.shape[1],device).world
    else:
        sd=torch.load(model_path,map_location="cpu",weights_only=True); context_dim=sd["context_proj.weight"].shape[1] if "context_proj.weight" in sd else 0; world=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],context_dim).to(device); world.load_state_dict(sd); world.eval().requires_grad_(False)
    actions=torch.as_tensor(encode_genes(world,state,device)[supported],device=device); head=Head(latent).to(device); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3); history=[]; saved=[]
    @torch.no_grad()
    def evaluate(epoch):
        pred=[]; head.eval()
        for lo in range(0,len(vi),batch):
            a=torch.as_tensor(valid[vi[lo:lo+batch]],device=device); b=torch.as_tensor(valid[vj[lo:lo+batch]],device=device); pred.append(head(actions,a,b).cpu())
        pred=torch.cat(pred).numpy(); truth=(half1[valid[vi],valid[vj]].astype("float32")-mean)/scale; row={"epoch":epoch,"validation_pairs":len(vi),"huber":float(torch.nn.functional.huber_loss(torch.from_numpy(pred),torch.from_numpy(truth))),"pearson":float(np.corrcoef(truth,pred)[0,1]) if pred.std()>0 else 0.,"spearman":float(spearmanr(truth,pred).statistic) if pred.std()>0 else 0.}; history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()}); print(json.dumps(row),flush=True)
    evaluate(0)
    for epoch in range(epochs):
        rng=np.random.default_rng(789+epoch); a=rng.choice(train,epoch_pairs); b=rng.choice(train,epoch_pairs); same=a==b
        while same.any(): b[same]=rng.choice(train,same.sum()); same=a==b
        head.train()
        for lo in range(0,epoch_pairs,batch):
            aa=torch.as_tensor(a[lo:lo+batch],device=device); bb=torch.as_tensor(b[lo:lo+batch],device=device); y=torch.as_tensor((half0[a[lo:lo+batch],b[lo:lo+batch]].astype("float32")-mean)/scale,device=device); loss=torch.nn.functional.huber_loss(head(actions,aa,bb),y); opt.zero_grad(); loss.backward(); opt.step()
        evaluate(epoch+1)
    selected=max(range(len(history)),key=lambda q:(history[q]["pearson"],history[q]["spearman"])); metric=history[selected]; head.load_state_dict(saved[selected]); advanced=metric["pearson"]>=.15 and metric["spearman"]>=.15; out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":head.state_dict(),"target_mean":mean,"target_scale":scale,"supported_genes":torch.as_tensor(supported)},out/f"depmap_expression_silencing_{kind}_head.pt"); result={"schema":"sl-predict-depmap-expression-silencing-decoder-v1","representation":kind,"parameters":sum(p.numel() for p in head.parameters()),"training_genes":len(train),"possible_training_pairs":len(train)*(len(train)-1)//2,"selected":metric,"advanced":bool(advanced),"double_perturbation_data_used":False,"sl_labels_used":False,"history":history}; (out/f"depmap_expression_silencing_{kind}_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2)); return result


if __name__=="__main__": build(int(sys.argv[1]) if len(sys.argv)>1 else 2000000)
