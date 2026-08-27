from pathlib import Path
import hashlib,json,sys
import numpy as np,torch
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; sys.path.insert(0,str(Path(__file__).parent))
from tcga_relation_decoder import Head
from world_model import SLPredict,encode_genes,load_residual_endpoint

def pairs(rng,n,count):
    keys=np.empty(0,"int64")
    while len(keys)<count:
        x=rng.integers(0,n,(int(1.2*(count-len(keys)))+1000,2)); x.sort(1); x=x[x[:,0]!=x[:,1]]; keys=np.unique(np.r_[keys,x[:,0]*n+x[:,1]])
    keys=keys[:count]; return keys//n,keys%n

def build(sample_pairs=2000000):
    z=np.load(OUT/"basal_context.npz"); permitted=z["train_cell"]; ids=z["model_ids"].astype(str); parity=np.asarray([hashlib.sha256(x.encode()).digest()[0]&1 for x in ids]); halves=[permitted&(parity==q) for q in (0,1)]; known=z["dependency_known"]; eligible=z["train_gene"]&np.asarray([(known[h].mean(0)>=.8) for h in halves]).all(0); genes=np.flatnonzero(eligible); device="cuda" if torch.cuda.is_available() else "cpu"; matrices=[]
    for h in halves:
        k=known[h][:,genes]; x=z["dependency"][h][:,genes].astype("float32"); mean=(x*k).sum(0)/k.sum(0); x=np.where(k,x-mean,0); x/=np.sqrt((x*x).sum(0)).clip(1e-6); t=torch.as_tensor(x,device=device); matrices.append((t.T@t).clamp(-1,1).half().cpu().numpy()); del t
        if device=="cuda":torch.cuda.empty_cache()
    rng=np.random.default_rng(751); i,j=pairs(rng,len(genes),min(sample_pairs,len(genes)*(len(genes)-1)//2)); a=matrices[0][i,j].astype("float32"); b=matrices[1][i,j].astype("float32"); qa,qb=np.quantile(a,.99),np.quantile(b,.99); pearson=float(np.corrcoef(a,b)[0,1]); spearman=float(spearmanr(a,b).statistic); enrichment=float(np.mean((a>=qa)&(b>=qb))/.0001); admitted=pearson>=.15 and spearman>=.15 and enrichment>=3; np.savez_compressed(OUT/"depmap_codependency.npz",genes=genes.astype("int16"),half0=matrices[0],half1=matrices[1]); result={"schema":"sl-predict-depmap-codependency-v1","cells_per_half":[int(x.sum()) for x in halves],"eligible_genes":len(genes),"eligible_pairs":len(genes)*(len(genes)-1)//2,"validation_sample_pairs":len(i),"split_half_pearson":pearson,"split_half_spearman":spearman,"top_one_percent_overlap_enrichment":enrichment,"admitted":bool(admitted),"double_perturbation_data_used":False,"sl_labels_used":False}; (OUT/"depmap_codependency.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); return result

def fit(epochs=12,epoch_pairs=1000000,batch=8192):
    torch.manual_seed(753); np.random.seed(753); device="cuda" if torch.cuda.is_available() else "cpu"; source=json.loads((OUT/"depmap_codependency.json").read_text()); assert source["admitted"]; data=np.load(OUT/"depmap_codependency.npz"); supported=data["genes"].astype("int64"); half0=data["half0"]; half1=data["half1"]; train=np.flatnonzero(supported%5!=0); valid=np.flatnonzero(supported%5==0); rng=np.random.default_rng(753); si,sj=pairs(rng,len(train),min(2000000,len(train)*(len(train)-1)//2)); stats=half0[train[si],train[sj]].astype("float32"); mean=float(stats.mean()); scale=float(stats.std()); vi,vj=np.triu_indices(len(valid),1); state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); model_dir=OUT/"native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; endpoint=load_residual_endpoint(model_dir/"world_model.pt",state.shape[1],device); actions=torch.as_tensor(encode_genes(endpoint.world,state,device)[supported],device=device); head=Head(actions.shape[1]).to(device); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3); history=[]; saved=[]
    @torch.no_grad()
    def evaluate(epoch):
        pred=[]; head.eval()
        for lo in range(0,len(vi),batch):
            a=torch.as_tensor(valid[vi[lo:lo+batch]],device=device); b=torch.as_tensor(valid[vj[lo:lo+batch]],device=device); pred.append(head(actions,a,b).cpu())
        pred=torch.cat(pred).numpy(); truth=(half1[valid[vi],valid[vj]].astype("float32")-mean)/scale; row={"epoch":epoch,"validation_pairs":len(vi),"huber":float(torch.nn.functional.huber_loss(torch.from_numpy(pred),torch.from_numpy(truth))),"pearson":float(np.corrcoef(truth,pred)[0,1]) if pred.std()>0 else 0.,"spearman":float(spearmanr(truth,pred).statistic) if pred.std()>0 else 0.}; history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()}); print(json.dumps(row),flush=True)
    evaluate(0)
    for epoch in range(epochs):
        rng=np.random.default_rng(753+epoch); a=rng.choice(train,epoch_pairs); b=rng.choice(train,epoch_pairs); same=a==b
        while same.any():b[same]=rng.choice(train,same.sum()); same=a==b
        head.train()
        for lo in range(0,epoch_pairs,batch):
            aa=torch.as_tensor(a[lo:lo+batch],device=device); bb=torch.as_tensor(b[lo:lo+batch],device=device); y=torch.as_tensor((half0[a[lo:lo+batch],b[lo:lo+batch]].astype("float32")-mean)/scale,device=device); loss=torch.nn.functional.huber_loss(head(actions,aa,bb),y); opt.zero_grad(); loss.backward(); opt.step()
        evaluate(epoch+1)
    selected=max(range(len(history)),key=lambda q:(history[q]["pearson"],history[q]["spearman"])); metric=history[selected]; head.load_state_dict(saved[selected]); advanced=metric["pearson"]>=.15 and metric["spearman"]>=.15; torch.save({"state_dict":head.state_dict(),"target_mean":mean,"target_scale":scale,"supported_genes":torch.as_tensor(supported)},model_dir/"depmap_codependency_head.pt"); result={"schema":"sl-predict-depmap-codependency-decoder-v1","parameters":sum(p.numel() for p in head.parameters()),"training_genes":len(train),"possible_training_pairs":len(train)*(len(train)-1)//2,"selected":metric,"advanced":bool(advanced),"double_perturbation_data_used":False,"sl_labels_used":False,"history":history}; (model_dir/"depmap_codependency_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2)); return result

def fit_world(model_path,out_dir,d=768,latent=256,layers=8,epochs=12,epoch_pairs=1000000,batch=8192):
    torch.manual_seed(753); np.random.seed(753); device="cuda" if torch.cuda.is_available() else "cpu"; data=np.load(OUT/"depmap_codependency.npz"); supported=data["genes"].astype("int64"); half0=data["half0"]; half1=data["half1"]; train=np.flatnonzero(supported%5!=0); valid=np.flatnonzero(supported%5==0); rng=np.random.default_rng(753); si,sj=pairs(rng,len(train),min(2000000,len(train)*(len(train)-1)//2)); stats=half0[train[si],train[sj]].astype("float32"); mean=float(stats.mean()); scale=float(stats.std()); vi,vj=np.triu_indices(len(valid),1); state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); sd=torch.load(model_path,map_location="cpu",weights_only=True); context_dim=sd["context_proj.weight"].shape[1] if "context_proj.weight" in sd else 0; world=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],context_dim).to(device); world.load_state_dict(sd); world.eval().requires_grad_(False); actions=torch.as_tensor(encode_genes(world,state,device)[supported],device=device); head=Head(latent).to(device); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3); history=[]; saved=[]
    @torch.no_grad()
    def evaluate(epoch):
        pred=[]; head.eval()
        for lo in range(0,len(vi),batch):
            a=torch.as_tensor(valid[vi[lo:lo+batch]],device=device); b=torch.as_tensor(valid[vj[lo:lo+batch]],device=device); pred.append(head(actions,a,b).cpu())
        pred=torch.cat(pred).numpy(); truth=(half1[valid[vi],valid[vj]].astype("float32")-mean)/scale; row={"epoch":epoch,"validation_pairs":len(vi),"huber":float(torch.nn.functional.huber_loss(torch.from_numpy(pred),torch.from_numpy(truth))),"pearson":float(np.corrcoef(truth,pred)[0,1]) if pred.std()>0 else 0.,"spearman":float(spearmanr(truth,pred).statistic) if pred.std()>0 else 0.}; history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()}); print(json.dumps(row),flush=True)
    evaluate(0)
    for epoch in range(epochs):
        rng=np.random.default_rng(753+epoch); a=rng.choice(train,epoch_pairs); b=rng.choice(train,epoch_pairs); same=a==b
        while same.any():b[same]=rng.choice(train,same.sum()); same=a==b
        head.train()
        for lo in range(0,epoch_pairs,batch):
            aa=torch.as_tensor(a[lo:lo+batch],device=device); bb=torch.as_tensor(b[lo:lo+batch],device=device); y=torch.as_tensor((half0[a[lo:lo+batch],b[lo:lo+batch]].astype("float32")-mean)/scale,device=device); loss=torch.nn.functional.huber_loss(head(actions,aa,bb),y); opt.zero_grad(); loss.backward(); opt.step()
        evaluate(epoch+1)
    selected=max(range(len(history)),key=lambda q:(history[q]["pearson"],history[q]["spearman"])); metric=history[selected]; head.load_state_dict(saved[selected]); advanced=metric["pearson"]>=.15 and metric["spearman"]>=.15; out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":head.state_dict(),"target_mean":mean,"target_scale":scale,"supported_genes":torch.as_tensor(supported)},out/"depmap_codependency_head.pt"); result={"schema":"sl-predict-scaled-depmap-codependency-decoder-v1","parameters":sum(p.numel() for p in head.parameters()),"training_genes":len(train),"possible_training_pairs":len(train)*(len(train)-1)//2,"selected":metric,"advanced":bool(advanced),"double_perturbation_data_used":False,"sl_labels_used":False,"history":history}; (out/"depmap_codependency_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2)); return result

if __name__=="__main__":{"build":build,"fit":fit}[sys.argv[1]]()
