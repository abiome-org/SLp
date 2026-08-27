from pathlib import Path
import csv,gzip,json,sys,zipfile
from collections import Counter
import numpy as np,pandas as pd,torch
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; RAW=ROOT/"data/yeast_sga"; sys.path.insert(0,str(Path(__file__).parent))
from tcga_relation_decoder import Head
from world_model import encode_genes,load_residual_endpoint

def mapping():
    symbols=pd.read_csv(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv").symbol.astype(str).str.upper()
    human={x:i for i,x in enumerate(symbols)}; rows=[]
    with gzip.open(RAW/"alliance_orthology_8.3.0.tsv.gz","rt") as f:
        for r in csv.DictReader((x for x in f if not x.startswith("#")),delimiter="\t"):
            if {r["Gene1SpeciesName"],r["Gene2SpeciesName"]}!={"Homo sapiens","Saccharomyces cerevisiae"} or r["IsBestScore"]!="Yes" or r["IsBestRevScore"]!="Yes":continue
            h,y=((r["Gene1Symbol"],r["Gene2Symbol"]) if r["Gene1SpeciesName"]=="Homo sapiens" else (r["Gene2Symbol"],r["Gene1Symbol"]))
            h,y=h.upper(),y.upper()
            if h in human:rows.append((h,y))
    rows=sorted(set(rows)); hc=Counter(x for x,y in rows); yc=Counter(y for x,y in rows)
    return {y:human[x] for x,y in rows if hc[x]==1 and yc[y]==1}

def build():
    ymap=mapping(); n=9845; keys=[]; values=[]; archive=zipfile.ZipFile(RAW/"costanzo2016_matrix.zip")
    for name in [x.filename for x in archive.infolist() if x.filename.endswith(".cdt")]:
        with archive.open(name) as f: header=[f.readline().decode(errors="replace").rstrip("\r\n").split("\t") for _ in range(6)]
        mapped=np.asarray([ymap.get((a or b).upper(),ymap.get(b.upper(),-1)) for a,b in zip(header[3][6:],header[2][6:])]); use=np.flatnonzero(mapped>=0); cols=[2,3,*list(6+use)]; frame=pd.read_csv(archive.open(name),sep="\t",skiprows=6,header=None,usecols=cols,na_values=[""])
        target=mapped[use]
        for row in frame.itertuples(index=False,name=None):
            a=ymap.get(str(row[1]).upper(),ymap.get(str(row[0]).upper(),-1)); x=np.asarray(row[2:],"float32"); keep=np.isfinite(x)&(target!=a)
            if a>=0 and keep.any():keys.append(a*n+target[keep]); values.append(x[keep])
    key=np.concatenate(keys).astype("int64"); value=np.concatenate(values).astype("float32"); order=np.argsort(key); key,value=key[order],value[order]; unique,start,count=np.unique(key,return_index=True,return_counts=True); total=np.add.reduceat(value,start); directed=total/count; a,b=unique//n,unique%n; lower=a<b; rev=b[lower]*n+a[lower]; at=np.searchsorted(unique,rev); reciprocal=(at<len(unique))&(unique[np.minimum(at,len(unique)-1)]==rev); fwd=directed[lower][reciprocal]; backward=directed[at[reciprocal]]; qf,qb=np.quantile(fwd,.01),np.quantile(backward,.01); pearson=float(np.corrcoef(fwd,backward)[0,1]); spearman=float(spearmanr(fwd,backward).statistic); enrichment=float(np.mean((fwd<=qf)&(backward<=qb))/.0001); pair_key=np.minimum(a,b)*n+np.maximum(a,b); po=np.argsort(pair_key); pair_key,total,count=pair_key[po],total[po],count[po]; pkey,pstart=np.unique(pair_key,return_index=True); consensus=np.add.reduceat(total,pstart)/np.add.reduceat(count,pstart); pair=np.column_stack((pkey//n,pkey%n)).astype("int16"); admitted=len(fwd)>=10000 and pearson>=.2 and spearman>=.2 and enrichment>=3; np.savez_compressed(OUT/"yeast_sga.npz",pairs=pair,epsilon=consensus.astype("float32")); result={"schema":"sl-predict-yeast-sga-v1","strict_one_to_one_human_genes":len(ymap),"mapped_directed_measurements":len(value),"mapped_directed_pairs":len(unique),"mapped_consensus_pairs":len(pair),"reciprocal_pairs":len(fwd),"reciprocal_pearson":pearson,"reciprocal_spearman":spearman,"negative_top_one_percent_overlap_enrichment":enrichment,"admitted":bool(admitted),"human_double_perturbation_data_used":False,"sl_labels_used":False}; (OUT/"yeast_sga.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); return result

def fit(epochs=12,epoch_pairs=1000000,batch=8192):
    torch.manual_seed(757); np.random.seed(757); device="cuda" if torch.cuda.is_available() else "cpu"; assert json.loads((OUT/"yeast_sga.json").read_text())["admitted"]; data=np.load(OUT/"yeast_sga.npz"); pair=data["pairs"].astype("int64"); raw=data["epsilon"].astype("float32"); train=np.flatnonzero((pair%5!=0).all(1)); valid=np.flatnonzero((pair%5==0).all(1)); mean=float(raw[train].mean()); scale=float(raw[train].std()); target=(raw-mean)/scale; state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); model_dir=OUT/"native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; endpoint=load_residual_endpoint(model_dir/"world_model.pt",state.shape[1],device); actions=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); head=Head(actions.shape[1]).to(device); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3); history=[]; saved=[]
    @torch.no_grad()
    def evaluate(epoch):
        pred=[]; head.eval()
        for lo in range(0,len(valid),batch):
            p=torch.as_tensor(pair[valid[lo:lo+batch]],device=device); pred.append(head(actions,p[:,0],p[:,1]).cpu())
        pred=torch.cat(pred).numpy(); truth=target[valid]; row={"epoch":epoch,"validation_pairs":len(valid),"huber":float(torch.nn.functional.huber_loss(torch.from_numpy(pred),torch.from_numpy(truth))),"pearson":float(np.corrcoef(truth,pred)[0,1]) if pred.std()>0 else 0.,"spearman":float(spearmanr(truth,pred).statistic) if pred.std()>0 else 0.}; history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()}); print(json.dumps(row),flush=True)
    evaluate(0)
    for epoch in range(epochs):
        chosen=np.random.default_rng(757+epoch).choice(train,epoch_pairs); head.train()
        for lo in range(0,len(chosen),batch):
            ix=chosen[lo:lo+batch]; p=torch.as_tensor(pair[ix],device=device); y=torch.as_tensor(target[ix],device=device); loss=torch.nn.functional.huber_loss(head(actions,p[:,0],p[:,1]),y); opt.zero_grad(); loss.backward(); opt.step()
        evaluate(epoch+1)
    selected=max(range(len(history)),key=lambda q:(history[q]["pearson"],history[q]["spearman"])); metric=history[selected]; head.load_state_dict(saved[selected]); advanced=metric["pearson"]>=.10 and metric["spearman"]>=.10; torch.save({"state_dict":head.state_dict(),"target_mean":mean,"target_scale":scale},model_dir/"yeast_sga_head.pt"); result={"schema":"sl-predict-yeast-sga-decoder-v1","parameters":sum(p.numel() for p in head.parameters()),"training_pairs":len(train),"selected":metric,"advanced":bool(advanced),"human_double_perturbation_data_used":False,"sl_labels_used":False,"history":history}; (model_dir/"yeast_sga_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2)); return result

if __name__=="__main__":{"build":build,"fit":fit}[sys.argv[1]]()
