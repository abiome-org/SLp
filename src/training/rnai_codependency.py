import hashlib,json
from pathlib import Path
import numpy as np,torch
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"

def sample(rng,n,count):
    keys=np.empty(0,"int64")
    while len(keys)<count:
        x=rng.integers(0,n,(int(1.2*(count-len(keys)))+1000,2)); x.sort(1); x=x[x[:,0]!=x[:,1]]; keys=np.unique(np.r_[keys,x[:,0]*n+x[:,1]])
    keys=keys[:count]; return keys//n,keys%n

def main(count=2000000):
    z=np.load(OUT/"rnai_dependency.npz"); ids=z["model_ids"].astype(str); values=z["rnai"].astype("float32"); known=z["known"]; genes=z["gene_index"].astype("int64"); parity=np.asarray([hashlib.sha256(x.encode()).digest()[0]&1 for x in ids]); halves=[]; eligible=np.ones(len(genes),bool)
    for q in (0,1): eligible&=known[parity==q].mean(0)>=.8
    genes=genes[eligible]; values=values[:,eligible]; known=known[:,eligible]; device="cuda" if torch.cuda.is_available() else "cpu"; matrices=[]
    for q in (0,1):
        k=known[parity==q]; x=values[parity==q]; mean=np.where(k,x,0).sum(0)/k.sum(0); x=np.where(k,x-mean,0); x/=np.sqrt((x*x).sum(0)).clip(1e-6); t=torch.as_tensor(x,device=device); matrices.append((t.T@t).clamp(-1,1).half().cpu().numpy()); del t
    rng=np.random.default_rng(941); i,j=sample(rng,len(genes),min(count,len(genes)*(len(genes)-1)//2)); a=matrices[0][i,j].astype("float32"); b=matrices[1][i,j].astype("float32"); qa,qb=np.quantile(a,.99),np.quantile(b,.99); internal={"pearson":float(np.corrcoef(a,b)[0,1]),"spearman":float(spearmanr(a,b).statistic),"top_one_percent_overlap_enrichment":float(np.mean((a>=qa)&(b>=qb))/.0001)}
    d=np.load(OUT/"depmap_codependency.npz"); pos={g:k for k,g in enumerate(d["genes"].astype("int64"))}; keep=np.asarray([(genes[x] in pos and genes[y] in pos) for x,y in zip(i,j)]); di=np.asarray([pos[genes[x]] for x in i[keep]]); dj=np.asarray([pos[genes[y]] for y in j[keep]]); r=(a[keep]+b[keep])/2; c=(d["half0"][di,dj].astype("float32")+d["half1"][di,dj].astype("float32"))/2; cross={"pairs":len(r),"pearson":float(np.corrcoef(r,c)[0,1]),"spearman":float(spearmanr(r,c).statistic)}; admitted=len(genes)>=8000 and internal["pearson"]>=.15 and internal["spearman"]>=.15 and internal["top_one_percent_overlap_enrichment"]>=3 and cross["pearson"]>=.10 and cross["spearman"]>=.10
    np.savez_compressed(OUT/"rnai_codependency.npz",genes=genes.astype("int16"),half0=matrices[0],half1=matrices[1]); result={"schema":"sl-predict-rnai-codependency-v1","cells_per_half":[int((parity==q).sum()) for q in (0,1)],"eligible_genes":len(genes),"validation_pairs":len(a),"split_half":internal,"cross_platform_crispr":cross,"admitted":bool(admitted),"double_perturbation_data_used":False,"sl_labels_used":False}; (OUT/"rnai_codependency.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
