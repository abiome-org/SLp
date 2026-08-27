import csv, json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"

def pairs(rng,n,count):
    keys=np.empty(0,"int64")
    while len(keys)<count:
        x=rng.integers(0,n,(int(1.2*(count-len(keys)))+1000,2)); x.sort(1); x=x[x[:,0]!=x[:,1]]; keys=np.unique(np.r_[keys,x[:,0]*n+x[:,1]])
    keys=keys[:count]; return keys//n,keys%n

def cosine(z,a,b,batch=16384):
    out=[]
    for lo in range(0,len(a),batch): out.append((z[torch.as_tensor(a[lo:lo+batch],device=z.device)]*z[torch.as_tensor(b[lo:lo+batch],device=z.device)]).sum(1).cpu())
    return torch.cat(out).numpy()

def overlap(a,b,q=.99): return float(np.mean((a>=np.quantile(a,q))&(b>=np.quantile(b,q)))/(1-q)**2)

def main(sample_pairs=2_000_000):
    rows=list(csv.DictReader(open(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"))); symbols=np.asarray([r["symbol"].upper() for r in rows]); raw=torch.load(ROOT/"data/models/weights/SE-600M/protein_embeddings.pt",map_location="cpu",weights_only=True); codep=np.load(OUT/"depmap_codependency.npz"); pos=np.full(len(symbols),-1,"int32"); pos[codep["genes"].astype("int64")]=np.arange(len(codep["genes"])); eligible=np.flatnonzero(np.asarray([g in raw for g in symbols])&(pos>=0)); i,j=pairs(np.random.default_rng(769),len(eligible),min(sample_pairs,len(eligible)*(len(eligible)-1)//2)); a,b=eligible[i],eligible[j]
    device="cuda" if torch.cuda.is_available() else "cpu"; z=torch.nn.functional.normalize(torch.stack([raw[g] for g in symbols[eligible]]).to(device),dim=1); protein=cosine(z,i,j); state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); function=np.mean([np.sum(state[a,lo:lo+32]*state[b,lo:lo+32],1)/(np.linalg.norm(state[a,lo:lo+32],axis=1)*np.linalg.norm(state[b,lo:lo+32],axis=1)+1e-6) for lo in (1688,1720,1752,1784)],axis=0); dependency=(codep["half0"][pos[a],pos[b]].astype("float32")+codep["half1"][pos[a],pos[b]].astype("float32"))/2
    result={"schema":"sl-predict-exact-protein-redundancy-v1","eligible_genes":len(eligible),"validation_sample_pairs":len(a),"protein_function_spearman":float(spearmanr(protein,function).statistic),"protein_function_top_one_percent_overlap_enrichment":overlap(protein,function),"protein_codependency_top_one_percent_overlap_enrichment":overlap(protein,dependency),"function_codependency_top_one_percent_overlap_enrichment":overlap(function,dependency),"double_perturbation_data_used":False,"sl_labels_used":False}; result["admitted"]=bool(result["protein_function_spearman"]>=.10 and result["protein_function_top_one_percent_overlap_enrichment"]>=5 and result["protein_codependency_top_one_percent_overlap_enrichment"]>=3 and result["function_codependency_top_one_percent_overlap_enrichment"]>=3); (OUT/"exact_protein_redundancy.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
