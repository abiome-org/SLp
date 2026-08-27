from pathlib import Path
import json,sys
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"


def pairs(rng,n,count):
    keys=np.empty(0,"int64")
    while len(keys)<count:
        x=rng.integers(0,n,(int(1.2*(count-len(keys)))+1000,2)); x.sort(1); x=x[x[:,0]!=x[:,1]]; keys=np.unique(np.r_[keys,x[:,0]*n+x[:,1]])
    keys=keys[:count]; return keys//n,keys%n


def normalized(x): return x/np.linalg.norm(x,axis=1,keepdims=True).clip(1e-6)


def cosine(z,a,b,batch=100000):
    out=np.empty(len(a),"float32")
    for lo in range(0,len(a),batch): out[lo:lo+batch]=(z[a[lo:lo+batch]]*z[b[lo:lo+batch]]).sum(1)
    return out


def overlap(a,b,q=.99): return float(np.mean((a>=np.quantile(a,q))&(b>=np.quantile(b,q)))/(1-q)**2)


def build(sample_pairs=2000000):
    feature=np.load(OUT/"features_spectral_safe.npz"); state=feature["state"].astype("float32"); codep=np.load(OUT/"depmap_codependency.npz"); pos=np.full(len(state),-1,"int32"); pos[codep["genes"].astype("int64")]=np.arange(len(codep["genes"])); eligible=np.flatnonzero(feature["esm_hit"]&(pos>=0)); i,j=pairs(np.random.default_rng(769),len(eligible),min(sample_pairs,len(eligible)*(len(eligible)-1)//2)); a,b=eligible[i],eligible[j]; protein=cosine(normalized(state[:,768:1024]),a,b); function=np.mean([cosine(normalized(state[:,lo:lo+32]),a,b) for lo in (1688,1720,1752,1784)],axis=0); c0,c1=pos[a],pos[b]; dependency=(codep["half0"][c0,c1].astype("float32")+codep["half1"][c0,c1].astype("float32"))/2; spearman=float(spearmanr(protein,function).statistic); pf=overlap(protein,function); pc=overlap(protein,dependency); fc=overlap(function,dependency); admitted=spearman>=.10 and pf>=5 and pc>=3 and fc>=3; result={"schema":"sl-predict-functional-redundancy-v1","eligible_genes":len(eligible),"validation_sample_pairs":len(a),"protein_function_spearman":spearman,"protein_function_top_one_percent_overlap_enrichment":pf,"protein_codependency_top_one_percent_overlap_enrichment":pc,"function_codependency_top_one_percent_overlap_enrichment":fc,"admitted":bool(admitted),"double_perturbation_data_used":False,"sl_labels_used":False}; (OUT/"functional_redundancy.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); return result


if __name__=="__main__": build(int(sys.argv[1]) if len(sys.argv)>1 else 2000000)
