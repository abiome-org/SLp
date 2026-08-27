from pathlib import Path
import csv,hashlib,json
import numpy as np,pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; DATA=ROOT/"data/tcga_pancan"

def residual(x,cancer,burden):
    r=x.astype("float32"); z=burden.astype("float32")
    for c in np.unique(cancer):
        at=cancer==c; r[:,at]-=r[:,at].mean(1,keepdims=True); z[at]-=z[at].mean()
    beta=(r@z)/(z@z+1e-8); r-=beta[:,None]*z; r/=np.linalg.norm(r,axis=1,keepdims=True).clip(1e-8); return r

def run():
    mutation=DATA/"mc3.v0.2.8.PUBLIC.nonsilentGene.xena.gz"; phenotype=pd.read_csv(DATA/"TCGA_phenotype_denseDataOnlyDownload.tsv.gz",sep="\t",dtype={"sample":str}); phenotype=phenotype[phenotype.sample_type_id.eq(1)].drop_duplicates("sample").set_index("sample"); header=pd.read_csv(mutation,sep="\t",nrows=0).columns; samples=np.asarray([x for x in header[1:] if x in phenotype.index]); dtype={x:np.int8 for x in samples}; dtype["sample"]=str; frame=pd.read_csv(mutation,sep="\t",usecols=["sample",*samples],index_col=0,dtype=dtype); burden=np.log1p(frame.to_numpy().sum(0)); meta=list(csv.DictReader(open(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"))); symbols=np.asarray([r["symbol"].upper() for r in meta]); rows={str(x).upper():i for i,x in enumerate(frame.index)}; hit=np.asarray([x in rows for x in symbols]); x=frame.to_numpy()[[rows[g] for g in symbols[hit]]]; cancer=phenotype.loc[samples,"_primary_disease"].astype(str).to_numpy(); half=np.asarray([hashlib.sha256(s[:12].encode()).digest()[0]%2 for s in samples]); counts=np.stack([x[:,half==h].sum(1) for h in (0,1)]); supported=(counts>=20).all(0); x=x[supported]; genes=np.flatnonzero(hit)[supported]; estimates=[]
    for h in (0,1):
        at=half==h; r=residual(x[:,at],cancer[at],burden[at]); estimates.append(-(r@r.T)[np.triu_indices(len(r),1)])
    pearson=float(np.corrcoef(*estimates)[0,1]); spearman=float(spearmanr(*estimates).statistic); n=len(estimates[0]); k=max(1,int(.01*n)); top=[set(np.argpartition(v,-k)[-k:]) for v in estimates]; overlap=len(top[0]&top[1]); enrichment=overlap/(.01*k); result={"schema":"sl-predict-tcga-mutual-exclusivity-v1","primary_tumors":len(samples),"cancer_types":len(np.unique(cancer)),"mapped_genes":int(hit.sum()),"supported_genes":len(genes),"supported_pairs":n,"minimum_half_count":20,"half_counts":np.bincount(half).tolist(),"pearson":pearson,"spearman":spearman,"top_one_percent_pairs":k,"top_overlap":overlap,"top_overlap_enrichment":enrichment,"advanced":bool(n>=100000 and pearson>=.15 and spearman>=.15 and enrichment>=2),"double_perturbation_data_used":False,"sl_labels_used":False}; np.savez_compressed(OUT/"tcga_mutual_exclusivity.npz",genes=genes,half0=estimates[0].astype("float32"),half1=estimates[1].astype("float32")); (OUT/"tcga_mutual_exclusivity.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); return result

def graph(k=32,dimensions=32):
    data=np.load(OUT/"tcga_mutual_exclusivity.npz"); genes=data["genes"]; n=len(genes); tri=np.triu_indices(n,1); value=np.minimum(data["half0"],data["half1"]); matrix=np.zeros((n,n),"float32"); matrix[tri]=value; matrix+=matrix.T; rows=[]; cols=[]; vals=[]
    for i in range(n):
        at=np.argpartition(matrix[i],-k)[-k:]; at=at[matrix[i,at]>0]; rows.extend([i]*len(at)); cols.extend(at); vals.extend(matrix[i,at])
    adjacency=sparse.csr_matrix((vals,(rows,cols)),shape=(n,n)); adjacency=adjacency.maximum(adjacency.T)+sparse.eye(n); degree=np.asarray(adjacency.sum(1)).ravel()**-.5; normalized=sparse.diags(degree)@adjacency@sparse.diags(degree); values,vectors=eigsh(normalized,k=dimensions,which="LA",v0=np.linspace(1,2,n)); order=np.argsort(values)[::-1]; z=(vectors[:,order]*values[order]).astype("float32"); z=(z-z.mean(0))/z.std(0).clip(1e-6); view=np.zeros((9845,dimensions),"float32"); view[genes]=z; base=np.load(OUT/"features_spectral_safe.npz"); state=base["state"].astype("float32"); assert np.max(np.abs(state[:,1024:1424]))==0; state[:,1024:1024+dimensions]=view; np.savez_compressed(OUT/"features_spectral_tcga.npz",state=state.astype("float16"),pairs=base["pairs"],relations=base["relations"],gf_hit=base["gf_hit"],esm_hit=base["esm_hit"],tcga_hit=np.isin(np.arange(len(state)),genes)); audit={"schema":"sl-predict-tcga-mutual-exclusivity-graph-v1","supported_genes":n,"directed_top_k":k,"undirected_edges":int((adjacency.nnz-n)//2),"dimensions":dimensions,"leading_eigenvalues":values[order].tolist(),"inserted_block":[1024,1024+dimensions],"sl_labels_used":False}; (OUT/"features_spectral_tcga.json").write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2)); return audit

if __name__=="__main__":
    import sys
    {"audit":run,"graph":graph}.get(sys.argv[1] if len(sys.argv)>1 else "audit")()
