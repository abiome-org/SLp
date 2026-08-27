import hashlib,io,json,re,struct,zlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"

def norm(x): return re.sub(r"V\d+(?:\d+)?$","",re.sub(r"[^A-Z0-9]","",str(x).upper()))
def sample(rng,n,count):
    keys=np.empty(0,"int64")
    while len(keys)<count:
        x=rng.integers(0,n,(int(1.2*(count-len(keys)))+1000,2)); x.sort(1); x=x[x[:,0]!=x[:,1]]; keys=np.unique(np.r_[keys,x[:,0]*n+x[:,1]])
    keys=keys[:count]; return keys//n,keys%n
def matrix(x,k,device):
    mean=np.where(k,x,0).sum(0)/k.sum(0); x=np.where(k,x-mean,0); x/=np.sqrt((x*x).sum(0)).clip(1e-6); t=torch.as_tensor(x,device=device); y=(t.T@t).clamp(-1,1).half().cpu().numpy(); del t; return y
def agreement(a,b):
    qa,qb=np.quantile(a,.99),np.quantile(b,.99); return {"pearson":float(np.corrcoef(a,b)[0,1]),"spearman":float(spearmanr(a,b).statistic),"top_one_percent_overlap_enrichment":float(np.mean((a>=qa)&(b>=qb))/.0001)}

def main(count=2000000):
    record=(ROOT/"data/project_score/corrected_logFCs.record").read_bytes(); n,x=struct.unpack_from("<HH",record,26); frame=pd.read_csv(io.BytesIO(zlib.decompress(record[30+n+x:],-15)),sep="\t"); source_models=np.asarray(frame.columns[1:].astype(str)); source_genes=frame.Gene.astype(str).str.upper().to_numpy(); source=frame.iloc[:,1:].to_numpy("float32").T
    meta=pd.read_csv(ROOT/"data/depmap24q2/Model.csv"); basal=np.load(OUT/"basal_context.npz"); br={x:i for i,x in enumerate(basal["model_ids"].astype(str))}; meta=meta[meta.ModelID.isin(br)].copy(); aliases=set()
    for col in ("CellLineName","StrippedCellLineName"):
        aliases.update(norm(v) for v in meta[col].dropna())
    aliases.update(norm(str(v).split("_",1)[0]) for v in meta.CCLEName.dropna()); source_names={norm(v) for v in source_models}; overlap=meta.apply(lambda r:any(norm(r[c]) in source_names for c in ("CellLineName","StrippedCellLineName")) or norm(str(r.CCLEName).split("_",1)[0]) in source_names,axis=1); excluded=set(meta.loc[overlap,"ModelID"]); keep=basal["train_cell"]&~np.isin(basal["model_ids"].astype(str),list(excluded)); dep=basal["dependency"][keep].astype("float32"); dep_known=basal["dependency_known"][keep]
    universe=pd.read_csv(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv").symbol.astype(str).str.upper(); uid={g:i for i,g in enumerate(universe)}; sr={g:i for i,g in enumerate(source_genes)}; genes=np.asarray([i for i,g in enumerate(universe) if g in sr],"int64"); cols=np.asarray([sr[universe.iloc[i]] for i in genes]); source=source[:,cols]; known=np.isfinite(source); parity=np.asarray([hashlib.sha256(x.encode()).digest()[0]&1 for x in source_models]); eligible=np.asarray([(known[parity==q].mean(0)>=.8) for q in (0,1)]).all(0)&(dep_known[:,genes].mean(0)>=.8); genes=genes[eligible]; source=source[:,eligible]; known=known[:,eligible]; dep=dep[:,genes]; dep_known=dep_known[:,genes]; device="cuda" if torch.cuda.is_available() else "cpu"; project=[matrix(source[parity==q],known[parity==q],device) for q in (0,1)]; independent=matrix(dep,dep_known,device)
    rng=np.random.default_rng(947); i,j=sample(rng,len(genes),min(count,len(genes)*(len(genes)-1)//2)); a=project[0][i,j].astype("float32"); b=project[1][i,j].astype("float32"); c=independent[i,j].astype("float32"); internal=agreement(a,b); cross=agreement((a+b)/2,c); admitted=len(source_models)>=300 and keep.sum()>=700 and len(genes)>=8000 and all(internal[k]>=v for k,v in {"pearson":.15,"spearman":.15,"top_one_percent_overlap_enrichment":3}.items()) and all(cross[k]>=v for k,v in {"pearson":.15,"spearman":.15,"top_one_percent_overlap_enrichment":3}.items()); np.savez_compressed(OUT/"project_score_codependency.npz",genes=genes.astype("int16"),half0=project[0],half1=project[1],independent=independent); result={"schema":"sl-predict-project-score-codependency-v1","project_score_models":len(source_models),"matched_models_excluded_from_depmap":len(excluded),"independent_depmap_models":int(keep.sum()),"eligible_genes":len(genes),"validation_pairs":len(a),"project_score_split_half":internal,"cross_panel":cross,"admitted":bool(admitted),"double_perturbation_data_used":False,"sl_labels_used":False}; (OUT/"project_score_codependency.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
