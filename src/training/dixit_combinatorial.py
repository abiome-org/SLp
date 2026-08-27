from pathlib import Path
import argparse, hashlib, json
import numpy as np, pandas as pd
from scipy import sparse
from scipy.io import mmread

ROOT=Path(__file__).resolve().parents[2]
SPECS=(("GSM2396856","dc_3hr","BMDC_3H","KO",3.),("GSM2396857","dc_0hr","BMDC_0H","KO",0.),("GSM2396860","k562_tfs_highmoi","K562_HIGH_MOI","CRISPRi",336.))

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def axis(p):return pd.read_csv(p,index_col=0).iloc[:,0].astype(str).to_numpy()

def panel(source,meta):
 m=pd.read_csv(meta); ens={str(e).split(".")[0]:str(s).upper() for e,s in zip(m.ensembl_gene_id,m.symbol) if pd.notna(e)}; canon=lambda x:ens.get(str(x).split(".")[0],str(x).upper()); common=None
 for p in sorted(Path(source).glob("*.npz"))+sorted((ROOT/"data/perturbseq_sources").glob("*.npz")):
  names={canon(x) for x in np.load(p)["feature_name"]}; common=names if common is None else common&names
 return common

def build(raw,assignments,source,output):
 raw=Path(raw); output=Path(output); a=pd.read_csv(assignments); keep_panel=panel(source,ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"); audit=[]
 for sample,stem,context,mode,hours in SPECS:
  cells=axis(raw/f"{sample}_{stem}_cellnames.csv.gz"); genes=axis(raw/f"{sample}_{stem}_genenames.csv.gz"); symbols=np.asarray([x.split("_",1)[1].upper() if "_" in x else x.upper() for x in genes]); features=sorted(keep_panel&set(symbols)); fmap={x:i for i,x in enumerate(features)}; selected=np.asarray([fmap.get(x,-1) for x in symbols]); d=a[(a.sample_id==sample)&a.in_count_matrix&a.retained_condition_support].copy(); d["condition"]=d.canonical_targets.fillna("ctrl").str.replace(";","+",regex=False); totals=d.condition.value_counts(); reps={c:min(4,int(n)//4) for c,n in totals.items() if n>=4}; d=d[d.condition.isin(reps)]; kept=d.condition.value_counts(); keys=sorted((c,r) for c,n in reps.items() for r in range(n)); key_id={x:i for i,x in enumerate(keys)}; cell_id={x:i for i,x in enumerate(cells)}; group=np.full(len(cells),-1,"int32")
  for row in d.itertuples():
   if row.cell_barcode in cell_id:group[cell_id[row.cell_barcode]]=key_id[(row.condition,int.from_bytes(hashlib.sha256(row.cell_barcode.encode()).digest()[:4],"big")%reps[row.condition])]
  matrix=raw/f"{sample}_{stem}.mtx.txt.gz"; raw_x=mmread(matrix).tocoo(); lib=np.bincount(raw_x.col,weights=raw_x.data,minlength=len(cells)); take=selected[raw_x.row]>=0; x=sparse.coo_matrix((raw_x.data[take].astype("float32"),(raw_x.col[take],selected[raw_x.row[take]])),shape=(len(cells),len(features))).tocsr(); declared=(*raw_x.shape,raw_x.nnz); del raw_x; x=x.multiply((1e4/lib)[:,None]).tocsr(); np.log1p(x.data,out=x.data); valid=group>=0; counts=np.bincount(group[valid],minlength=len(keys)); g=sparse.csr_matrix((np.ones(valid.sum()),(group[valid],np.flatnonzero(valid))),shape=(len(keys),len(cells))); future=(g@x).toarray()/counts[:,None]; conditions=np.asarray([k[0] for k in keys]); control=future[conditions=="ctrl"].mean(0); endpoints=[c.split("+") if c!="ctrl" else [] for c in conditions]; card=np.asarray([len(x) for x in endpoints],"int8"); ea=np.asarray([x[0] if x else "" for x in endpoints]); eb=np.asarray([x[1] if len(x)>1 else "" for x in endpoints]); out=output/f"dixit_{context.lower()}.npz"; output.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,source_id=f"dixit2016_{context.lower()}",context_id=context,mode=mode,duration_hours=np.float32(hours),condition=conditions,endpoint_a=ea,endpoint_b=eb,cardinality=card,role=np.full(len(keys),"train"),pseudoreplicate=np.asarray([k[1] for k in keys],"int16"),cell_count=counts.astype("int32"),feature_name=np.asarray(features),future_state=future.astype("float32"),control_mean=control.astype("float32")); audit.append({"source":out.stem,"input_sha256":sha(matrix),"matrix_shape":declared,"features":len(features),"pseudobulks":len(keys),"cells":int(counts.sum()),"pair_conditions":int((kept.index.str.contains("\\+")).sum()),"pair_cells":int(kept[kept.index.str.contains("\\+")].sum()),"cardinalities":{str(n):int((card==n).sum()) for n in np.unique(card)}})
 (output/"manifest.json").write_text(json.dumps({"schema":"sl-predict-dixit-combinatorial-v1","sources":audit,"normalization":"per-cell log1p counts per 10,000","replicates":"one to four barcode-hash pseudoreplicates, at least four cells each","assignment_source_sha256":sha(assignments),"outcome_files_read":False,"sl_labels_used":False},indent=2)); print(json.dumps(audit,indent=2))

if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--raw",default=ROOT/"data/raw/cart_v3_pair_state/dixit_gse90063"); p.add_argument("--assignments",default=ROOT/"data/processed/cart_v3_pair_state/dixit_gse90063_v1/cell_assignment_support.csv.gz"); p.add_argument("--source",default=ROOT/"data/processed/perturbseq_world_v1"); p.add_argument("--output",default=ROOT/"data/perturbseq_sources/dixit_combinatorial"); x=p.parse_args(); build(x.raw,x.assignments,x.source,x.output)
