from pathlib import Path
import argparse, csv, gzip, hashlib, json, warnings
import h5py, numpy as np, pandas as pd, rdata
from sklearn.decomposition import PCA

ROOT=Path(__file__).resolve().parents[2]; MOIS=("0.1","0.2","0.5","1.0","3.0","5.0")

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def panel(source):
 meta=pd.read_csv(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"); ens={str(e).split(".")[0]:str(s).upper() for e,s in zip(meta.ensembl_gene_id,meta.symbol) if pd.notna(e)}; canon=lambda x:ens.get(str(x).split(".")[0],str(x).upper()); common=None
 paths=sorted(Path(source).glob("*.npz"))+sorted((ROOT/"data/perturbseq_sources").glob("*.npz"))+sorted((ROOT/"data/perturbseq_sources/dixit_combinatorial").glob("*.npz"))
 for p in paths:
  names={canon(x) for x in np.load(p)["feature_name"]}; common=names if common is None else common&names
 return common

def build(raw,source,output):
 raw=Path(raw); output=Path(output); output.mkdir(parents=True,exist_ok=True); keep=panel(source); guide={r["index"]:r["gene_symbol"].upper() for r in csv.DictReader(gzip.open(raw/"GSE337988_NGS6194_crispr_demux_index_map.csv.gz","rt"))}; meta=pd.read_csv(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"); ids={str(s).upper():int(i) for s,i in zip(meta.symbol,meta.unified_id)}; held=set(json.loads((ROOT/"results/sl_predict/slkb_outcomes_intervention_external.json").read_text())["symbols"]); val=lambda x:int.from_bytes(hashlib.sha256(x.encode()).digest()[:4],"big")%5==0; audit=[]; records=[]; pca_rows=[]; full_features=None
 for sid,moi in enumerate(MOIS):
  rds=raw/f"GSE337988_pilot_processed_objects_MOI_{moi}_se.rds"; h5=raw/f"GSE337988_pilot_processed_objects_MOI_{moi}_assays.h5"
  with warnings.catch_warnings(): warnings.simplefilter("ignore"); obj=rdata.read_rds(rds)
  cd=obj.colData.listData; symbols=np.char.upper(np.asarray(obj.rowRanges.elementMetadata.listData["symbol"],str)); fmap={x:i for i,x in enumerate(symbols)}; features=np.asarray(sorted(keep&set(symbols))); cols=np.asarray([fmap[x] for x in features]); cells=np.asarray(cd["cell"],str); bad=np.asarray(cd["is.bad"],bool); assignments=np.asarray(cd["assignment"],str); members=[]; kind=np.full(len(cells),-1,"int8")
  for i,text in enumerate(assignments):
   gids=[] if text in ("","NA") else text.split(","); targets=[guide.get(x) for x in gids]
   if bad[i] or not gids or any(x is None for x in targets):members.append(()); continue
   if all(x=="NTC" for x in targets):kind[i]=0; members.append(()); continue
   genes=tuple(sorted(set(targets)))
   if "NTC" not in targets and len(genes)==len(targets) and len(genes)<=8:kind[i]=len(genes); members.append(genes)
   else:members.append(())
  role=np.full(len(cells),-1,"int8")
  for i,genes in enumerate(members):
   if not genes or any(x in held or x not in ids for x in genes):continue
   v=[val(x) for x in genes]; role[i]=1 if all(v) else (0 if not any(v) else -1)
  state=np.empty((len(cells),len(features)),"float32")
  with h5py.File(h5) as f:
   x=f["assay001"]; assert x.shape==(len(cells),len(symbols))
   assert full_features is None or np.array_equal(full_features,symbols); full_features=symbols if full_features is None else full_features
   control_rows=np.flatnonzero(kind==0); raw_control=x[control_rows]; control_lib=raw_control.sum(1).clip(1); full_control=np.log1p(raw_control*(1e4/control_lib[:,None])).mean(0).astype("float32")
   sample=np.sort(np.concatenate([np.random.default_rng(731+sid+n).choice(ix,min(250,len(ix)),replace=False) for n in range(1,9) if len(ix:=np.flatnonzero((role==0)&(kind==n)))])); raw_sample=x[sample]; sample_lib=raw_sample.sum(1).clip(1); pca_rows.append(np.log1p(raw_sample*(1e4/sample_lib[:,None]))-full_control)
   for start in range(0,len(cells),512):
    raw_x=x[start:start+512]; lib=raw_x.sum(1).clip(1); state[start:start+len(raw_x)]=np.log1p(raw_x[:,cols]*(1e4/lib[:,None]))
  control=state[kind==0].mean(0); max_card=int(kind.max())
  use=(kind>0)&(kind<=2); condition=np.asarray(["+".join(members[i]) for i in np.flatnonzero(use)]); indices=np.flatnonzero(use); unique,count=np.unique(condition,return_counts=True); reps={c:min(4,int(n)//4) for c,n in zip(unique,count) if n>=4}; keys=sorted((c,r) for c,n in reps.items() for r in range(n)); key_id={x:i for i,x in enumerate(keys)}; group=np.full(len(indices),-1,"int32")
  for j,(i,c) in enumerate(zip(indices,condition)):
   if c in reps:group[j]=key_id[(c,int.from_bytes(hashlib.sha256(cells[i].encode()).digest()[:4],"big")%reps[c])]
  selected=group>=0; counts=np.bincount(group[selected],minlength=len(keys)); future=np.zeros((len(keys),len(features)),"float32"); np.add.at(future,group[selected],state[indices[selected]]); future/=counts[:,None]; conditions=np.asarray([x[0] for x in keys]); endpoints=[x.split("+") for x in conditions]; card=np.asarray([len(x) for x in endpoints],"int8"); out=output/f"gse337988_dld1_moi_{moi.replace('.','_')}.npz"; np.savez_compressed(out,source_id=f"gse337988_dld1_moi_{moi.replace('.','_')}",context_id=f"DLD1_MOI_{moi.replace('.','_')}",mode="CRISPRi",duration_hours=np.float32(120),condition=conditions,endpoint_a=np.asarray([x[0] for x in endpoints]),endpoint_b=np.asarray([x[1] if len(x)>1 else "" for x in endpoints]),cardinality=card,role=np.full(len(keys),"train"),pseudoreplicate=np.asarray([x[1] for x in keys],"int16"),cell_count=counts.astype("int32"),feature_name=features,future_state=future,control_mean=control.astype("float32"))
  strict={str(n):int((kind==n).sum()) for n in range(max_card+1)}; audit.append({"source":out.stem,"rds_sha256":sha(rds),"h5_sha256":sha(h5),"cells":len(cells),"expression_genes":len(symbols),"common_genes":len(features),"strict_cells_by_cardinality":strict,"pair_cells":int((kind==2).sum()),"pair_conditions":int(len(unique)),"pair_conditions_ge4":len(reps),"pseudobulks":len(keys),"set_train_cells":int((role==0).sum()),"set_validation_cells":int((role==1).sum())}); records.append((sid,h5,members,kind,role,full_control))
  print(json.dumps(audit[-1]),flush=True); del obj,state
 pool=np.concatenate(pca_rows); pca=PCA(64,svd_solver="randomized",random_state=731).fit(pool); scale=pca.transform(pool).std(0).clip(.05); set_rows=[]
 for sid,h5,members,kind,role,control in records:
  keep_rows=np.flatnonzero(role>=0); member_ids=np.full((len(keep_rows),8),-1,"int16")
  for j,i in enumerate(keep_rows):member_ids[j,:kind[i]]=[ids[x] for x in members[i]]
  target=np.empty((len(keep_rows),64),"float32")
  with h5py.File(h5) as f:
   x=f["assay001"]
   for start in range(0,len(keep_rows),512):
    rows=keep_rows[start:start+512]; raw_x=x[rows]; lib=raw_x.sum(1).clip(1); target[start:start+len(rows)]=pca.transform(np.log1p(raw_x*(1e4/lib[:,None]))-control)/scale
  set_rows.append((member_ids,kind[keep_rows],np.full(len(keep_rows),sid,"int8"),role[keep_rows],target))
 members=np.concatenate([x[0] for x in set_rows]); cardinality=np.concatenate([x[1] for x in set_rows]); source_ids=np.concatenate([x[2] for x in set_rows]); roles=np.concatenate([x[3] for x in set_rows]); target=np.concatenate([x[4] for x in set_rows]); np.savez_compressed(output/"gse337988_dld1_set_cells.npz",members=members,cardinality=cardinality,source=source_ids,role=roles,target=target,feature_name=full_features,sources=np.asarray([x["source"] for x in audit]),pca_mean=pca.mean_.astype("float32"),pca_components=pca.components_.astype("float32"),target_scale=scale.astype("float32"))
 manifest={"schema":"sl-predict-gse337988-pilot-v1","sources":audit,"set_cells":len(target),"set_train_cells":int((roles==0).sum()),"set_validation_cells":int((roles==1).sum()),"set_state_dimensions":64,"set_state_explained_variance":float(pca.explained_variance_ratio_.sum()),"normalization":"per-cell log1p counts per 10,000; source-matched NTC mean subtracted; shared PCA fitted only on deterministic fitting-cell samples","pair_replicates":"one to four cell-hash pseudoreplicates, at least four cells per exact condition","assignment":"strict distinct non-NTC guide-to-gene assignments; mixed NTC, repeated-gene and unassigned cells excluded","benchmark_membership_used_for_exclusion_only":True,"outcome_files_read":False,"sl_labels_used":False}; (output/"manifest.json").write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2))

if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--raw",default=ROOT/"data/perturbseq_sources/gse337988_audit"); p.add_argument("--source",default=ROOT/"data/processed/perturbseq_world_v1"); p.add_argument("--output",default=ROOT/"data/perturbseq_sources/gse337988_pilot"); a=p.parse_args(); build(a.raw,a.source,a.output)
