from pathlib import Path
import argparse, csv, gzip, hashlib, json, time, warnings
import numpy as np, rdata

ROOT=Path(__file__).resolve().parents[2]

def array(x):return np.asarray(x.values if hasattr(x,"values") else x)

def extract(rds,output):
 start=time.time()
 with warnings.catch_warnings():warnings.simplefilter("ignore"); obj=rdata.read_rds(rds)
 cd=obj.colData.listData; keys=("Sample","Barcode","demux_type","assignment","promoter","gRNA_number","guide_name","assignment.fishash","demux_type.fishash","MOI","Timepoint_hrs","total.counts","is.bad.cell","cell"); arrays={k.replace(".","_"):array(cd[k]) for k in keys}; arrays["feature_name"]=array(obj.rowRanges.elementMetadata.listData["symbol"]); out=Path(output); out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,**arrays); audit={"schema":"sl-predict-gse337988-scaled-metadata-v1","cells":len(arrays["cell"]),"expression_features":len(arrays["feature_name"]),"fields":{k:str(v.dtype) for k,v in arrays.items()},"examples":{k:arrays[k][:5].tolist() for k in ("assignment","assignment_fishash","demux_type","demux_type_fishash","guide_name")},"parse_seconds":time.time()-start}; out.with_suffix(".json").write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2))

def split_gene(x):return int.from_bytes(hashlib.sha256(x.encode()).digest()[:4],"big")%5==0

def audit(metadata,guide_map,output):
 import pandas as pd
 z=np.load(metadata,allow_pickle=True); mapping={r["index"]:r["sample"].upper() for r in csv.DictReader(gzip.open(guide_map,"rt"))}; meta=pd.read_csv(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"); ids={str(s).upper():int(i) for s,i in zip(meta.symbol,meta.unified_id)}; held=set(json.loads((ROOT/"results/sl_predict/slkb_outcomes_intervention_external.json").read_text())["symbols"]); text=z["guide_name"].astype(str); bad=z["is_bad_cell"].astype(bool); members=np.full((len(text),8),-1,"int16"); card=np.full(len(text),-1,"int8"); role=np.full(len(text),-1,"int8"); controls=[]; mapped=total=0; strict_genes=set(); unknown_cells=0
 for i,value in enumerate(text):
  if bad[i] or value in ("","NA","nan","None"):continue
  guides=value.split(","); targets=[mapping.get(x) for x in guides]; total+=len(guides); mapped+=sum(x is not None for x in targets)
  if any(x is None for x in targets):unknown_cells+=1; continue
  if all(x=="NTC" for x in targets):card[i]=0; controls.append(i); continue
  genes=tuple(sorted(set(targets)))
  if "NTC" in targets or len(genes)!=len(targets) or len(genes)>8 or any(x not in ids for x in genes):continue
  card[i]=len(genes); members[i,:len(genes)]=[ids[x] for x in genes]
  if any(x in held for x in genes):continue
  val=[split_gene(x) for x in genes]; role[i]=1 if all(val) else (0 if not any(val) else -1); strict_genes.update(genes)
 keep=np.flatnonzero(role>=0); out=Path(output); out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,cell_index=keep,members=members[keep],cardinality=card[keep],role=role[keep],control_index=np.asarray(controls,"int32"),feature_name=z["feature_name"]); counts={str(n):int((card==n).sum()) for n in range(9)}; result={"schema":"sl-predict-gse337988-scaled-assignment-v1","cells":len(text),"good_cells":int((~bad).sum()),"mapped_assignment_fraction":mapped/max(1,total),"unknown_assignment_cells":unknown_cells,"strict_cells_by_cardinality":counts,"strict_exact_multi_gene_cells":int((card>=2).sum()),"ntc_control_cells":len(controls),"intervention_isolated_cells":len(keep),"training_cells":int((role==0).sum()),"fully_gene_cold_validation_cells":int((role==1).sum()),"intervention_isolated_multi_gene_cells":int(((role>=0)&(card>=2)).sum()),"intervention_isolated_target_genes":len(strict_genes),"expression_features":len(z["feature_name"]),"benchmark_membership_used_for_exclusion_only":True,"outcome_files_read":False,"sl_labels_used":False}; out.with_suffix(".json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

def build_state(assignments,metadata,h5,response_pack,output):
 import h5py
 from sklearn.decomposition import PCA
 a=np.load(assignments); m=np.load(metadata,allow_pickle=True); response=set(np.load(response_pack,allow_pickle=True)["response_gene"].astype(str)); feature=m["feature_name"].astype(str); fmap={x:i for i,x in enumerate(feature)}; selected=np.asarray(sorted(response&set(feature))); cols=np.asarray([fmap[x] for x in selected]); order=np.argsort(cols); cols=cols[order]; selected=selected[order]; assert len(selected)>=2000; keep=a["cell_index"].astype("int64"); controls=a["control_index"].astype("int64"); all_rows=np.unique(np.r_[controls,keep]); row_map=np.full(len(m["cell"]),-1,"int32"); row_map[all_rows]=np.arange(len(all_rows)); state=np.empty((len(all_rows),len(selected)),"float32"); total=m["total_counts"].astype("float32").clip(1)
 with h5py.File(h5) as f:
  x=f["assay001"]; assay_shape=x.shape; assert assay_shape==(len(m["cell"]),len(feature))
  for start in range(0,len(row_map),1000):
   stop=min(len(row_map),start+1000); local=np.flatnonzero(row_map[start:stop]>=0)
   if len(local):state[row_map[start+local]]=np.log1p(x[start:stop,cols][local]*(1e4/total[start+local,None]))
 control=state[row_map[controls]].mean(0); card=a["cardinality"]; role=a["role"]; sample=[]
 for n in range(1,9):
  ix=np.flatnonzero((role==0)&(card==n))
  if len(ix):sample.append(np.random.default_rng(731+n).choice(ix,min(2000,len(ix)),replace=False))
 sample=np.concatenate(sample); fit=state[row_map[keep[sample]]]-control; pca=PCA(64,svd_solver="randomized",random_state=731).fit(fit); latent_scale=pca.transform(fit).std(0).clip(.05); target=(pca.transform(state[row_map[keep]]-control)/latent_scale).astype("float32"); out=Path(output); out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,members=a["members"],cardinality=card,role=role,target=target,feature_name=selected,pca_mean=pca.mean_.astype("float32"),pca_components=pca.components_.astype("float32"),target_scale=latent_scale.astype("float32")); result={"schema":"sl-predict-gse337988-scaled-state-v1","assay_sha256":"7fb76618bf426416c033d501D0DD143BA029973E8834D11638FC84B1F60C3B6D".lower(),"assay_shape":[*assay_shape],"aligned_cells":len(row_map),"fixed_response_panel_genes":len(selected),"rows":len(target),"train_rows":int((role==0).sum()),"validation_rows":int((role==1).sum()),"train_multi_rows":int(((role==0)&(card>=2)).sum()),"validation_pair_rows":int(((role==1)&(card==2)).sum()),"pca_fit_rows":len(sample),"state_dimensions":64,"explained_variance":float(pca.explained_variance_ratio_.sum()),"normalization":"per-cell log1p counts per 10,000 using aligned RNA total counts; NTC mean subtraction; PCA fitted only on cardinality-balanced fitting cells","benchmark_membership_used_for_exclusion_only":True,"outcome_files_read":False,"sl_labels_used":False}; out.with_suffix(".json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--rds",default=ROOT/"data/perturbseq_sources/gse337988_audit/GSE337988_sublib2_processed_objects_Med_se.rds"); p.add_argument("--output",default=ROOT/"data/perturbseq_sources/gse337988_audit/gse337988_med_metadata.npz"); p.add_argument("--audit",action="store_true"); p.add_argument("--build-state",action="store_true"); p.add_argument("--metadata",default=ROOT/"data/perturbseq_sources/gse337988_audit/gse337988_med_metadata.npz"); p.add_argument("--guide-map",default=ROOT/"data/perturbseq_sources/gse337988_audit/GSE337988_NGS6475_crispr_demux_index_map_prod.csv.gz"); p.add_argument("--audit-output",default=ROOT/"data/perturbseq_sources/gse337988_audit/gse337988_med_assignments.npz"); p.add_argument("--h5",default=ROOT/"data/perturbseq_sources/gse337988_audit/GSE337988_sublib2_processed_objects_Med_assays.h5"); p.add_argument("--response-pack",default=ROOT/"results/sl_predict/gse337988_moi_state.npz"); p.add_argument("--state-output",default=ROOT/"results/sl_predict/gse337988_scaled_med_state.npz"); a=p.parse_args(); build_state(a.audit_output,a.metadata,a.h5,a.response_pack,a.state_output) if a.build_state else (audit(a.metadata,a.guide_map,a.audit_output) if a.audit else extract(a.rds,a.output))
