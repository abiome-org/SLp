from pathlib import Path
import csv, gzip, hashlib, json, re, urllib.request
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/"data/depmap24q2"; OUT=ROOT/"results/sl_predict"
FILES=("OmicsExpressionProteinCodingGenesTPMLogp1.csv","CRISPRGeneEffect.csv")


def norm(x): return re.sub("[^A-Z0-9]","",str(x).upper())


def mapping():
    source=np.load(OUT/"slkb_outcomes_intervention_external.npz",allow_pickle=True)
    cells=sorted({norm(str(x).split("|")[-1]) for x in source["contexts"]})
    rows=list(csv.DictReader((DATA/"Model.csv").open(encoding="utf-8-sig"))); found={}
    for cell in cells:
        exact=[r for r in rows if norm(r["StrippedCellLineName"])==cell]; hits=exact or [r for r in rows if cell in {norm(r[k].split("_")[0]) for k in ("CellLineName","CCLEName")}]
        if len(hits)==1: found[cell]=hits[0]["ModelID"]
    return source,cells,found


def download():
    DATA.mkdir(parents=True,exist_ok=True); manifest=json.loads((DATA/"manifest.json").read_text()); _,_,models=mapping(); wanted=set(models.values())
    for name in FILES:
        item=next(x for x in manifest["files"] if x["name"]==name); out=DATA/name.replace(".csv",".selected.csv.gz"); md5=hashlib.md5(); kept=0
        with urllib.request.urlopen(item["download_url"]) as response, gzip.open(out,"wt",newline="",encoding="utf-8") as dst:
            for i,line in enumerate(response):
                md5.update(line); first=line.split(b",",1)[0].decode().strip('"\ufeff')
                if i==0 or first in wanted: dst.write(line.decode("utf-8")); kept+=i>0
        if md5.hexdigest()!=item["supplied_md5"]: raise ValueError(f"checksum mismatch: {name}")
        print({"file":name,"source_bytes":item["size"],"selected_rows":kept,"md5":md5.hexdigest()})


def matrix(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        reader=csv.reader(f); header=next(reader); rows=list(reader)
    return [x.split(" (")[0] for x in header[1:]], {r[0]:np.asarray([float(x) if x else np.nan for x in r[1:]],"float32") for r in rows}


def streamed(name,keep_ids=None):
    manifest=json.loads((DATA/"manifest.json").read_text()); item=next(x for x in manifest["files"] if x["name"]==name); meta=list(csv.DictReader((ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv").open())); symbols=[str(r["symbol"]).upper() for r in meta]; md5=hashlib.md5(); rows={}
    with urllib.request.urlopen(item["download_url"]) as response:
        header=next(response); md5.update(header); names=[x.split(" (")[0].upper() for x in next(csv.reader([header.decode("utf-8")]))[1:]]; columns={g:i+1 for i,g in enumerate(names)}; positions=[columns.get(g) for g in symbols]
        for line in response:
            md5.update(line); model=line.split(b",",1)[0].decode().strip('"')
            if keep_ids is None or model in keep_ids:
                row=next(csv.reader([line.decode("utf-8")])); rows[model]=np.asarray([float(row[i]) if i is not None and row[i] else np.nan for i in positions],"float32")
    if md5.hexdigest()!=item["supplied_md5"]: raise ValueError(f"checksum mismatch: {name}")
    print({"file":name,"rows":len(rows),"md5":md5.hexdigest()}); return symbols,rows


def pack(gene_aligned=False):
    source,cells,models=mapping(); observed=set(source["context"].tolist()); train={norm(str(source["contexts"][i]).split("|")[-1]) for i in observed}; held=set(json.loads((OUT/"slkb_outcomes_intervention_external.json").read_text())["symbols"]); states=[]; audits={}
    meta=list(csv.DictReader((ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv").open())); symbols=[str(r["symbol"]).upper() for r in meta]; gene_states=[]
    for seed,name in enumerate(FILES,123):
        genes,rows=matrix(DATA/name.replace(".csv",".selected.csv.gz")); train_ids=[models[c] for c in train if c in models and models[c] in rows]; x=np.stack([rows[i] for i in train_ids]); mean=np.nanmean(x,0); sd=np.nanstd(x,0); good=np.isfinite(mean)&(sd>1e-4); keep=good&np.asarray([g.upper() not in held for g in genes])
        rng=np.random.default_rng(seed); projection=rng.choice((-1.,1.),(keep.sum(),64)).astype("float32")/np.sqrt(keep.sum()); embedded={}; aligned={}; columns={g.upper():i for i,g in enumerate(genes)}; present=[(i,columns[g]) for i,g in enumerate(symbols) if g in columns and good[columns[g]]]
        for cell,model in models.items():
            if model in rows:
                z=np.zeros_like(mean); np.divide(rows[model]-mean,sd,out=z,where=good); z=np.nan_to_num(z).clip(-5,5); embedded[cell]=(z[keep]@projection).astype("float32"); aligned[cell]=np.zeros(len(symbols),"float32"); aligned[cell][[i for i,_ in present]]=z[[j for _,j in present]]
        center=np.stack([embedded[c] for c in train if c in embedded]).mean(0); scale=np.stack([embedded[c] for c in train if c in embedded]).std(0).clip(.1); states.append({c:(v-center)/scale for c,v in embedded.items()}); audits[name]={"genes":int(good.sum()),"training_models":len(train_ids),"available_cells":sorted(embedded)}
        gene_states.append(aligned)
    context_state=[]; known=[]; gene_context=[]
    for context in source["contexts"]:
        cell=norm(str(context).split("|")[-1]); masks=[float(cell in x) for x in states]; context_state.append(np.r_[*(x.get(cell,np.zeros(64,"float32")) for x in states),masks]); known.append(any(masks))
        if gene_aligned: gene_context.append(np.column_stack([x.get(cell,np.zeros(len(symbols),"float32")) for x in gene_states]))
    context_state=np.asarray(context_state,"float32"); name="slkb_outcomes_intervention_depmap_gene" if gene_aligned else "slkb_outcomes_intervention_depmap"; arrays={"pairs":source["pairs"],"context":source["context"],"target":source["target"],"contexts":source["contexts"],"context_state":context_state,"context_known":np.asarray(known)}
    if gene_aligned: arrays["gene_context_state"]=np.asarray(gene_context,"float32")
    np.savez_compressed(OUT/f"{name}.npz",**arrays)
    audit={"release":"DepMap Public 24Q2","doi":"10.25452/figshare.plus.25880521.v1","dimensions":context_state.shape[1],"held_out_genes_excluded_from_global_projection":len(held),"gene_aligned_dimensions":2 if gene_aligned else 0,"gene_aligned_inputs":"independent single-gene measurements; no pair outcomes" if gene_aligned else None,"resolved_cells":models,"unresolved_cells":sorted(set(cells)-set(models)),"known_contexts":int(sum(known)),"total_contexts":len(known),"modalities":audits}; (OUT/f"{name}.json").write_text(json.dumps(audit,indent=2)); print(audit)


def world_pack():
    source,cells,models=mapping(); symbols,dependency=streamed(FILES[1]); _,expression=streamed(FILES[0],set(dependency)); common=sorted(set(dependency)&set(expression)); held=set(json.loads((OUT/"slkb_outcomes_intervention_external.json").read_text())["symbols"]); targets={models[x] for x in ("A549","JURKAT","K562")}; fit=np.asarray([x not in targets for x in common]); ex=np.stack([expression[x] for x in common]); de=np.stack([dependency[x] for x in common])
    emean=np.nanmean(ex[fit],0); esd=np.nanstd(ex[fit],0); dmean=np.nanmean(de[fit],0); dsd=np.nanstd(de[fit],0); egood=np.isfinite(emean)&(esd>1e-4); dgood=np.isfinite(dmean)&(dsd>1e-4); ez=np.zeros_like(ex); dz=np.zeros_like(de); np.divide(ex-emean,esd,out=ez,where=egood); np.divide(de-dmean,dsd,out=dz,where=dgood); ez=np.nan_to_num(ez).clip(-5,5); dz=np.nan_to_num(dz).clip(-5,5)
    safe=egood&~np.isin(np.asarray(symbols),tuple(held)); rng=np.random.default_rng(731); projection=rng.choice((-1.,1.),(safe.sum(),128)).astype("float32")/np.sqrt(safe.sum()); state=ez[:,safe]@projection; center=state[fit].mean(0); scale=state[fit].std(0).clip(.1); state=(state-center)/scale; train_gene=dgood&~np.isin(np.asarray(symbols),tuple(held)); name="depmap_world"; np.savez_compressed(OUT/f"{name}.npz",model_ids=np.asarray(common),cell_state=state.astype("float32"),dependency=dz.astype("float16"),dependency_known=(np.isfinite(de)&dgood),train_cell=fit,train_gene=train_gene)
    index=dict(zip(common,range(len(common)))); context_state=np.stack([state[index[models.get(norm(str(x).split('|')[-1]),'')]] if models.get(norm(str(x).split('|')[-1])) in index else np.zeros(128,"float32") for x in source["contexts"]]); known=np.linalg.norm(context_state,axis=1)>0; outcome="slkb_outcomes_intervention_depmap_world"; np.savez_compressed(OUT/f"{outcome}.npz",pairs=source["pairs"],context=source["context"],target=source["target"],contexts=source["contexts"],context_state=context_state,context_known=known)
    audit={"release":"DepMap Public 24Q2","models":len(common),"training_models":int(fit.sum()),"target_models_excluded":sorted(targets),"training_genes":int(train_gene.sum()),"held_out_genes_excluded":len(held),"state_dimensions":128,"known_contexts":int(known.sum())}; (OUT/f"{name}.json").write_text(json.dumps(audit,indent=2)); (OUT/f"{outcome}.json").write_text(json.dumps(audit,indent=2)); print(audit)


def project_context(model_ids=("ACH-002475",),name="hap1_context"):
    _,_,models=mapping(); symbols,dependency=streamed(FILES[1]); wanted=set(dependency)|set(model_ids); _,expression=streamed(FILES[0],wanted); common=sorted(set(dependency)&set(expression)); held=set(json.loads((OUT/"slkb_outcomes_intervention_external.json").read_text())["symbols"]); targets={models[x] for x in ("A549","JURKAT","K562")}; fit=np.asarray([x not in targets for x in common]); ex=np.stack([expression[x] for x in common]); emean=np.nanmean(ex[fit],0); esd=np.nanstd(ex[fit],0); good=np.isfinite(emean)&(esd>1e-4); safe=good&~np.isin(np.asarray(symbols),tuple(held)); rng=np.random.default_rng(731); projection=rng.choice((-1.,1.),(safe.sum(),128)).astype("float32")/np.sqrt(safe.sum()); ez=np.zeros_like(ex); np.divide(ex-emean,esd,out=ez,where=good); base=np.nan_to_num(ez).clip(-5,5)[:,safe]@projection; center=base[fit].mean(0); scale=base[fit].std(0).clip(.1); states=[]
    resolved=[model for model in model_ids if model in expression]
    for model in resolved:
        x=expression[model]; z=np.zeros_like(emean); np.divide(x-emean,esd,out=z,where=good); states.append((np.nan_to_num(z).clip(-5,5)[safe]@projection-center)/scale)
    reference=np.load(OUT/"depmap_world.npz"); index={x:i for i,x in enumerate(reference["model_ids"].astype(str))}; check=np.stack([(base[common.index(x)]-center)/scale-reference["cell_state"][index[x]] for x in common if x in index]); out=OUT/f"{name}.npz"; np.savez_compressed(out,model_ids=np.asarray(resolved),cell_state=np.asarray(states,"float32")); audit={"schema":"depmap-world-context-projection-v1","release":"DepMap Public 24Q2","model_ids":resolved,"unavailable_model_ids":sorted(set(model_ids)-set(resolved)),"state_dimensions":128,"held_out_genes_excluded":len(held),"reference_models":len(common),"max_abs_reference_error":float(np.abs(check).max())}; out.with_suffix(".json").write_text(json.dumps(audit,indent=2)); print(audit)


def raw_dependency_pack():
    symbols,rows=streamed(FILES[1]); world=np.load(OUT/"depmap_world.npz"); models=world["model_ids"].astype(str); raw=np.stack([rows[x] for x in models]); out=OUT/"depmap_tolerance.npz"; np.savez_compressed(out,model_ids=models,cell_state=world["cell_state"],dependency=raw.astype("float16"),dependency_known=np.isfinite(raw),train_cell=world["train_cell"],train_gene=world["train_gene"]); audit={"schema":"depmap-raw-gene-effect-v1","release":"DepMap Public 24Q2","models":len(models),"genes":len(symbols),"measured":int(np.isfinite(raw).sum()),"training_genes":int(world["train_gene"].sum()),"target_models_excluded":models[~world["train_cell"]].tolist()}; out.with_suffix(".json").write_text(json.dumps(audit,indent=2)); print(audit)


def dependency_landscape_pack(dimensions=64):
    from sklearn.decomposition import PCA
    world=np.load(OUT/"depmap_world.npz"); cells=world["train_cell"]; permitted=world["train_gene"]; uid=np.arange(len(permitted)); train=permitted&(uid%5!=0); valid=permitted&(uid%5==0); x=world["dependency"][cells].astype("float32").T; known=world["dependency_known"][cells].T; coverage=known.mean(1); excluded=(~permitted)&(coverage>=.8); unavailable=(~permitted)&(~excluded); x=np.where(known,x,0.); pca=PCA(dimensions,svd_solver="randomized",random_state=731).fit(x[train]); raw=pca.transform(x); scale=raw[train].std(0).clip(.05); target=(raw/scale).astype("float32"); name="dependency_landscape"; np.savez_compressed(OUT/f"{name}.npz",target=target,train=train,valid=valid,excluded=excluded,unavailable=unavailable,known_fraction=coverage.astype("float32"),components=pca.components_.astype("float32"),mean=pca.mean_.astype("float32"),scale=scale.astype("float32")); audit={"schema":"depmap-dependency-landscape-v1","release":"DepMap Public 24Q2","dimensions":dimensions,"training_cells":int(cells.sum()),"target_cells_excluded":world["model_ids"][~cells].astype(str).tolist(),"fit_genes":int(train.sum()),"selection_genes":int(valid.sum()),"intervention_isolated_genes":int(excluded.sum()),"unavailable_excluded_genes":int(unavailable.sum()),"fit_known_fraction":float(known[train].mean()),"selection_known_fraction":float(known[valid].mean()),"isolated_known_fraction":float(known[excluded].mean()),"explained_variance":float(pca.explained_variance_ratio_.sum()),"pca_fit_genes":"firewall-permitted unified_id modulo 5 nonzero only","sl_labels_used":False}; (OUT/f"{name}.json").write_text(json.dumps(audit,indent=2)); print(audit)


def interaction_residual_pack():
    source=np.load(OUT/"slkb_outcomes_intervention_depmap_world.npz",allow_pickle=True); raw=np.load(OUT/"depmap_tolerance.npz"); _,_,models=mapping(); index={x:i for i,x in enumerate(raw["model_ids"].astype(str))}; held=np.arange(raw["dependency"].shape[1])%5==0; keep=np.zeros(len(source["pairs"]),bool); expected=np.zeros(len(keep),"float32"); contexts=[]
    for c,name in enumerate(source["contexts"]):
        model=models.get(norm(str(name).split("|")[-1]),""); m=index.get(model,-1); rows=np.flatnonzero(source["context"]==c)
        if m<0 or not raw["train_cell"][m] or not len(rows):continue
        pairs=source["pairs"][rows]; good=raw["dependency_known"][m,pairs].all(1); rows,pairs=rows[good],pairs[good]
        if not len(rows):continue
        single=raw["dependency"][m,pairs].astype("float32").sum(1); single=(single-single.mean())/max(.1,float(single.std())); fit=(~held[pairs]).all(1); design=np.column_stack((np.ones(fit.sum()),single[fit])); intercept,slope=np.linalg.lstsq(design,source["target"][rows[fit],0],rcond=None)[0]; expected[rows]=intercept+slope*single; keep[rows]=True; contexts.append({"context":str(name),"model_id":model,"rows":len(rows),"fit_rows":int(fit.sum()),"slope":float(slope),"correlation":float(np.corrcoef(source["target"][rows,0],single)[0,1])})
    target=source["target"][keep].copy(); target[:,0]-=expected[keep]; pairs=source["pairs"][keep]; train=(~held[pairs]).all(1); valid=held[pairs].all(1); name="slkb_outcomes_depmap_additive_residual"; np.savez_compressed(OUT/f"{name}.npz",pairs=pairs,context=source["context"][keep],target=target,contexts=source["contexts"],context_state=source["context_state"],context_known=source["context_known"],single_expected=expected[keep]); audit={"schema":"slkb-depmap-additive-residual-v1","rows":len(pairs),"train_rows":int(train.sum()),"both_genes_new_rows":int(valid.sum()),"both_genes_new_pairs":int(len(np.unique(pairs[valid,0].astype("int64")*len(held)+pairs[valid,1]))),"target_models_excluded":raw["model_ids"][~raw["train_cell"]].astype(str).tolist(),"binary_sl_labels_used":False,"contexts":contexts}; (OUT/f"{name}.json").write_text(json.dumps(audit,indent=2)); print(audit)


if __name__=="__main__":
    import sys
    if sys.argv[1]=="project_context":project_context(tuple(sys.argv[3:]),sys.argv[2])
    else:{"download":download,"pack":pack,"gene_pack":lambda:pack(True),"world_pack":world_pack,"hap1_context":project_context,"raw_dependency":raw_dependency_pack,"dependency_landscape":dependency_landscape_pack,"interaction_residual":interaction_residual_pack}[sys.argv[1]]()
