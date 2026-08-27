from pathlib import Path
import csv, math, zipfile

ROOT = Path(__file__).resolve().parents[2]


def values(raw):
    line = raw.decode("utf-8").strip(); inner = line[line.index("VALUES(")+7:-2]
    return next(csv.reader([inner], delimiter=",", quotechar="'", doublequote=True))


def named_counts(counts, names):
    return [(name, float(count)) for name, count in zip(names.split(";"), counts.split(";")) if count and count != "NULL"]


def main():
    source = ROOT / "data/slkb/SQL_Dumps.zip"; out = ROOT / "results/sl_predict/slkb_outcomes.tsv"
    libraries = {}
    with zipfile.ZipFile(source).open("SQL_Dumps/SLKB-sqlite3_dump.sql") as f:
        for raw in f:
            if raw.startswith(b"INSERT INTO cdko_sgrna_counts VALUES"):
                v = values(raw)
                for phase, counts, names in (("T0",v[5],v[6]),("TEnd",v[7],v[8])):
                    for name, count in named_counts(counts,names):
                        key=(v[10],v[11],phase,name); libraries[key]=libraries.get(key,0.)+count
    depletion = {}; writer_file = None
    with zipfile.ZipFile(source).open("SQL_Dumps/SLKB-sqlite3_dump.sql") as f:
        for raw in f:
            if raw.startswith(b"INSERT INTO cdko_sgrna_counts VALUES"):
                v = values(raw); key = (int(v[3]), v[10], v[11])
                logs=[]
                for phase,counts,names,sign in (("T0",v[5],v[6],-1),("TEnd",v[7],v[8],1)):
                    x=[math.log2(count/libraries[(v[10],v[11],phase,name)]*1e6+1) for name,count in named_counts(counts,names)]
                    if x: logs.append(sign*sum(x)/len(x))
                score = sum(logs) if len(logs)==2 else float("nan")
                total, n = depletion.get(key, (0., 0)); depletion[key] = (total+score, n+1)
            elif raw.startswith(b"INSERT INTO cdko_original_sl_results VALUES"):
                if writer_file is None:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    writer_file = out.open("w", newline="", encoding="utf-8")
                    writer = csv.writer(writer_file, delimiter="\t")
                    writer.writerow(("gene1","gene2","study","cell","label","sl_score","log2_depletion","guides"))
                v = values(raw); total, n = depletion.get((int(v[1]), v[3], v[4]), (float("nan"), 0))
                writer.writerow((v[5],v[6],v[3],v[4],int(v[7].lower()=="sl"),v[8],total/n if n else "",n))
    if writer_file: writer_file.close()
    import pandas as pd, numpy as np
    meta=pd.read_csv(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv")
    gene={str(s).upper():int(i) for s,i in zip(meta.symbol,meta.unified_id)}
    d=pd.read_csv(out,sep="\t"); d=d[d.gene1.str.upper().isin(gene)&d.gene2.str.upper().isin(gene)].copy()
    pairs=np.column_stack((d.gene1.str.upper().map(gene),d.gene2.str.upper().map(gene))).astype("int16")
    pairs.sort(1); n=len(meta); candidates=set((pairs[:,0].astype("int64")*n+pairs[:,1]).tolist()); blocked=set()
    for path in (ROOT/"data/feng2024/data").glob("data_split*/CV*.npy"):
        obj=np.load(path,allow_pickle=True)
        for item in obj.flat:
            if isinstance(item,np.ndarray) and item.ndim==2 and item.shape[1]==2:
                p=item.astype("int32"); p.sort(1); blocked.update(k for k in (p[:,0].astype("int64")*n+p[:,1]) if k in candidates)
    keep=np.array([int(a)*n+int(b) not in blocked for a,b in pairs])
    pairs=pairs[keep]; d=d[keep].reset_index(drop=True)
    keys=pairs[:,0].astype("int64")*n+pairs[:,1]; unique,inverse=np.unique(keys,return_inverse=True); label=np.zeros(len(unique),"int8"); np.maximum.at(label,inverse,d.label.to_numpy("int8"))
    np.savez_compressed(out.with_name("slkb_labels.npz"),pairs=np.column_stack((unique//n,unique%n)).astype("int16"),label=label)
    finite=d.log2_depletion.notna().to_numpy(); pairs=pairs[finite]; d=d[finite].reset_index(drop=True)
    context=(d.study.astype(str)+"|"+d.cell).astype("category"); raw=d.log2_depletion.astype("float32")
    target=((raw-raw.groupby(context).transform("mean"))/raw.groupby(context).transform("std").clip(lower=.1)).clip(-8,8).to_numpy("float32")
    score=pd.to_numeric(d.sl_score,errors="coerce").astype("float32"); grouped=score.groupby(context)
    strength=((score-grouped.transform("median"))/grouped.transform(lambda x:(x-x.median()).abs().median()).clip(lower=.01)).abs().clip(0,8).to_numpy("float32")
    np.savez_compressed(out.with_suffix(".npz"),pairs=pairs,context=context.cat.codes.to_numpy("int16"),
                        target=np.column_stack((target,strength)).astype("float32"),contexts=context.cat.categories.to_numpy())


def strict_external():
    import pickle, pandas as pd, numpy as np
    out=ROOT/"results/sl_predict"; pack=np.load(out/"slkb_outcomes.npz",allow_pickle=True); meta=pd.read_csv(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"); ids=dict(zip(meta.symbol.astype(str),meta.unified_id.astype(int))); n=len(meta); blocked=set()
    for path in (ROOT/"data/models/SLAMR/data_slb_filtered").glob("*/*_scenario3_fold5_seed88.pkl"):
        for _,valid,test in pickle.load(open(path,"rb")):
            for group in (valid,test):
                for a,partners in group.items():
                    if a in ids:
                        blocked.update(min(ids[a],ids[b])*n+max(ids[a],ids[b]) for b,_,_ in partners if b in ids)
    pairs=pack["pairs"]; keep=np.array([int(a)*n+int(b) not in blocked for a,b in pairs]); np.savez_compressed(out/"slkb_outcomes_strict_external.npz",pairs=pairs[keep],context=pack["context"][keep],target=pack["target"][keep],contexts=pack["contexts"]); print({"trajectories":len(pairs),"retained":int(keep.sum()),"excluded":int((~keep).sum()),"blocked_pairs":len(blocked)})


def intervention_external():
    import json, pickle, pandas as pd, numpy as np
    out=ROOT/"results/sl_predict"; pack=np.load(out/"slkb_outcomes.npz",allow_pickle=True); meta=pd.read_csv(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"); ids=dict(zip(meta.symbol.astype(str),meta.unified_id.astype(int))); held=set()
    for path in (ROOT/"data/models/SLAMR/data_slb_filtered").glob("*/*_scenario3_fold5_seed88.pkl"):
        for _,valid,test in pickle.load(open(path,"rb")):
            for group in (valid,test):
                for a,partners in group.items():
                    if a in ids: held.add(ids[a])
                    held.update(ids[b] for b,_,_ in partners if b in ids)
    pairs=pack["pairs"]; keep=~np.isin(pairs,tuple(held)).any(1); name="slkb_outcomes_intervention_external"; np.savez_compressed(out/f"{name}.npz",pairs=pairs[keep],context=pack["context"][keep],target=pack["target"][keep],contexts=pack["contexts"])
    audit={"trajectories":len(pairs),"retained":int(keep.sum()),"excluded":int((~keep).sum()),"held_out_genes":len(held),"symbols":sorted(meta.loc[meta.unified_id.isin(held),"symbol"].astype(str))}; (out/f"{name}.json").write_text(json.dumps(audit,indent=2)); print(audit|{"symbols":audit["symbols"][:10]})


def context_external():
    import ast, json, numpy as np
    out=ROOT/"results/sl_predict"; source=np.load(out/"slkb_outcomes_intervention_external.npz",allow_pickle=True); means={}
    for path in (ROOT/"data/models/SLAMR/data_slb_filtered/LLM_emb").glob("*_gpt-5.1_*_desc_embedding.csv"):
        rows=[np.asarray(ast.literal_eval(r["Embedding"]),"float32") for r in csv.DictReader(path.open(encoding="utf-8")) if r["gene_name"]!="gene_name"]; cell=path.name.split("_gpt-5.1_",1)[1].split("_desc_",1)[0].upper(); x=np.mean(rows,0); means[cell]=x/(np.linalg.norm(x)+1e-8)
    rng=np.random.default_rng(123); proj=rng.normal(0,1/np.sqrt(1536),(1536,32)).astype("float32"); cell_state={k:(v@proj).astype("float32") for k,v in means.items()}; state=np.stack([cell_state.get(str(x).split("|")[-1].upper(),np.zeros(32,"float32")) for x in source["contexts"]]); known=np.linalg.norm(state,axis=1)>0; name="slkb_outcomes_intervention_context"; np.savez_compressed(out/f"{name}.npz",pairs=source["pairs"],context=source["context"],target=source["target"],contexts=source["contexts"],context_state=state,context_known=known); audit={"dimensions":32,"known_cells":sorted(means),"known_contexts":int(known.sum()),"total_contexts":len(known)}; (out/f"{name}.json").write_text(json.dumps(audit,indent=2)); print(audit)


if __name__ == "__main__":
    import sys
    {"strict_external":strict_external,"intervention_external":intervention_external,"context_external":context_external}.get(sys.argv[1] if len(sys.argv)>1 else "",main)()
