import csv, gzip, hashlib, json, urllib.parse, urllib.request
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; DATA=ROOT/"data/ensembl"
ATTR=["ensembl_gene_id","external_gene_name","hsapiens_paralog_ensembl_gene","hsapiens_paralog_associated_gene_name","hsapiens_paralog_subtype","hsapiens_paralog_orthology_type","hsapiens_paralog_perc_id","hsapiens_paralog_perc_id_r1"]

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest().upper()
def content_sha(path):
    h=hashlib.sha256()
    with gzip.open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest().upper()
def auroc(pos,neg):
    y=np.r_[np.ones(len(pos)),np.zeros(len(neg))]; r=rankdata(np.r_[pos,neg]); n=len(pos)
    return float((r[:n].sum()-n*(n+1)/2)/(n*len(neg)))

def download():
    attrs="".join(f'<Attribute name="{x}"/>' for x in ATTR); query=f'<?xml version="1.0" encoding="UTF-8"?><Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="0" count="" datasetConfigVersion="0.6"><Dataset name="hsapiens_gene_ensembl" interface="default">{attrs}</Dataset></Query>'
    url="https://jun2026.archive.ensembl.org/biomart/martservice?query="+urllib.parse.quote(query); request=urllib.request.Request(url,headers={"User-Agent":"SL-Predict public-source audit"}); raw=urllib.request.urlopen(request,timeout=1800).read()
    if raw.startswith(b"Query ERROR") or not raw.endswith(b"\n"): raise RuntimeError(raw[-500:].decode(errors="replace"))
    DATA.mkdir(parents=True,exist_ok=True); (DATA/"human_paralogs_r116.tsv.gz").write_bytes(gzip.compress(raw,compresslevel=9,mtime=0))

def controls(i,j,pairs,seed):
    rng=np.random.default_rng(seed); flip=rng.random(len(i))<.5
    a=np.where(flip,j,i); b=np.where(flip,i,j); b=b[rng.permutation(len(b))]
    keep=np.asarray([x!=y and (min(x,y),max(x,y)) not in pairs for x,y in zip(a,b)])
    return a[keep],b[keep]

def main(fetch=False):
    raw=DATA/"human_paralogs_r116.tsv.gz"
    if fetch or not raw.exists(): download()
    meta=list(csv.DictReader(open(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv",encoding="utf-8")))
    emap={r["ensembl_gene_id"].split(".")[0]:k for k,r in enumerate(meta)}; found={}; source_rows=0; types={}
    with gzip.open(raw,"rt",encoding="utf-8") as f:
        reader=csv.reader(f,delimiter="\t"); header=next(reader)
        if len(header)!=len(ATTR): raise ValueError(header)
        for row in reader:
            if len(row)!=len(ATTR) or not row[2]: continue
            source_rows+=1; a=emap.get(row[0].split(".")[0]); b=emap.get(row[2].split(".")[0])
            if a is None or b is None or a==b: continue
            key=(min(a,b),max(a,b)); ids=[float(x) for x in row[6:8] if x]
            ident=float(np.mean(ids)) if ids else np.nan
            if key not in found or ident>found[key]: found[key]=ident
            types[row[5]]=types.get(row[5],0)+1
    pairs=set(found); i=np.asarray([x[0] for x in sorted(pairs)],dtype="int16"); j=np.asarray([x[1] for x in sorted(pairs)],dtype="int16"); identity=np.asarray([found[x] for x in sorted(pairs)],dtype="float32")
    dep=np.load(OUT/"depmap_codependency.npz"); pos=np.full(len(meta),-1,dtype="int32"); pos[dep["genes"].astype("int64")]=np.arange(len(dep["genes"])); keep=(pos[i]>=0)&(pos[j]>=0); ei=i[keep].astype("int64"); ej=j[keep].astype("int64"); ident=identity[keep]; di=pos[ei]; dj=pos[ej]
    scores=[dep[f"half{k}"][di,dj].astype("float32") for k in range(2)]; seed_metrics={}
    for seed in (811,812):
        ni,nj=controls(ei,ej,pairs,seed); ndi=pos[ni]; ndj=pos[nj]
        seed_metrics[str(seed)]={"control_pairs":len(ni),**{f"half{k}_auroc":auroc(scores[k],dep[f"half{k}"][ndi,ndj].astype("float32")) for k in range(2)}}
    lower=[min(x[f"half{k}_auroc"] for x in seed_metrics.values()) for k in range(2)]; finite=np.isfinite(ident)
    rho=[float(spearmanr(ident[finite],scores[k][finite]).statistic) for k in range(2)]
    protocol=json.loads((OUT/"ensembl_paralog_protocol.json").read_text()); limits=protocol["admission"]
    exact=bool(np.all(i<j) and len(pairs)==len(i)); genes=len(set(i.tolist()+j.tolist()))
    admitted=genes>=limits["minimum_unique_mapped_genes"] and len(i)>=limits["minimum_unique_mapped_pairs"] and len(ei)>=limits["minimum_depmap_evaluated_pairs"] and exact and min(lower)>=limits["minimum_depmap_half0_auroc"] and min(rho)>=limits["minimum_identity_codependency_spearman_each_half"]
    np.savez_compressed(OUT/"ensembl_paralogs.npz",i=i,j=j,identity=identity)
    result={"schema":"sl-predict-ensembl-paralog-v1","ensembl_release":116,"source_rows":source_rows,"mapped_homology_type_rows":types,"unique_mapped_pairs":len(i),"unique_mapped_genes":genes,"depmap_evaluated_pairs":len(ei),"exact_unordered_deduplication":exact,"identity_finite_pairs":int(finite.sum()),"control_seed_metrics":seed_metrics,"lower_auroc_by_half":lower,"identity_codependency_spearman_by_half":rho,"admitted":bool(admitted),"double_perturbation_data_used":False,"sl_labels_used":False,"benchmark_pairs_used":False,"source_gzip_sha256":sha(raw),"source_content_sha256":content_sha(raw),"attribute_manifest_sha256":sha(DATA/"biomart_attributes_r116.tsv"),"protocol_sha256":sha(OUT/"ensembl_paralog_protocol.json")}
    (OUT/"ensembl_paralogs.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=="__main__":
    import sys; main("--fetch" in sys.argv)
