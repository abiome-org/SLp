import json,zipfile
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/"data/horlbeck2018"; OUT=ROOT/"results/sl_predict"
FILES={"k562":("K562.zip","CRISPRi_K562_replicateAverage_GIscores_genes_inclnegs.txt"),"jurkat":("Jurkat.zip","CRISPRi_Jurkat_emap_gene_filt.txt")}

def load(cell):
    archive,name=FILES[cell]
    with zipfile.ZipFile(DATA/archive) as z: x=pd.read_csv(z.open(name),sep="\t",index_col=0)
    common=x.index.astype(str).intersection(x.columns.astype(str)); x=x.loc[common,common].apply(pd.to_numeric,errors="coerce"); keep=~x.index.str.lower().str.startswith("negative_control"); return x.loc[keep,keep]

def summarize(x,universe):
    a=x.to_numpy("float32"); i,j=np.triu_indices(len(x),1); finite=np.isfinite(a[i,j]); both=np.isfinite(a)&np.isfinite(a.T); error=float(np.max(np.abs(a[both]-a.T[both]))) if both.any() else float("inf"); return {"genes":len(x),"mapped_genes":sum(g.upper() in universe for g in x.index),"finite_pairs":int(finite.sum()),"symmetry_max_abs_error":error},(x.index.to_numpy(),a,i[finite],j[finite],a[i[finite],j[finite]])

def main():
    universe=set(pd.read_csv(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv").symbol.astype(str).str.upper()); frames={c:load(c) for c in FILES}; stats={}; packs={}
    for c,x in frames.items(): stats[c],packs[c]=summarize(x,universe)
    common=frames["k562"].index.intersection(frames["jurkat"].index); a=frames["k562"].loc[common,common].to_numpy("float32"); b=frames["jurkat"].loc[common,common].to_numpy("float32"); i,j=np.triu_indices(len(common),1); good=np.isfinite(a[i,j])&np.isfinite(b[i,j]); a,b=a[i[good],j[good]],b[i[good],j[good]]; q=.01; enrichment=float(np.mean((a<=np.quantile(a,q))&(b<=np.quantile(b,q)))/(q*q)); cross={"genes":len(common),"pairs":len(a),"pearson":float(np.corrcoef(a,b)[0,1]),"spearman":float(spearmanr(a,b).statistic),"bottom_one_percent_overlap_enrichment":enrichment}; admitted=stats["k562"]["finite_pairs"]>=100000 and stats["k562"]["mapped_genes"]>=400 and stats["jurkat"]["finite_pairs"]>=60000 and stats["jurkat"]["mapped_genes"]>=300 and all(v["symmetry_max_abs_error"]<=1e-6 for v in stats.values()) and cross["spearman"]>=.30 and enrichment>=3
    np.savez_compressed(OUT/"horlbeck_processed_gimap.npz",k562_genes=packs["k562"][0],k562_matrix=packs["k562"][1].astype("float16"),jurkat_genes=packs["jurkat"][0],jurkat_matrix=packs["jurkat"][1].astype("float16")); result={"schema":"sl-predict-horlbeck-processed-gimap-v1","cells":stats,"cross_cell":cross,"admitted":bool(admitted),"sl_labels_used":False}; (OUT/"horlbeck_processed_gimap.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
