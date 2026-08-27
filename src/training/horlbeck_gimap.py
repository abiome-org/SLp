import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"

def tail(a,b,q=.01): return float(np.mean((a<=np.quantile(a,q))&(b<=np.quantile(b,q)))/(q*q))
def guide_gene(x): return np.where(x.str.startswith("negative_control"),"negative_control",x.str.extract(r"^(.+?)_[+-]_",expand=False))

def main():
    path=ROOT/"data/horlbeck2018/GSE116198_sgRNA_pair_phenotypes.txt.gz"; frame=pd.read_csv(path,sep="\t",skiprows=4,header=None,usecols=[0,4,5,13,14]); frame.columns=["pair","jurkat_r1","jurkat_r2","k562_r1","k562_r2"]; guides=frame.pair.str.split("++",n=1,expand=True,regex=False); frame["guide_a"],frame["guide_b"]=guides[0],guides[1]; frame["gene_a"],frame["gene_b"]=guide_gene(frame.guide_a),guide_gene(frame.guide_b); gene_rows=(frame.gene_a!="negative_control")&(frame.gene_b!="negative_control")&frame.gene_a.notna()&frame.gene_b.notna()&(frame.gene_a!=frame.gene_b); pair=np.where(frame.gene_a<frame.gene_b,frame.gene_a+"|"+frame.gene_b,frame.gene_b+"|"+frame.gene_a); values={}
    for cell in ("jurkat","k562"):
        reps=[]
        for rep in (1,2):
            col=f"{cell}_r{rep}"; baseline=float(frame.loc[(frame.gene_a=="negative_control")&(frame.gene_b=="negative_control"),col].median()); sa=frame.loc[frame.gene_b=="negative_control"].groupby("guide_a")[col].median(); sb=frame.loc[frame.gene_a=="negative_control"].groupby("guide_b")[col].median(); residual=frame[col]-frame.guide_a.map(sa)-frame.guide_b.map(sb)+baseline; reps.append(pd.DataFrame({"pair":pair[gene_rows],"value":residual[gene_rows]}).groupby("pair").value.median())
        common=reps[0].index.intersection(reps[1].index); a=reps[0][common].to_numpy("float32"); b=reps[1][common].to_numpy("float32"); finite=np.isfinite(a)&np.isfinite(b); values[cell]=(common[finite],a[finite],b[finite])
    shared=values["jurkat"][0].intersection(values["k562"][0]); genes=sorted({g for p in shared for g in p.split("|")}); pos={g:i for i,g in enumerate(genes)}; pairs=np.asarray([[pos[g] for g in p.split("|")] for p in shared],"int16"); maps=[]; rows={}
    for cell in ("jurkat","k562"):
        index={p:i for i,p in enumerate(values[cell][0])}; a=np.asarray([values[cell][1][index[p]] for p in shared]); b=np.asarray([values[cell][2][index[p]] for p in shared]); maps.append(((a+b)/2).astype("float32")); rows[cell]={"pairs":len(shared),"replicate_pearson":float(np.corrcoef(a,b)[0,1]),"replicate_spearman":float(spearmanr(a,b).statistic),"bottom_one_percent_overlap_enrichment":tail(a,b)}
    universe=set(pd.read_csv(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv").symbol.astype(str).str.upper()); mapped=sum(g.upper() in universe for g in genes); admitted=len(shared)>=100000 and mapped>=400 and all(x["replicate_pearson"]>=.5 and x["replicate_spearman"]>=.5 and x["bottom_one_percent_overlap_enrichment"]>=10 for x in rows.values()); np.savez_compressed(OUT/"horlbeck_gimap.npz",genes=np.asarray(genes),pairs=pairs,jurkat=maps[0],k562=maps[1]); result={"schema":"sl-predict-horlbeck-gimap-v1","input_rows":len(frame),"unique_nonself_gene_pairs":len(shared),"genes":len(genes),"mapped_genes":mapped,"cells":rows,"cross_cell_pearson":float(np.corrcoef(maps[0],maps[1])[0,1]),"cross_cell_spearman":float(spearmanr(maps[0],maps[1]).statistic),"admitted":bool(admitted),"sl_labels_used":False}; (OUT/"horlbeck_gimap.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
