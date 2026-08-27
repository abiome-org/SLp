import json,pickle
from pathlib import Path
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; MU=ROOT/"data/models/MuSL/processed_data"

def canon(a,b): return np.where(a<b,a+"|"+b,b+"|"+a)

def main():
    x=pd.read_csv(ROOT/"data/spidr2025/41586_2025_8815_MOESM5_ESM.csv"); genes=x.gene_combination.str.split(";",n=1,expand=True); x["a"],x["b"]=genes[0].str.upper(),genes[1].str.upper(); x["pair"]=canon(x.a,x.b); score=pd.to_numeric(x["sens.score"],errors="coerce"); unique=x.pair.nunique(); conflicts=int(x.assign(score=score).groupby("pair").score.nunique().gt(1).sum()); all_genes=set(x.a)|set(x.b); universe=set(pd.read_csv(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv").symbol.astype(str).str.upper())
    mirror=pd.read_csv(MU/"spidr_res_filtered.csv"); mirror["pair"]=canon(mirror["Gene A"].str.upper(),mirror["Gene B"].str.upper()); joined=mirror.merge(x[["pair","sens.score"]],on="pair"); mirror_corr=float(np.corrcoef(joined["GEMINI sensitive"],joined["sens.score"])[0,1])
    meta=pd.read_csv(MU/"meta_table_7684.csv"); symbols=meta.symbol.astype(str).str.upper().to_numpy(); root=MU/"data/CV3_bins_32/fold_data"; benchmark=set()
    for seed in (42,432):
        for fold in pickle.load(open(root/f"test_pairs_seed{seed}.pkl","rb")):
            p=np.asarray(fold,"int64"); benchmark.update(canon(symbols[p[:,0]],symbols[p[:,1]]))
    source=set(x.pair); overlap=source&benchmark; benchmark_genes={g for p in benchmark for g in p.split("|")}; result={"schema":"sl-predict-spidr-public-access-v1","rows":len(x),"unique_nonself_pairs":unique,"genes":len(all_genes),"mapped_genes":len(all_genes&universe),"finite_scores":int(np.isfinite(score).sum()),"finite_fraction":float(np.isfinite(score).mean()),"conflicting_duplicate_pairs":conflicts,"local_filtered_mirror":{"rows":len(mirror),"matched_pairs":len(joined),"pearson":mirror_corr},"musl_overlap_without_labels":{"benchmark_pairs":len(benchmark),"exact_pairs":len(overlap),"source_genes_in_benchmark":len(all_genes&benchmark_genes)},"admitted":bool(unique>=145000 and len(all_genes)>=540 and len(all_genes&universe)>=500 and np.isfinite(score).mean()>=.999 and conflicts==0 and len(joined)==len(mirror) and mirror_corr>=.999),"sl_benchmark_labels_used":False}; (OUT/"spidr_public_access.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
