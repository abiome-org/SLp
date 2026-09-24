"""Assemble SBSL feature tables (features.py + discover_run.py) into CSVs for train.R, and convert train.R output to
results/models/<name>_<split>.parquet.
usage: python prep_csv.py csv <split>          -> external/models/sbsl/_slb/<split>.csv
       python prep_csv.py out <split> <name> <model.csv>"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "elisl"))
import common as C  # noqa: E402

KEY = ["gene_a", "gene_b", "cancer"]
WORK = C.ROOT / "external/models/sbsl/_slb"
FEATS = ["copathway_participation", "dsl_mutex_amp", "dsl_mutex_del", "dsl_mutex_mut", "dsl_mutex", "exp_corr",
         "exp_corr_pvalue", "exp_corr_normal", "exp_corr_normal_pvalue", "diff_exp_pvalue", "diff_exp_logfc",
         "avana_codep", "avana_codep_pvalue", "avana_dep", "avana_dep_pvalue", "avana_avg", "d2_codep",
         "d2_codep_pvalue", "d2_dep", "d2_dep_pvalue", "d2_avg", "discover_mutex", "mut_logrank.pval",
         "mrna_logrank.pval", "mutex_alt", "gtex_corr", "gtex_corr.pvalue"]


def csv(split):
    X = pd.read_parquet(WORK / "feat" / C.BENCH.name / f"{split}.parquet")
    d = pd.read_parquet(WORK / "cache/keys_discover.parquet")
    X = X.merge(d, on=KEY, how="left")
    # hypergeometric mutex tests are undefined (NaN) when the cancer type has no samples with both GISTIC and MC3
    # calls (LAML in PanCanAtlas); SBSL's formula gives 1 - P(X >= 0) = 0 when there are no alterations -> use 0
    mx = ["dsl_mutex_amp", "dsl_mutex_del", "dsl_mutex_mut", "mutex_alt"]
    X[mx] = X[mx].fillna(0.0)
    lab = C.human(split)
    if "label" in lab:
        X["SL"] = X.example_id.map(dict(zip(lab.example_id, lab.label))).astype(int)
    else:
        X["SL"] = 0
    X[["example_id", "context_id", "SL"] + FEATS].to_csv(WORK / "feat" / C.BENCH.name / f"{split}.csv", index=False)
    C.log(f"{split}.csv: {len(X):,} rows; rows with any NA feature: {X[FEATS].isna().any(axis=1).sum():,}")


def out(split, name, model_csv):
    s = pd.read_csv(model_csv)
    p = C.ROOT / "results/models" / f"{name}_{split}.parquet"
    s[["example_id", "score"]].astype({"score": "float64"}).to_parquet(p, index=False)
    C.log(f"wrote {p} ({len(s):,} rows)")


if __name__ == "__main__":
    if sys.argv[1] == "csv":
        csv(sys.argv[2])
    else:
        out(*sys.argv[2:5])
