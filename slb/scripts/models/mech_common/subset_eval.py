"""SLB dev metrics restricted to the examples a model actually covers (uses slbench's own eval machinery).

    uv run python scripts/models/mech_common/subset_eval.py results/models/fba_dev.parquet --mask in_model
The mask column (bool) in the prediction file marks covered examples. Reports per-species balanced AUROC
(SLB weighting), unadjusted AUROC, n and positives on the covered subset. Dev only.
"""
import argparse

import numpy as np
import polars as pl

from slbench.evaluate import _auc, headline, load_gold

ap = argparse.ArgumentParser()
ap.add_argument("preds")
ap.add_argument("--mask", default="in_model")
ap.add_argument("--split", default="dev")
a = ap.parse_args()
p = pl.read_parquet(a.preds).select("example_id", pl.col("score").cast(pl.Float64), pl.col(a.mask).cast(pl.Boolean))
df = load_gold(a.split).join(p, on="example_id", how="inner").filter(pl.col(a.mask))
print(f"covered subset of {a.split}: n={df.height:,} pos={int(df['label'].sum()):,}")
if df.height:
    s, parts = headline(df)
    print(f"covered-subset SLB (mean of species with data): {s:.4f}  " + "  ".join(f"{k}={v:.4f}" for k, v in parts.items()))
    for sp in df["species"].unique(maintain_order=True).to_list():
        d = df.filter(pl.col("species") == sp)
        if d.height:
            print(f"  {sp:6s} n={d.height:>7,} pos={int(d['label'].sum()):>5,} SLB-AUROC={_auc(d)[0]:.4f} "
                  f"unadj={_auc(d, balanced=False)[0]:.4f}")
