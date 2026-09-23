"""Turn SLp-1.1 pair scores into SLB prediction files (unscorable examples get a constant, i.e. ties)."""
import sys

import polars as pl

from slpbench.evaluate import load_split

split, col = sys.argv[1], sys.argv[2]  # col: sl_score | label_free
s = pl.read_csv(f"results/slp11_{split}_scores.tsv", separator="\t", has_header=False,
                new_columns=["gene_a", "gene_b", "sl_score", "label_free"], schema_overrides={"sl_score": pl.Float64, "label_free": pl.Float64})
d = load_split(split).select("example_id", "species", "gene_a", "gene_b")
d = d.join(s, on=["gene_a", "gene_b"], how="left")
cov = d.filter(pl.col("species") == "human")[col].is_not_null().mean()
fill = d[col].median()
d.select("example_id", pl.col(col).fill_null(fill).alias("score")).write_parquet(f"results/slp11_{col}_{split}.parquet")
print(f"{split} {col}: human coverage {cov:.1%}")
