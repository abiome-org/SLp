"""Turn SLp-1.1 pair scores into SLB prediction files (unscorable examples get a constant, i.e. ties)."""
import sys

import polars as pl

from slbench.evaluate import inputs_path

split, col = sys.argv[1], sys.argv[2]  # col: sl_score | label_free; optional argv[3] scores tsv, argv[4] output
s = pl.read_csv(sys.argv[3] if len(sys.argv) > 3 else f"results/slb/slp11_{split}_scores.tsv", separator="\t", has_header=False,
                new_columns=["gene_a", "gene_b", "sl_score", "label_free"], schema_overrides={"sl_score": pl.Float64, "label_free": pl.Float64})
d = pl.read_parquet(inputs_path(split)).select("example_id", "species", "gene_a", "gene_b")
d = d.join(s.unique(["gene_a", "gene_b"]), on=["gene_a", "gene_b"], how="left")
cov = d.filter(pl.col("species") == "human")[col].is_not_null().mean()
fill = d[col].median()
out = sys.argv[4] if len(sys.argv) > 4 else f"results/slb/slp11_{col}_{split}.parquet"
d.select("example_id", pl.col(col).fill_null(fill).alias("score")).write_parquet(out)
print(f"{split} {col}: human coverage {cov:.1%}")
