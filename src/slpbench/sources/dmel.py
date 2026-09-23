"""Drosophila S2-cell combinatorial RNAi genetic-interaction maps (Boutros lab)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from . import finalize

RAW = Path("data/raw")

# Positive: significant aggravating interaction on cell count, at each paper's own FDR
# (Fischer 2015: 1%; Heigwer 2023: 10%).
# Negative: clearly non-significant and |pi| below the median |pi| of the screen.
NEG_FDR = 0.25


def fischer2015() -> pl.DataFrame:
    """Fischer et al. 2015 eLife: 1,293 targets x 72 queries, S2 cells, pi-score on cell count."""
    import rdata

    c = rdata.conversion.convert(rdata.parser.parse_file(RAW / "fischer2015_dmel/Interactions.rda"))["Interactions"]
    pi = c["piscore"].sel(phenotype="4x.count")
    pa = c["padj"].sel(phenotype="4x.count")
    tsym2fb = dict(zip(c["Anno"]["target"]["Symbol"], c["Anno"]["target"]["TID"]))
    qsym2fb = dict(zip(c["Anno"]["query"]["Symbol"], c["Anno"]["query"]["TID"]))
    t = [tsym2fb[str(s)] for s in pi.coords["target"].values]
    q = [qsym2fb[str(s)] for s in pi.coords["query"].values]
    T, Q = np.meshgrid(t, q, indexing="ij")
    df = pl.DataFrame({"gene_a": T.ravel(), "gene_b": Q.ravel(),
                       "score": pi.values.ravel().astype(float), "signif": pa.values.ravel().astype(float)})
    df = df.filter(pl.col("score").is_not_nan() & pl.col("signif").is_not_nan())
    return _label(df, "fischer2015", "S2", "RNAi", "pi_cellcount", "padj_BH", pos_fdr=0.01)


def heigwer2023() -> pl.DataFrame:
    """Heigwer et al. 2023 Cell Systems: genome-scale S2 RNAi GI map; keep the cell-count feature."""
    path = RAW / "heigwer2023_dmel/interactions_stat_tested_bias_corrected.csv.gz"
    lf = pl.scan_csv(path, infer_schema_length=10000)
    df = (
        lf.filter(pl.col("feature") == HEIGWER_COUNT_FEATURE)
        .select(pl.col("fbgn").alias("gene_a"), pl.col("query_name").alias("gene_b"),
                pl.col("mpi").cast(pl.Float64).alias("score"), pl.col("fdr").cast(pl.Float64).alias("signif"))
        .collect()
    )
    return _label(df, "heigwer2023", "S2", "RNAi", "mean_pi_cellcount", "fdr", pos_fdr=0.10, resolve=True)


HEIGWER_COUNT_FEATURE = "cells"


def _label(df, source, context, mechanism, score_name, signif_name, pos_fdr, resolve=False) -> pl.DataFrame:
    abs_med = df["score"].abs().median()
    df = df.with_columns(
        pl.lit("dmel").alias("species"), pl.lit(source).alias("source"), pl.lit(context).alias("context"),
        pl.lit(mechanism).alias("mechanism"), pl.lit(score_name).alias("score_name"),
        pl.lit(signif_name).alias("signif_name"),
        pl.when((pl.col("signif") < pos_fdr) & (pl.col("score") < 0)).then(1)
        .when((pl.col("signif") > NEG_FDR) & (pl.col("score").abs() < abs_med)).then(0)
        .otherwise(None).cast(pl.Int8).alias("label"),
    )
    return finalize(df, "dmel" if resolve else None)
