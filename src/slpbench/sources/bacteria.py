"""Bacterial pairwise CRISPRi screens."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from . import finalize

RAW = Path("data/raw")


def dualcrispri2025() -> pl.DataFrame:
    """Dual CRISPRi-seq, Cell Systems 2025: S. pneumoniae D39V, ~869 operon-targeting sgRNAs pairwise.

    Units are sgRNA target operons; only sgRNAs whose target is a single gene are kept, so
    every example is gene x gene. IDs are the authors' D39V gene names / SPV_ locus tags.
    Positive: authors' interactionSum == Negative. Negative: authors' Neutral, excluding
    essential x essential pairs (both knockdowns already at the fitness floor, so "neutral"
    is uninformative) and requiring |epsilon| below the median.
    """
    df = pl.read_csv(RAW / "dualcrispri2025_spneumo/mmc4.csv", infer_schema_length=0, null_values=["NA", ""])
    df = df.filter(~pl.col("SG1.targets").str.contains(",") & ~pl.col("SG2.targets").str.contains(","))
    df = df.with_columns(pl.col("epsilonSum").cast(pl.Float64), pl.col("padj").cast(pl.Float64))
    med = df["epsilonSum"].abs().median()
    df = df.select(
        pl.lit("spne").alias("species"), pl.lit("dualcrispri2025").alias("source"), pl.lit("D39V").alias("context"),
        pl.lit("CRISPRi").alias("mechanism"),
        pl.col("SG1.targets").alias("gene_a"), pl.col("SG2.targets").alias("gene_b"),
        pl.col("epsilonSum").alias("score"), pl.lit("epsilon_log2FC").alias("score_name"),
        pl.col("padj").alias("signif"), pl.lit("padj").alias("signif_name"),
        pl.when(pl.col("interactionSum") == "Negative").then(1)
        .when((pl.col("interactionSum") == "Neutral") & (pl.col("pairs") != "E-E")
              & (pl.col("epsilonSum").abs() < med)).then(0)
        .otherwise(None).cast(pl.Int8).alias("label"),
    )
    return finalize(df)
