"""Per-source parsers. Each exposes `load() -> pl.DataFrame` in the MEASUREMENT schema."""

from __future__ import annotations

import polars as pl

# One row per (source, context, unordered gene pair) measurement.
MEASUREMENT_SCHEMA = {
    "species": pl.String,     # human | scer | spom | dmel
    "source": pl.String,      # study key, e.g. costanzo2016
    "context": pl.String,     # cell line / strain+condition; human contexts are cell-line names
    "mechanism": pl.String,   # CRISPR-KO | CRISPR-Cas12a | CRISPRi | RNAi | deletion | hypomorph
    "gene_a": pl.String,      # canonical IDs, gene_a < gene_b
    "gene_b": pl.String,
    "score": pl.Float64,      # native interaction score; negative = aggravating / synthetic sick-lethal
    "score_name": pl.String,
    "signif": pl.Float64,     # p / FDR / q as published (null if the source has none)
    "signif_name": pl.String,
    "label": pl.Int8,         # 1 = SL/sick (strong negative GI), 0 = measured non-interaction, null = ambiguous
}


def finalize(df: pl.DataFrame, species_resolve: str | None = None) -> pl.DataFrame:
    """Resolve IDs, order pairs, drop self-pairs/unmapped, collapse duplicate measurements.

    Duplicates of the same pair within (source, context) — e.g. A-query×B-array and
    B-query×A-array — are merged: score = mean, signif = min, label = consensus
    (conflicting 1/0 labels become null).
    """
    from slpbench import ids

    if species_resolve:
        df = df.with_columns(
            ids.resolve(species_resolve, df["gene_a"]).alias("gene_a"),
            ids.resolve(species_resolve, df["gene_b"]).alias("gene_b"),
        )
    df = df.filter(pl.col("gene_a").is_not_null() & pl.col("gene_b").is_not_null() & (pl.col("gene_a") != pl.col("gene_b")))
    df = df.with_columns(
        pl.min_horizontal("gene_a", "gene_b").alias("gene_a"),
        pl.max_horizontal("gene_a", "gene_b").alias("gene_b"),
    )
    keys = ["species", "source", "context", "mechanism", "gene_a", "gene_b", "score_name", "signif_name"]
    df = df.group_by(keys).agg(
        pl.col("score").mean(),
        pl.col("signif").min(),
        pl.col("label").min().alias("_lmin"),
        pl.col("label").max().alias("_lmax"),
        pl.col("label").null_count().alias("_lnull"),
    ).with_columns(
        pl.when((pl.col("_lmin") == pl.col("_lmax")) & (pl.col("_lnull") == 0)).then(pl.col("_lmin"))
        .otherwise(None).cast(pl.Int8).alias("label")
    )
    return df.select(list(MEASUREMENT_SCHEMA)).cast(MEASUREMENT_SCHEMA)


def neutral_mask(score: pl.Expr, signif: pl.Expr | None, abs_cut: pl.Expr | float, signif_cut: float | None) -> pl.Expr:
    """Measured non-interaction: small |score| and (if available) clearly non-significant."""
    m = score.abs() < abs_cut
    if signif is not None and signif_cut is not None:
        m = m & (signif > signif_cut)
    return m
