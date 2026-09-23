"""Within-species paralog pairs with protein sequence identity (Ensembl 116).

identity = max(%id query->target, %id target->query) / 100, over all Ensembl paralog types
(within_species_paralog and older other_paralog duplications).
"""

from __future__ import annotations

import functools
from pathlib import Path

import polars as pl

from slpbench import ids

RAW = Path("data/raw/paralogs")

BIOMART = {"human": "ensembl_hsapiens_paralogs.tsv", "scer": "ensembl_scerevisiae_paralogs.tsv",
           "dmel": "ensembl_dmelanogaster_paralogs.tsv"}


@functools.cache
def paralogs() -> pl.DataFrame:
    parts = []
    for sp, f in BIOMART.items():
        df = pl.read_csv(RAW / f, separator="\t", infer_schema_length=0, quote_char=None)
        df = df.filter(~df[df.columns[0]].str.starts_with("[success]"))
        df = df.select(
            pl.col(df.columns[0]).alias("a"), pl.col(df.columns[2]).alias("b"),
            pl.max_horizontal(pl.col(df.columns[6]).cast(pl.Float64), pl.col(df.columns[7]).cast(pl.Float64)).alias("identity"),
        ).drop_nulls()
        parts.append(_canon(df, sp))
    c = pl.read_csv(RAW / "compara116_spombe_homologies.tsv.gz", separator="\t", infer_schema_length=0,
                    columns=["gene_stable_id", "homology_type", "homology_gene_stable_id", "homology_species",
                             "identity", "homology_identity"])
    c = c.filter((pl.col("homology_species") == "schizosaccharomyces_pombe") & pl.col("homology_type").str.contains("paralog"))
    c = c.select(pl.col("gene_stable_id").alias("a"), pl.col("homology_gene_stable_id").alias("b"),
                 pl.max_horizontal(pl.col("identity").cast(pl.Float64), pl.col("homology_identity").cast(pl.Float64)).alias("identity"))
    parts.append(_canon(c, "spom"))
    return pl.concat(parts)


def _canon(df: pl.DataFrame, sp: str) -> pl.DataFrame:
    """Map Ensembl gene IDs to SLB canonical IDs; one row per unordered pair (max identity)."""
    if sp in ("human", "scer", "spom"):
        df = df.with_columns(ids.resolve(sp, df["a"]).alias("a"), ids.resolve(sp, df["b"]).alias("b"))
    df = df.drop_nulls().filter(pl.col("a") != pl.col("b")).with_columns(
        pl.min_horizontal("a", "b").alias("a"), pl.max_horizontal("a", "b").alias("b"), (pl.col("identity") / 100))
    return df.group_by("a", "b").agg(pl.col("identity").max()).with_columns(pl.lit(sp).alias("species")) \
        .select("species", "a", "b", "identity")


@functools.cache
def identity_lookup() -> dict[tuple[str, str, str], float]:
    return {(s, a, b): w for s, a, b, w in paralogs().iter_rows()}
