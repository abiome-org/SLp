"""Audit training data against SLB's held-out families.

Input: a parquet/csv of training records with columns `species`, `gene_a`, `gene_b` (pairs; e.g.
combinatorial screen measurements or SL labels you fit on). Gene IDs must be SLB canonical IDs
(human HGNC symbols, SGD ORFs, PomBase systematic IDs, FBgn) — use `slbench.ids.resolve`.

A record leaks if either gene belongs to a dev or test family. Genes not in the benchmark are
checked through their homology family too (a paralog/ortholog of a test gene also leaks).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from slbench import families
from slbench.build import bucket
from slbench.evaluate import BENCH


def family_buckets(extra_genes: pl.DataFrame | None = None) -> pl.DataFrame:
    fam = pl.read_parquet(BENCH / "held_out_families.parquet")
    if extra_genes is None or extra_genes.height == 0:
        return fam
    known = fam.select("species", "gene")
    new = extra_genes.join(known, on=["species", "gene"], how="anti").unique()
    if new.height == 0:
        return fam
    # Re-derive families including the new genes so homologs of held-out genes are caught.
    allg = pl.concat([known, new]).unique()
    f2 = families.assign(allg).with_columns(pl.col("family").map_elements(bucket, return_dtype=pl.String).alias("bucket"))
    # A new gene inherits the strictest bucket among benchmark genes sharing its family.
    rank = {"test": 0, "dev": 1, "train": 2}
    fam_b = f2.join(fam.select("species", "gene", pl.col("bucket").alias("b0")), on=["species", "gene"], how="left") \
        .group_by("family").agg(pl.col("b0").drop_nulls().replace_strict(rank, return_dtype=pl.Int8).min().alias("r"))
    inv = {v: k for k, v in rank.items()}
    f2 = f2.join(fam_b, on="family", how="left").with_columns(
        pl.when(pl.col("r").is_not_null()).then(pl.col("r").replace_strict(inv, return_dtype=pl.String))
        .otherwise(pl.col("bucket")).alias("bucket")).drop("r")
    return f2


def check(records: pl.DataFrame, allow_dev: bool = False) -> pl.DataFrame:
    genes = pl.concat([records.select("species", pl.col("gene_a").alias("gene")),
                       records.select("species", pl.col("gene_b").alias("gene"))]).unique()
    fb = family_buckets(genes).select("species", "gene", "bucket")
    r = records.join(fb.rename({"gene": "gene_a", "bucket": "bucket_a"}), on=["species", "gene_a"], how="left") \
        .join(fb.rename({"gene": "gene_b", "bucket": "bucket_b"}), on=["species", "gene_b"], how="left")
    bad = ["test"] + ([] if allow_dev else ["dev"])
    return r.filter(pl.col("bucket_a").is_in(bad) | pl.col("bucket_b").is_in(bad))


def main(path: str, allow_dev: bool = False) -> int:
    p = Path(path)
    rec = pl.read_parquet(p) if p.suffix == ".parquet" else pl.read_csv(p)
    leaks = check(rec.select("species", "gene_a", "gene_b"), allow_dev)
    print(f"{rec.height:,} records checked; {leaks.height:,} touch held-out families")
    if leaks.height:
        print(leaks.group_by("species", "bucket_a", "bucket_b").len().sort("len", descending=True))
        return 1
    return 0
