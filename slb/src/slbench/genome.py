"""Gene coordinates for the yeasts: genetic-linkage filtering and overlapping-ORF families.

scer: SGD_features.tab (every feature with a systematic name and coordinates).
spom: PomBase 2026-09-01 per-chromosome CDS spans, else gene spans (reference/fetch/pombase_coords.tsv).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import polars as pl

RAW = Path("data/raw")
LINKAGE_KB = 200  # same-chromosome pairs closer than this are unscored (linkage distorts double-mutant fitness)
OVERLAP_MIN_BP = 1  # physically overlapping ORFs share one family (deleting one disrupts the other)

_SGD_COLS = ["sgdid", "type", "qualifier", "orf", "std", "alias", "parent", "secondary", "chrom", "start", "stop",
             "strand", "gpos", "cver", "sver", "desc"]


@cache
def coords(species: str) -> pl.DataFrame:
    """gene, chrom, start, end (1-based, start <= end), qualifier (SGD Verified/Dubious/..., else null)."""
    if species == "scer":
        f = pl.read_csv(RAW / "ids/SGD_features.tab", separator="\t", has_header=False, new_columns=_SGD_COLS,
                        infer_schema_length=0, quote_char=None)
        f = f.filter(pl.col("orf").is_not_null() & pl.col("start").is_not_null() & (pl.col("type") != "CDS")
                     & pl.col("parent").str.starts_with("chromosome"))
        s, e = pl.col("start").cast(pl.Int64), pl.col("stop").cast(pl.Int64)
        f = f.select(pl.col("orf").alias("gene"), pl.col("chrom"), pl.min_horizontal(s, e).alias("start"),
                     pl.max_horizontal(s, e).alias("end"), pl.col("qualifier"))
    elif species == "spom":
        parts = []
        for kind in ("cds", "gene"):  # coding span where there is one (deletions remove the CDS), else the gene
            for p in sorted((RAW / "pombase/coords").glob(f"chromosome_*.{kind}.coords.tsv")):
                d = pl.read_csv(p, separator="\t", has_header=False, new_columns=["gene", "start", "stop", "strand"])
                parts.append(d.with_columns(pl.lit(p.name.split(".")[0]).alias("chrom")))
        f = pl.concat(parts).select("gene", "chrom", pl.min_horizontal("start", "stop").alias("start"),
                                    pl.max_horizontal("start", "stop").alias("end"),
                                    pl.lit(None, dtype=pl.String).alias("qualifier"))
    else:
        raise ValueError(f"no coordinates for {species}")
    return f.unique("gene", keep="first").sort("gene")


def pair_distance(pairs: pl.DataFrame, species: str) -> pl.Series:
    """Gap in bp between the two genes of each row (0 if they overlap); null if on different chromosomes or unknown."""
    c = coords(species)
    d = pairs.select("gene_a", "gene_b")
    for g in ("a", "b"):
        d = d.join(c.select(pl.col("gene").alias(f"gene_{g}"), pl.col("chrom").alias(f"c{g}"),
                            pl.col("start").alias(f"s{g}"), pl.col("end").alias(f"e{g}")),
                   on=f"gene_{g}", how="left", maintain_order="left")
    gap = pl.max_horizontal(pl.lit(0), pl.max_horizontal("sa", "sb") - pl.min_horizontal("ea", "eb"))
    return d.select(pl.when(pl.col("ca") == pl.col("cb")).then(gap).otherwise(None).alias("distance"))["distance"]


def overlapping_pairs(species: str, genes: set[str]) -> list[tuple[str, str]]:
    """Pairs of the given genes whose coordinates overlap by at least OVERLAP_MIN_BP."""
    c = coords(species).filter(pl.col("gene").is_in(sorted(genes))).sort("chrom", "start")
    out = []
    for _, g in c.group_by("chrom"):
        rows = g.sort("start").select("gene", "start", "end").rows()
        for i, (gi, si, ei) in enumerate(rows):
            for gj, sj, ej in rows[i + 1:]:
                if sj > ei - OVERLAP_MIN_BP + 1:
                    break
                if min(ei, ej) - max(si, sj) + 1 >= OVERLAP_MIN_BP:
                    out.append((gi, gj))
    return out
