"""Extra homology edges for the family graph: new species, S. pneumoniae, and bacteria <-> eukaryote orthology.

Companion to families.py (unchanged). extra_edges() returns edges in the families.edges() schema
(u, v, kind, weight) with nodes "species:gene" (canonical IDs: ids.py / ids_extra.py). The lead can
combine them as pl.concat([families.edges(), families_extra.extra_edges()]).unique().

Edge sources (built by scripts/orthology_extra/build_edges.py):
  diamond_rbh      reciprocal best DIAMOND hits between every pair of the 12 proteomes (orthologs, weight 1.0)
  alliance         Alliance combined orthology for pairs involving mouse / worm (orthologs, weight 1.0)
  ensembl_biomart  Ensembl 116 paralogs for mouse / worm, max(%id, %id_r1) >= 30% (paralogs, weight = identity)
  diamond_self     DIAMOND within-species hits, nident / min(qlen, slen) >= 30%, for calb, spne and
                   the other bacteria (paralogs, weight = identity)
  diamond_self_global  DIAMOND within-species hits for human/scer/spom/dmel, nident / max(qlen, slen) >= 30%
                   (full-length paralogs Ensembl misses)
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

EDGES = Path("data/interim/orthology_extra/edges.parquet")
BASE_SPECIES = {"human", "scer", "spom", "dmel"}  # species whose homology families.edges() already covers
ALL_SOURCES = ("diamond_rbh", "alliance", "ensembl_biomart", "diamond_self", "diamond_self_global")


def extra_edges_with_source(sources: tuple[str, ...] = ALL_SOURCES, base_pairs: bool = True,
                            species: set[str] | None = None) -> pl.DataFrame:
    """Edges with their source column.

    base_pairs: keep DIAMOND RBH orthologs between two base species (human/scer/spom/dmel). These add
      homology links that the curated base resources miss (e.g. spom <-> dmel direct, or RBH pairs not
      supported by >= 3 Alliance methods); they are conservative for leakage but merge some curated families.
    species: if given, drop edges touching a species outside this set.
    """
    e = pl.read_parquet(EDGES).filter(pl.col("source").is_in(list(sources)))
    su = pl.col("u").str.split(":").list.first()
    sv = pl.col("v").str.split(":").list.first()
    if not base_pairs:
        e = e.filter(~(su.is_in(list(BASE_SPECIES)) & sv.is_in(list(BASE_SPECIES))))
    if species is not None:
        e = e.filter(su.is_in(list(species)) & sv.is_in(list(species)))
    return e


def extra_edges(sources: tuple[str, ...] = ALL_SOURCES, base_pairs: bool = True,
                species: set[str] | None = None) -> pl.DataFrame:
    """Same schema as families.edges(): u, v, kind, weight (one row per unordered pair and kind)."""
    e = extra_edges_with_source(sources, base_pairs, species)
    return e.group_by("u", "v", "kind").agg(pl.col("weight").max()).select("u", "v", "kind", "weight").sort("u", "v", "kind")


def all_edges(**kw) -> pl.DataFrame:
    """families.edges() plus extra_edges(**kw)."""
    from slbench import families

    return pl.concat([families.edges(), extra_edges(**kw)]).unique()
