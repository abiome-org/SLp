"""Audit training data against SLB's held-out families.

Input: a parquet/csv of training records with columns `species`, `gene_a`, `gene_b` (pairs; e.g.
combinatorial screen measurements or SL labels you fit on). Species are SLB codes (see SPECIES); genes
may be any identifier the SLB resolvers know (symbols in any case, previous symbols, aliases, Ensembl,
Entrez, ORFs, locus tags) and are mapped to canonical IDs with `slbench.ids_extra.resolver`.

A record leaks if either gene belongs to a dev or test family. Genes not in the benchmark are
checked through the homology graph (a paralog/ortholog of a test gene also leaks). Each gene gets a
status: test/dev/train (its family's bucket), `none` (homology graph component without benchmark
genes: clean) or `unknown` (unresolvable, or absent from the homology graph: not verified clean).
"""

from __future__ import annotations

import functools
from pathlib import Path

import polars as pl

from slbench import families, ids, ids_extra
from slbench.evaluate import BENCH

SPECIES = sorted(set(ids.RESOLVERS) | set(ids_extra.RESOLVERS))
RANK = {"test": 0, "dev": 1, "train": 2}


@functools.cache
def _graph() -> tuple[families.DSU, dict[str, str]]:
    """Homology components over families.edges(), and component root -> strictest benchmark bucket."""
    fam = pl.read_parquet(BENCH / "held_out_families.parquet")
    dsu = families.DSU()
    for u, v in families.edges().select("u", "v").iter_rows():
        dsu.union(u, v)
    comp: dict[str, str] = {}
    for sp, g, b in fam.select("species", "gene", "bucket").iter_rows():
        r = dsu.find(f"{sp}:{g}")
        if r not in comp or RANK[b] < RANK[comp[r]]:
            comp[r] = b
    return dsu, comp


def gene_status(genes: pl.DataFrame) -> pl.DataFrame:
    """genes: species, gene (raw). Returns species, gene, gene_id (canonical or null), bucket."""
    bad = sorted(set(genes["species"].drop_nulls().unique()) - set(SPECIES)) + (["null"] if genes["species"].null_count() else [])
    if bad:
        raise ValueError(f"unknown species {bad}; valid species codes: {', '.join(SPECIES)}")
    fam = pl.read_parquet(BENCH / "held_out_families.parquet")
    known = {(s, g): b for s, g, b in fam.select("species", "gene", "bucket").iter_rows()}
    dsu, comp = _graph()
    rows = []
    for sp, g in genes.unique().iter_rows():
        gid = None if g is None else ids_extra.resolver(sp)(g)
        if gid is None and (sp, g) in known:
            gid = g
        if gid is None:
            b = "unknown"
        elif (sp, gid) in known:
            b = known[sp, gid]
        elif f"{sp}:{gid}" in dsu.p:
            b = comp.get(dsu.find(f"{sp}:{gid}"), "none")
        else:
            b = "unknown"
        rows.append((sp, g, gid, b))
    return pl.DataFrame(rows, schema={"species": pl.String, "gene": pl.String, "gene_id": pl.String,
                                      "bucket": pl.String}, orient="row")


def annotate(records: pl.DataFrame) -> pl.DataFrame:
    records = records.with_columns(pl.col("species", "gene_a", "gene_b").cast(pl.String))
    genes = pl.concat([records.select("species", pl.col("gene_a").alias("gene")),
                       records.select("species", pl.col("gene_b").alias("gene"))]).unique()
    st = gene_status(genes)
    return records.join(st.rename({"gene": "gene_a", "gene_id": "gene_a_id", "bucket": "bucket_a"}),
                        on=["species", "gene_a"], how="left", nulls_equal=True) \
        .join(st.rename({"gene": "gene_b", "gene_id": "gene_b_id", "bucket": "bucket_b"}),
              on=["species", "gene_b"], how="left", nulls_equal=True)


def _bad(allow_dev: bool) -> list[str]:
    return ["test"] + ([] if allow_dev else ["dev"])


def check(records: pl.DataFrame, allow_dev: bool = False) -> pl.DataFrame:
    """Records touching a held-out family."""
    bad = _bad(allow_dev)
    return annotate(records).filter(pl.col("bucket_a").is_in(bad) | pl.col("bucket_b").is_in(bad))


def main(path: str, allow_dev: bool = False, allow_unknown: bool = False) -> int:
    p = Path(path)
    rec = pl.read_parquet(p) if p.suffix == ".parquet" else pl.read_csv(p, infer_schema_length=0)
    try:
        r = annotate(rec.select("species", "gene_a", "gene_b"))
    except ValueError as e:
        print(f"error: {e}")
        return 2
    bad = _bad(allow_dev)
    leak = pl.col("bucket_a").is_in(bad) | pl.col("bucket_b").is_in(bad)
    unk = ~leak & ((pl.col("bucket_a") == "unknown") | (pl.col("bucket_b") == "unknown"))
    leaks, unknown = r.filter(leak), r.filter(unk)
    print(f"{r.height:,} records checked; {leaks.height:,} touch held-out families; "
          f"{unknown.height:,} involve unknown genes (not verified clean); "
          f"{r.height - leaks.height - unknown.height:,} clean")
    with pl.Config(tbl_rows=50, fmt_str_lengths=40):
        if leaks.height:
            print(leaks.group_by("species", "bucket_a", "bucket_b").len().sort("len", descending=True))
        if unknown.height:
            g = pl.concat([unknown.filter(pl.col(f"bucket_{s}") == "unknown").select("species", pl.col(f"gene_{s}").alias("gene"),
                                                                                      pl.col(f"gene_{s}_id").alias("gene_id"))
                           for s in "ab"]).unique().sort("species", "gene")
            print(f"{g.height:,} unknown genes (unresolved, or resolved but absent from the homology graph):")
            print(g)
    return 1 if leaks.height or (unknown.height and not allow_unknown) else 0
