"""Single-gene and homology tables the build writes into features/, so any copy of the benchmark is self-contained.

  line_effects.parquet      depmap_id, gene, effect   DepMap 24Q4 Chronos effect of each benchmark human gene in each
                                                      benchmark cell line (context-specific fitness covariate)
  paralogs.parquet          species, a, b, identity   Ensembl paralog identity between benchmark genes (a < b)
  codependency.parquet      gene_a, gene_b, codependency   DepMap profile correlation of every benchmark human pair
  homology_status.parquet   node, status              every node ("species:gene") of the homology graph and the
                                                      strictest split of a benchmark gene in its component
                                                      (test/dev/train, or none) — used by check-leakage
  gene_aliases.parquet      species, tier, key, gene_id   name -> canonical ID tables of the SLB resolvers
                                                      (tier 0: primary names; tier >= 1: unambiguous fallbacks)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

FILES = ("line_effects", "paralogs", "codependency", "homology_status", "gene_aliases")


def build(genes: pl.DataFrame, fam: pl.DataFrame, pairs: pl.DataFrame, contexts: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """genes: species, gene; fam: species, gene, bucket; pairs: species, gene_a, gene_b (every benchmark pair)."""
    from slbench import families, fitness, homology, ids, ids_extra, leakage

    human = set(genes.filter(pl.col("species") == "human")["gene"])
    dep_ids = tuple(sorted(contexts["depmap_id"].drop_nulls().unique()))
    line, _ = fitness.depmap_long(dep_ids)
    out = {"line_effects": line.filter(pl.col("gene").is_in(sorted(human))).sort("depmap_id", "gene")}

    par = homology.paralogs().join(genes.rename({"gene": "a"}), on=["species", "a"]) \
        .join(genes.rename({"gene": "b"}), on=["species", "b"])
    out["paralogs"] = par.sort("species", "a", "b")

    z, col = depmap_z()
    hp = pairs.filter(pl.col("species") == "human").select("gene_a", "gene_b").unique()
    hp = hp.filter(pl.col("gene_a").is_in(list(col)) & pl.col("gene_b").is_in(list(col)))
    ia = np.array([col[g] for g in hp["gene_a"]]); ib = np.array([col[g] for g in hp["gene_b"]])
    cod = np.einsum("ij,ij->j", z[:, ia], z[:, ib]) / z.shape[0] if len(ia) else np.array([])
    out["codependency"] = hp.with_columns(pl.Series("codependency", cod.astype(np.float64))).sort("gene_a", "gene_b")

    dsu = families.DSU()
    for u, v in pl.concat([families.edges(), families.overlap_edges(genes)]).select("u", "v").iter_rows():
        dsu.union(u, v)
    rank = {"test": 0, "dev": 1, "train": 2}
    comp: dict[str, str] = {}
    for sp, g, b in fam.select("species", "gene", "bucket").iter_rows():
        r = dsu.find(f"{sp}:{g}")
        if r not in comp or rank[b] < rank[comp[r]]:
            comp[r] = b
    nodes = sorted(dsu.p)
    out["homology_status"] = pl.DataFrame({"node": nodes, "status": [comp.get(dsu.find(n), "none") for n in nodes]})

    rows = []
    for sp in leakage.SPECIES:
        r = ids_extra.resolver(sp)
        rows += [(sp, 0, k, v) for k, v in r.primary.items() if v]
        for t, fb in enumerate(r.fallbacks, start=1):
            rows += [(sp, t, k, next(iter(v))) for k, v in fb.items() if len(v) == 1]
    out["gene_aliases"] = pl.DataFrame(rows, schema={"species": pl.String, "tier": pl.Int8, "key": pl.String,
                                                     "gene_id": pl.String}, orient="row").unique().sort("species", "tier", "key")
    return out


def depmap_z() -> tuple[np.ndarray, dict[str, int]]:
    """Standardised DepMap gene-effect matrix (lines x genes) for co-dependency."""
    from slbench import ids

    df = pl.read_csv(Path("data/raw/depmap/CRISPRGeneEffect_24Q4.csv"))
    h = ids.human()
    genes = {}
    for j, c in enumerate(df.columns[1:]):
        sym = h(c.split(" (")[0]) or h(c.split("(")[-1].rstrip(")"))
        if sym and sym not in genes:
            genes[sym] = j
    x = df.drop(df.columns[0]).to_numpy().astype(np.float32)
    x = np.where(np.isnan(x), np.nanmean(x, axis=0), x)
    return (x - x.mean(0)) / (x.std(0) + 1e-6), genes


def read(bench: Path, name: str) -> pl.DataFrame:
    p = bench / "features" / f"{name}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing: rebuild the benchmark, or use a bundle that ships features/")
    return pl.read_parquet(p)


class TableResolver:
    """The SLB resolver semantics (exact then upper-case key, primary names before unambiguous fallbacks) over
    gene_aliases.parquet."""

    def __init__(self, table: pl.DataFrame, species: str):
        t = table.filter(pl.col("species") == species)
        self.tiers = [dict(zip(g["key"], g["gene_id"])) for _, g in sorted(t.group_by("tier"), key=lambda kv: kv[0][0])]

    def __call__(self, name: str | None) -> str | None:
        if name is None:
            return None
        key = str(name).strip()
        for tier in self.tiers:
            for k in (key, key.upper()):
                if k in tier:
                    return tier[k]
        return None
