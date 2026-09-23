"""Gene families used as the unit of train/dev/test assignment.

A family is a connected component of a graph whose nodes are "species:gene" and whose edges are
  * close paralogs within a species (Ensembl 116; human, S. cerevisiae, S. pombe, D. melanogaster),
    kept when max protein sequence identity >= PARALOG_MIN_IDENTITY;
  * reciprocal-best orthologs across human, S. cerevisiae and D. melanogaster (Alliance combined);
  * curated S. pombe orthologs to human and S. cerevisiae (PomBase).
Holding out a whole family holds out a gene, its close paralogs and its orthologs in every species.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl

RAW = Path("data/raw")
PARALOG_MIN_IDENTITY = 0.30
MAX_FAMILY = 400  # components larger than this are broken up (see _cap)

ALLIANCE_TAXA = {"NCBITaxon:9606": "human", "NCBITaxon:559292": "scer", "NCBITaxon:7227": "dmel"}


class DSU:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        p = self.p
        p.setdefault(x, x)
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def edges() -> pl.DataFrame:
    from slpbench import ids

    from slpbench.homology import paralogs

    out = []
    par = paralogs().filter(pl.col("identity") >= PARALOG_MIN_IDENTITY)
    out += [(f"{sp}:{a}", f"{sp}:{b}", "paralog", w) for sp, a, b, w in par.iter_rows()]

    # Alliance orthology: reciprocal best only
    hgnc = pl.read_csv(RAW / "ids/hgnc_complete_set.txt", separator="\t", infer_schema_length=0, quote_char=None,
                       columns=["hgnc_id", "symbol"])
    hgnc = dict(zip(hgnc["hgnc_id"], hgnc["symbol"]))
    sgd = ids.scer()
    with gzip.open(RAW / "orthology/ORTHOLOGY-ALLIANCE_COMBINED.tsv.gz", "rt") as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if header is None:
                header = p
                continue
            s1, s2 = ALLIANCE_TAXA.get(p[2]), ALLIANCE_TAXA.get(p[6])
            if not s1 or not s2 or s1 == s2 or p[11] != "Yes" or p[12] != "Yes":
                continue
            g1, g2 = _alliance_id(p[0], s1, hgnc, sgd), _alliance_id(p[4], s2, hgnc, sgd)
            if g1 and g2:
                out.append((f"{s1}:{g1}", f"{s2}:{g2}", "ortholog", 1.0))

    spom = ids.spom()
    for line in open(RAW / "orthology/pombe-human-orthologs.tsv"):
        a, b = line.rstrip("\n").split("\t")[:2]
        a, b = spom(a), hgnc.get(b)
        if a and b:
            out.append((f"spom:{a}", f"human:{b}", "ortholog", 1.0))
    for line in open(RAW / "orthology/pombe-cerevisiae-orthologs.tsv"):
        a, b = line.rstrip("\n").split("\t")[:2]
        a, b = spom(a), sgd(b)
        if a and b:
            out.append((f"spom:{a}", f"scer:{b}", "ortholog", 1.0))
    return pl.DataFrame(out, schema=["u", "v", "kind", "weight"], orient="row").unique()


def _alliance_id(gid: str, species: str, hgnc: dict, sgd) -> str | None:
    if species == "human":
        return hgnc.get(gid)
    if species == "scer":
        return sgd(gid.removeprefix("SGD:"))
    if species == "dmel":
        return gid.removeprefix("FB:")
    return None


def assign(genes: pl.DataFrame) -> pl.DataFrame:
    """genes: columns species, gene. Returns species, gene, family (stable string ID)."""
    dsu = DSU()
    nodes = [f"{s}:{g}" for s, g in genes.iter_rows()]
    for n in nodes:
        dsu.find(n)
    e = edges()
    node_set = set(nodes)
    # Only edges touching at least one benchmark gene matter; keep bridges through other genes
    # (e.g. a human gene linking two yeast paralogs) since they carry homology information.
    for u, v, kind, w in e.iter_rows():
        dsu.union(u, v)
    fam = {n: dsu.find(n) for n in nodes}
    fam = _cap(fam, e, node_set)
    return pl.DataFrame({"species": genes["species"], "gene": genes["gene"], "family": [fam[n] for n in nodes]})


def _cap(fam: dict[str, str], e: pl.DataFrame, node_set: set[str]) -> dict[str, str]:
    """Split oversized components by re-running union-find with orthologs + tighter paralogs."""
    from collections import Counter

    sizes = Counter(fam.values())
    big = {f for f, n in sizes.items() if n > MAX_FAMILY}
    if not big:
        return fam
    members = [n for n, f in fam.items() if f in big]
    for thresh in (0.4, 0.5, 0.6, 0.8, 1.01):
        dsu = DSU()
        for n in members:
            dsu.find(n)
        sub = e.filter((pl.col("kind") == "ortholog") | (pl.col("weight") >= thresh))
        mset = set(members)
        for u, v, kind, w in sub.iter_rows():
            if u in mset or v in mset:
                dsu.union(u, v)
        new = {n: "cap:" + dsu.find(n) for n in members}
        if max(Counter(new.values()).values()) <= MAX_FAMILY:
            break
    fam = dict(fam)
    fam.update(new)
    return fam
