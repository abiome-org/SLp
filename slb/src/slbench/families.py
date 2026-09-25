"""Gene families used as the unit of train/dev/test assignment.

A family is a connected component of a graph whose nodes are "species:gene" and whose edges are
  * close paralogs within a species (Ensembl 116; human, S. cerevisiae, S. pombe, D. melanogaster),
    kept when max protein sequence identity >= PARALOG_MIN_IDENTITY;
  * orthologs across human, S. cerevisiae and D. melanogaster (Alliance combined) that are reciprocal
    best hits or supported by >= ORTHOLOG_MIN_ALGORITHMS prediction methods;
  * curated S. pombe orthologs to human and S. cerevisiae (PomBase);
  * families_extra: DIAMOND reciprocal-best-hit orthologs across 12 proteomes (5 bacteria, yeasts,
    C. albicans, worm, fly, mouse, human), Alliance orthologs for mouse/worm, and paralogs >= 30%
    identity for mouse, worm, C. albicans and the bacteria.
Holding out a whole family holds out a gene, its close paralogs and its orthologs in every species.

A family is named after its smallest node from one of the original species (NAMING_SPECIES), so
adding genes from new species does not rename, and thereby re-bucket, an existing family.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl

RAW = Path("data/raw")
PARALOG_MIN_IDENTITY = 0.30
ORTHOLOG_MIN_ALGORITHMS = 3
HCOP_MIN_SUPPORT = 2  # HCOP human-yeast orthologs asserted by at least this many databases
MAX_FAMILY = 400  # fail if a component exceeds this; never break homology edges

NAMING_SPECIES = ("human", "scer", "spom", "dmel", "spne")

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
    from slbench import ids

    from slbench.homology import paralogs

    out = []
    par = paralogs().filter(pl.col("identity") >= PARALOG_MIN_IDENTITY)
    out += [(f"{sp}:{a}", f"{sp}:{b}", "paralog", w) for sp, a, b, w in par.iter_rows()]

    # Alliance orthology: reciprocal best, or supported by >= ORTHOLOG_MIN_ALGORITHMS prediction methods
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
            if not s1 or not s2 or s1 == s2:
                continue
            if not ((p[11] == "Yes" and p[12] == "Yes") or int(p[9]) >= ORTHOLOG_MIN_ALGORITHMS):
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
    for sp, fn, col in (("scer", "hcop_human_scerevisiae.txt.gz", "s.cerevisiae_ensembl_gene"),
                        ("spom", "hcop_human_spombe.txt.gz", "s.pombe_ensembl_gene")):
        h = pl.read_csv(RAW / "orthology" / fn, separator="\t", infer_schema_length=0, quote_char=None)
        h = h.filter(pl.col("support").str.split(",").list.len() >= HCOP_MIN_SUPPORT)
        res = sgd if sp == "scer" else spom
        for hid, g in zip(h["hgnc_id"], h[col]):
            a, b = hgnc.get(hid), res(g)
            if a and b:
                out.append((f"human:{a}", f"{sp}:{b}", "ortholog", 1.0))
    from slbench.families_extra import extra_edges

    base = pl.DataFrame(out, schema=["u", "v", "kind", "weight"], orient="row")
    return pl.concat([base, extra_edges()]).unique()


def overlap_edges(genes: pl.DataFrame) -> pl.DataFrame:
    """Physically overlapping yeast ORFs: deleting one usually disrupts the other, so they share a family."""
    from slbench import genome

    out = []
    for sp in ("scer", "spom"):
        g = set(genes.filter(pl.col("species") == sp)["gene"])
        out += [(f"{sp}:{a}", f"{sp}:{b}", "overlap", 1.0) for a, b in genome.overlapping_pairs(sp, g)]
    return pl.DataFrame(out, schema=["u", "v", "kind", "weight"], orient="row")


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
    e = pl.concat([edges(), overlap_edges(genes)])
    # Only edges touching at least one benchmark gene matter; keep bridges through other genes
    # (e.g. a human gene linking two yeast paralogs) since they carry homology information.
    for u, v, kind, w in e.iter_rows():
        dsu.union(u, v)
    label: dict[str, str] = {}
    for n in list(dsu.p):
        r = dsu.find(n)
        if n.split(":", 1)[0] in NAMING_SPECIES and (r not in label or n < label[r]):
            label[r] = n
    fam = {n: label.get(dsu.find(n), dsu.find(n)) for n in nodes}
    _check_family_size(fam)
    return pl.DataFrame({"species": genes["species"], "gene": genes["gene"], "family": [fam[n] for n in nodes]})


def _check_family_size(fam: dict[str, str]) -> None:
    """Keep every qualifying homology edge inside a family, including large components."""
    from collections import Counter

    sizes = Counter(fam.values())
    big = [(f, n) for f, n in sizes.items() if n > MAX_FAMILY]
    if big:
        raise ValueError(f"homology component exceeds MAX_FAMILY={MAX_FAMILY}: {big[:5]}")
