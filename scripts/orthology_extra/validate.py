"""Validation of the extra homology edges (numbers quoted in notes/data/orthology.md).

1. DIAMOND RBH vs curated orthology (families.edges(): Alliance for human/scer/dmel, PomBase for spom;
   Alliance for mmus/cele): pair precision, gene coverage, and component-level recall (curated pair ends up in
   one component using only sequence-derived edges: RBH + DIAMOND self paralogs >= 30%).
2. DIAMOND self identity (nident / min(qlen, slen)) vs Ensembl paralog identity (max(%id, %id_r1)).
3. Spot checks of known orthologs, incl. bacteria <-> eukaryotes.
4. eggNOG 5.0 LUCA-level OGs: fraction of cross-kingdom RBH pairs whose two proteins share an OG.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl

from slpbench import families
from slpbench.homology import paralogs

W = Path("data/interim/orthology_extra")
e = pl.read_parquet(W / "edges.parquet")
base = families.edges()
self_all = pl.read_parquet(W / "diamond_self_paralogs_all.parquet")
out: dict = {}


def sp(c):
    return pl.col(c).str.split(":").list.first()


def pairs(df, a, b):
    df = df.filter(((sp("u") == a) & (sp("v") == b)) | ((sp("u") == b) & (sp("v") == a)))
    return {tuple(sorted(x)) for x in df.select("u", "v").iter_rows()}


def components(edge_iter):
    d = families.DSU()
    for u, v in edge_iter:
        d.union(u, v)
    return d


rbh = e.filter(pl.col("source") == "diamond_rbh")
seq_dsu = components(list(rbh.select("u", "v").iter_rows()) +
                     list(self_all.select("u", "v").iter_rows()))
res = []
for a, b, ref in [("human", "scer", base), ("human", "dmel", base), ("dmel", "scer", base), ("human", "spom", base),
                  ("scer", "spom", base), ("cele", "human", e.filter(pl.col("source") == "alliance")),
                  ("human", "mmus", e.filter(pl.col("source") == "alliance")),
                  ("cele", "scer", e.filter(pl.col("source") == "alliance"))]:
    r = pairs(rbh, a, b)
    A = pairs(ref.filter(pl.col("kind") == "ortholog"), a, b)
    ga = {x for p in A for x in p}
    same = sum(1 for u, v in A if u in seq_dsu.p and v in seq_dsu.p and seq_dsu.find(u) == seq_dsu.find(v))
    res.append({"pair": f"{a}-{b}", "rbh_pairs": len(r), "curated_pairs": len(A),
                "rbh_precision": round(len(r & A) / len(r), 3),
                "curated_genes_with_rbh": round(len({x for p in r for x in p} & ga) / len(ga), 3),
                "curated_pairs_same_seq_component": round(same / len(A), 3)})
out["rbh_vs_curated"] = res

p = paralogs()
par = []
for s in ["scer", "spom", "human"]:
    ens = p.filter(pl.col("species") == s).select((pl.lit(s + ":") + pl.col("a")).alias("u"),
                                                  (pl.lit(s + ":") + pl.col("b")).alias("v"), pl.col("identity").alias("ens"))
    dd = self_all.filter(pl.col("u").str.starts_with(s + ":")).rename({"identity": "dia"})
    j = ens.join(dd, on=["u", "v"], how="full", coalesce=True).fill_null(0)
    E, D = j["ens"] >= 0.3, j["dia"] >= 0.3
    both = j.filter((pl.col("ens") > 0) & (pl.col("dia") > 0))
    par.append({"species": s, "ensembl_ge30": int(E.sum()), "diamond_ge30": int(D.sum()), "both": int((E & D).sum()),
                "pearson_on_shared": round(both.select(pl.corr("ens", "dia")).item(), 3),
                "median_diamond_minus_ensembl": round(float((both["dia"] - both["ens"]).median()), 3)})
out["paralog_identity_vs_ensembl"] = par

allE = pl.concat([base.with_columns(pl.lit("base").alias("source")), e])
spot = {}
for n in ["human:POLA1", "ecol:b0060", "spne:rplB", "spne:rpoB", "spne:gyrB", "spne:eno", "spne:rpsD", "spne:rplL",
          "spne:dnaA", "spne:ftsZ", "spne:ffh", "spne:tuf", "spne:polC", "spne:secA"]:
    x = allE.filter(((pl.col("u") == n) | (pl.col("v") == n)) & (pl.col("kind") == "ortholog"))
    spot[n] = sorted({v if u == n else u for u, v in x.select("u", "v").iter_rows()})
out["spot_checks"] = spot

# eggNOG LUCA-level OGs (validation only)
EGG = Path("data/raw/orthology_extra/eggnog5_1_members.tsv.gz")
TAX = {"9606": "human", "4932": "scer", "511145": "ecol", "224308": "bsub", "83332": "mtub"}
if EGG.exists():
    fa = {}
    for s in TAX.values():
        for line in open(W / f"fasta/{s}.fa"):
            if line.startswith(">"):
                fa[line[1:].strip()] = 1
    from slpbench import ids, ids_extra

    res_ = {"human": None, "scer": ids.scer(), "ecol": ids_extra.ecol(), "bsub": ids_extra.bsub(), "mtub": ids_extra.mtub()}
    ens2sym = {}
    for line in gzip.open("data/raw/orthology_extra/proteomes/human.pep.fa.gz", "rt"):
        if line.startswith(">"):
            f = line[1:].split()
            g = [t for t in f if t.startswith("gene:")]
            if g:
                ens2sym[f[0].split(".")[0]] = ids.human()(g[0][5:].split(".")[0])
    og: dict[str, set] = {}
    with gzip.open(EGG, "rt") as f:
        for line in f:
            p_ = line.rstrip("\n").split("\t")
            for m in p_[4].split(","):
                t, _, pid = m.partition(".")
                s = TAX.get(t)
                if not s:
                    continue
                g = ens2sym.get(pid) if s == "human" else res_[s](pid)
                if g:
                    og.setdefault(f"{s}:{g}", set()).add(p_[1])
    r = rbh.filter(sp("u").is_in(list(TAX.values())) & sp("v").is_in(list(TAX.values())))
    stats = {}
    for u, v in r.select("u", "v").iter_rows():
        su, sv = u.split(":")[0], v.split(":")[0]
        if (su in ("ecol", "bsub", "mtub")) == (sv in ("ecol", "bsub", "mtub")):
            continue
        k = "-".join(sorted([su, sv]))
        st = stats.setdefault(k, [0, 0, 0])
        if u in og and v in og:
            st[0] += 1
            st[1] += bool(og[u] & og[v])
        else:
            st[2] += 1
    out["eggnog_luca_og_agreement"] = {k: {"both_in_eggnog": a, "same_og": b, "frac": round(b / max(a, 1), 3),
                                           "not_in_eggnog": c} for k, (a, b, c) in sorted(stats.items())}
(W / "validation.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
