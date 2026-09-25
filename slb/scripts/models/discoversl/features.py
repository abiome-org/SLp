"""DiscoverSL (Das et al. 2019, Bioinformatics 35:701) features for SLB human pairs, re-implemented from the
package source (DiscoverSL_2.0 R/DiscoverSL.r) on local TCGA PanCanAtlas data because the package fetched
TCGA from the retired cBioPortal CGDS API (cgdsr).

For an ordered pair (gene1 = primary, gene2 = partner), pan-cancer TCGA primary tumours:
  PValue             DiffExp: two-sided Welch t-test of gene2 log2 expression, gene1-mutated vs not
                     (the package used edgeR exactTest on RSEM counts; EB++ values are normalised, so a t-test)
  correlation.pvalue Pearson correlation p-value of gene1 vs gene2 expression
  Mutex              Fisher sumlog of (1 - P[co-occurrence >= observed]) for mutation, amplification (GISTIC>=2)
                     and deep deletion (GISTIC<=-2), exactly as predictSL() does
  PvalPathway        hypergeometric over the package's KEGG + REACTOME gene sets, then 1 - p (as predictSL())
Both orientations are computed; downstream models take the max score over orientations.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import omics  # noqa: E402
import slb  # noqa: E402

GENESETS = slb.ROOT / "external/models/discoversl/export/genesets.tsv"


def sumlog(ps):
    ps = np.clip(np.asarray(ps, float), 1e-300, 1.0)
    x = -2 * np.log(ps).sum(0)
    return stats.chi2.sf(x, 2 * ps.shape[0])


def compute(pairs: pd.DataFrame) -> pd.DataFrame:
    """pairs: columns g1, g2 (ordered). Returns the 4 DiscoverSL features."""
    ex = omics.tcga_expr()
    mut = omics.tcga_mut()
    cna = omics.tcga_cna()
    samples = ex.index.intersection(mut.index).intersection(cna.index)
    ex, mut, cna = ex.loc[samples], mut.loc[samples], cna.loc[samples]
    n = len(samples)
    M = (mut.fillna(0).to_numpy(np.float32) > 0).astype(np.float32)
    AMP = (cna.to_numpy(np.float32) >= 2).astype(np.float32)
    DEL = (cna.to_numpy(np.float32) <= -2).astype(np.float32)
    mi = {g: i for i, g in enumerate(mut.columns)}
    ci = {g: i for i, g in enumerate(cna.columns)}
    ei = {g: i for i, g in enumerate(ex.columns)}
    E = ex.to_numpy(np.float32)
    out = pd.DataFrame(index=pairs.index, columns=["PValue", "correlation.pvalue", "Mutex", "PvalPathway"],
                       dtype=float)
    # DiffExp: group by primary gene
    for g1, idx in pairs.groupby("g1").groups.items():
        if g1 not in mi:
            continue
        m = M[:, mi[g1]] > 0
        if m.sum() < 4 or (~m).sum() < 4:
            continue
        cols = np.array([ei.get(g, -1) for g in pairs.loc[idx, "g2"]])
        ok = cols >= 0
        if ok.any():
            x = E[:, cols[ok]]
            t = stats.ttest_ind(x[m], x[~m], axis=0, equal_var=False, nan_policy="omit")
            v = np.full(len(idx), np.nan)
            v[ok] = np.asarray(t.pvalue, float)
            out.loc[idx, "PValue"] = v
    # correlation p
    a = np.array([ei.get(g, -1) for g in pairs.g1])
    b = np.array([ei.get(g, -1) for g in pairs.g2])
    ok = (a >= 0) & (b >= 0)
    Z = np.nan_to_num((E - np.nanmean(E, 0)) / (np.nanstd(E, 0) + 1e-9))
    r = np.einsum("ij,ij->j", Z[:, a[ok]], Z[:, b[ok]]) / n
    tstat = r * np.sqrt((n - 2) / np.clip(1 - r ** 2, 1e-12, None))
    cp = np.full(len(pairs), np.nan)
    cp[ok] = 2 * stats.t.sf(np.abs(tstat), n - 2)
    out["correlation.pvalue"] = cp

    # Mutex: phyper(both-1, k1, n-k1, k2, lower.tail=FALSE) = P[X >= both]; predictSL uses 1 - that
    def cooc(X, gi):
        a = np.array([gi.get(g, -1) for g in pairs.g1])
        b = np.array([gi.get(g, -1) for g in pairs.g2])
        ok = (a >= 0) & (b >= 0)
        p = np.full(len(pairs), np.nan)
        both = np.einsum("ij,ij->j", X[:, a[ok]], X[:, b[ok]])
        k1, k2 = X[:, a[ok]].sum(0), X[:, b[ok]].sum(0)
        p[ok] = stats.hypergeom.sf(both - 1, n, k1, k2)
        return np.round(p, 2)
    pm, pa, pd_ = cooc(M, mi), cooc(AMP, ci), cooc(DEL, ci)
    allp = np.vstack([1 - pm, 1 - pa, 1 - pd_])
    mx = np.full(len(pairs), np.nan)
    okm = ~np.isnan(allp).any(0)
    mx[okm] = sumlog(allp[:, okm])
    out["Mutex"] = mx
    # pathway
    gs = pd.read_csv(GENESETS, sep="\t", header=None, names=["name", "url", "genes"])
    gs = gs[gs.name.str.startswith("REACTOME_") | gs.name.str.startswith("KEGG_")]
    res = slb.symbol_resolver()
    sets = [set(res(x) or x for x in str(s).split(",")) for s in gs.genes]
    genes = sorted(set(pairs.g1) | set(pairs.g2))
    gi = {g: i for i, g in enumerate(genes)}
    P = np.zeros((len(sets), len(genes)), np.float32)
    for j, s in enumerate(sets):
        for g in s:
            if g in gi:
                P[j, gi[g]] = 1
    a = np.array([gi[g] for g in pairs.g1])
    b = np.array([gi[g] for g in pairs.g2])
    both = np.einsum("ij,ij->j", P[:, a], P[:, b])
    k1, k2 = P[:, a].sum(0), P[:, b].sum(0)
    pp = stats.hypergeom.sf(both - 1, len(sets), k1, k2)
    out["PvalPathway"] = 1 - pp
    return out


def pair_features(split: str) -> pd.DataFrame:
    """Both orientations for every human pair of the split; cached per benchmark version."""
    cache = omics.CACHE / f"discoversl_{slb.BENCH.name}_{split}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    d = slb.load(split)
    d = d[d.species == "human"]
    k = slb.pair_key(d.gene_a, d.gene_b).drop_duplicates()
    ab = k.str.split("|", expand=True)
    pairs = pd.concat([pd.DataFrame({"key": k.values, "g1": ab[0].values, "g2": ab[1].values}),
                       pd.DataFrame({"key": k.values, "g1": ab[1].values, "g2": ab[0].values})], ignore_index=True)
    f = compute(pairs)
    out = pd.concat([pairs, f], axis=1)
    out.to_parquet(cache)
    return out


if __name__ == "__main__":
    for s in sys.argv[1:] or [slb.SPLIT]:
        f = pair_features(s)
        print(s, f.shape, f.iloc[:, 3:].notna().mean().round(3).to_dict(), file=sys.stderr)
