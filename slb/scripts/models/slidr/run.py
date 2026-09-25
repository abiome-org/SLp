"""SLIdR (Srivatsa et al. 2022, Nat Commun 13:2699; cbg-ethz/slidr) re-implemented in Python for SLB pairs.

Test (identical to slidr::identifySLHits / getPval): in cell lines where the 'driver' gene is altered, rank every
gene's viability within each line, normalise by the number of genes and sum over altered lines; the Irwin-Hall
CDF of that rank sum is mut_pvalue (small = the partner is consistently among the most essential genes).
The same statistic in the driver-WT lines gives a two-sided WT_pvalue; SLIdR keeps hits whose partner is NOT
significant in WT lines (WT_pvalue > 0.1). Pan-cancer (SLIdR also runs per primary site).
Score per ordered pair = -log10(mut_pvalue), minus 1000 when WT_pvalue <= 0.1 (partner generically essential);
pair score = max over the two orientations. No SL labels. Human only (needs cell-line screens + genotypes).

Variants (alteration definition x viability screen):
  slidr__rnai        damaging mutation or deep deletion (DepMap 24Q4), DEMETER2 RNAi viabilities (as in the paper,
                     which used DRIVE shRNA + CCLE mutations/CN)
  slidr__crispr      same alterations, DepMap CRISPR (Chronos) viabilities
  slidr__crispr_lossexpr  extension: alteration = damaging mutation, deep deletion OR no expression (TPM log1p < 1),
                     which makes the test applicable to paralog pairs (De Kegel-style)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import omics  # noqa: E402
import slb  # noqa: E402


def ih_cdf(x, n):
    x = np.asarray(x, float)
    if n <= 20:
        out = np.empty_like(x)
        for i, xi in enumerate(x):
            k = np.arange(0, int(np.floor(xi)) + 1)
            s = ((-1.0) ** k) * np.array([float(__import__("math").comb(n, int(kk))) for kk in k]) * (xi - k) ** n
            out[i] = s.sum() / float(__import__("math").factorial(n))
        return np.clip(out, 0, 1)
    return stats.norm.cdf(x, n / 2, np.sqrt(n / 12))


def slidr_scores(pairs, viab, alt):
    lines = viab.index.intersection(alt.index)
    V = viab.loc[lines]
    # rank of each gene's viability within each line, normalised by #genes (NaN -> middle)
    R = V.rank(axis=1, na_option="keep").div(V.notna().sum(axis=1), axis=0).fillna(0.5).to_numpy(np.float32)
    gi = {g: i for i, g in enumerate(V.columns)}
    A = alt.loc[lines]
    ai = {g: i for i, g in enumerate(A.columns)}
    Am = A.to_numpy(bool)
    best = np.full(len(pairs), np.nan)
    for x, y in [("a", "b"), ("b", "a")]:
        sc = np.full(len(pairs), np.nan)
        for g, idx in pairs.groupby(x).groups.items():
            if g not in ai:
                continue
            m = Am[:, ai[g]]
            n, w = int(m.sum()), int((~m).sum())
            if n < 2 or w < 2:
                continue
            cols = np.array([gi.get(p, -1) for p in pairs.loc[idx, y]])
            ok = cols >= 0
            if not ok.any():
                continue
            rs_mut = R[m][:, cols[ok]].sum(0)
            rs_wt = R[~m][:, cols[ok]].sum(0)
            p_mut = np.clip(ih_cdf(rs_mut, n), 1e-300, 1)
            cw = ih_cdf(rs_wt, w)
            p_wt = 2 * np.minimum(cw, 1 - cw)
            s = -np.log10(p_mut) - 1000 * (p_wt <= 0.1)
            v = np.full(len(idx), np.nan)
            v[ok] = s
            sc[pairs.index.get_indexer(idx)] = v
        best = np.fmax(best, sc)
    return best


def main(split):
    d = slb.load(split)
    h = d[d.species == "human"].copy()
    h["key"] = slb.pair_key(h.gene_a, h.gene_b).values
    keys = pd.Series(h.key.unique())
    ab = keys.str.split("|", expand=True)
    pairs = pd.DataFrame({"a": ab[0].values, "b": ab[1].values})
    mut = omics.ccle_mut() > 0
    cn = omics.ccle_cn()
    expr = omics.ccle_expr()
    genes = mut.columns.union(cn.columns)
    lines = mut.index.intersection(cn.index)
    deep = (cn.reindex(index=lines, columns=genes) < 0.5).fillna(False)  # OmicsCNGene log2(rel CN + 1) < 0.5
    alt = mut.reindex(index=lines, columns=genes).fillna(False) | deep
    lines_e = lines.intersection(expr.index)
    loss = alt.loc[lines_e] | (expr.reindex(index=lines_e, columns=genes) < 1).fillna(False)
    for name, viab, al in [("slidr__rnai", omics.rnai(), alt), ("slidr__crispr", omics.crispr(), alt),
                           ("slidr__crispr_lossexpr", omics.crispr(), loss)]:
        s = pd.Series(slidr_scores(pairs, viab, al), index=keys.values)
        out = d[["example_id"]].copy()
        out["score"] = np.nan
        out.loc[h.index, "score"] = s.reindex(h.key).to_numpy()
        print(name, "scored human rows:", int(out.score.notna().sum()), file=sys.stderr)
        slb.write(out, name, split)
        slb.evaluate(name, split)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else slb.SPLIT)
