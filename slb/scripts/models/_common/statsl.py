"""Vectorised pair statistics behind the classic data-driven SL inference methods.

For each unordered human gene pair (A, B) this computes, from single-gene data only:
  ess_<screen>_<alt>   functional examination (DAISY step 2 / ISLE in-vitro step / SLIdR-like): one-sided
                       Mann-Whitney z that B is more essential in cell lines where A is inactive (alt = low
                       expression tertile, low copy-number tertile, damaging mutation) than in the others;
                       max over the two directions. screen = crispr (DepMap Chronos 24Q4) | rnai (DEMETER2).
  sof_<alt>            survival-of-the-fittest / under-represented co-inactivation in TCGA primary tumours
                       (DAISY step 1, ISLE step I): z of the hypergeometric lower tail of #samples with both
                       genes inactive (alt = expr: low tertile within cancer type; cna: GISTIC <= -1).
  coexp_tcga, coexp_ccle  Spearman co-expression (DAISY step 3).
  surv_<alt>           ISLE step II: stratified (by cancer type) log-rank z for co-inactivation; positive =
                       patients with both genes inactive survive longer (the SL signature).
  phylo_dist           ISLE step III: feature-weighted squared distance of Tabach et al. 2013 phylogenetic
                       profiles (weights shipped with ISLE); smaller = more co-evolved.
All z are oriented so that larger = more SL-like. NaN where data are missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omics  # noqa: E402
import slb  # noqa: E402


def _rank_cols(x: np.ndarray) -> np.ndarray:
    """Column-wise average ranks, NaNs filled with the column's mid rank."""
    r = np.apply_along_axis(lambda c: stats.rankdata(np.where(np.isnan(c), np.nanmedian(c), c)), 0, x)
    return r.astype(np.float32)


def ess_tests(pairs: pd.DataFrame, screen: pd.DataFrame, alts: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """Functional examination. alts[name] = boolean (lines x genes) 'inactive' matrix."""
    out = {}
    lines = screen.index
    R = _rank_cols(screen.to_numpy(dtype=np.float32))  # low effect (more essential) = low rank
    gi = {g: i for i, g in enumerate(screen.columns)}
    n = len(lines)
    for name, alt in alts.items():
        al = alt.reindex(index=lines)
        valid = al.notna().any(axis=1).to_numpy()
        A = al.fillna(False).astype(bool)
        agi = {g: i for i, g in enumerate(A.columns)}
        Am = A.to_numpy()
        z = np.full(len(pairs), np.nan)
        for d, (x, y) in enumerate([("a", "b"), ("b", "a")]):
            zz = np.full(len(pairs), np.nan)
            for g, idx in pairs.groupby(x).groups.items():
                if g not in agi:
                    continue
                m = Am[:, agi[g]] & valid
                n1 = int(m.sum())
                n2 = int(valid.sum()) - n1
                if n1 < 3 or n2 < 3:
                    continue
                partners = pairs.loc[idx, y].to_numpy()
                cols = np.array([gi.get(p, -1) for p in partners])
                ok = cols >= 0
                if not ok.any():
                    continue
                Rv = R[valid][:, cols[ok]]
                mv = m[valid]
                rr = np.apply_along_axis(stats.rankdata, 0, Rv) if Rv.shape[0] != n else Rv  # re-rank on valid lines
                U = rr[mv].sum(0) - n1 * (n1 + 1) / 2
                mu = n1 * n2 / 2
                sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
                zi = (mu - U) / sd  # positive: partner more essential (lower effect) in altered lines
                tmp = np.full(len(idx), np.nan)
                tmp[ok] = zi
                zz[pairs.index.get_indexer(idx)] = tmp
            z = np.fmax(z, zz)
        out[name] = z
    return out


def _hyper_z(x, N, Ka, Kb):
    p = stats.hypergeom.cdf(x, N, Ka, Kb)
    return -stats.norm.ppf(np.clip(p, 1e-300, 1 - 1e-16))  # positive = fewer co-inactive than expected


def sof(pairs: pd.DataFrame, L: pd.DataFrame) -> np.ndarray:
    """Hypergeometric under-representation of co-inactivation; L boolean samples x genes (NaN = no data)."""
    gi = {g: i for i, g in enumerate(L.columns)}
    X = L.fillna(False).to_numpy(dtype=np.float32)
    ia = np.array([gi.get(g, -1) for g in pairs.a])
    ib = np.array([gi.get(g, -1) for g in pairs.b])
    ok = (ia >= 0) & (ib >= 0)
    z = np.full(len(pairs), np.nan)
    N = X.shape[0]
    K = X.sum(0)
    both = np.einsum("ij,ij->j", X[:, ia[ok]], X[:, ib[ok]])
    z[ok] = _hyper_z(both, N, K[ia[ok]], K[ib[ok]])
    return z


def spearman(pairs: pd.DataFrame, E: pd.DataFrame) -> np.ndarray:
    gi = {g: i for i, g in enumerate(E.columns)}
    R = _rank_cols(E.to_numpy(dtype=np.float32))
    R = (R - R.mean(0)) / (R.std(0) + 1e-9)
    ia = np.array([gi.get(g, -1) for g in pairs.a])
    ib = np.array([gi.get(g, -1) for g in pairs.b])
    ok = (ia >= 0) & (ib >= 0)
    r = np.full(len(pairs), np.nan)
    r[ok] = np.einsum("ij,ij->j", R[:, ia[ok]], R[:, ib[ok]]) / R.shape[0]
    return r


def logrank(pairs: pd.DataFrame, L: pd.DataFrame, clin: pd.DataFrame, chunk: int = 4000) -> np.ndarray:
    """Stratified log-rank (score test of a Cox model cov ~ strata(type)) for cov = both genes inactive."""
    c = clin.reindex(L.index)
    keep = c["OS.time"].notna() & c["OS"].notna() & c["type"].notna()
    L = L[keep.values]
    c = c[keep]
    gi = {g: i for i, g in enumerate(L.columns)}
    X = L.fillna(False).to_numpy(dtype=np.float32)
    ia = np.array([gi.get(g, -1) for g in pairs.a])
    ib = np.array([gi.get(g, -1) for g in pairs.b])
    ok = np.where((ia >= 0) & (ib >= 0))[0]
    z = np.full(len(pairs), np.nan)
    strata = []
    for t, idx in c.groupby("type").indices.items():
        tt = c["OS.time"].to_numpy()[idx]
        ev = c["OS"].to_numpy()[idx].astype(np.float32)
        o = np.argsort(-tt, kind="stable")  # descending time -> cumulative sum = risk set
        strata.append((idx[o], ev[o], tt[o]))
    for s in range(0, len(ok), chunk):
        sel = ok[s:s + chunk]
        C = X[:, ia[sel]] * X[:, ib[sel]]
        O_E = np.zeros(len(sel))
        V = np.zeros(len(sel))
        for idx, ev, tt in strata:
            Cs = C[idx]
            n_risk = np.arange(1, len(idx) + 1, dtype=np.float32)
            n1 = np.cumsum(Cs, 0)
            # ties: use risk set at the last occurrence of each time in descending order
            last = np.r_[tt[1:] != tt[:-1], True]
            # deaths per distinct time handled approximately per-sample (Breslow-like, fine for ranking)
            e = ev[:, None]
            p = n1 / n_risk[:, None]
            O_E += (e * (Cs - p)).sum(0)
            V += (e * p * (1 - p)).sum(0)
        z[sel] = -O_E / np.sqrt(np.maximum(V, 1e-9))  # positive = fewer deaths than expected among co-inactive
    return z


def phylo(pairs: pd.DataFrame) -> np.ndarray:
    import pyreadr
    ph = pyreadr.read_r(str(slb.ROOT / "external/models/isle/data/yuval.phylogenetic.profile.RData"))["phylo"]
    w = pyreadr.read_r(str(slb.ROOT / "external/models/isle/data/feature.weight.RData"))["feature.weight"].to_numpy()[0]
    res = slb.symbol_resolver()
    ph["sym"] = [res(g) or g for g in ph.genes]
    ph = ph.drop_duplicates("sym").set_index("sym")
    P = ph.iloc[:, 3:].to_numpy(dtype=float) if ph.shape[1] - 3 == len(w) else ph.select_dtypes("number").to_numpy()
    P = P[:, -len(w):]
    gi = {g: i for i, g in enumerate(ph.index)}
    ia = np.array([gi.get(g, -1) for g in pairs.a])
    ib = np.array([gi.get(g, -1) for g in pairs.b])
    ok = (ia >= 0) & (ib >= 0)
    d = np.full(len(pairs), np.nan)
    d[ok] = ((P[ia[ok]] - P[ib[ok]]) ** 2) @ w
    return d


def compute(pairs: pd.DataFrame) -> pd.DataFrame:
    """pairs: DataFrame with columns a, b (HGNC symbols). Returns pairs + all statistics."""
    pairs = pairs.reset_index(drop=True)
    out = pairs.copy()
    model = pd.read_csv(slb.RAW / "depmap/Model_24Q4.csv", usecols=["ModelID", "OncotreeLineage"]).set_index("ModelID")
    expr = omics.ccle_expr()
    cn = omics.ccle_cn()
    mut = omics.ccle_mut()
    lin = model.OncotreeLineage.reindex(expr.index).fillna("other")
    expr_low = omics.tertiles_by_group(expr, pd.Series(["all"] * len(expr), index=expr.index)) == 0
    expr_low = expr_low.where(expr.notna())
    cn_low = omics.tertiles_by_group(cn, pd.Series(["all"] * len(cn), index=cn.index)) == 0
    alts = {"expr": expr_low, "cn": cn_low, "mut": mut.astype(bool)}
    print("functional examination ...", file=sys.stderr)
    for sname, scr in [("crispr", omics.crispr()), ("rnai", omics.rnai())]:
        for k, v in ess_tests(pairs, scr, alts).items():
            out[f"ess_{sname}_{k}"] = v
    print("TCGA ...", file=sys.stderr)
    te = omics.tcga_expr()
    clin = omics.tcga_clin()
    types = clin["type"].reindex(te.index).fillna("NA")
    te_low = (omics.tertiles_by_group(te, types) == 0).where(te.notna())
    tc = omics.tcga_cna()
    tc_low = (tc <= -1)
    out["sof_expr"] = sof(pairs, te_low)
    out["sof_cna"] = sof(pairs, tc_low)
    out["coexp_tcga"] = spearman(pairs, te)
    out["coexp_ccle"] = spearman(pairs, expr)
    print("survival ...", file=sys.stderr)
    out["surv_expr"] = logrank(pairs, te_low, clin)
    out["surv_cna"] = logrank(pairs, tc_low, clin)
    out["phylo_dist"] = phylo(pairs)
    return out


def pair_stats(splits=("train", "dev", "test")) -> pd.DataFrame:
    """Statistics for every unordered human pair in the given splits; cached per benchmark version."""
    cache = omics.CACHE / "pairstats.parquet"  # pair statistics do not depend on the benchmark version
    have = pd.read_parquet(cache) if cache.exists() else None
    ps = []
    for s in splits:
        d = slb.load(s)
        d = d[d.species == "human"]
        k = slb.pair_key(d.gene_a, d.gene_b)
        ps.append(pd.DataFrame({"key": k.values}))
    keys = pd.concat(ps).key.drop_duplicates()
    if have is not None:
        keys = keys[~keys.isin(have.key)]
    if len(keys):
        ab = keys.str.split("|", expand=True)
        new = compute(pd.DataFrame({"a": ab[0].values, "b": ab[1].values}))
        new.insert(0, "key", keys.values)
        have = new if have is None else pd.concat([have, new], ignore_index=True)
        have.to_parquet(cache)
    return have


if __name__ == "__main__":
    import time
    t = time.time()
    s = pair_stats(tuple(sys.argv[1:]) or ("dev",))
    print(s.describe().T.to_string(), file=sys.stderr)
    print(f"{time.time() - t:.0f}s", file=sys.stderr)
