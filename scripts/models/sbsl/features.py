"""SBSL (Seale et al., Bioinformatics 2022) molecular pair features for the human rows of an SLB split.

Re-implements r/features/*.R and r/experiments/generate_features.R of github.com/joanagoncalveslab/SBSL (commit
6f4a934) per (gene1, gene2, cancer type), gene1 = SLB gene_a. Single-gene inputs are shared with the ELISL adapter
(scripts/models/elisl/omics_data.py: DepMap 24Q4, DEMETER2, TCGA PanCanAtlas, GTEx v8) + MSigDB v7.0 KEGG/Reactome/PID.

  copathway_participation   hypergeometric P(X>=k) of shared MSigDB c2.cp KEGG+Reactome+PID v7.0 gene sets
  dsl_mutex_amp/_del/_mut   DiscoverSL-style mutual exclusivity P(X<k) (hypergeometric) of GISTIC amp (=2),
                            GISTIC del (=-2), MC3 non-silent mutation in tumours of the cancer type
  dsl_mutex                 Fisher combination (metap::sumlog) of the three; 1 on error/warning (e.g. a p of 0)
  mutex_alt                 same test on 'altered' = mut | amp | del
  discover_mutex            DISCOVER (Canisius et al. 2016) pairwise mutual-exclusivity test (R package discover,
                            run in docker slb/sbsl by discover.R), computed afterwards and merged in
  exp_corr(_pvalue), exp_corr_normal(_pvalue)  Pearson of TCGA tumour / normal expression (linear scale); 0 / 1 if undefined
  gtex_corr(.pvalue)        Pearson of GTEx v8 TPM in the matched healthy tissue; 0 / 1 if undefined
  diff_exp_logfc/_pvalue    expression of gene2 in tumours with vs without gene1 mutated (only if >3 mutated),
                            else 0 / 1. SBSL ran edgeR exactTest on raw counts (GSE62944); SLB substitute: log2 EB++
                            difference + Welch t-test (no raw counts in PanCanAtlas)
  avana_codep/_pvalue, avana_avg, avana_dep/_pvalue   DepMap CRISPR gene effect (SBSL: Avana 19Q3, here 24Q4 Chronos)
                            in lines of the cancer type: Pearson co-dependency, mean dependency, Wilcoxon of gene2
                            effect in gene1-mutated (>=3) vs other lines (W, p); 1 / 0 on error or warning
  d2_*                      same with DEMETER2 combined RNAi
  mut_logrank.pval          log-rank p of patients with both genes mutated vs rest
  mrna_logrank.pval         log-rank p of patients with both genes under-expressed (bottom 5% within type) vs rest
MUTEX (Babur et al.; Java, per-cancer ranked groups) is not reproduced -> constant, removed by caret's nzv.

usage: python features.py <split>          (run with external/models/elisl/.venv/bin/python)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "elisl"))
import common as C  # noqa: E402
import omics_data as O  # noqa: E402
from features import pair_corr, KEY  # noqa: E402  (ELISL adapter helpers)

warnings.filterwarnings("ignore")
WORK = C.ROOT / "external/models/sbsl/_slb"
FEAT = WORK / "feat" / C.BENCH.name
FEAT.mkdir(parents=True, exist_ok=True)
(WORK / "cache").mkdir(parents=True, exist_ok=True)
NJOBS = int(os.environ.get("SBSL_CPUS", "8"))


def rnai():
    return C.cached("d2_effect", lambda: O._depmap_csv("../demeter2/D2_gene_effect.csv"))


def crispr_effect():
    return C.cached("crispr_effect", lambda: O._depmap_csv("CRISPRGeneEffect_24Q4.csv"))


# ------------------------------------------------------------------ helpers
def corr_defaults(r, p):
    bad = ~np.isfinite(r) | ~np.isfinite(p) | ((r == 0) & (p == 0))
    return np.where(bad, 0.0, r), np.where(bad, 1.0, p)


def hyper_upper(k, K, n, N):
    """R phyper(k - 1, K, N - K, n, lower.tail = FALSE) = P(X >= k)."""
    return stats.hypergeom.sf(k - 1, N, K, n)


def mutex_p(A: pd.DataFrame, g1, g2):
    """1 - P(X >= k) over the samples (rows) of boolean matrix A; genes absent -> 0 alterations."""
    cols = {g: i for i, g in enumerate(A.columns)}
    a = A.to_numpy()
    N = a.shape[0]
    tot = a.sum(0)
    out = np.empty(len(g1))
    for j, (x, y) in enumerate(zip(g1, g2)):
        i1, i2 = cols.get(x), cols.get(y)
        K = tot[i1] if i1 is not None else 0
        n = tot[i2] if i2 is not None else 0
        k = int((a[:, i1] & a[:, i2]).sum()) if (i1 is not None and i2 is not None) else 0
        out[j] = 1 - hyper_upper(k, K, n, N)
    return out


def sumlog(ps: np.ndarray):
    """metap::sumlog on rows; SBSL returns 1 on any error/warning (metap warns and drops p <= 0)."""
    out = np.ones(len(ps))
    ok = (ps > 0).all(1) & (ps <= 1).all(1)
    chi = -2 * np.log(ps[ok]).sum(1)
    out[ok] = stats.chi2.sf(chi, 2 * ps.shape[1])
    return out


def dependency_feats(D: pd.DataFrame, mut: pd.DataFrame, lines, kk, prefix):
    """SBSL dependency_scores_analysis.R: data restricted to lines of the lineage."""
    X = D.loc[D.index.intersection(lines)]
    cols = {g: i for i, g in enumerate(X.columns)}
    x = X.to_numpy(dtype=np.float64)
    M = mut.reindex(index=X.index).fillna(0) > 0
    mcols = {g: i for i, g in enumerate(M.columns)}
    m = M.to_numpy()
    r, p = pair_corr(X, kk.gene_a, kk.gene_b)
    miss = np.array([(a not in cols) or (b not in cols) for a, b in zip(kk.gene_a, kk.gene_b)])
    r, p = corr_defaults(r, p)
    r[miss], p[miss] = 0.0, 1.0
    mean = np.nanmean(x, 0) if x.shape[0] else np.full(x.shape[1], np.nan)
    avg = np.array([np.nanmean([mean[cols[a]], mean[cols[b]]]) if (a in cols and b in cols) else np.nan
                    for a, b in zip(kk.gene_a, kk.gene_b)])
    W = np.zeros(len(kk))
    P = np.ones(len(kk))
    for j, (a, b) in enumerate(zip(kk.gene_a, kk.gene_b)):
        if b not in cols or a not in mcols:
            continue
        mu = m[:, mcols[a]]
        if mu.sum() < 3:
            continue
        v = x[:, cols[b]]
        v1, v0 = v[mu], v[~mu]
        v1, v0 = v1[~np.isnan(v1)], v0[~np.isnan(v0)]
        if len(v1) == 0 or len(v0) == 0:
            continue
        allv = np.r_[v1, v0]
        if len(np.unique(allv)) < len(allv):  # R wilcox.test warns on ties -> SBSL default (p 1, W 0)
            continue
        W[j] = stats.mannwhitneyu(v1, v0, alternative="two-sided").statistic
        # R: exact p for n < 50 without ties, else normal approximation with continuity correction
        if len(v1) < 50 and len(v0) < 50:
            P[j] = _exact_wilcox_p(W[j], len(v1), len(v0))
        else:
            P[j] = stats.mannwhitneyu(v1, v0, alternative="two-sided", use_continuity=True).pvalue
    return {f"{prefix}_codep": r, f"{prefix}_codep_pvalue": p, f"{prefix}_dep": W, f"{prefix}_dep_pvalue": P,
            f"{prefix}_avg": avg}


_WCACHE = {}


def _exact_wilcox_p(w, m, n):
    """Two-sided exact p of the Wilcoxon rank-sum W (R's pwilcox) via the DP count of rank-sum configurations."""
    key = (m, n)
    if key not in _WCACHE:
        # null distribution of U: f(m, n, u) = f(m-1, n, u-n) + f(m, n-1, u)
        f = {}

        def dist(mm, nn):
            if (mm, nn) in f:
                return f[(mm, nn)]
            if mm == 0 or nn == 0:
                d = np.zeros(mm * nn + 1)
                d[0] = 1
            else:
                a = dist(mm - 1, nn)
                b = dist(mm, nn - 1)
                d = np.zeros(mm * nn + 1)
                d[nn:nn + len(a)] += a
                d[:len(b)] += b
            f[(mm, nn)] = d
            return d
        sys.setrecursionlimit(10000)
        d = dist(m, n)
        _WCACHE[key] = np.cumsum(d) / d.sum()
    cdf = _WCACHE[key]
    w = int(round(w))
    mid = m * n / 2
    if w > mid:
        p = 1 - (cdf[w - 1] if w >= 1 else 0)
    else:
        p = cdf[w]
    return min(1.0, 2 * p)


def logrank_p(time, event, group):
    from lifelines.statistics import logrank_test
    if group.sum() == 0 or (~group).sum() == 0:
        return 1.0
    try:
        p = logrank_test(time[group], time[~group], event[group], event[~group]).p_value
        return 1.0 if not np.isfinite(p) else float(p)
    except Exception:
        return 1.0


# ------------------------------------------------------------------ per cancer type
def per_cancer(cancer, kk, data):
    E, CN, MU, clin, ptype, G, pw, dep, d2, clmut, dmap = data
    genes = sorted(set(kk.gene_a) | set(kk.gene_b))
    stype = pd.Series([ptype.get(s[:12]) for s in E.index], index=E.index)
    tum, nor = O.is_tumour(E.index), O.is_normal(E.index)
    sel = stype.notna().values if cancer == C.PANCAN else (stype == cancer).values
    tsel, nsel = tum & sel, nor & sel
    Eg = [g for g in genes if g in E.columns]
    Et = E.loc[tsel, Eg]
    r = kk[KEY].copy()
    # pathway co-participation
    P = pw.reindex(columns=genes).fillna(False)
    N = P.shape[0]
    Ks = P.sum(0)
    arr = P.to_numpy()
    gi = {g: i for i, g in enumerate(genes)}
    k = np.array([(arr[:, gi[a]] & arr[:, gi[b]]).sum() for a, b in zip(kk.gene_a, kk.gene_b)])
    r["copathway_participation"] = hyper_upper(k, Ks[kk.gene_a].values, Ks[kk.gene_b].values, N)
    # mutual exclusivity (tumour samples with CN and mutation calls)
    cn_s = [s for s in CN.index if s in set(MU.index) and (ptype.get(s[:12]) == cancer or
                                                          (cancer == C.PANCAN and ptype.get(s[:12]) is not None))]
    cn_s = [s for s in cn_s if O.is_tumour([s])[0]]
    cnm = CN.loc[cn_s].reindex(columns=genes)
    amp, dele = cnm == 2, cnm == -2
    mut = MU.loc[cn_s].reindex(columns=genes).fillna(0) > 0
    pa = mutex_p(amp, kk.gene_a, kk.gene_b)
    pd_ = mutex_p(dele, kk.gene_a, kk.gene_b)
    pm = mutex_p(mut, kk.gene_a, kk.gene_b)
    r["dsl_mutex_amp"], r["dsl_mutex_del"], r["dsl_mutex_mut"] = pa, pd_, pm
    r["dsl_mutex"] = sumlog(np.c_[pa, pd_, pm])
    r["mutex_alt"] = mutex_p(amp | dele | mut, kk.gene_a, kk.gene_b)
    # expression correlations
    lin = lambda X: np.power(2.0, X) - 1
    r["exp_corr"], r["exp_corr_pvalue"] = corr_defaults(*pair_corr(lin(Et), kk.gene_a, kk.gene_b))
    En = E.loc[nsel, Eg]
    if len(En) >= 3:
        r["exp_corr_normal"], r["exp_corr_normal_pvalue"] = corr_defaults(*pair_corr(lin(En), kk.gene_a, kk.gene_b))
    else:
        r["exp_corr_normal"], r["exp_corr_normal_pvalue"] = 0.0, 1.0
    gs = [s for s in O.gtex_samples(cancer) if s in G.index]
    r["gtex_corr"], r["gtex_corr.pvalue"] = corr_defaults(
        *pair_corr(G.loc[gs, [g for g in genes if g in G.columns]], kk.gene_a, kk.gene_b))
    # differential expression of gene2 by gene1 mutation (tumours with expression)
    Mt = MU.reindex(index=Et.index, columns=genes).fillna(0) > 0
    ex = Et.to_numpy(dtype=np.float64)
    ec = {g: i for i, g in enumerate(Eg)}
    lfc, dp = np.zeros(len(kk)), np.ones(len(kk))
    mt = Mt.to_numpy()
    for j, (a, b) in enumerate(zip(kk.gene_a, kk.gene_b)):
        mu = mt[:, gi[a]]
        if mu.sum() <= 3 or b not in ec:
            continue
        v = ex[:, ec[b]]
        v1, v0 = v[mu & ~np.isnan(v)], v[~mu & ~np.isnan(v)]
        if len(v1) < 2 or len(v0) < 2:
            continue
        lfc[j] = v1.mean() - v0.mean()
        t = stats.ttest_ind(v1, v0, equal_var=False)
        dp[j] = t.pvalue if np.isfinite(t.pvalue) else 1.0
    r["diff_exp_logfc"], r["diff_exp_pvalue"] = lfc, dp
    # cell-line dependency (lines of the cancer type)
    lines = dmap.index if cancer == C.PANCAN else dmap.index[(dmap == cancer).values]
    for D, pre in [(dep, "avana"), (d2, "d2")]:
        for kname, v in dependency_feats(D.reindex(columns=[g for g in genes if g in D.columns]), clmut, lines, kk,
                                         pre).items():
            r[kname] = v
    # survival log-rank tests
    cl = clin[clin.index.isin(Et.index)]
    cl = cl.dropna(subset=["OS", "OS.time"]).drop_duplicates("patient")
    pats = cl.patient.values
    tt, ev = cl["OS.time"].to_numpy(float), cl["OS"].to_numpy(float)
    mp = Mt.copy()
    mp.index = [s[:12] for s in mp.index]
    mp = mp.groupby(level=0).any().reindex(pats).fillna(False).to_numpy()
    grp = stype[tsel].values
    q = Et.groupby(grp).transform(lambda s: s < s.quantile(0.05)) if cancer == C.PANCAN else (Et < Et.quantile(0.05))
    q = q.reindex(columns=genes).fillna(False)
    q.index = [s[:12] for s in q.index]
    up = q.groupby(level=0).any().reindex(pats).fillna(False).to_numpy()
    r["mut_logrank.pval"] = [logrank_p(tt, ev, mp[:, gi[a]] & mp[:, gi[b]]) for a, b in zip(kk.gene_a, kk.gene_b)]
    r["mrna_logrank.pval"] = [logrank_p(tt, ev, up[:, gi[a]] & up[:, gi[b]]) for a, b in zip(kk.gene_a, kk.gene_b)]
    C.log(f"sbsl {cancer}: {len(kk):,} keys, {tsel.sum()} tumours, {len(cn_s)} CN+mut, {len(lines)} lines")
    return r


def pathways():
    sets = []
    for f in ["c2.cp.kegg.v7.0.symbols.gmt", "c2.cp.pid.v7.0.symbols.gmt", "c2.cp.reactome.v7.0.symbols.gmt"]:
        for line in open(C.RAW / "msigdb_v7" / f):
            x = line.rstrip("\n").split("\t")
            sets.append((x[0], set(x[2:])))
    res = C.resolver()
    genes = sorted({res.get(g, g) for _, s in sets for g in s})
    gi = {g: i for i, g in enumerate(genes)}
    M = np.zeros((len(sets), len(genes)), dtype=bool)
    for i, (_, s) in enumerate(sets):
        M[i, [gi[res.get(g, g)] for g in s]] = True
    return pd.DataFrame(M, columns=genes)


def build(keys):
    from joblib import Parallel, delayed
    data = (O.tcga_expr(), O.tcga_cna(), O.tcga_mut(), O.tcga_clin(), O.patient_type(), O.gtex(), pathways(),
            crispr_effect(), rnai(), O.cl_mut(), C.depmap_tcga())
    groups = list(keys.groupby("cancer"))
    # PANCAN is the largest group: run it alone, the rest in parallel
    out = []
    for cancer, kk in groups:
        out.append(per_cancer(cancer, kk.reset_index(drop=True), data))
    return pd.concat(out, ignore_index=True)


def main(split):
    rows = C.pairs_with_type(C.human(split))
    keys = rows[KEY].drop_duplicates().reset_index(drop=True)
    p = WORK / "cache" / "keys_sbsl.parquet"
    have = pd.read_parquet(p) if p.exists() else None
    todo = keys if have is None else keys.merge(have[KEY], on=KEY, how="left", indicator=True).query(
        "_merge == 'left_only'")[KEY]
    C.log(f"sbsl features: {len(keys):,} keys, computing {len(todo):,}")
    if len(todo):
        new = build(todo.reset_index(drop=True))
        have = new if have is None else pd.concat([have, new], ignore_index=True)
        have.to_parquet(p)
    X = rows[["example_id", "context_id"] + KEY].merge(have, on=KEY, how="left")
    X.to_parquet(FEAT / f"{split}.parquet")
    C.log(f"wrote {FEAT / f'{split}.parquet'} {X.shape}")


if __name__ == "__main__":
    main(sys.argv[1])
