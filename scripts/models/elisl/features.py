"""ELISL feature sets for the human rows of an SLB split.

Re-implements (vectorised, same definitions) ELISL's feature_generation/*.py + embedding/{dependency,tissue}_embedding.py:
  seq_1024               |SeqVec(g1) - SeqVec(g2)|, -1 for all 1024 dims if either gene has no embedding
  ppi_ec                 |node2vec(g1) - node2vec(g2)|, -1 if either gene is missing from the graph
  crispr_dependency_mut  dep0..3: mean dependency of g1 in lines of the cancer type with / without a g2 non-silent
                         mutation, and of g2 with / without g1 mutated (NaN if undefined)
  crispr_dependency_expr same with 'g altered' = |expression z| >= 1.96
  tissue                 gene1_expr_m1/m0, gene2_expr_m1/m0 (mean TCGA expression z of g1 in patients with / without
                         g2 mutated, ...), gtex_coexp(+p), tumor_coexp(+p), normal_coexp(+p) (Pearson),
                         tumor_cocnv(+p) (Spearman on GISTIC), surv (Cox p-value of g1&g2 co-alteration, strata
                         sex, race, age quartile; alteration = mutation or |z|>=1.96 or |GISTIC|>=2)
Row order and pair orientation follow SLB (gene_a = gene1, gene_b = gene2). Cancer type per row: common.context_tcga().
Outputs _slb/feat/<split>_<family>.parquet (example_id, context_id + feature columns). Per-(pair, cancer) values are
cached in _slb/cache/keys_<family>.parquet so later splits only compute new keys.

usage: python features.py <split>
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import common as C
import omics_data as O

warnings.filterwarnings("ignore")
NJOBS = int(os.environ.get("ELISL_CPUS", "8"))
FEAT = C.WORK / "feat" / C.BENCH.name
FEAT.mkdir(parents=True, exist_ok=True)
KEY = ["gene_a", "gene_b", "cancer"]


# ------------------------------------------------------------------ generic helpers
def cond_means(V: pd.DataFrame, A: pd.DataFrame):
    """m1[j, i] = mean_{rows altered in gene j} V[:, i];  m0[j, i] = mean over the other rows (NaN-skipping)."""
    v = V.to_numpy(dtype=np.float64)
    nn = ~np.isnan(v)
    v0 = np.where(nn, v, 0.0)
    a = A.to_numpy(dtype=np.float64)
    s1 = a.T @ v0
    n1 = a.T @ nn.astype(np.float64)
    t = v0.sum(0)[None, :]
    n = nn.sum(0)[None, :].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = s1 / n1
        m0 = (t - s1) / (n - n1)
    return m1, m0


def pair_corr(X: pd.DataFrame, g1, g2, rank=False, chunk=1500):
    """Pearson (or Spearman if rank) r and two-sided p for column pairs, pairwise-complete observations.
    Genes absent from X -> (0, 0) exactly like ELISL's zero-initialised arrays; undefined r -> NaN."""
    if rank:
        X = X.rank()
    cols = {g: i for i, g in enumerate(X.columns)}
    x = X.to_numpy(dtype=np.float64)
    i1 = np.array([cols.get(g, -1) for g in g1])
    i2 = np.array([cols.get(g, -1) for g in g2])
    ok = (i1 >= 0) & (i2 >= 0)
    r = np.zeros(len(g1))
    p = np.zeros(len(g1))
    idx = np.where(ok)[0]
    for s in range(0, len(idx), chunk):
        b = idx[s:s + chunk]
        a1 = x[:, i1[b]]
        a2 = x[:, i2[b]]
        m = ~(np.isnan(a1) | np.isnan(a2))
        n = m.sum(0).astype(np.float64)
        a1 = np.where(m, a1, 0.0)
        a2 = np.where(m, a2, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu1 = a1.sum(0) / n
            mu2 = a2.sum(0) / n
            d1 = np.where(m, a1 - mu1, 0.0)
            d2 = np.where(m, a2 - mu2, 0.0)
            rr = (d1 * d2).sum(0) / np.sqrt((d1 ** 2).sum(0) * (d2 ** 2).sum(0))
            rr = np.clip(rr, -1, 1)
            df = n - 2
            tt = rr * np.sqrt(df / np.maximum(1 - rr ** 2, 1e-300))
            pp = 2 * stats.t.sf(np.abs(tt), df)
            pp = np.where(np.abs(rr) >= 1, 0.0, pp)
        rr[n < 3] = np.nan
        pp[n < 3] = np.nan
        r[b] = rr
        p[b] = pp
    return r, p


def lookup(m, genes, g_row, g_col):
    gi = {g: i for i, g in enumerate(genes)}
    return m[[gi[g] for g in g_row], [gi[g] for g in g_col]]


# ------------------------------------------------------------------ context-free families
def emb_diff(pairs: pd.DataFrame, emb: pd.DataFrame, prefix: str) -> pd.DataFrame:
    e = emb.to_numpy(dtype=np.float32)
    gi = {g: i for i, g in enumerate(emb.index)}
    i1 = np.array([gi.get(g, -1) for g in pairs.gene_a])
    i2 = np.array([gi.get(g, -1) for g in pairs.gene_b])
    out = np.full((len(pairs), e.shape[1]), -1.0, dtype=np.float32)
    ok = (i1 >= 0) & (i2 >= 0)
    out[ok] = np.abs(e[i1[ok]] - e[i2[ok]])
    C.log(f"{prefix}: {ok.sum():,}/{len(ok):,} pairs have both embeddings")
    return pd.DataFrame(out, columns=[f"{prefix}{i}" for i in range(e.shape[1])], index=pairs.index)


# ------------------------------------------------------------------ cell-line families
def cl_family(keys: pd.DataFrame, alt: str) -> pd.DataFrame:
    dep = O.dep()
    dm = C.depmap_tcga().reindex(dep.index)
    out = []
    for cancer, kk in keys.groupby("cancer"):
        lines = dep.index if cancer == C.PANCAN else dep.index[(dm == cancer).values]
        genes = sorted(set(kk.gene_a) | set(kk.gene_b))
        V = dep.loc[lines].reindex(columns=genes)
        if alt == "mut":
            A = O.cl_mut().reindex(index=lines, columns=genes).fillna(0) > 0
        else:
            A = O.cl_exprz().reindex(index=lines, columns=genes).abs() >= 1.96
        m1, m0 = cond_means(V, A)
        r = kk[KEY].copy()
        r["dep0"] = lookup(m1, genes, kk.gene_b, kk.gene_a)
        r["dep1"] = lookup(m0, genes, kk.gene_b, kk.gene_a)
        r["dep2"] = lookup(m1, genes, kk.gene_a, kk.gene_b)
        r["dep3"] = lookup(m0, genes, kk.gene_a, kk.gene_b)
        C.log(f"cl_{alt} {cancer}: {len(lines)} lines, {len(kk):,} keys")
        out.append(r)
    return pd.concat(out)


# ------------------------------------------------------------------ tissue family
def cox_wald_p(time, event, strata_codes, X, max_iter=50, tol=1e-9):
    """Wald p-value of a single binary covariate in a stratified Cox model with Efron ties, vectorised over the
    columns of X (patients x pairs, bool). Same estimator as lifelines CoxPHFitter(strata=...).fit(...)
    .summary['p'] used by ELISL (penalizer 0); validated against lifelines in notes/models/elisl.md.
    Non-finite / non-converged fits (e.g. complete separation) give p -> 1 as lifelines' huge SE does."""
    from scipy.stats import norm
    order = np.lexsort((-time, strata_codes))           # by stratum, then time descending
    t, e, st = time[order], event[order].astype(bool), strata_codes[order]
    X = X[order].astype(np.float64)
    B = X.shape[1]
    # risk-set boundaries: groups of (stratum, time); risk set of a group = all rows of the stratum up to the
    # last row of that group (descending time)
    key_change = np.r_[True, (st[1:] != st[:-1]) | (t[1:] != t[:-1])]
    gid = np.cumsum(key_change) - 1
    G = gid[-1] + 1
    strat_start = np.r_[True, st[1:] != st[:-1]]
    # cumulative counts within stratum
    cs1 = np.cumsum(X, axis=0)
    base1 = np.maximum.accumulate(np.where(strat_start, np.arange(len(t)), 0))
    cs_all = np.arange(1, len(t) + 1, dtype=np.float64)
    last = np.r_[np.where(key_change)[0][1:] - 1, len(t) - 1]         # last row of each group
    first_of_stratum = base1[last]
    n_at = cs_all[last] - first_of_stratum                             # at-risk counts per group
    prev1 = cs1[np.maximum(first_of_stratum - 1, 0)] * (first_of_stratum > 0)[:, None]
    n1_at = cs1[last] - prev1                                         # (G, B)
    n0_at = n_at[:, None] - n1_at
    # deaths per group
    ev = e.astype(np.float64)
    d_all = np.bincount(gid, weights=ev, minlength=G)
    d1 = np.zeros((G, B))
    np.add.at(d1, gid[e], X[e])
    d0 = d_all[:, None] - d1
    keep = d_all > 0
    n1_at, n0_at, d1, d0, dd = n1_at[keep], n0_at[keep], d1[keep], d0[keep], d_all[keep]
    maxd = int(dd.max()) if len(dd) else 0
    beta = np.zeros(B)
    sum_dx = d1.sum(0)
    info = np.zeros(B)
    for _ in range(max_iter):
        w = np.exp(beta)[None, :]
        S0 = n0_at + n1_at * w
        S1 = n1_at * w
        D0 = d0 + d1 * w
        D1 = d1 * w
        U = sum_dx.copy()
        I = np.zeros(B)
        for l in range(maxd):
            m = (dd > l)[:, None]
            f = l / dd[:, None]
            a = S0 - f * D0
            b = S1 - f * D1
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where(m, b / a, 0.0)
            U -= r.sum(0)
            I += np.where(m, r - r ** 2, 0.0).sum(0)
        step = np.where(I > 1e-12, U / np.maximum(I, 1e-12), 0.0)
        step = np.clip(step, -5, 5)
        beta = np.clip(beta + step, -30, 30)
        info = I
        if np.all(np.abs(step) < tol):
            break
    with np.errstate(invalid="ignore", divide="ignore"):
        z = beta * np.sqrt(info)
    p = 2 * norm.sf(np.abs(z))
    p[~np.isfinite(p) | (info <= 1e-12)] = 1.0
    return p


def survival(kk, alt_pat: pd.DataFrame, surv: pd.DataFrame, strata, chunk=512):
    """alt_pat: patients x genes bool (patients = surv.index). ELISL: p=1 if no / all patients co-altered."""
    genes = {g: i for i, g in enumerate(alt_pat.columns)}
    a = alt_pat.to_numpy()
    P = a.shape[0]
    codes = pd.Series(list(zip(*[surv[c].astype(str) for c in strata]))).astype("category").cat.codes.to_numpy()
    tm = surv.OS_MONTHS.to_numpy(float)
    ev = surv.OS_STATUS.to_numpy(int)
    pv = np.ones(len(kk))
    ia = np.array([genes.get(g, -1) for g in kk.gene_a])
    ib = np.array([genes.get(g, -1) for g in kk.gene_b])
    ok = np.where((ia >= 0) & (ib >= 0))[0]
    for s in range(0, len(ok), chunk):
        b = ok[s:s + chunk]
        co = a[:, ia[b]] & a[:, ib[b]]
        n = co.sum(0)
        fit = (n > 0) & (n < P)
        if fit.any():
            pv[b[fit]] = cox_wald_p(tm, ev, codes, co[:, fit])
    return pv


def tissue_family(keys: pd.DataFrame) -> pd.DataFrame:
    E = O.tcga_expr()
    CN = O.tcga_cna()
    MU = O.tcga_mut()
    clin = O.tcga_clin()
    ptype = O.patient_type()
    G = O.gtex()
    samp_type = pd.Series([ptype.get(s[:12]) for s in E.index], index=E.index)
    tum = O.is_tumour(E.index)
    nor = O.is_normal(E.index)
    out = []
    for cancer, kk in keys.groupby("cancer"):
        genes = sorted(set(kk.gene_a) | set(kk.gene_b))
        if cancer == C.PANCAN:
            tsel = tum & samp_type.notna().values
            nsel = nor & samp_type.notna().values
        else:
            tsel = tum & (samp_type == cancer).values
            nsel = nor & (samp_type == cancer).values
        Et = E.loc[tsel].reindex(columns=genes)
        # std expression: per-gene z within the cancer type (PANCAN: within each type)
        grp = samp_type[tsel]
        gb = Et.groupby(grp.values)
        Zt = (Et - gb.transform("mean")) / gb.transform("std")
        Mt = MU.reindex(index=Et.index, columns=genes).fillna(0) > 0
        r = kk[KEY].copy()
        m1, m0 = cond_means(Zt, Mt)
        r["gene1_expr_m1"] = lookup(m1, genes, kk.gene_b, kk.gene_a)
        r["gene1_expr_m0"] = lookup(m0, genes, kk.gene_b, kk.gene_a)
        r["gene2_expr_m1"] = lookup(m1, genes, kk.gene_a, kk.gene_b)
        r["gene2_expr_m0"] = lookup(m0, genes, kk.gene_a, kk.gene_b)
        gs = [s for s in O.gtex_samples(cancer) if s in G.index]
        r["gtex_coexp"], r["gtex_coexp_p"] = pair_corr(G.loc[gs, [g for g in genes if g in G.columns]], kk.gene_a, kk.gene_b)
        lin = lambda X: np.power(2.0, X) - 1  # EB++ xena is log2(x+1); ELISL correlated normalised RSEM (linear)
        r["tumor_coexp"], r["tumor_coexp_p"] = pair_corr(lin(E.loc[tsel, [g for g in genes if g in E.columns]]), kk.gene_a, kk.gene_b)
        En = E.loc[nsel, [g for g in genes if g in E.columns]]
        if len(En) >= 3:
            r["normal_coexp"], r["normal_coexp_p"] = pair_corr(lin(En), kk.gene_a, kk.gene_b)
        else:  # no matched normals (e.g. LAML, SKCM): ELISL leaves 0
            r["normal_coexp"], r["normal_coexp_p"] = 0.0, 0.0
        cn_samples = [s for s in CN.index if s in set(Et.index)] or [s for s in CN.index if ptype.get(s[:12]) == cancer]
        CNt = CN.loc[cn_samples, [g for g in genes if g in CN.columns]]
        r["tumor_cocnv"], r["tumor_cocnv_p"] = pair_corr(CNt, kk.gene_a, kk.gene_b, rank=True)
        # survival
        cl = clin[clin.index.isin(Et.index) | clin.index.isin(CNt.index)].copy()
        cl = cl[O.is_tumour(cl.index)].dropna(subset=["age", "OS", "OS.time"]).drop_duplicates("patient")
        surv = pd.DataFrame({"OS_MONTHS": cl["OS.time"].values / 30.4375, "OS_STATUS": cl["OS"].astype(int).values,
                             "SEX": cl.gender.fillna("NA").values, "RACE": cl.race.fillna("NA").values,
                             "AGE_STRATA": pd.qcut(cl.age, q=[0, .25, .5, .75, 1.], duplicates="drop").astype(str).values,
                             "TYPE": cl.type.values}, index=cl.patient.values)
        strata = ["SEX", "RACE", "AGE_STRATA"] + (["TYPE"] if cancer == C.PANCAN else [])
        alt = (Mt | (Zt.abs() >= 1.96))
        alt.index = [s[:12] for s in alt.index]
        cna_alt = CNt.reindex(columns=genes).abs() >= 2
        cna_alt.index = [s[:12] for s in cna_alt.index]
        alt = alt.groupby(level=0).any().reindex(surv.index).fillna(False)
        cna_alt = cna_alt.groupby(level=0).any().reindex(surv.index).fillna(False)
        alt = alt | cna_alt
        r["surv"] = survival(kk, alt, surv, strata)
        C.log(f"tissue {cancer}: {tsel.sum()} tumours, {nsel.sum()} normals, {len(gs)} GTEx, {len(surv)} surv pts, "
              f"{len(kk):,} keys")
        out.append(r)
    return pd.concat(out)


# ------------------------------------------------------------------ driver
def keyed(name, keys, build):
    p = C.CACHE / f"keys_{name}.parquet"
    have = pd.read_parquet(p) if p.exists() else None
    todo = keys if have is None else keys.merge(have[KEY], on=KEY, how="left", indicator=True).query(
        "_merge == 'left_only'")[KEY]
    C.log(f"{name}: {len(keys):,} keys, computing {len(todo):,}")
    if len(todo):
        new = build(todo.reset_index(drop=True))
        have = new if have is None else pd.concat([have, new], ignore_index=True)
        have.to_parquet(p)
    return keys.merge(have, on=KEY, how="left")


def main(split):
    rows = C.pairs_with_type(C.human(split))
    keys = rows[KEY].drop_duplicates().reset_index(drop=True)
    base = rows[["example_id", "context_id", "gene_a", "gene_b", "cancer"]]
    fam = sys.argv[2:] or ["seq_1024", "ppi_ec", "crispr_dependency_mut", "crispr_dependency_expr", "tissue"]
    for f in fam:
        out = FEAT / f"{split}_{f}.parquet"
        if f == "seq_1024":
            X = emb_diff(rows, pd.read_parquet(C.CACHE / "seqvec.parquet"), "seq")
        elif f == "ppi_ec":
            X = emb_diff(rows, pd.read_parquet(C.CACHE / "node2vec.parquet"), "ppi_ec")
        else:
            build = {"crispr_dependency_mut": lambda k: cl_family(k, "mut"),
                     "crispr_dependency_expr": lambda k: cl_family(k, "expr"),
                     "tissue": tissue_family}[f]
            X = base[KEY].merge(keyed(f, keys, build), on=KEY, how="left").drop(columns=KEY)
            X.index = rows.index
        pd.concat([base[["example_id", "context_id"]], X], axis=1).to_parquet(out)
        C.log(f"wrote {out} {X.shape}")


if __name__ == "__main__":
    main(sys.argv[1])
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)  # py3.8 + pyarrow 4 occasionally segfaults at interpreter teardown after all outputs are written
