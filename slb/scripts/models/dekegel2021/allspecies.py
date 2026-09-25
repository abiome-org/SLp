"""dekegel2021__allspecies: De Kegel 2021-style paralog random forest for every SLB species.

The 22 original features are human-specific (GTEx, CORUM, DepMap, yeast orthologs). This variant keeps the
model (RandomForest 600 trees, max_depth 3, max_features 0.5, min_samples_leaf 8) and rebuilds the subset of
feature *families* that exist in any species from the shared bundle (data/interim/bundle/<species>/):
  sequence   : paralog identity (Ensembl 116 via slbench.homology for human/scer/spom/dmel, DIAMOND/Ensembl paralog
               edges from slbench.families_extra for the other species; 0 = not paralogs), is_paralog,
               family_size (max over the two genes), closest (either gene is the other's most similar paralog),
               ESM-2 cosine similarity (bundle esm2.parquet, if present)
  neighbourhood: interact (PPI edge, bundle ppi = STRING without experimental/text-mining + BioGRID physical),
               n_total_ppi, fet_ppi_overlap (-log10 Fisher p of shared neighbours), shared_ppi_jaccard,
               shared_ppi_mean_essentiality (mean bundle single-gene fitness of shared neighbours),
               colocalisation (share a GO CC term), shared GO BP Jaccard (GO without IGI evidence)
One RF per species, fitted on that species' SLB train rows only. Species come from the split (no hard-coding).
usage: uv run python scripts/models/dekegel2021/allspecies.py [split]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import slb  # noqa: E402

BUNDLE = slb.ROOT / "data/interim/bundle"
FEATS = ["identity", "is_paralog", "family_size", "closest", "esm2_cos", "interact", "n_total_ppi", "fet_ppi_overlap",
         "shared_ppi_jaccard", "shared_ppi_mean_essentiality", "colocalisation", "go_bp_jaccard"]


def paralog_edges(sp: str) -> pd.DataFrame:
    import os
    cwd = os.getcwd()
    os.chdir(slb.ROOT)  # slbench modules use repo-relative data paths
    try:
        from slbench import homology
        p = homology.paralogs().filter(pl.col("species") == sp).select("a", "b", "identity").to_pandas()
        if p.empty:
            from slbench import families_extra
            e = families_extra.extra_edges().filter(pl.col("kind") == "paralog").to_pandas()
            e = e[e.u.str.startswith(sp + ":") & e.v.str.startswith(sp + ":")]
            a, b = e.u.str.split(":", n=1).str[1], e.v.str.split(":", n=1).str[1]
            p = pd.DataFrame({"a": np.minimum(a, b), "b": np.maximum(a, b), "identity": e.weight.to_numpy()})
            p = p.groupby(["a", "b"], as_index=False).identity.max()
    finally:
        os.chdir(cwd)
    return p


def features(d: pd.DataFrame, sp: str) -> pd.DataFrame:
    x = pd.DataFrame(index=d.index)
    a, b = d.gene_a.astype(str).to_numpy(), d.gene_b.astype(str).to_numpy()
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    par = paralog_edges(sp)
    ident = dict(zip(zip(par.a, par.b), par.identity))
    x["identity"] = [ident.get((u, v), 0.0) for u, v in zip(lo, hi)]
    x["is_paralog"] = (x.identity > 0).astype(float)
    fam = pd.concat([par.a, par.b]).value_counts()
    x["family_size"] = np.maximum(fam.reindex(a).fillna(0).to_numpy(), fam.reindex(b).fillna(0).to_numpy())
    best = pd.concat([par[["a", "b", "identity"]], par.rename(columns={"a": "b", "b": "a"})[["a", "b", "identity"]]])
    top = best.sort_values("identity", ascending=False).drop_duplicates("a").set_index("a").b
    x["closest"] = ((top.reindex(a).to_numpy() == b) | (top.reindex(b).to_numpy() == a)).astype(float)
    # ESM-2
    x["esm2_cos"] = np.nan
    ep = BUNDLE / sp / "esm2.parquet"
    if ep.exists():
        e = pd.read_parquet(ep).set_index("gene")
        E = e.to_numpy(np.float32)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        gi = {g: i for i, g in enumerate(e.index)}
        ia = np.array([gi.get(g, -1) for g in a])
        ib = np.array([gi.get(g, -1) for g in b])
        ok = (ia >= 0) & (ib >= 0)
        v = np.full(len(a), np.nan)
        v[ok] = (E[ia[ok]] * E[ib[ok]]).sum(1)
        x["esm2_cos"] = v
    # PPI
    for c in ["interact", "n_total_ppi", "fet_ppi_overlap", "shared_ppi_jaccard", "shared_ppi_mean_essentiality"]:
        x[c] = np.nan
    pp = BUNDLE / sp / "ppi.parquet"
    if pp.exists():
        p = pd.read_parquet(pp)
        ch = [c for c in p.columns if c.startswith("string_")]
        strong = (p[ch].max(axis=1) >= 400) | (p.biogrid_phys > 0)
        p = p[strong]
        nb = pd.concat([p[["gene_a", "gene_b"]], p.rename(columns={"gene_a": "gene_b", "gene_b": "gene_a"})[
            ["gene_a", "gene_b"]]]).groupby("gene_a").gene_b.apply(set).to_dict()
        N = len(nb)
        fit = {}
        fp = BUNDLE / sp / "fitness.parquet"
        if fp.exists():
            f = pd.read_parquet(fp)
            fit = dict(zip(f.gene, f.effect))
        inter, ntot, fet, jac, ess = [], [], [], [], []
        for u, v in zip(a, b):
            na, nb_ = nb.get(u, set()) - {v}, nb.get(v, set()) - {u}
            sh = na & nb_
            un = na | nb_
            inter.append(float(v in nb.get(u, set())))
            ntot.append(len(un))
            jac.append(len(sh) / len(un) if un else 0.0)
            if na and nb_:
                table = [[len(sh), len(na - sh)], [len(nb_ - sh), max(N - len(un), 0)]]
                fet.append(-np.log10(max(stats.fisher_exact(table, alternative="greater")[1], 1e-300)))
            else:
                fet.append(0.0)
            vals = [fit[g] for g in sh if g in fit]
            ess.append(np.mean(vals) if vals else np.nan)
        x["interact"], x["n_total_ppi"], x["fet_ppi_overlap"] = inter, ntot, fet
        x["shared_ppi_jaccard"], x["shared_ppi_mean_essentiality"] = jac, ess
    # GO
    x["colocalisation"], x["go_bp_jaccard"] = np.nan, np.nan
    gp = BUNDLE / sp / "go.parquet"
    if gp.exists():
        g = pd.read_parquet(gp)
        cc = g[g.aspect == "C"].groupby("gene").term.apply(set).to_dict()
        bp = g[g.aspect == "P"].groupby("gene").term.apply(set).to_dict()
        x["colocalisation"] = [float(bool(cc.get(u, set()) & cc.get(v, set()))) for u, v in zip(a, b)]
        jj = []
        for u, v in zip(a, b):
            s1, s2 = bp.get(u, set()), bp.get(v, set())
            jj.append(len(s1 & s2) / len(s1 | s2) if (s1 | s2) else np.nan)
        x["go_bp_jaccard"] = jj
    return x


def feats_rows(d: pd.DataFrame, sp: str) -> pd.DataFrame:
    """features() on unique pairs, broadcast back to rows."""
    u = d[["gene_a", "gene_b"]].drop_duplicates().reset_index(drop=True)
    f = pd.concat([u, features(u, sp)], axis=1)
    return d[["gene_a", "gene_b"]].merge(f, on=["gene_a", "gene_b"], how="left").set_index(d.index)


def main(split):
    tr_all, ev_all = slb.load("train"), slb.load(split)
    out = ev_all[["example_id"]].copy()
    out["score"] = np.nan
    for sp in sorted(ev_all.species.unique()):
        tr = tr_all[tr_all.species == sp]
        ev = ev_all[ev_all.species == sp]
        if tr.label.sum() < 10:
            print(f"{sp}: too few train positives ({int(tr.label.sum())}), skipped", file=sys.stderr)
            continue
        xt, xe = feats_rows(tr, sp), feats_rows(ev, sp)
        cols = [c for c in FEATS if xt[c].notna().any()]
        rf = RandomForestClassifier(n_estimators=600, random_state=8, max_features=0.5, max_depth=3, min_samples_leaf=8,
                                    n_jobs=8)
        rf.fit(xt[cols].fillna(-1).to_numpy(), tr.label.to_numpy())
        out.loc[ev.index, "score"] = rf.predict_proba(xe[cols].fillna(-1).to_numpy())[:, 1]
        imp = sorted(zip(rf.feature_importances_, cols), reverse=True)[:5]
        print(f"{sp}: train {len(tr):,} ({int(tr.label.sum())} SL), paralog rows {int(xt.is_paralog.sum()):,}; "
              + ", ".join(f"{n}={v:.2f}" for v, n in imp), file=sys.stderr)
    slb.write(out, "dekegel2021__allspecies", split)
    slb.evaluate("dekegel2021__allspecies", split)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else slb.SPLIT)
