"""DepMap 'driver-loss -> paralog dependency' scores, unsupervised (no SL labels), human only.

  deltadep__mut      paralogSL / Delta Dependency (Mo & Zhu 2026, github tjogzt/paralogSL, compute_dd):
                     DD(D,P) = mean Chronos effect of P in D-WT lines - mean in D-mutant lines, pan-cancer.
                     D-mutant = damaging mutation (DepMap 24Q4 damaging matrix; the package example uses any
                     annotated mutation). Pair score = max over the two orientations.
  depmap_ols__loss   DepMap-based in-silico paralog screen (RespAsahikawaMedicalUniv, 25Q3 OLS scanner):
                     OLS  effect(P) ~ loss(D) + lineage + expr(P) + CN(P); score = -t(loss) (P more essential when
                     D is lost). loss(D) = damaging mutation OR deep deletion (log2 rel CN < 0.5) OR no expression
                     (TPM log1p < 1). Pair score = max over orientations. Needs >= 3 lost and >= 3 intact lines.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import omics  # noqa: E402
import slb  # noqa: E402


def main(split):
    d = slb.load(split)
    h = d[d.species == "human"].copy()
    h["key"] = slb.pair_key(h.gene_a, h.gene_b).values
    keys = pd.Series(h.key.unique())
    ab = keys.str.split("|", expand=True)
    pairs = pd.DataFrame({"a": ab[0].values, "b": ab[1].values})
    G = omics.crispr()
    mut = omics.ccle_mut().reindex(G.index) > 0
    cn = omics.ccle_cn().reindex(G.index)
    ex = omics.ccle_expr().reindex(G.index)
    lin = pd.read_csv(slb.RAW / "depmap/Model_24Q4.csv", usecols=["ModelID", "OncotreeLineage"]).set_index(
        "ModelID").OncotreeLineage.reindex(G.index).fillna("other")
    L = pd.get_dummies(lin, drop_first=True).to_numpy(float)
    genes = mut.columns.union(cn.columns).union(ex.columns)
    loss = (mut.reindex(columns=genes).fillna(False).astype(bool) | (cn.reindex(columns=genes) < 0.5)
            | (ex.reindex(columns=genes) < 1))
    dd = np.full(len(pairs), np.nan)
    ols = np.full(len(pairs), np.nan)
    for i, (a, b) in enumerate(zip(pairs.a, pairs.b)):
        for D, P in [(a, b), (b, a)]:
            if P not in G.columns:
                continue
            y = G[P].to_numpy(float)
            if D in mut.columns:
                m = mut[D].fillna(False).to_numpy(dtype=bool)
                if m.sum() >= 3 and (~m).sum() >= 3:
                    v = np.nanmean(y[~m]) - np.nanmean(y[m])
                    dd[i] = np.nanmax([dd[i], v])
            if D in loss.columns:
                if P not in ex.columns or P not in cn.columns:
                    continue
                l = loss[D].to_numpy(dtype=bool)
                ok = ~np.isnan(y) & ex[P].notna().to_numpy() & cn[P].notna().to_numpy()
                if l[ok].sum() < 3 or (~l[ok]).sum() < 3:
                    continue
                X = np.column_stack([np.ones(ok.sum()), l[ok].astype(float), ex[P].to_numpy()[ok], cn[P].to_numpy()[ok], L[ok]])
                beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
                r = y[ok] - X @ beta
                dof = max(ok.sum() - np.linalg.matrix_rank(X), 1)
                s2 = r @ r / dof
                try:
                    cov = s2 * np.linalg.pinv(X.T @ X)
                    t = beta[1] / np.sqrt(cov[1, 1])
                    ols[i] = np.nanmax([ols[i], -t])
                except Exception:
                    pass
    for name, v in [("deltadep__mut", dd), ("depmap_ols__loss", ols)]:
        s = pd.Series(v, index=keys.values)
        out = d[["example_id"]].copy()
        out["score"] = np.nan
        out.loc[h.index, "score"] = s.reindex(h.key).to_numpy()
        slb.write(out, name, split)
        slb.evaluate(name, split)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else slb.SPLIT)
