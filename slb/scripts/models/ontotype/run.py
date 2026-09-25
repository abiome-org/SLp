"""Ontotype model (Yu et al. 2016, Cell Systems, "Translation of genotype to phenotype by a hierarchy of cell
subsystems"; code michaelkyu/ontotype), retrained on SLB train.

The ontotype of a genotype is, for every ontology term, the number of disrupted genes annotated to it
(here 0, 1 or 2 for a double mutant). Yu et al. fed ontotypes of GO (or CliXO data-driven ontologies)
to a random forest trained on Costanzo 2010 double-mutant fitness. Here: GO (go-basic, is_a + part_of,
propagated; IGI evidence dropped; terms with >= 6 genes and <= 30% of annotated genes), and a
gradient-boosted classifier (LightGBM on the sparse ontotype; faster than a random forest at 2M rows)
fitted on SLB train labels only. Per-species model; generic over any species with GO annotations.

    SLB_BENCH=... SLB_SPLIT=dev scripts/models/ontotype/run.sh      (GO from the shared bundle)
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp_
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import bundle_io as B  # noqa: E402


def gene_terms(species: str, min_genes=6, max_frac=0.3, idx: dict | None = None):
    """gene -> term-index array. With idx (pooled variant) the term index is shared across species."""
    gt = B.go_propagated(species)
    if idx is None:
        cnt = defaultdict(int)
        for ts in gt.values():
            for t in ts:
                cnt[t] += 1
        n = len(gt)
        keep = sorted(t for t, c in cnt.items() if min_genes <= c <= max_frac * n)
        idx = {t: i for i, t in enumerate(keep)}
    return {g: np.array(sorted(idx[t] for t in ts if t in idx), dtype=np.int32) for g, ts in gt.items()}, len(idx)


def shared_index(species: list[str], min_genes=6, max_frac=0.3) -> dict:
    """Union over species of the terms each species keeps."""
    keep = set()
    for s in species:
        gt = B.go_propagated(s)
        cnt = defaultdict(int)
        for ts in gt.values():
            for t in ts:
                cnt[t] += 1
        keep |= {t for t, c in cnt.items() if min_genes <= c <= max_frac * len(gt)}
    return {t: i for i, t in enumerate(sorted(keep))}


def ontotype(a_list, b_list, gt, nterms):
    rows, cols, vals = [], [], []
    empty = np.array([], dtype=np.int32)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        ta, tb = gt.get(a, empty), gt.get(b, empty)
        u, c = np.unique(np.concatenate([ta, tb]), return_counts=True)
        rows.append(np.full(len(u), i, dtype=np.int32))
        cols.append(u)
        vals.append(c.astype(np.float32))
    return sp_.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(a_list), nterms))


def params():
    return dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
                feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                num_threads=int(os.environ.get("SLB_THREADS", 8)), seed=0)


def pooled(tr, ev):
    """allspecies variant: one model over all species on a shared GO-term space (+ species-agnostic),
    so species without train labels (new SLB species) are scored by transfer."""
    sps = [s for s in sorted(set(tr.species) | set(ev.species)) if B.has_bundle(s, "go")]
    idx = shared_index(sps)
    gts = {s: gene_terms(s, idx=idx)[0] for s in sps}
    trs = tr[tr.species.isin(sps)]
    xs, ys, ga, gb = [], [], [], []
    for s in sps:
        t = trs[trs.species == s]
        if len(t):
            xs.append(ontotype(t.gene_a, t.gene_b, gts[s], len(idx)))
            ys.append(t.label.to_numpy().astype(int))
            ga.append((s + "|" + t.gene_a).to_numpy())
            gb.append((s + "|" + t.gene_b).to_numpy())
    xt, y = sp_.vstack(xs).tocsr(), np.concatenate(ys)
    ga, gb = np.concatenate(ga), np.concatenate(gb)
    rng = np.random.default_rng(0)
    g = np.array(sorted(set(ga) | set(gb)))
    val = rng.choice(g, int(0.15 * len(g)), replace=False)
    ia, ib = np.isin(ga, val), np.isin(gb, val)
    va, trm = ia & ib, ~ia & ~ib
    dt = lgb.Dataset(xt[np.flatnonzero(trm)], y[trm])
    m = lgb.train(params(), dt, 2000, valid_sets=[lgb.Dataset(xt[np.flatnonzero(va)], y[va], reference=dt)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    auc = roc_auc_score(y[va], m.predict(xt[np.flatnonzero(va)], num_iteration=m.best_iteration))
    m2 = lgb.train(params(), lgb.Dataset(xt, y), max(m.best_iteration, 50))
    print(f"pooled over {sps}: {len(idx)} terms, internal-val AUROC {auc:.4f}", flush=True)
    res = []
    for s in ev.species.unique():
        if s in gts:
            e = ev[ev.species == s]
            res.append(pd.DataFrame({"example_id": e.example_id.to_numpy(),
                                     "score": m2.predict(ontotype(e.gene_a, e.gene_b, gts[s], len(idx)))}))
    return pd.concat(res)


def main():
    tr = B.load_train()
    ev = B.load_split()
    if "--pooled" in sys.argv:
        out = ev[["example_id"]].merge(pooled(tr, ev), on="example_id", how="left")
        p = B.out_path("ontotype", variant="pooled")
        out.to_parquet(p)
        print("wrote", p, f"scored {out.score.notna().sum():,}/{len(out):,}")
        return
    res = []
    for s in ev.species.unique():
        t, e = tr[tr.species == s], ev[ev.species == s]
        if not B.has_bundle(s, "go") or t.label.sum() == 0:
            print(f"{s}: no GO bundle or no train labels; unscored")
            continue
        gt, nt = gene_terms(s)
        xt = ontotype(t.gene_a, t.gene_b, gt, nt)
        xe = ontotype(e.gene_a, e.gene_b, gt, nt)
        y = t.label.to_numpy().astype(int)
        rng = np.random.default_rng(0)
        g = np.array(sorted(set(t.gene_a) | set(t.gene_b)))
        val = rng.choice(g, int(0.15 * len(g)), replace=False)
        ia, ib = np.isin(t.gene_a.to_numpy(), val), np.isin(t.gene_b.to_numpy(), val)
        va, trm = ia & ib, ~ia & ~ib
        params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
                      feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                      num_threads=int(os.environ.get("SLB_THREADS", 8)), seed=0)
        rounds, auc = 100, float("nan")
        if 0 < y[va].sum() < va.sum():
            dt = lgb.Dataset(xt[np.flatnonzero(trm)], y[trm])
            dv = lgb.Dataset(xt[np.flatnonzero(va)], y[va], reference=dt)
            m = lgb.train(params, dt, 2000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
            auc = roc_auc_score(y[va], m.predict(xt[np.flatnonzero(va)], num_iteration=m.best_iteration))
            rounds = max(m.best_iteration, 50)
        m2 = lgb.train(params, lgb.Dataset(xt, y), rounds)
        res.append(pd.DataFrame({"example_id": e.example_id.to_numpy(), "score": m2.predict(xe)}))
        print(f"{s}: {nt} terms, train {len(t):,}, internal-val AUROC {auc:.4f}, rounds {rounds}", flush=True)
    out = ev[["example_id"]].merge(pd.concat(res), on="example_id", how="left")
    p = B.out_path("ontotype")
    out.to_parquet(p)
    print("wrote", p, f"scored {out.score.notna().sum():,}/{len(out):,}")


if __name__ == "__main__":
    main()
