"""Multi-network GI classifier in the style of Wong et al. 2004 (PNAS, "Combining biological networks to
predict genetic interactions") and Pandey et al. 2010 (BMC Syst Biol, "An integrative multi-network and
multi-classifier approach to predict genetic interactions"): pair features from GO co-annotation,
physical PPI and STRING functional-association channels, fed to gradient boosting fitted on SLB train only.

Inputs: the shared bundle data/interim/bundle/<species>/{go,ppi,fitness}.parquet (GO without IGI;
BioGRID physical only; STRING neighbourhood/fusion/co-occurrence/co-expression/database, i.e. NOT the
experimental channel, which carries genetic-interaction assays, and not textmining). Species-generic:
every species in the split that has a bundle is scored.

Features for a pair (a, b):
  GO (BP/CC/MF, propagated over is_a/part_of): n shared terms, Jaccard, max information content of a
     shared term (Resnik), n shared direct terms
  PPI: BioGRID physical evidence count, 5 STRING channels, shared physical / STRING(>=400) neighbours
     and their Jaccard, degrees (min/max)
  Single-gene: bundle single-loss effect (min/max; permitted input), n direct GO terms (min/max)
  same_family flag from the benchmark
Variants: default = one LightGBM per species (species without train labels fall back to the pooled
model); `pooled` = one model over all species with a species code feature.
Early stopping on an internal validation set of train pairs whose both genes are in a random 15% of
that species' train genes (dev labels are never used for fitting or model selection).

    SLB_BENCH=... SLB_SPLIT=dev scripts/models/go_ppi_gbm/run.sh
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import bundle_io as B  # noqa: E402

CH = ["neighborhood", "fusion", "cooccurence", "coexpression", "database"]
NS = {"biological_process": "bp", "cellular_component": "cc", "molecular_function": "mf"}
FEAT = [f"go_{k}_{x}" for k in ("bp", "cc", "mf") for x in ("n_shared", "jaccard", "max_ic", "n_shared_direct")] \
    + ["biogrid_phys"] + [f"string_{c}" for c in CH] \
    + ["phys_shared_nb", "phys_nb_jaccard", "string_shared_nb", "string_nb_jaccard",
       "phys_deg_min", "phys_deg_max", "string_deg_min", "string_deg_max", "go_n_min", "go_n_max",
       "fit_min", "fit_max", "same_family", "species_code"]


class SpeciesData:
    def __init__(self, sp: str, code: int):
        self.code = code
        _, ns = B.go_dag()
        self.direct = B.go_direct(sp) if B.has_bundle(sp, "go") else {}
        prop = B.go_propagated(sp) if self.direct else {}
        self.prop = {k: {} for k in NS.values()}
        for g, ts in prop.items():
            for k in NS.values():
                self.prop[k][g] = {t for t in ts if NS.get(ns.get(t)) == k}
        n = max(len(prop), 1)
        cnt: dict[str, int] = {}
        for ts in prop.values():
            for t in ts:
                cnt[t] = cnt.get(t, 0) + 1
        self.ic = {t: -math.log(c / n) for t, c in cnt.items()}
        e = B.ppi(sp)
        self.edge, self.phys, self.strn = {}, {}, {}
        if len(e):
            vals = e[["biogrid_phys"] + [f"string_{c}" for c in CH]].to_numpy()
            for a, b, v in zip(e.gene_a, e.gene_b, vals):
                self.edge[(a, b)] = v
                if v[0] > 0:
                    self.phys.setdefault(a, set()).add(b)
                    self.phys.setdefault(b, set()).add(a)
                if v[1:].max() >= 400:
                    self.strn.setdefault(a, set()).add(b)
                    self.strn.setdefault(b, set()).add(a)
        self.fit = B.fitness(sp)

    def features(self, d: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(d), len(FEAT)), dtype=np.float32)
        empty: set = set()
        zero = np.zeros(1 + len(CH))
        for i, (a, b, fam) in enumerate(zip(d.gene_a, d.gene_b, d.same_family)):
            f = []
            for k in ("bp", "cc", "mf"):
                sa, sb = self.prop[k].get(a, empty), self.prop[k].get(b, empty)
                sh = sa & sb
                un = len(sa) + len(sb) - len(sh)
                f += [len(sh), len(sh) / un if un else 0.0, max((self.ic[t] for t in sh), default=0.0),
                      len(self.direct.get(a, empty) & self.direct.get(b, empty))]
            f += list(self.edge.get((a, b) if a < b else (b, a), zero))
            for nb in (self.phys, self.strn):
                na, nb_ = nb.get(a, empty), nb.get(b, empty)
                sh = len(na & nb_)
                un = len(na) + len(nb_) - sh
                f += [sh, sh / un if un else 0.0]
            da, db = len(self.phys.get(a, empty)), len(self.phys.get(b, empty))
            sa_, sb_ = len(self.strn.get(a, empty)), len(self.strn.get(b, empty))
            ga, gb = len(self.direct.get(a, empty)), len(self.direct.get(b, empty))
            fa, fb = self.fit.get(a, np.nan), self.fit.get(b, np.nan)
            f += [min(da, db), max(da, db), min(sa_, sb_), max(sa_, sb_), min(ga, gb), max(ga, gb),
                  np.fmin(fa, fb), np.fmax(fa, fb), float(fam), self.code]
            out[i] = f
        return out


def fit(x, y, ga, gb, seed=0):
    rng = np.random.default_rng(seed)
    g = np.array(sorted(set(ga) | set(gb)))
    val = rng.choice(g, max(1, int(0.15 * len(g))), replace=False)
    ia, ib = np.isin(ga, val), np.isin(gb, val)
    va, tr = ia & ib, ~ia & ~ib
    params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                  num_threads=int(os.environ.get("SLB_THREADS", 8)), seed=seed)
    rounds, auc = 300, float("nan")
    if 0 < y[va].sum() < va.sum() and y[tr].sum() > 0:
        dt = lgb.Dataset(x[tr], y[tr], feature_name=FEAT)
        m = lgb.train(params, dt, 2000, valid_sets=[lgb.Dataset(x[va], y[va], reference=dt)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        rounds = max(m.best_iteration, 50)
        auc = roc_auc_score(y[va], m.predict(x[va], num_iteration=m.best_iteration))
    return lgb.train(params, lgb.Dataset(x, y, feature_name=FEAT), rounds), auc, rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooled", action="store_true")
    a = ap.parse_args()
    tr = B.load_train()
    ev = B.load_split()
    allsp = sorted(set(ev.species.unique()) | set(tr.species.unique()))
    sd = {s: SpeciesData(s, i) for i, s in enumerate(allsp) if B.has_bundle(s, "go") or B.has_bundle(s, "ppi")}
    species = [s for s in ev.species.unique() if s in sd]
    print("bundle species:", list(sd), "| scored:", species, flush=True)
    tr = tr[tr.species.isin(list(sd))]
    trs = pd.concat([tr[tr.species == s] for s in sd if (tr.species == s).any()])
    xtr = np.vstack([sd[s].features(trs[trs.species == s]) for s in sd if (trs.species == s).any()])
    ytr = trs.label.to_numpy().astype(int)
    gkey_a = (trs.species + "|" + trs.gene_a).to_numpy()
    gkey_b = (trs.species + "|" + trs.gene_b).to_numpy()
    out = pd.Series(np.nan, index=ev.example_id.to_numpy())
    pooled = None
    if a.pooled or any(s not in set(trs.species) for s in species):
        pooled, auc, r = fit(xtr, ytr, gkey_a, gkey_b)
        print(f"pooled: internal-val AUROC {auc:.4f} rounds {r}", flush=True)
    for s in species:
        e = ev[ev.species == s]
        xe = sd[s].features(e)
        m_tr = (trs.species == s).to_numpy()
        if a.pooled or m_tr.sum() == 0:
            m = pooled
        else:
            m, auc, r = fit(xtr[m_tr], ytr[m_tr], gkey_a[m_tr], gkey_b[m_tr])
            imp = sorted(zip(m.feature_importance("gain"), FEAT), reverse=True)[:5]
            print(f"{s}: internal-val AUROC {auc:.4f} rounds {r}; top gain: " + ", ".join(c for _, c in imp), flush=True)
        out[e.example_id.to_numpy()] = m.predict(xe)
    res = pd.DataFrame({"example_id": ev.example_id.to_numpy(), "score": out.to_numpy()})
    p = B.out_path("go_ppi_gbm", variant="pooled" if a.pooled else None)
    res.to_parquet(p)
    print("wrote", p, f"scored {res.score.notna().sum():,}/{len(res):,}")


if __name__ == "__main__":
    main()
