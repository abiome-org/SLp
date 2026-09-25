"""Train ELISL's ELRRF (original class, external/models/elisl/src/models/ELRRF.py) on SLB train human rows and
score an SLB split.

Faithful to ELISL's single-cancer *test* protocol (ELRRF.single_cancer_test_experiment, task '..._trus'):
  - 5 feature sets (seq_1024, ppi_ec, crispr_dependency_mut, crispr_dependency_expr, tissue) + their early
    concatenation ('type2' combination) -> 6 LightGBM random forests (boosting_type='rf', ELRRF defaults:
    400 trees, 165 leaves, colsample 0.8, subsample 0.632, min_child_samples 10, reg_lambda 5), no grid search
  - StandardScaler + VarianceThreshold(1e-4) fitted on train, NaN features -> 0 (ELISL: fillna(0))
  - balance_strategy 'undersample_train', n_split=5 repeats (fit_predict_2set), each repeat undersamples the
    majority class; with a 'cancer' column ELISL balances within each cancer -> here within each SLB context
  - late integration: probabilities of the 6 models averaged with weights = training AUPRC (combine_preds)
SLB changes: one pooled model over all human contexts (ELISL trained one model per cancer type; many SLB contexts
have too few positives for that) and the final score is the mean of the 5 repeats' ensembled probabilities
(ELISL reported the mean metric over repeats).

usage: python elrrf.py <split> <out.parquet>
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import common as C

sys.path.insert(0, str(C.ROOT / "external/models/elisl"))
import src.models.ELRRF as _E  # noqa: E402
from src.models.ELRRF import ELRRF  # noqa: E402

# In the released ELRRF.py, balance_cancer_by_index / balance_by_index (used by fit_predict_2set) sit inside a
# triple-quoted block and are therefore undefined; re-activate that exact original source text.
_src = open(_E.__file__).read()
_a = _src.index("def balance_cancer_by_index(")
_b = _src.index("def combine_preds(")
exec(compile(_src[_a:_b].rstrip().rstrip("'").rstrip(), _E.__file__, "exec"), _E.__dict__)

FAMS = ["seq_1024", "ppi_ec", "crispr_dependency_mut", "crispr_dependency_expr", "tissue"]
NJ = int(os.environ.get("ELISL_CPUS", "8"))
FEAT = C.WORK / "feat" / C.BENCH.name


def load(split, labels):
    feats = {}
    for f in FAMS:
        d = pd.read_parquet(FEAT / f"{split}_{f}.parquet")
        d = d.drop(columns=["context_id"]).rename(columns={"example_id": "pair_name"})
        d.insert(1, "class", d.pair_name.map(labels).fillna(0).astype(int).values)
        feats[f] = d.fillna(0)
    return feats


def main(split, out):
    t0 = time.time()
    tr = C.human("train")
    te = C.human(split)
    lab_tr = dict(zip(tr.example_id, tr.label.astype(int)))
    lab_te = dict(zip(te.example_id, te.label.fillna(0).astype(int))) if "label" in te else {}
    Ftr, Fte = load("train", lab_tr), load(split, lab_te)
    for f in FAMS:
        assert (Ftr[f].pair_name.values == tr.example_id.values).all()
        assert (Fte[f].pair_name.values == te.example_id.values).all()
    elrrf = ELRRF(use_single=True, grid_searched=False, balance_strategy="undersample_train", use_comb=False,
                  thold=0.5, process=True, n_jobs=NJ)
    for f in FAMS:
        elrrf.add_dataset(f, Ftr[f], Fte[f])
    comb_tr = pd.concat([Ftr[FAMS[0]]] + [Ftr[f].drop(columns=["pair_name", "class"]) for f in FAMS[1:]], axis=1)
    comb_te = pd.concat([Fte[FAMS[0]]] + [Fte[f].drop(columns=["pair_name", "class"]) for f in FAMS[1:]], axis=1)
    elrrf.add_dataset("&".join(FAMS), comb_tr, comb_te)
    s_tr = pd.DataFrame({"pair_name": tr.example_id.values, "class": tr.label.astype(int).values,
                         "cancer": tr.context_id.values})
    s_te = pd.DataFrame({"pair_name": te.example_id.values, "class": [lab_te.get(e, 0) for e in te.example_id]})
    C.log(f"train {len(s_tr):,} rows ({s_tr['class'].sum():,} pos), {split} {len(s_te):,} rows")
    results = elrrf.fit_predict_2set(s_tr, s_te, n_split=5)
    per_fold = []
    for fold, models in results.items():
        probs, w = [], []
        for name, r in models.items():
            if name == "time":
                continue
            probs.append(np.asarray(r["probabilities"], dtype=float))
            w.append(float(r["tr_auc"]))
            C.log(f"fold {fold} {name}: train AUPRC {r['tr_auc']:.3f}, n_train {len(r['train_names'])}")
        w = np.array(w) / np.sum(w)
        per_fold.append((np.vstack(probs) * w[:, None]).sum(0))
    score = np.mean(per_fold, axis=0)
    res = pd.DataFrame({"example_id": te.example_id.values, "score": score})
    res.to_parquet(out)
    C.log(f"wrote {out} ({len(res):,} rows) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
