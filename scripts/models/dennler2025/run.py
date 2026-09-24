"""Dennler et al. (Ryan lab, 2025) paralog similarity classifiers on SLB (human paralog pairs).

  dennler2025_released : released 'SL | All 36 features XGB' predictions (Zenodo 14973633; XGB fitted on
                         DepMap-derived paralog SL labels, not on combinatorial screens).
  dennler2025_slbtrain : same XGB pipeline + 36 sequence/structure/PLM features, refit on SLB train labels.
  dennler2025_slbtrain_all58 : 36 features + the 22 De Kegel 2021 features, refit on SLB train.
Non-paralog human pairs get score 0; other species are left missing.
usage: <venv>/python scripts/models/dennler2025/run.py [split]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dekegel2021"))
import slb  # noqa: E402
from features import FEATURES as DK22  # noqa: E402

F36 = ['min_sequence_identity', 'rank_struct', 'selfSP_struct', 'taxid_struct', 'fident_struct', 'bits_struct',
       'alntmscore_struct', 'qtmscore_struct', 'ttmscore_struct', 'alnlen_struct', 'evalue_struct', 'prob_struct',
       'lddt_struct', 'rank_seq', 'selfSP_seq', 'taxid_seq',
       'esm2_beginning_of_sequence_cosine', 'esm2_beginning_of_sequence_euclidean',
       'esm2_beginning_of_sequence_manhattan', 'esm2_beginning_of_sequence_ts_ss', 'esm2_end_of_sequence_cosine',
       'esm2_end_of_sequence_euclidean', 'esm2_end_of_sequence_manhattan', 'esm2_end_of_sequence_ts_ss',
       'esm2_mean_of_residue_tokens_cosine', 'esm2_mean_of_residue_tokens_euclidean',
       'esm2_mean_of_residue_tokens_manhattan', 'esm2_mean_of_residue_tokens_ts_ss',
       'esm2_mean_of_special_tokens_cosine', 'esm2_mean_of_special_tokens_euclidean',
       'esm2_mean_of_special_tokens_manhattan', 'esm2_mean_of_special_tokens_ts_ss',
       'ProtT5_per-protein_cosine', 'ProtT5_per-protein_euclidean', 'ProtT5_per-protein_manhattan',
       'ProtT5_per-protein_ts_ss']
REL = "SL | All 36 features XGB"


def table() -> pd.DataFrame:
    cols = list(dict.fromkeys(["sorted_gene_pair", "A1", "A2", "A1_ensembl", "A2_ensembl"] + F36 + DK22))
    f = pd.read_csv(slb.RAW / "ryan_paralog_features/ens111_human_allFeatures.csv", usecols=cols)
    p = pd.read_csv(slb.RAW / "ryan_paralog_features/ens111_human_allPredictions.csv", usecols=["sorted_gene_pair", REL])
    f = f.merge(p, on="sorted_gene_pair", how="left")
    res = slb.symbol_resolver()
    a = [res(e) or res(s) or s for e, s in zip(f.A1_ensembl, f.A1)]
    b = [res(e) or res(s) or s for e, s in zip(f.A2_ensembl, f.A2)]
    f["key"] = slb.pair_key(pd.Series(a), pd.Series(b)).values
    for c in set(F36 + DK22):
        f[c] = pd.to_numeric(f[c].replace({True: 1, False: 0, "True": 1, "False": 0}), errors="coerce")
    return f.drop_duplicates("key").set_index("key")


def rows(split, f):
    d = slb.load(split)
    h = d[d.species == "human"].copy()
    h["key"] = slb.pair_key(h.gene_a, h.gene_b).values
    x = f.reindex(h.key.values)
    x.index = h.index
    return d, h, x


def xgb():
    return Pipeline([("scaler", StandardScaler()),
                     ("classifier", XGBClassifier(n_estimators=600, random_state=8, learning_rate=0.1,
                                                  colsample_bytree=0.5, eval_metric="logloss", n_jobs=8))])


def main(split="dev"):
    f = table()
    _, htr, xtr = rows("train", f)
    d, h, x = rows(split, f)
    have = x[F36[0]].notna()
    mtr = xtr[F36[0]].notna()
    outs = {"dennler2025__released": x[REL].where(have)}
    for name, feats in [("dennler2025__slbtrain", F36), ("dennler2025__slbtrain_all58", F36 + DK22)]:
        m = xgb().fit(xtr.loc[mtr, feats].values, htr.loc[mtr, "label"].values)
        s = pd.Series(np.nan, index=h.index)
        s[have] = m.predict_proba(x.loc[have, feats].values)[:, 1]
        outs[name] = s
    print(f"train rows with features {mtr.sum():,} ({int(htr.label[mtr].sum())} SL); {split} human {have.sum():,}/{len(h):,}",
          file=sys.stderr)
    for name, s in outs.items():
        out = d[["example_id", "species"]].copy()
        out["score"] = np.nan
        out.loc[h.index, "score"] = s.fillna(0).values
        slb.write(out, name, split)
        if split == "dev":
            slb.evaluate(name)


if __name__ == "__main__":
    main(*(sys.argv[1:] or [slb.SPLIT]))
