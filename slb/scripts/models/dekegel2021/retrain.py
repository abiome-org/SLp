"""Retrain the De Kegel 2021 RF (same 22 features, same hyper-parameters) on SLB train labels only.

Training rows: human SLB train rows whose pair has paralog features (one row per context x pair; the
model is context-agnostic like the original). Scores dev/test rows with features; others NaN.
usage: uv run python scripts/models/dekegel2021/retrain.py <train.csv> <eval.csv> <out.csv>
"""
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from features import FEATURES

tr = pd.read_csv(sys.argv[1])
tr = tr[tr.has_features]
ev = pd.read_csv(sys.argv[2])
rf = RandomForestClassifier(n_estimators=600, random_state=8, max_features=0.5, max_depth=3, min_samples_leaf=8,
                            n_jobs=8)
rf.fit(tr[FEATURES].fillna(0).values, tr.label.values)
print(f"trained on {len(tr):,} rows ({int(tr.label.sum()):,} SL)", file=sys.stderr)
m = ev.has_features.astype(bool)
ev["score"] = float("nan")
ev.loc[m, "score"] = rf.predict_proba(ev.loc[m, FEATURES].fillna(0).values)[:, 1]
ev[["example_id", "score"]].to_csv(sys.argv[3], index=False)
imp = sorted(zip(rf.feature_importances_, FEATURES), reverse=True)
print("importances:", ", ".join(f"{n}={v:.3f}" for v, n in imp[:8]), file=sys.stderr)
