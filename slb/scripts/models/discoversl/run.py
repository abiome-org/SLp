"""DiscoverSL on SLB (human only: needs TCGA mutation/CNA/expression).
  discoversl__released : released RF (leaky: trained on literature SL/non-SL pairs) on re-computed features.
  discoversl__slbtrain : same RF form (regression, 700 trees, mtry=1, same 4 features) refit on SLB train labels.
Pair score = max over the two orientations (gene1 primary / gene2 partner).
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import slb  # noqa: E402
from features import pair_features  # noqa: E402

F = ["PValue", "Mutex", "correlation.pvalue", "PvalPathway"]
W = slb.ROOT / "external/models/discoversl/export"


def emit(d, h, s, name, split):
    out = d[["example_id"]].copy()
    out["score"] = np.nan
    out.loc[h.index, "score"] = s.reindex(h.key).to_numpy()
    slb.write(out, name, split)
    slb.evaluate(name, split)


def main(split):
    d = slb.load(split)
    h = d[d.species == "human"].copy()
    h["key"] = slb.pair_key(h.gene_a, h.gene_b).values
    f = pair_features(split)
    # released model (R, Docker)
    tmp_in, tmp_out = W / f"feat_{split}.csv", W / f"pred_{split}.csv"
    f.to_csv(tmp_in, index=False)
    subprocess.run(["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}", "-v", f"{slb.ROOT / 'external/models/discoversl'}:/d",
                    "-v", f"{Path(__file__).resolve().parent}:/code", "slb/r-stats", "Rscript", "/code/score_pretrained.R",
                    f"/d/export/{tmp_in.name}", f"/d/export/{tmp_out.name}"], check=True)
    p = pd.read_csv(tmp_out)
    emit(d, h, p.groupby("key").score.max(), "discoversl__released", split)
    # retrain on SLB train
    tr = slb.load("train")
    tr = tr[tr.species == "human"].copy()
    tr["key"] = slb.pair_key(tr.gene_a, tr.gene_b).values
    ft = pair_features("train")
    # one training row per (row, orientation) like the package's directional pairs; label from the SLB row
    X = tr[["key", "label"]].merge(ft, on="key").dropna(subset=F)
    rf = RandomForestRegressor(n_estimators=700, max_features=1, n_jobs=8, random_state=0, min_samples_leaf=5)
    rf.fit(X[F].to_numpy(), X.label.to_numpy())
    ok = f[F].notna().all(1)
    f["score"] = np.nan
    f.loc[ok, "score"] = rf.predict(f.loc[ok, F].to_numpy())
    print(f"retrain rows {len(X):,} ({int(X.label.sum())} SL)", file=sys.stderr)
    emit(d, h, f.groupby("key").score.max(), "discoversl__slbtrain", split)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else slb.SPLIT)
