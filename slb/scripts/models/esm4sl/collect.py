"""Collect ESM4SL test_logits.csv files into results/models/<name>_<split>.parquet (example_id, score).

usage: python collect.py <name> <split> <workdir1> [<workdir2> ...]
Each workdir holds test.csv (row order = test dataloader order) and out/csv_log/version_*/test_logits.csv.
Rows of the split that are not in any workdir's test.csv stay missing (eval median-fills them).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import slb  # noqa: E402

name, split, wds = sys.argv[1], sys.argv[2], sys.argv[3:]
parts = []
for wd in map(Path, wds):
    t = pd.read_csv(wd / "test.csv")
    logs = sorted((wd / "out/csv_log").glob("version_*/test_logits.csv"), key=lambda p: p.stat().st_mtime)
    if not logs:
        print(f"{wd}: no test_logits.csv", file=sys.stderr)
        continue
    lg = pd.read_csv(logs[-1])
    assert len(lg) == len(t), (wd, len(lg), len(t))
    # the test loader is unshuffled and never swaps genes -> same order as test.csv
    assert (lg.gene1.values == t["0"].values).all() and (lg.gene2.values == t["1"].values).all()
    parts.append(pd.DataFrame({"example_id": t.example_id, "score": lg.probs}))
d = slb.load(split)[["example_id", "species"]]
s = d.merge(pd.concat(parts) if parts else pd.DataFrame(columns=["example_id", "score"]), on="example_id", how="left")
slb.write(s, name, split)
print(slb.coverage(s), file=sys.stderr)
