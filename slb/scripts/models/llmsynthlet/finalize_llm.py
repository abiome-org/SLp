"""CSV from llm_run.py -> results/models/<name>_<split>.parquet (+ dev eval). Per-context scores are matched on
(context_id, pair); context-free scores on the pair. Non-human rows are left missing.
usage: uv run python scripts/models/llmsynthlet/finalize_llm.py <csv> <name> [split]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import slb  # noqa: E402

csv, name = sys.argv[1], sys.argv[2]
split = sys.argv[3] if len(sys.argv) > 3 else slb.SPLIT
r = pd.read_csv(csv)
r["context_id"] = r.context_id.fillna("")
d = slb.load(split)
out = d[["example_id", "species"]].copy()
out["score"] = np.nan
allsp = r.key.str.contains(":").any()
h = d.species == d.species if allsp else d.species == "human"
key = slb.pair_key(d.gene_a[h], d.gene_b[h]).values
if allsp:
    key = (d.species[h] + ":").values + key
if (r.context_id != "").any():
    s = r.groupby(["context_id", "key"]).score.mean()
    idx = pd.MultiIndex.from_arrays([d.context_id[h].values, key])
    out.loc[h, "score"] = s.reindex(idx).to_numpy()
else:
    s = r.groupby("key").score.mean()
    out.loc[h, "score"] = s.reindex(key).to_numpy()
print("coverage:", slb.coverage(out), "| unparsable:", int(r.score.isna().sum()), file=sys.stderr)
slb.write(out, name, split)
slb.evaluate(name, split)
