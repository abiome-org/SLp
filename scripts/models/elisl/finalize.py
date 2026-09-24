"""Write an adapter's native (human) predictions via the shared slb.write (fills unscored species, writes
<name>_<split>.coverage.json) and evaluate dev.  Runs in the repo's uv env:
    uv run python scripts/models/elisl/finalize.py <pred.parquet|csv with example_id,score> <name> <split>"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import slb  # noqa: E402

pred, name, split = sys.argv[1:4]
p = pd.read_parquet(pred) if pred.endswith(".parquet") else pd.read_csv(pred)
d = slb.load(split)[["example_id"]].merge(p[["example_id", "score"]], on="example_id", how="left")
slb.write(d, name, split)
slb.evaluate(name, split)
