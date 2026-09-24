"""Turn a per-row CSV (example_id, score) into results/models/<name>_<split>.parquet covering the whole split.

Rows the model cannot score (non-human species, pairs that are not Ensembl paralogs) get `fill`:
'nan' leaves them missing (eval median-fills), a number assigns that score.
usage: uv run python scripts/models/dekegel2021/finalize.py <scores.csv> <name> <split> [fill]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import slb  # noqa: E402

s = pd.read_csv(sys.argv[1])
name, split = sys.argv[2], sys.argv[3]
fill = sys.argv[4] if len(sys.argv) > 4 else "nan"
d = slb.load(split)[["example_id", "species", "same_family"]]
d = d.merge(s, on="example_id", how="left")
print("native coverage:", slb.coverage(d), file=sys.stderr)
if fill != "nan":
    d.loc[(d.species == "human") & d.score.isna(), "score"] = float(fill)
slb.write(d, name, split)
