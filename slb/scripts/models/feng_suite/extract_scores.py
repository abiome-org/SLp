"""Read a saved N x N Feng score matrix and write SLB predictions (example_id, score) for every example of
species SLB_SPECIES (default human) in split SLB_SPLIT (default dev). Labels are never read.
Usage: extract_scores.py <score_matrix.npy> <slb_gene_index.parquet> <out.parquet>"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(os.environ.get("SLB_BENCH", ROOT / "data/slb"))
if not BENCH.is_absolute():
    BENCH = ROOT / BENCH
split = os.environ.get("SLB_SPLIT", "dev")
sp_ = os.environ.get("SLB_SPECIES", "human")
mat_path, gi_path, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
f = BENCH / f"{split}.parquet"
if not f.exists():
    f = BENCH / f"{split}_inputs.parquet"
d = pd.read_parquet(f, columns=["example_id", "species", "gene_a", "gene_b"])
d = d[d.species == sp_]
gi = pd.read_parquet(gi_path)
u = dict(zip(gi.gene, gi.unified_id))
M = np.load(mat_path, mmap_mode="r")
a, b = d.gene_a.map(u), d.gene_b.map(u)
ok = a.notna() & b.notna()
a, b = a[ok].astype(int).values, b[ok].astype(int).values
s = (np.asarray(M[a, b], dtype=np.float64) + np.asarray(M[b, a], dtype=np.float64)) / 2
res = pd.DataFrame({"example_id": d.example_id[ok].values, "score": s})
out.parent.mkdir(parents=True, exist_ok=True)
res.to_parquet(out, index=False)
print(f"wrote {out}: {len(res)} of {len(d)} {sp_} {split} examples; non-finite {(~np.isfinite(s)).sum()}; "
      f"score min/median/max {np.nanmin(s):.4g}/{np.nanmedian(s):.4g}/{np.nanmax(s):.4g}; n unique {len(np.unique(s))}")
