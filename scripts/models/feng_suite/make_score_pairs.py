"""Pairs a Feng-suite model must score (fit + valid train pairs and every human pair of dev and test inputs), written
to preprocessed_data/slb_score_pairs.npy; used by the patched SLGNN final_test to avoid scoring all N^2/2 pairs."""
import os
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(os.environ.get("SLB_BENCH", ROOT / "data/bench/slb1.2"))
if not BENCH.is_absolute():
    BENCH = ROOT / BENCH
WORK = Path(os.environ.get("SLB_WORK", ROOT / "external/models/_slb_work" / BENCH.name))
FD = WORK / "feng/data"
z = np.load(FD / "data_split/slb_split_arrays.npz")
gi = pd.read_parquet(FD / "slb_gene_index.parquet"); u = dict(zip(gi.gene, gi.unified_id))
pp = [z["fit_pos"], z["fit_neg"], z["val_pos"], z["val_neg"]]
for f in ["dev.parquet", "test_inputs.parquet"]:
    if (BENCH / f).exists():
        d = pd.read_parquet(BENCH / f, columns=["species", "gene_a", "gene_b"]); d = d[d.species == "human"]
        a, b = d.gene_a.map(u).values, d.gene_b.map(u).values
        pp.append(np.stack([np.minimum(a, b), np.maximum(a, b)], 1))
p = np.unique(np.concatenate(pp).astype(np.int64), axis=0); p = p[p[:, 0] != p[:, 1]]
np.save(FD / "preprocessed_data/slb_score_pairs.npy", p); print("score pairs", len(p))
