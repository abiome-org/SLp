"""Build the benchmark copies ("arms") of the cross-species transfer track.

Each arm is data/transfer/<arm>/: a copy of data/slb/ in which only train.parquet differs; every other
file (inputs, features, hidden/) is a symlink to data/slb/. So the held-out families and the scorer
are unchanged and only the human train labels change:

  full        SLB train as is (H+, the reference)
  h0          no human pair labels (H0, the claim): train on the non-human species only
  human_only  human pair labels only (checks whether the non-human data helps human at all)
  dose_pXX_sN human train pairs among a random subset of human train genes (seed N), sized so that
              about XX% of human train positives remain; non-human train rows are kept in full

Dose arms sample genes rather than pairs: a pair is kept when both of its genes were drawn, which mimics
screening fewer genes rather than labelling a random subset of every screen. (Sampling whole families
instead keeps same-family pairs at rate q but others at q^2, which skews the kept set towards paralogs.)

    uv run python scripts/transfer/arms.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/slb"
OUT = ROOT / "data/transfer"
DOSES = (0.01, 0.03, 0.1, 0.3)
SEEDS = (0, 1)


def _link(arm: Path, train: pd.DataFrame, meta: dict) -> None:
    arm.mkdir(parents=True, exist_ok=True)
    for p in SRC.iterdir():
        if p.name == "train.parquet":
            continue
        q = arm / p.name
        if not q.is_symlink():
            q.symlink_to(p.resolve())
    train.to_parquet(arm / "train.parquet", index=False)
    h = train[train.species == "human"]
    meta |= {"train_rows": len(train), "human_rows": len(h), "human_pos": int(h.label.sum()),
             "nonhuman_pos": int(train.label.sum() - h.label.sum())}
    (arm / "arm.json").write_text(json.dumps(meta, indent=1))
    print(f"{arm.name:18s} human rows {len(h):>7,}  human pos {meta['human_pos']:>5,}  total rows {len(train):,}")


def dose_mask(h: pd.DataFrame, target: float, seed: int) -> tuple[np.ndarray, float]:
    """Keep pairs whose two genes are both drawn at rate q; q is bisected so the kept share of positives
    is close to `target` (for one fixed random order of genes, so doses nest within a seed)."""
    ga, gb = h.gene_a.to_numpy(), h.gene_b.to_numpy()
    genes = np.array(sorted(set(ga) | set(gb)))
    u = dict(zip(genes, np.random.default_rng(seed).random(len(genes))))
    ua, ub = np.array([u[x] for x in ga]), np.array([u[x] for x in gb])
    pos = h.label.to_numpy() == 1
    lo, hi = 0.0, 1.0
    for _ in range(40):
        q = (lo + hi) / 2
        m = (ua < q) & (ub < q)
        lo, hi = (q, hi) if m[pos].mean() < target else (lo, q)
    return (ua < hi) & (ub < hi), hi


def main():
    tr = pd.read_parquet(SRC / "train.parquet")
    hum = (tr.species == "human").to_numpy()
    _link(OUT / "full", tr, {"arm": "full"})
    _link(OUT / "h0", tr[~hum], {"arm": "h0"})
    _link(OUT / "human_only", tr[hum], {"arm": "human_only"})
    h = tr[hum]
    for s in SEEDS:
        for d in DOSES:
            m, q = dose_mask(h, d, s)
            keep = pd.concat([tr[~hum], h[m]])
            _link(OUT / f"dose_p{round(d * 100):02d}_s{s}", keep,
                  {"arm": "dose", "target_pos_frac": d, "seed": s, "gene_rate": q,
                   "human_genes_kept": len(set(h.gene_a[m]) | set(h.gene_b[m]))})


if __name__ == "__main__":
    main()
