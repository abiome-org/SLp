"""Shared helpers for models-features adapters (scripts/models/<name>/).

Every adapter reads the SLB benchmark from $SLB_BENCH (default data/slb), never touches
hidden/, trains only on train.parquet labels, and writes results/models/<bench>/<name>_<split>.parquet
with columns (example_id, score).
"""

from __future__ import annotations

import os as _os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    _os.environ.setdefault(_v, "8")  # shared machine: cap thread pools (effective if imported before numpy)

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(os.environ.get("SLB_BENCH", ROOT / "data/slb"))
if not BENCH.is_absolute():
    BENCH = ROOT / BENCH
RAW = ROOT / "data/raw"
SPLIT = os.environ.get("SLB_SPLIT", "dev")
RESULTS = ROOT / "results/models" / BENCH.name


def load(split: str) -> pd.DataFrame:
    """train / dev (with labels) or test (inputs only). Never reads hidden/."""
    if split.startswith("test"):
        return pd.read_parquet(BENCH / f"{split}_inputs.parquet")
    return pd.read_parquet(BENCH / f"{split}.parquet")


def contexts() -> pd.DataFrame:
    return pd.read_parquet(BENCH / "contexts.parquet")


def pair_key(a: pd.Series, b: pd.Series) -> pd.Series:
    """Unordered pair key 'A|B' with A<B."""
    a = a.astype(str).values
    b = b.astype(str).values
    lo = np.where(a < b, a, b)
    hi = np.where(a < b, b, a)
    return pd.Series([f"{x}|{y}" for x, y in zip(lo, hi)])


def write(df: pd.DataFrame, name: str, split: str, score_col: str = "score") -> Path:
    """Write results/<...>/<name>_<split>.parquet. Species the model does not score at all get a constant 0.0
    (= AUROC 0.5 in every stratum, the same as median filling; needed because the evaluator refuses files with
    > 50% missing); unscored rows of a partially covered species get that species' median native score.
    Native per-species coverage is written to <name>_<split>.coverage.json."""
    import json
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{name}_{split}.parquet"
    d = df[["example_id", score_col]].rename(columns={score_col: "score"}).astype({"score": "float64"})
    sp = load(split).set_index("example_id").species.reindex(d.example_id).to_numpy()
    cov = {}
    for s_ in pd.unique(sp):
        m = sp == s_
        n = int(d.score[m].notna().sum())
        cov[str(s_)] = {"scored": n, "rows": int(m.sum())}
        if n == 0:
            d.loc[m, "score"] = 0.0
            cov[str(s_)]["constant_fill"] = True
        elif n < m.sum():  # unscored rows of a covered species -> that species' median native score
            d.loc[m & d.score.isna().to_numpy(), "score"] = float(d.score[m].median())
            cov[str(s_)]["median_fill"] = int(m.sum() - n)
    d.to_parquet(out, index=False)
    (RESULTS / f"{name}_{split}.coverage.json").write_text(json.dumps(cov, indent=1))
    print(f"wrote {out}  native coverage " + "; ".join(f"{k}: {v['scored']:,}/{v['rows']:,}" for k, v in cov.items()),
          file=sys.stderr)
    return out


def evaluate(name: str, split: str = "dev") -> None:
    """Run slbench eval on dev and save the report to results/models/<name>_dev.txt (no-op for other splits)."""
    if split != "dev":
        return None
    p = RESULTS / f"{name}_{split}.parquet"
    r = subprocess.run(["uv", "run", "slbench", "eval", str(p), "--split", split, "--allow-missing",
                        "--out", str(RESULTS / f"{name}_{split}.json")],
                       cwd=ROOT, capture_output=True, text=True, env={**os.environ, "SLB_BENCH": str(BENCH)})
    txt = r.stdout + ("\n" + r.stderr if r.returncode else "")
    (RESULTS / f"{name}_{split}.txt").write_text(txt)
    print("\n".join(txt.splitlines()[:4]))
    return None


def coverage(df: pd.DataFrame, score_col: str = "score") -> str:
    lines = []
    for sp, g in df.groupby("species"):
        lines.append(f"{sp}: {g[score_col].notna().sum():,}/{len(g):,}")
    return "; ".join(lines)


def hgnc() -> pd.DataFrame:
    """HGNC complete set: symbol, ensembl_gene_id, entrez_id, uniprot_ids, prev/alias symbols."""
    return pd.read_csv(RAW / "ids/hgnc_complete_set.txt", sep="\t", low_memory=False,
                       usecols=["symbol", "ensembl_gene_id", "entrez_id", "uniprot_ids", "prev_symbol",
                                "alias_symbol", "locus_group"])


def symbol_resolver():
    """Map any (current/previous/alias) human symbol, Ensembl gene id or Entrez id -> current HGNC symbol."""
    h = hgnc()
    m = {}
    for col in ("alias_symbol", "prev_symbol"):
        for s, v in zip(h.symbol, h[col]):
            if isinstance(v, str):
                for x in v.split("|"):
                    m.setdefault(x, s)
    for s, e, z in zip(h.symbol, h.ensembl_gene_id, h.entrez_id):
        m[s] = s
        if isinstance(e, str):
            m[e] = s
        if z == z:
            m[str(int(z))] = s
    return lambda x: m.get(str(x)) if x == x else None
