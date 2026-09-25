"""Bundle I/O helpers for the battery (models-mechanistic): split I/O + data/interim/bundle/<species>/ loaders.

Environment: SLB_BENCH (default data/slb), SLB_SPLIT (default dev), SLB_BUNDLE (default
data/interim/bundle). Pure pandas/numpy/pyarrow so it works in every model venv.
"""

from __future__ import annotations

import os
from collections import defaultdict
from functools import cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(os.environ.get("SLB_BENCH", ROOT / "data/slb"))
if not BENCH.is_absolute():
    BENCH = ROOT / BENCH
SPLIT = os.environ.get("SLB_SPLIT", "dev")
BUNDLE = Path(os.environ.get("SLB_BUNDLE", ROOT / "data/interim/bundle"))
RESULTS = Path(os.environ["SLB_OUT"]) if os.environ.get("SLB_OUT") else (
    ROOT / "results/models" / BENCH.name)
if not RESULTS.is_absolute():
    RESULTS = ROOT / RESULTS


def split_path(split: str = SPLIT) -> Path:
    p = BENCH / f"{split}.parquet"
    return p if p.exists() else BENCH / f"{split}_inputs.parquet"


def load_split(split: str = SPLIT, columns=None) -> pd.DataFrame:
    return pd.read_parquet(split_path(split), columns=columns)


def load_train() -> pd.DataFrame:
    t = pd.read_parquet(BENCH / "train.parquet")
    return t[t.label.notna()]


def out_path(name: str, split: str = SPLIT, variant: str | None = None) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    return RESULTS / (f"{name}__{variant}_{split}.parquet" if variant else f"{name}_{split}.parquet")


def has_bundle(species: str, part: str) -> bool:
    return (BUNDLE / species / f"{part}.parquet").exists()


@cache
def go_dag() -> tuple[dict, dict]:
    """(parents: term -> set(parent terms), namespace: term -> namespace)."""
    e = pd.read_parquet(BUNDLE / "_go/edges.parquet")
    t = pd.read_parquet(BUNDLE / "_go/terms.parquet")
    par = defaultdict(set)
    for c, p in zip(e.child, e.parent):
        par[c].add(p)
    return dict(par), dict(zip(t.term, t.namespace))


def go_direct(species: str) -> dict[str, set]:
    g = pd.read_parquet(BUNDLE / species / "go.parquet", columns=["gene", "term"])
    out = defaultdict(set)
    for a, t in zip(g.gene, g.term):
        out[a].add(t)
    return dict(out)


def go_propagated(species: str) -> dict[str, set]:
    par, _ = go_dag()
    memo: dict[str, frozenset] = {}

    def anc(t):
        if t not in memo:
            s = {t}
            for p in par.get(t, ()):
                s |= anc(p)
            memo[t] = frozenset(s)
        return memo[t]

    out = {}
    for g, ts in go_direct(species).items():
        s = set()
        for t in ts:
            s |= anc(t)
        out[g] = s
    return out


def fitness(species: str) -> dict[str, float]:
    if not has_bundle(species, "fitness"):
        return {}
    f = pd.read_parquet(BUNDLE / species / "fitness.parquet")
    return dict(zip(f.gene, f.effect))


def ppi(species: str) -> pd.DataFrame:
    return pd.read_parquet(BUNDLE / species / "ppi.parquet") if has_bundle(species, "ppi") else pd.DataFrame(
        columns=["gene_a", "gene_b"])
