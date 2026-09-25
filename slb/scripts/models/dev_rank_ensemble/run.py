"""Frozen dev-selected rank blend; consumes only base-model predictions.

Core weights are GO/PPI GBM 2, Ontotype 1, SynLeaF KG 1, De Kegel RF 1.
The loss variant adds DepMap OLS at weight 1 in human only. Scores are percentile
ranks within species before blending; this uses no labels from the target split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(os.environ.get("SLB_BENCH", "data/slb"))
if not BENCH.is_absolute():
    BENCH = ROOT / BENCH
SOURCES = ("go_ppi_gbm", "ontotype", "synleaf__allspecies", "dekegel2021__allspecies")
WEIGHTS = (2.0, 1.0, 1.0, 1.0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def blend(split: str, variant: str) -> tuple[pl.DataFrame, dict]:
    if BENCH.name != "slb":
        raise ValueError("this frozen recipe was selected on the canonical benchmark (data/slb)")
    path = BENCH / ("train.parquet" if split == "train" else f"{split}_inputs.parquet")
    d = pl.read_parquet(path).select("example_id", "species")
    ids, species = d["example_id"], d["species"].to_numpy()
    src = list(SOURCES) + (["depmap_ols__loss"] if variant == "loss" else [])
    ranks = {}
    hashes = {}
    for name in src:
        p = ROOT / "results/models" / BENCH.name / f"{name}_{split}.parquet"
        if not p.is_file():
            raise FileNotFoundError(f"base-model predictions required: {p}")
        f = pl.read_parquet(p).select("example_id", pl.col("score").cast(pl.Float64))
        if f.height != d.height or f["example_id"].null_count() or f["example_id"].n_unique() != f.height:
            raise ValueError(f"{p} must have one unique score for every example")
        x = d.select("example_id").join(f, on="example_id", how="left", validate="1:1", maintain_order="left")
        values = x["score"].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{p} has missing or non-finite scores")
        r = np.empty(len(values), dtype=float)
        for sp in np.unique(species):
            mask = species == sp
            r[mask] = rankdata(values[mask], method="average") / mask.sum()
        ranks[name] = r
        hashes[name] = sha256(p)
    core = sum(w * ranks[n] for n, w in zip(SOURCES, WEIGHTS)) / sum(WEIGHTS)
    if variant == "loss":
        mask = species == "human"
        core[mask] = (sum(WEIGHTS) * core[mask] + ranks["depmap_ols__loss"][mask]) / (sum(WEIGHTS) + 1)
    out = pl.DataFrame({"example_id": ids, "score": core})
    recipe = {"benchmark_manifest_sha256": sha256(BENCH / "manifest.json"), "split": split,
              "variant": variant, "core_weights": dict(zip(SOURCES, WEIGHTS)),
              "human_extra": {"depmap_ols__loss": 1.0} if variant == "loss" else {},
              "normalization": "mean percentile rank per species", "input_sha256": hashes}
    return out, recipe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=os.environ.get("SLB_SPLIT", "dev"), choices=("dev", "test"))
    ap.add_argument("--variant", default="core", choices=("core", "loss"))
    a = ap.parse_args()
    pred, recipe = blend(a.split, a.variant)
    dest = ROOT / "results/models" / BENCH.name / f"dev_rank_ensemble__{a.variant}_{a.split}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    pred.write_parquet(dest)
    dest.with_suffix(".recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    print(f"wrote {dest} ({pred.height:,} rows)")


if __name__ == "__main__":
    main()
