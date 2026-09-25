"""Reproduce the frozen dev search ledger; this script reads dev labels only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

from slbench import evaluate as E

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "results/models" / E.BENCH.name
BASE = {
    "go": "go_ppi_gbm", "onto": "ontotype", "syn": "synleaf__allspecies",
    "deK": "dekegel2021__allspecies", "musl": "musl__allspecies",
    "slm": "slmgae__allspecies", "mvg": "mvgcn_isl__allspecies",
    "esm": "esm4sl__allspecies", "dep": "depmap_ols__loss",
}


def main() -> None:
    if E.BENCH.name != "slb":
        raise ValueError("this ledger is frozen for the canonical benchmark (data/slb)")
    gold = E.load_gold("dev")
    species = gold["species"].to_numpy()
    ranks = {}
    hashes = {}
    for short, model in BASE.items():
        path = MODEL_DIR / f"{model}_dev.parquet"
        pred = E.read_predictions(path)
        joined, _ = E.validated_join(gold.select("example_id"), pred)
        values = joined["score"].to_numpy()
        r = np.empty(gold.height, dtype=float)
        for sp in np.unique(species):
            mask = species == sp
            r[mask] = rankdata(values[mask], method="average") / mask.sum()
        ranks[short] = r
        hashes[short] = E.file_sha256(path)

    rows = []

    def add(name: str, weights: dict[str, float], parent: str = "", extra: tuple | None = None) -> None:
        x = sum(w * ranks[n] for n, w in weights.items()) / sum(weights.values())
        if extra:
            model, sp, w = extra
            mask = species == sp
            x[mask] = (sum(weights.values()) * x[mask] + w * ranks[model][mask]) / (sum(weights.values()) + w)
        score, parts = E.headline(gold.with_columns(pl.Series("score", x)))
        rows.append({"name": name, "parent": parent, "weights": weights,
                     "extra": {"model": extra[0], "species": extra[1], "weight": extra[2]} if extra else None,
                     "hypothesis": "the added predictor provides complementary pair ranking",
                     "selection": "finalist" if name == "go2+onto+syn+deK" else "screened",
                     "dev_score": score, "species": parts})
        print(f"{name:28s} {score:.5f}", flush=True)

    for name in ("go", "onto", "syn", "deK", "musl", "slm", "mvg", "esm"):
        add(name, {name: 1})
    others = [
        ["onto"], ["syn"], ["deK"], ["musl"], ["onto", "syn"],
        ["onto", "deK"], ["onto", "musl"], ["syn", "deK"],
        ["onto", "syn", "deK"], ["onto", "syn", "deK", "musl"],
    ]
    for os_ in others:
        for gw in (1, 2):
            name = f"go{gw}+" + "+".join(os_)
            add(name, {"go": gw, **{n: 1 for n in os_}}, parent="go")
    core = {"go": 2, "onto": 1, "syn": 1, "deK": 1}
    for name in ("dep", "slm", "mvg"):
        for w in (0.5, 1.0):
            sp = "human" if name == "dep" else "scer,spom"
            # For yeast ablations, add the source on both yeast species, leaving human unchanged.
            x = sum(v * ranks[n] for n, v in core.items()) / sum(core.values())
            for target in sp.split(","):
                mask = species == target
                x[mask] = (sum(core.values()) * x[mask] + w * ranks[name][mask]) / (sum(core.values()) + w)
            score, parts = E.headline(gold.with_columns(pl.Series("score", x)))
            rows.append({"name": f"core+{name}{w}", "parent": "core", "weights": core,
                         "extra": {"model": name, "species": sp, "weight": w},
                         "hypothesis": f"{name} adds ranking signal in {sp}",
                         "selection": "finalist" if name == "dep" and w == 1.0 else "screened",
                         "dev_score": score, "species": parts})
            print(f"core+{name}{w:24g} {score:.5f}", flush=True)
    out = {"benchmark": E.bench_id(), "manifest_sha256": E.file_sha256(E.BENCH / "manifest.json"),
           "scorer_version": E.SCORER_VERSION, "source_predictions_sha256": hashes, "attempts": rows}
    path = ROOT / "reference/dev_rank_ensemble_dev_search.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
