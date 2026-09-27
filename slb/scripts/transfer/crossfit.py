"""Larger readouts: H0 vs H+ on human pairs, with H+ cross-fitted.

Scopes: `train` scores the human train pairs; `all` scores every scored human pair (train, dev and test),
which is the transfer track's main readout since H0 never sees human labels. In `all`, H+ fold models
train on human dev/test labels: fine inside this track (fixed recipe, no tuning), but these models and
their predictions must never feed the leaderboard.

The primary track (score.py) has about 900 human positives, so its paired H0 − H+ interval is roughly
±0.04 wide: too wide to show non-inferiority at 0.03 even if the true difference is 0. SLB's human train
split has 3,486 positives. H0 never sees human labels, so they are held out for H0 as they are. H+ is
cross-fitted: human train families are hashed into K folds, and a pair whose genes fall in folds {i, j}
is scored by a model trained on every non-human train row plus the human train rows touching neither
fold i nor fold j (K(K+1)/2 models). No human pair's genes or their paralogs are in its own H+ model.

Caveat: the non-human orthologs of these genes are in SLB train (for `all`, removing them would drop
69-78% of yeast train rows), so both arms see yeast labels of orthologous pairs: ortholog-open for both,
equally. `score` also reports pairs with no measured non-human ortholog pair, so lookup cannot pass for
transfer.

    uv run python scripts/transfer/crossfit.py build --scope all
    uv run python scripts/transfer/crossfit.py fit --scope all --pair 0 3     # or --h0
    uv run python scripts/transfer/crossfit.py score --scope all
Bench: data/transfer/_xh[_all][_<bench>]/ (split "xh"; balance weights refitted on it with fitness.propensity).
TRANSFER_BENCH=data/slb_next runs it on a candidate build (outputs suffixed _slb_next).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / os.environ.get("TRANSFER_BENCH", "data/slb")  # e.g. data/slb_next for a candidate build
TAG = "" if SRC.name == "slb" else f"_{SRC.name}"
SCOPE = "train"


def xh() -> Path:
    return ROOT / (f"data/transfer/_xh{TAG}" if SCOPE == "train" else f"data/transfer/_xh_all{TAG}")


def out_dir() -> Path:
    return ROOT / (f"results/transfer/crossfit{TAG}" if SCOPE == "train" else f"results/transfer/crossfit_all{TAG}")


K = 5
MODEL = "gbm_nocode"


def fold_of(families: pd.Series) -> np.ndarray:
    return np.array([int(hashlib.sha256(f.encode()).hexdigest(), 16) % K for f in families])


def human_train() -> pd.DataFrame:
    """Scored human pairs of the scope, with fold ids of both genes' families."""
    t = pd.read_parquet(SRC / "train.parquet")
    parts = [t[t.species == "human"]]
    if SCOPE == "all":  # every scored human pair, including the one-gene-held-out (semi) splits
        for s in ("dev", "test", "dev_semi", "test_semi"):
            x = pd.read_parquet(SRC / f"{s}_inputs.parquet").merge(
                pd.read_parquet(SRC / f"hidden/{s}_labels.parquet"), on="example_id")
            parts.append(x[(x.species == "human") & x.label.notna()][t.columns])
    h = pd.concat(parts).reset_index(drop=True)
    # every pair here is evaluation data, so the benchmark's bar for evaluation labels applies to all of them
    import polars as pl

    from slbench.build import EVAL_MIN_RELIABILITY
    from slbench.evaluate import label_reliability
    r = label_reliability(pl.from_pandas(h[["context_id", "sources"]]), SRC).to_numpy()
    h = h[r >= EVAL_MIN_RELIABILITY].reset_index(drop=True)
    f = pd.read_parquet(SRC / "held_out_families.parquet")
    fam = dict(zip(f[f.species == "human"].gene, f[f.species == "human"].family))
    h["fold_a"], h["fold_b"] = fold_of(h.gene_a.map(fam)), fold_of(h.gene_b.map(fam))
    return h


def build():
    import polars as pl

    from slbench.fitness import propensity, with_degree
    xh().mkdir(parents=True, exist_ok=True)
    (xh() / "hidden").mkdir(exist_ok=True)
    for p in SRC.iterdir():
        if p.name != "hidden" and not (xh() / p.name).exists():
            (xh() / p.name).symlink_to(p.resolve())
    h = human_train()
    cols = ["example_id", "species", "context_id", "ancestry_group", "gene_a", "gene_b", "same_family"]
    h[cols].to_parquet(xh() / "xh_inputs.parquet", index=False)
    h[["example_id", "label", "sources"]].to_parquet(xh() / "hidden/xh_labels.parquet", index=False)
    # SLB balance weights, fitted on this split as the build does for dev/test (row counts here are over
    # scored rows only: train.parquet carries no unscored rows)
    part = pl.from_pandas(h[cols + ["label", "sources"]])
    scored = with_degree(part)
    w = scored.select("example_id").with_columns(propensity(scored, pl.read_parquet(SRC / "contexts.parquet")))
    w.write_parquet(xh() / "hidden/xh_propensity.parquet")
    print(f"xh ({SCOPE}): {len(h):,} human pairs, {int(h.label.sum()):,} positives; "
          f"pairs per fold set: {h.groupby(['fold_a', 'fold_b']).size().to_dict()}")


def fit(pair: tuple[int, int] | None):
    sys.path.insert(0, str(ROOT / "scripts/transfer"))
    import run_model as R
    tr = pd.read_parquet(SRC / "train.parquet")
    h = human_train()
    OUT = out_dir()
    OUT.mkdir(parents=True, exist_ok=True)
    if pair is None:  # H0: no human rows; scores every human pair of the scope
        sc = R.run_gbm(tr[tr.species != "human"], h, MODEL)
        name = "h0"
    else:
        i, j = pair
        out_folds = {i, j}
        touch = h.fold_a.isin(out_folds) | h.fold_b.isin(out_folds)
        keep = pd.concat([tr[tr.species != "human"], h[~touch][tr.columns]])
        target = h[(h[["fold_a", "fold_b"]].min(axis=1) == min(i, j)) & (h[["fold_a", "fold_b"]].max(axis=1) == max(i, j))]
        sc = R.run_gbm(keep, target, MODEL)
        name = f"hplus_{min(i, j)}{max(i, j)}"
        print(f"{name}: human rows kept {int((~touch).sum()):,}, scored {len(target):,}")
    pd.DataFrame({"example_id": sc.index, "score": sc.to_numpy()}).to_parquet(OUT / f"{name}.parquet", index=False)


def score(reps: int):
    XH, OUT = xh(), out_dir()
    os.environ["SLB_BENCH"] = str(XH)
    from slbench import baselines
    from slbench import evaluate as E
    assert E.BENCH == XH
    import polars as pl
    gold, ids = E.load_gold("xh"), E.input_ids("xh")
    ia, ib, n = E._family_index(gold)
    rng = np.random.default_rng(2)
    ws = [E._family_weights(rng.poisson(1.0, n), ia, ib) for _ in range(reps)]
    hp = pd.concat([pd.read_parquet(p) for p in sorted(OUT.glob("hplus_*.parquet"))])
    assert hp.example_id.is_unique and len(hp) == len(ids), (len(hp), len(ids))
    preds = {"h0": pl.read_parquet(OUT / "h0.parquet"), "hplus_crossfit": pl.from_pandas(hp)}
    for b in ("paralog_identity", "codependency", "fitness"):
        preds[b] = baselines.run(b, "xh")
    # ortholog-open lookup (measured GI of non-human ortholog pairs); its in_model flag marks the pairs
    # that have such a measurement, so every entry is also scored on the pairs that have none
    look = pl.read_parquet(OUT / "ortholog_gi_transfer.parquet")
    preds["ortholog_gi_transfer"] = look.select("example_id", "score")
    no_orth = gold.select("example_id").join(look.select("example_id", "in_model"), on="example_id", how="left",
                                             maintain_order="left")["in_model"].fill_null(False).not_()
    res, boots = {}, {}
    for k, p in preds.items():
        df, _ = E.validated_join(gold, p.select("example_id", pl.col("score").cast(pl.Float64)), inputs=ids)
        # all_lines: stratified balanced AUROC over every human context, including lines without an ancestry
        # estimate (HAP1, MCF10A, RPE1), which the ancestry-averaged SLB human score leaves out
        res[k] = {"human": E.species_score(df, "human"), "all_lines": E._auc(df)[0],
                  "human_no_ortholog_pair": E.species_score(df.filter(no_orth), "human"),
                  "paralog_pairs": E._auc(df.filter(pl.col("same_family")))[0],
                  "other_pairs": E._auc(df.filter(~pl.col("same_family")))[0]}
        boots[k] = np.array([E.species_score(df, "human", w) for w in ws])
        res[k]["ci95"] = [float(np.nanpercentile(boots[k], q)) for q in (2.5, 97.5)]
        boots[k + ":all_lines"] = np.array([E._auc(df, w)[0] for w in ws])
        res[k]["all_lines_ci95"] = [float(np.nanpercentile(boots[k + ":all_lines"], q)) for q in (2.5, 97.5)]
    comp = {}
    for a, b in [("h0", "hplus_crossfit")] + [("h0", x) for x in ("paralog_identity", "codependency", "fitness")] \
            + [("hplus_crossfit", x) for x in ("paralog_identity", "codependency", "fitness")]:
        d = boots[a] - boots[b]
        comp[f"{a}_minus_{b}"] = {"delta": res[a]["human"] - res[b]["human"],
                                  "ci95": [float(np.nanpercentile(d, q)) for q in (2.5, 97.5)]}
        d = boots[a + ":all_lines"] - boots[b + ":all_lines"]
        comp[f"{a}_minus_{b}:all_lines"] = {"delta": res[a]["all_lines"] - res[b]["all_lines"],
                                            "ci95": [float(np.nanpercentile(d, q)) for q in (2.5, 97.5)]}
    pos = int(gold.filter(pl.col("ancestry_group") != "unknown")["label"].sum())
    pos_all = int(gold["label"].sum())
    out = {"reps": reps, "scope": SCOPE, "human_pos": pos, "human_pos_all_lines": pos_all,
           "pos_with_ortholog_pair": int(gold.filter(~no_orth)["label"].sum()), "k": K, "model": MODEL, "entries": res, "comparisons": comp}
    (OUT / "report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "fit", "score"])
    ap.add_argument("--pair", type=int, nargs=2)
    ap.add_argument("--h0", action="store_true")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--scope", choices=["train", "all"], default="train")
    a = ap.parse_args()
    global SCOPE
    SCOPE = a.scope
    if a.cmd == "build":
        build()
    elif a.cmd == "fit":
        fit(None if a.h0 else tuple(a.pair))
    else:
        score(a.reps)


if __name__ == "__main__":
    main()
