"""Score predictions against SLB.

    uv run slpbench eval preds.parquet --split dev          # hill-climb here
    uv run slpbench eval preds.parquet --split test         # milestones only

Predictions: parquet or csv with columns `example_id` and `score` (higher = more likely SL).
Every example in the split must be scored.

SLB score, the one headline number:
  1. Stratum = context (cell line / strain) x screen. SL pairs are only compared with non-SL pairs
     from the same stratum, so hit-rate differences between cell lines or libraries earn nothing.
  2. Fitness-balanced: e = P(SL | both genes' single-loss effects, screen, context), fitted within
     the split (fitness.propensity). SL pairs are weighted 1 - e and non-SL pairs e (overlap
     weights), rescaled per stratum and class. Weighted SL and non-SL pairs then have the same
     single-gene fitness profile, so predicting "sick genes are SL" scores 0.5: only information
     beyond the two genes' fitness earns credit.
  3. Species score = that balanced AUROC; for human, the mean over ancestry groups with at least
     MIN_GROUP_POS positives. SLB score = mean over species.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import polars as pl

from slpbench.metrics import average_precision, stratified_auc

BENCH = Path(os.environ.get("SLB_BENCH", "data/bench/slb1.2"))
MIN_GROUP_POS = 20
SPECIES = ["human", "scer", "spom", "spne"]
SPECIES_NAMES = {"human": "H. sapiens", "scer": "S. cerevisiae", "spom": "S. pombe", "spne": "S. pneumoniae"}


def load_split(split: str) -> pl.DataFrame:
    if split.startswith("test"):
        x = pl.read_parquet(BENCH / f"{split}_inputs.parquet")
        return x.join(pl.read_parquet(BENCH / "hidden" / f"{split}_labels.parquet"), on="example_id", maintain_order="left")
    return pl.read_parquet(BENCH / f"{split}.parquet")


def load_gold(split: str) -> pl.DataFrame:
    """Split with labels and the SLB balance weight `_bw`: overlap weights (1 - e for SL pairs, e for
    non-SL pairs, e = fitness.propensity), rescaled so each class's weights in a stratum sum to its count."""
    d = load_split(split).join(pl.read_parquet(BENCH / "hidden" / f"{split}_propensity.parquet"),
                               on="example_id", how="left", maintain_order="left")
    e = pl.col("propensity")
    d = d.with_columns(pl.when(pl.col("label") == 1).then(1 - e).otherwise(e).alias("_bw"))
    by = ["context_id", "sources", "label"]
    return d.with_columns((pl.col("_bw") * pl.len().over(by) / pl.col("_bw").sum().over(by)).alias("_bw"))


def read_predictions(path: str | Path) -> pl.DataFrame:
    p = Path(path)
    df = pl.read_parquet(p) if p.suffix == ".parquet" else pl.read_csv(p)
    return df.select("example_id", pl.col("score").cast(pl.Float64))


def _codes(*cols: pl.Series) -> np.ndarray:
    key = pl.DataFrame(list(cols)).select(pl.concat_str(pl.all(), separator="\x1f").alias("k"))["k"]
    return key.cast(pl.Categorical).to_physical().to_numpy()


def _auc(df: pl.DataFrame, w: np.ndarray | None = None, balanced: bool = True) -> tuple[float, float]:
    ww = df["_bw"].to_numpy() if balanced else np.ones(df.height)
    if w is not None:
        ww = ww * w
    return stratified_auc(_codes(df["context_id"], df["sources"]), df["label"].to_numpy(), df["score"].to_numpy(), ww)


def _within_gene_auc(df: pl.DataFrame) -> tuple[float, float]:
    """Balanced AUROC stratified by (context, screen, gene): each example sits in two strata, one per gene."""
    both = pl.concat([
        df.select("context_id", "sources", pl.col("gene_a").alias("g"), "label", "score", "_bw"),
        df.select("context_id", "sources", pl.col("gene_b").alias("g"), "label", "score", "_bw"),
    ])
    return stratified_auc(_codes(both["context_id"], both["sources"], both["g"]), both["label"].to_numpy(),
                          both["score"].to_numpy(), both["_bw"].to_numpy())


def headline(df: pl.DataFrame, w: np.ndarray | None = None) -> tuple[float, dict]:
    parts = {}
    if w is not None:
        df = df.with_columns(pl.Series("_w", w))
    for sp in SPECIES:
        d = df.filter(pl.col("species") == sp)
        if d.height == 0:
            continue
        if sp == "human":
            groups = []
            for g, dg in d.partition_by("ancestry_group", as_dict=True).items():
                if (dg["label"] == 1).sum() >= MIN_GROUP_POS:
                    groups.append(_auc(dg, dg["_w"].to_numpy() if w is not None else None)[0])
            parts[sp] = float(np.nanmean(groups))
        else:
            parts[sp] = _auc(d, d["_w"].to_numpy() if w is not None else None)[0]
    return float(np.nanmean(list(parts.values()))), parts


def _row(name: str, d: pl.DataFrame) -> dict:
    y = d["label"].to_numpy()
    wg, wn = _within_gene_auc(d)
    prev = y.mean() if len(y) else float("nan")
    return {
        "stratum": name, "n": d.height, "pos": int(y.sum()), "slb_auroc": _auc(d)[0],
        "within_gene": wg if wn else float("nan"), "unadjusted_auroc": _auc(d, balanced=False)[0],
        "ap_lift": average_precision(y, d["score"].to_numpy()) / prev if prev > 0 else float("nan"),
    }


def _family_index(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    fams = pl.read_parquet(BENCH / "held_out_families.parquet").select("species", "gene", "family")
    key = df.select("example_id", "species", "gene_a", "gene_b") \
        .join(fams.rename({"gene": "gene_a", "family": "fa"}), on=["species", "gene_a"], how="left", maintain_order="left") \
        .join(fams.rename({"gene": "gene_b", "family": "fb"}), on=["species", "gene_b"], how="left", maintain_order="left")
    assert key["example_id"].equals(df["example_id"])
    uf = pl.concat([key["fa"], key["fb"]]).unique().to_list()
    idx = {f: i for i, f in enumerate(uf)}
    return np.array([idx[f] for f in key["fa"]]), np.array([idx[f] for f in key["fb"]]), len(uf)


def bootstrap(df: pl.DataFrame, reps: int, seed: int = 0) -> tuple[float, float]:
    """Cluster bootstrap over gene families: an example's weight is the product of its two
    families' (Poisson) resample counts."""
    ia, ib, n = _family_index(df)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        c = rng.poisson(1.0, n)
        vals.append(headline(df, (c[ia] * c[ib]).astype(float))[0])
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def compare(pa: pl.DataFrame, pb: pl.DataFrame, split: str, reps: int = 200, seed: int = 0) -> dict:
    """Paired family-cluster bootstrap of SLB(B) - SLB(A) on the same resamples."""
    gold = load_gold(split)
    da = gold.join(pa, on="example_id", how="left", maintain_order="left")
    db = gold.join(pb, on="example_id", how="left", maintain_order="left")
    if da["score"].null_count() or db["score"].null_count():
        raise SystemExit("both prediction files must score every example")
    sa, pa_parts = headline(da)
    sb, pb_parts = headline(db)
    ia, ib, n = _family_index(gold)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(reps):
        c = rng.poisson(1.0, n)
        w = (c[ia] * c[ib]).astype(float)
        deltas.append(headline(db, w)[0] - headline(da, w)[0])
    deltas = np.array(deltas)
    return {"split": split, "a": sa, "b": sb, "delta": sb - sa,
            "delta_ci95": [float(np.nanpercentile(deltas, 2.5)), float(np.nanpercentile(deltas, 97.5))],
            "p_b_not_better": float(np.mean(deltas <= 0)),
            "species_delta": {k: pb_parts[k] - pa_parts[k] for k in pa_parts}}


def evaluate(preds: pl.DataFrame, split: str, boot: int = 0, allow_missing: bool = False) -> dict:
    df = load_gold(split).join(preds, on="example_id", how="left", maintain_order="left")
    missing = df["score"].null_count() + df["score"].is_nan().sum()
    if missing:
        if not allow_missing:
            raise SystemExit(f"{missing:,} of {df.height:,} {split} examples have no score (use --allow-missing)")
        df = df.with_columns(pl.col("score").fill_nan(None).fill_null(pl.col("score").median()))
    score, parts = headline(df)
    res = {"benchmark": BENCH.name, "split": split, "slb_score": score, "species_scores": parts, "n": df.height,
           "missing_filled": int(missing), "strata": []}
    if boot:
        res["slb_score_ci95"] = bootstrap(df, boot)
    res["strata"].append(_row("ALL (flat)", df))
    for sp in SPECIES:
        d = df.filter(pl.col("species") == sp)
        if d.height:
            res["strata"].append(_row(f"species={sp}", d))
    h = df.filter(pl.col("species") == "human")
    for g in sorted(h["ancestry_group"].unique().to_list()):
        res["strata"].append(_row(f"human ancestry={g}", h.filter(pl.col("ancestry_group") == g)))
    for fam, lab in [(True, "same family (paralogs)"), (False, "different families")]:
        d = df.filter(pl.col("same_family") == fam)
        if d.height:
            res["strata"].append(_row(f"pair={lab}", d))
    for c in sorted(h["context_id"].unique().to_list()):
        d = h.filter(pl.col("context_id") == c)
        if (d["label"] == 1).sum() >= 5:
            res["strata"].append(_row(f"context={c}", d))
    return res


def format_report(res: dict) -> str:
    lines = [f"{res['benchmark']} {res['split']}  n={res['n']:,}"]
    ci = res.get("slb_score_ci95")
    lines.append(f"SLB score: {res['slb_score']:.4f}" + (f"  (95% CI {ci[0]:.4f}–{ci[1]:.4f})" if ci else ""))
    lines.append("  " + "  ".join(f"{SPECIES_NAMES[k]}={v:.4f}" for k, v in res["species_scores"].items()))
    if res["missing_filled"]:
        lines.append(f"  WARNING: {res['missing_filled']:,} missing scores filled with the median")
    lines.append("\nper stratum: SLB = fitness-balanced AUROC (what the score uses); wg = within-gene SLB;"
                 " unadj = AUROC without fitness balancing (diagnostic)")
    lines.append(f"{'stratum':44s} {'n':>9s} {'pos':>7s} {'SLB':>7s} {'wg':>7s} {'unadj':>7s} {'AP×':>6s}")
    for r in res["strata"]:
        lines.append(f"{r['stratum'][:44]:44s} {r['n']:>9,d} {r['pos']:>7,d} {r['slb_auroc']:>7.4f} "
                     f"{r['within_gene']:>7.4f} {r['unadjusted_auroc']:>7.4f} {r['ap_lift']:>6.2f}")
    return "\n".join(lines)


def save(res: dict, path: Path) -> None:
    path.write_text(json.dumps(res, indent=2, default=float))
