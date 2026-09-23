"""Score predictions against SLB-1.

    uv run slpbench eval preds.parquet --split dev          # hill-climb here
    uv run slpbench eval preds.parquet --split test         # milestones only

Predictions: parquet or csv with columns `example_id` and `score` (higher = more likely SL).
Every example in the split must be scored.

Headline (SLB score): the mean over species of a species score. A species score is the
context-stratified AUROC; for human it is the mean over ancestry groups (with at least
MIN_GROUP_POS positives) of the context-stratified AUROC within the group.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from slpbench.metrics import average_precision, stratified_auc

BENCH = Path("data/bench/slb1")
MIN_GROUP_POS = 10
SPECIES = ["human", "scer", "spom", "dmel", "spne"]
SPECIES_NAMES = {"human": "H. sapiens", "scer": "S. cerevisiae", "spom": "S. pombe",
                 "dmel": "D. melanogaster", "spne": "S. pneumoniae"}


def load_split(split: str) -> pl.DataFrame:
    if split.startswith("test"):
        x = pl.read_parquet(BENCH / f"{split}_inputs.parquet")
        return x.join(pl.read_parquet(BENCH / "hidden" / f"{split}_labels.parquet"), on="example_id", maintain_order="left")
    return pl.read_parquet(BENCH / f"{split}.parquet")


def read_predictions(path: str | Path) -> pl.DataFrame:
    p = Path(path)
    df = pl.read_parquet(p) if p.suffix == ".parquet" else pl.read_csv(p)
    return df.select("example_id", pl.col("score").cast(pl.Float64))


def _codes(*cols: pl.Series) -> np.ndarray:
    key = pl.DataFrame(list(cols)).select(pl.concat_str(pl.all(), separator="\x1f").alias("k"))["k"]
    return key.cast(pl.Categorical).to_physical().to_numpy()


def _auc(df: pl.DataFrame, w: np.ndarray | None = None, matched: bool = False) -> tuple[float, float]:
    keys = [df["context_id"]] + ([df["fbin_lo"].cast(pl.String), df["fbin_hi"].cast(pl.String)] if matched else [])
    return stratified_auc(_codes(*keys), df["label"].to_numpy(), df["score"].to_numpy(), w)


def attach_fitness_bins(df: pl.DataFrame) -> pl.DataFrame:
    """Add fbin_lo/fbin_hi: the two genes' single-loss-effect quintiles (unordered; 0 = unknown)."""
    g = pl.read_parquet(BENCH / "gene_single_effects.parquet").select("species", "gene", "fitness_bin")
    df = df.join(g.rename({"gene": "gene_a", "fitness_bin": "_ba"}), on=["species", "gene_a"], how="left") \
           .join(g.rename({"gene": "gene_b", "fitness_bin": "_bb"}), on=["species", "gene_b"], how="left")
    return df.with_columns(
        pl.min_horizontal(pl.col("_ba").fill_null(0), pl.col("_bb").fill_null(0)).alias("fbin_lo"),
        pl.max_horizontal(pl.col("_ba").fill_null(0), pl.col("_bb").fill_null(0)).alias("fbin_hi"),
    ).drop("_ba", "_bb")


def _within_gene_auc(df: pl.DataFrame) -> tuple[float, float]:
    """Stratify by (context, gene): each example sits in two strata, one per gene."""
    both = pl.concat([
        df.select("context_id", pl.col("gene_a").alias("g"), "label", "score"),
        df.select("context_id", pl.col("gene_b").alias("g"), "label", "score"),
    ])
    return stratified_auc(_codes(both["context_id"], both["g"]), both["label"].to_numpy(), both["score"].to_numpy())


def headline(df: pl.DataFrame, w: np.ndarray | None = None, matched: bool = False) -> tuple[float, dict]:
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
                    groups.append(_auc(dg, dg["_w"].to_numpy() if w is not None else None, matched)[0])
            parts[sp] = float(np.nanmean(groups))
        else:
            parts[sp] = _auc(d, d["_w"].to_numpy() if w is not None else None, matched)[0]
    return float(np.nanmean(list(parts.values()))), parts


def _row(name: str, d: pl.DataFrame) -> dict:
    y = d["label"].to_numpy()
    auc, _ = _auc(d)
    fm, _ = _auc(d, matched=True)
    wg, wn = _within_gene_auc(d)
    prev = y.mean() if len(y) else float("nan")
    return {
        "stratum": name, "n": d.height, "pos": int(y.sum()), "auroc": auc, "fitness_matched_auroc": fm,
        "within_gene_auroc": wg if wn else float("nan"),
        "ap_lift": average_precision(y, d["score"].to_numpy()) / prev if prev > 0 else float("nan"),
    }


def bootstrap(df: pl.DataFrame, reps: int, seed: int = 0) -> tuple[float, float]:
    """Cluster bootstrap over gene families: an example's weight is the product of its two
    families' resample counts."""
    rng = np.random.default_rng(seed)
    fams = pl.read_parquet(BENCH / "held_out_families.parquet").select("species", "gene", "family")
    d = df.join(fams.rename({"gene": "gene_a", "family": "fa"}), on=["species", "gene_a"]) \
          .join(fams.rename({"gene": "gene_b", "family": "fb"}), on=["species", "gene_b"])
    uf = pl.concat([d["fa"], d["fb"]]).unique()
    idx = {f: i for i, f in enumerate(uf.to_list())}
    ia = np.array([idx[f] for f in d["fa"]])
    ib = np.array([idx[f] for f in d["fb"]])
    vals = []
    for _ in range(reps):
        c = rng.poisson(1.0, len(uf))  # Poisson bootstrap ~ multinomial resampling
        vals.append(headline(d, (c[ia] * c[ib]).astype(float))[0])
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def compare(pa: pl.DataFrame, pb: pl.DataFrame, split: str, reps: int = 200, seed: int = 0) -> dict:
    """Paired family-cluster bootstrap of SLB(B) - SLB(A) on the same resamples."""
    gold = attach_fitness_bins(load_split(split))
    da = gold.join(pa, on="example_id", how="left")
    db = gold.join(pb, on="example_id", how="left")
    if da["score"].null_count() or db["score"].null_count():
        raise SystemExit("both prediction files must score every example")
    sa, pa_parts = headline(da)
    sb, pb_parts = headline(db)
    ma, _ = headline(da, matched=True)
    mb, _ = headline(db, matched=True)
    fams = pl.read_parquet(BENCH / "held_out_families.parquet").select("species", "gene", "family")
    key = gold.join(fams.rename({"gene": "gene_a", "family": "fa"}), on=["species", "gene_a"]) \
              .join(fams.rename({"gene": "gene_b", "family": "fb"}), on=["species", "gene_b"])
    assert key["example_id"].equals(gold["example_id"])
    uf = pl.concat([key["fa"], key["fb"]]).unique().to_list()
    idx = {f: i for i, f in enumerate(uf)}
    ia = np.array([idx[f] for f in key["fa"]])
    ib = np.array([idx[f] for f in key["fb"]])
    rng = np.random.default_rng(seed)
    deltas, mdeltas = [], []
    for _ in range(reps):
        c = rng.poisson(1.0, len(uf))
        w = (c[ia] * c[ib]).astype(float)
        deltas.append(headline(db, w)[0] - headline(da, w)[0])
        mdeltas.append(headline(db, w, True)[0] - headline(da, w, True)[0])
    deltas, mdeltas = np.array(deltas), np.array(mdeltas)
    return {"split": split, "a": sa, "b": sb, "delta": sb - sa,
            "delta_ci95": [float(np.nanpercentile(deltas, 2.5)), float(np.nanpercentile(deltas, 97.5))],
            "p_b_not_better": float(np.mean(deltas <= 0)),
            "matched_a": ma, "matched_b": mb, "matched_delta": mb - ma,
            "matched_delta_ci95": [float(np.nanpercentile(mdeltas, 2.5)), float(np.nanpercentile(mdeltas, 97.5))],
            "species_delta": {k: pb_parts[k] - pa_parts[k] for k in pa_parts}}


def evaluate(preds: pl.DataFrame, split: str, boot: int = 0, allow_missing: bool = False) -> dict:
    gold = attach_fitness_bins(load_split(split))
    df = gold.join(preds, on="example_id", how="left")
    missing = df["score"].null_count() + df["score"].is_nan().sum()
    if missing:
        if not allow_missing:
            raise SystemExit(f"{missing:,} of {df.height:,} {split} examples have no score (use --allow-missing)")
        df = df.with_columns(pl.col("score").fill_nan(None).fill_null(pl.col("score").median()))
    score, parts = headline(df)
    mscore, mparts = headline(df, matched=True)
    res = {"split": split, "slb_score": score, "species_scores": parts,
           "slb_matched": mscore, "species_matched": mparts, "n": df.height,
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
    lines = [f"SLB-1 {res['split']}  n={res['n']:,}"]
    ci = res.get("slb_score_ci95")
    lines.append(f"SLB score: {res['slb_score']:.4f}" + (f"  (95% CI {ci[0]:.4f}–{ci[1]:.4f})" if ci else ""))
    lines.append("  " + "  ".join(f"{SPECIES_NAMES[k]}={v:.4f}" for k, v in res["species_scores"].items()))
    lines.append(f"SLB fitness-matched: {res['slb_matched']:.4f}")
    lines.append("  " + "  ".join(f"{SPECIES_NAMES[k]}={v:.4f}" for k, v in res["species_matched"].items()))
    if res["missing_filled"]:
        lines.append(f"  WARNING: {res['missing_filled']:,} missing scores filled with the median")
    lines.append(f"\n{'stratum':44s} {'n':>9s} {'pos':>7s} {'AUROC':>7s} {'fmAUROC':>8s} {'wgAUROC':>8s} {'AP×':>6s}")
    for r in res["strata"]:
        lines.append(f"{r['stratum'][:44]:44s} {r['n']:>9,d} {r['pos']:>7,d} {r['auroc']:>7.4f} "
                     f"{r['fitness_matched_auroc']:>8.4f} {r['within_gene_auroc']:>8.4f} {r['ap_lift']:>6.2f}")
    return "\n".join(lines)


def save(res: dict, path: Path) -> None:
    path.write_text(json.dumps(res, indent=2, default=float))
