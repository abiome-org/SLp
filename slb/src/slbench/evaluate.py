"""Score predictions against SLB.

    uv run slbench eval preds.parquet --split dev          # hill-climb here
    uv run slbench eval preds.parquet --split test         # milestones only

Predictions: parquet or csv with columns `example_id` and `score` (higher = more likely SL).
Every example in the split must be scored.

SLB score, the one headline number:
  1. Stratum = context (cell line / strain) x screen. SL pairs are only compared with non-SL pairs
     from the same stratum, so hit-rate differences between cell lines or libraries earn nothing.
  2. Fitness-balanced: e = P(SL | both genes' single-loss effects, screen, context), fitted within
     the split (fitness.propensity). SL pairs are weighted 1 - e and non-SL pairs e (overlap
     weights), rescaled per stratum and class. Weighted SL and non-SL pairs then have the same
     fitted single-gene fitness design means; a flexible fitness-only model can retain residual
     signal, measured by the fitness_lgbm control.
  3. Species score = that balanced AUROC; for human, the mean over ancestry groups with at least
     MIN_GROUP_POS positives. SLB score = mean over the benchmark's headline species (manifest.json
     "headline_species"). Auxiliary species ("auxiliary_species": too few test positives or labels
     verified only within their own study) are scored and reported the same way, but not averaged in.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import polars as pl

from slbench.metrics import average_precision, stratified_auc

BENCH = Path(os.environ.get("SLB_BENCH", "data/slb"))
MIN_GROUP_POS = 20
SCORER_VERSION = "1.3.2"
REPORTS = Path("results/reports")  # generated markdown reports (gitignored)
SPECIES_NAMES = {"human": "H. sapiens", "scer": "S. cerevisiae", "spom": "S. pombe", "spne": "S. pneumoniae",
                 "dmel": "D. melanogaster", "cele": "C. elegans", "mmus": "M. musculus", "bsub": "B. subtilis",
                 "ecol": "E. coli"}


def bench_id(bench: Path | None = None) -> str:
    """Benchmark revision from manifest.json. Results record it; the directory name does not matter."""
    return json.loads(((bench or BENCH) / "manifest.json").read_text())["version"]


def write_report(name: str, text: str) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (path := REPORTS / name).write_text(text)
    return path


def species_tiers() -> tuple[list[str], list[str]]:
    """(headline, auxiliary) species of the benchmark."""
    m = json.loads((BENCH / "manifest.json").read_text())
    return m["headline_species"], m["auxiliary_species"]


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


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validated_join(gold: pl.DataFrame, preds: pl.DataFrame, allow_missing: bool = False,
                   max_missing_fraction: float = 0.5) -> tuple[pl.DataFrame, int]:
    """Validate a submission before joining, so it cannot change the evaluated row set."""
    if preds["example_id"].null_count():
        raise ValueError("prediction example_id contains null values")
    if preds["example_id"].n_unique() != preds.height:
        raise ValueError("prediction example_id contains duplicates")
    unknown = preds.join(gold.select("example_id"), on="example_id", how="anti")
    if unknown.height:
        raise ValueError(f"predictions contain {unknown.height:,} unknown example_id values")
    if np.isinf(preds["score"].to_numpy()).any():
        raise ValueError("predictions contain infinite scores")
    df = gold.join(preds, on="example_id", how="left", maintain_order="left", validate="1:1")
    if df.height != gold.height or not df["example_id"].equals(gold["example_id"]):
        raise ValueError("prediction join changed the gold row set")
    missing = int(df["score"].null_count() + df["score"].is_nan().sum())
    if missing and not allow_missing:
        raise ValueError(f"{missing:,} of {df.height:,} examples have no finite score (use --allow-missing)")
    if missing > df.height * max_missing_fraction:
        raise ValueError(f"{missing:,} of {df.height:,} examples have no score; check the benchmark version")
    if missing == df.height:
        raise ValueError("predictions contain no finite scores")
    if missing:
        df = df.with_columns(pl.col("score").fill_nan(None).fill_null(pl.col("score").median()))
    return df, missing


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


def species_score(d: pl.DataFrame, sp: str, w: np.ndarray | None = None) -> float:
    """Balanced AUROC; human = mean over ancestry groups. NaN when fewer than MIN_GROUP_POS positives."""
    if sp != "human":
        return _auc(d, w)[0] if (d["label"] == 1).sum() >= MIN_GROUP_POS else float("nan")
    groups = []
    for g, dg in d.with_row_index("_r").partition_by("ancestry_group", as_dict=True).items():
        if g[0] == "unknown":
            continue
        if (dg["label"] == 1).sum() >= MIN_GROUP_POS:
            groups.append(_auc(dg, w[dg["_r"].to_numpy()] if w is not None else None)[0])
    return float(np.nanmean(groups))


def headline(df: pl.DataFrame, w: np.ndarray | None = None, species: list[str] | None = None) -> tuple[float, dict]:
    """Mean species score over `species` (default: the headline species)."""
    parts = {}
    for sp in species if species is not None else species_tiers()[0]:
        m = (df["species"] == sp).to_numpy()
        if m.any():
            parts[sp] = species_score(df.filter(pl.Series(m)), sp, w[m] if w is not None else None)
    vals = [v for v in parts.values() if not np.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), parts


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
    # `unique()` has process-dependent order. Assign family indices canonically
    # so the fixed bootstrap seed selects the same families on every run.
    uf = pl.concat([key["fa"], key["fb"]]).unique().sort().to_list()
    idx = {f: i for i, f in enumerate(uf)}
    return np.array([idx[f] for f in key["fa"]]), np.array([idx[f] for f in key["fb"]]), len(uf)


def _family_weights(counts: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
    """Resample a same-family pair once, rather than squaring its family count."""
    return np.where(ia == ib, counts[ia], counts[ia] * counts[ib]).astype(float)


def bootstrap(df: pl.DataFrame, reps: int, seed: int = 0) -> tuple[float, float]:
    """Cluster bootstrap over gene families: an example's weight is the product of its two
    families' (Poisson) resample counts."""
    ia, ib, n = _family_index(df)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        c = rng.poisson(1.0, n)
        vals.append(headline(df, _family_weights(c, ia, ib))[0])
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def compare(pa: pl.DataFrame, pb: pl.DataFrame, split: str, reps: int = 200, seed: int = 0) -> dict:
    """Paired family-cluster bootstrap of SLB(B) - SLB(A) on the same resamples."""
    gold = load_gold(split)
    da, _ = validated_join(gold, pa)
    db, _ = validated_join(gold, pb)
    sa, pa_parts = headline(da)
    sb, pb_parts = headline(db)
    ia, ib, n = _family_index(gold)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(reps):
        c = rng.poisson(1.0, n)
        w = _family_weights(c, ia, ib)
        deltas.append(headline(db, w)[0] - headline(da, w)[0])
    deltas = np.array(deltas)
    return {"split": split, "a": sa, "b": sb, "delta": sb - sa,
            "delta_ci95": [float(np.nanpercentile(deltas, 2.5)), float(np.nanpercentile(deltas, 97.5))],
            "p_b_not_better": float(np.mean(deltas <= 0)),
            "species_delta": {k: pb_parts[k] - pa_parts[k] for k in pa_parts}}


def evaluate(preds: pl.DataFrame, split: str, boot: int = 0, allow_missing: bool = False) -> dict:
    df, missing = validated_join(load_gold(split), preds, allow_missing)
    score, parts = headline(df)
    aux = headline(df, species=species_tiers()[1])[1]
    res = {"benchmark": bench_id(), "manifest_sha256": file_sha256(BENCH / "manifest.json"),
           "scorer_version": SCORER_VERSION, "split": split, "slb_score": score, "species_scores": parts,
           "auxiliary_species_scores": aux, "n": df.height, "missing_filled": int(missing), "strata": []}
    if boot:
        res["slb_score_ci95"] = bootstrap(df, boot)
    res["strata"].append(_row("ALL (flat)", df))
    for sp in (*species_tiers()[0], *species_tiers()[1]):
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
    if res.get("auxiliary_species_scores"):
        lines.append("auxiliary (not in SLB score): " + "  ".join(
            f"{SPECIES_NAMES[k]}=" + ("n/a (<20 pos)" if np.isnan(v) else f"{v:.4f}")
            for k, v in res["auxiliary_species_scores"].items()))
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
    def clean(value):
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    path.write_text(json.dumps(clean(res), indent=2, allow_nan=False))
