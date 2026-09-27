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
  3. Noise ceiling: labels are measurements, and an independent re-measurement recovers them only at
     AUROC < 1 (each source's reliability, label_reliability.parquet, from reference/label_reliability.tsv).
     The ceiling of a set of strata is their reliability averaged with the same pair weights as the AUROC.
     Normalized score = (AUROC - 0.5) / (ceiling - 0.5): 0 = chance, 1 = as good as re-running the
     experiment. A model of the true biology can reach 1 and can exceed it, since a single re-measurement is
     itself noisy; only predicting the measurement noise of these screens would take it much further.
  4. Species score = normalized score; for human, the mean over ancestry groups (lines without an
     ancestry estimate form their own group, "unknown") with at least MIN_GROUP_POS positives, each group
     normalized by its own ceiling. SLB score = mean over the headline species (manifest.json
     "headline_species"). Auxiliary species ("auxiliary_species") are scored and reported the same way, but
     not averaged in. Raw balanced AUROCs and ceilings are reported alongside (slb_auroc, species_auroc,
     ceilings).
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
SCORER_VERSION = "3.0.0"
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


def inputs_path(split: str, bench: Path | None = None) -> Path:
    """The model-facing file of a split: train.parquet (labelled) or <split>_inputs.parquet (no labels)."""
    bench = bench or BENCH
    return bench / ("train.parquet" if split == "train" else f"{split}_inputs.parquet")


def load_split(split: str) -> pl.DataFrame:
    """A split with its labels (null = measured but unscored). Evaluation machinery: reads hidden/."""
    if split == "train":
        return pl.read_parquet(inputs_path(split))
    x = pl.read_parquet(inputs_path(split))
    return x.join(pl.read_parquet(BENCH / "hidden" / f"{split}_labels.parquet"), on="example_id", maintain_order="left")


def load_gold(split: str) -> pl.DataFrame:
    """Scored rows of a split with labels and the SLB balance weight `_bw`: overlap weights (1 - e for SL
    pairs, e for non-SL pairs, e = fitness.propensity), rescaled so each class's weights in a stratum sum to
    its count. Unscored (measured but ambiguous) rows are inputs only."""
    d = load_split(split).filter(pl.col("label").is_not_null()).join(pl.read_parquet(BENCH / "hidden" / f"{split}_propensity.parquet"),
                               on="example_id", how="left", maintain_order="left")
    e = pl.col("propensity")
    d = d.with_columns(pl.when(pl.col("label") == 1).then(1 - e).otherwise(e).alias("_bw"))
    by = ["context_id", "sources", "label"]
    d = d.with_columns((pl.col("_bw") * pl.len().over(by) / pl.col("_bw").sum().over(by)).alias("_bw"))
    return d.with_columns(label_reliability(d).alias("_rel"))


def label_reliability(d: pl.DataFrame, bench: Path | None = None) -> pl.Series:
    """Per row: reliability of its most reliable labelling source (`sources`), context-specific where given."""
    rel = pl.read_parquet((bench or BENCH) / "label_reliability.parquet")
    long = d.select(pl.int_range(pl.len()).alias("_i"), "context_id", pl.col("sources").str.split(",").alias("source")) \
        .explode("source")
    long = long.join(rel.filter(pl.col("context_id").is_not_null()).rename({"reliability": "r_ctx"}),
                     on=["source", "context_id"], how="left") \
        .join(rel.filter(pl.col("context_id").is_null()).drop("context_id").rename({"reliability": "r_src"}),
              on="source", how="left")
    best = long.group_by("_i").agg(pl.coalesce("r_ctx", "r_src").max().alias("r")).sort("_i")
    return best["r"]


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
                   max_missing_fraction: float = 0.5, inputs: pl.Series | None = None) -> tuple[pl.DataFrame, int]:
    """Validate a submission before joining, so it cannot change the evaluated row set.

    `inputs` is every example_id of the split's inputs (scored and unscored); a submission must score all of
    them, so neither its coverage nor the error messages depend on which rows are scored. Defaults to gold."""
    ids = (gold["example_id"] if inputs is None else inputs).to_frame("example_id")
    if preds["example_id"].null_count():
        raise ValueError("prediction example_id contains null values")
    if preds["example_id"].n_unique() != preds.height:
        raise ValueError("prediction example_id contains duplicates")
    unknown = preds.join(ids, on="example_id", how="anti")
    if unknown.height:
        raise ValueError(f"predictions contain {unknown.height:,} unknown example_id values")
    if np.isinf(preds["score"].to_numpy()).any():
        raise ValueError("predictions contain infinite scores")
    full = ids.join(preds, on="example_id", how="left", maintain_order="left", validate="1:1")
    missing = int(full["score"].null_count() + full["score"].is_nan().sum())
    if missing and not allow_missing:
        raise ValueError(f"{missing:,} of {full.height:,} input rows have no finite score (use --allow-missing)")
    if missing > full.height * max_missing_fraction:
        raise ValueError(f"{missing:,} of {full.height:,} input rows have no score; check the benchmark version")
    if missing == full.height:
        raise ValueError("predictions contain no finite scores")
    if missing:
        full = full.with_columns(pl.col("score").fill_nan(None).fill_null(pl.col("score").median()))
    df = gold.join(full, on="example_id", how="left", maintain_order="left", validate="1:1")
    if df.height != gold.height or not df["example_id"].equals(gold["example_id"]):
        raise ValueError("prediction join changed the gold row set")
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


def _ceiling(df: pl.DataFrame, w: np.ndarray | None = None) -> float:
    """Label reliability averaged over strata with the pair weights of stratified_auc (w_pos * w_neg)."""
    if "_rel" not in df.columns:
        return float("nan")
    ww = df["_bw"].to_numpy() * (w if w is not None else 1.0)
    codes, y, r = _codes(df["context_id"], df["sources"]), df["label"].to_numpy(), df["_rel"].to_numpy()
    n = codes.max() + 1 if len(codes) else 0
    wp = np.bincount(codes, weights=ww * (y == 1), minlength=n)
    wn = np.bincount(codes, weights=ww * (y == 0), minlength=n)
    rs = np.bincount(codes, weights=r, minlength=n) / np.maximum(np.bincount(codes, minlength=n), 1)
    pw = wp * wn
    return float((pw * rs).sum() / pw.sum()) if pw.sum() else float("nan")


def _groups(d: pl.DataFrame, sp: str) -> list[pl.DataFrame]:
    """Row sets averaged into a species score: the whole species, or human ancestry groups (incl. unknown)."""
    d = d.with_row_index("_r")
    parts = [d] if sp != "human" else list(d.partition_by("ancestry_group"))
    return [g for g in parts if (g["label"] == 1).sum() >= MIN_GROUP_POS]


def species_result(d: pl.DataFrame, sp: str, w: np.ndarray | None = None) -> dict:
    """{"auroc", "ceiling", "normalized"}: balanced AUROC, noise ceiling and (AUROC - .5) / (ceiling - .5),
    each averaged over the species' groups (human: ancestry groups, each normalized by its own ceiling).
    NaN when no group has MIN_GROUP_POS positives."""
    rows = []
    for g in _groups(d, sp):
        gw = w[g["_r"].to_numpy()] if w is not None else None
        a, c = _auc(g, gw)[0], _ceiling(g, gw)
        rows.append((a, c, (a - 0.5) / (c - 0.5) if c > 0.5 else float("nan")))
    if not rows:
        return {"auroc": float("nan"), "ceiling": float("nan"), "normalized": float("nan")}
    a, c, n = (float(np.nanmean(x)) if not np.all(np.isnan(x)) else float("nan") for x in np.array(rows).T)
    return {"auroc": a, "ceiling": c, "normalized": n}


def species_score(d: pl.DataFrame, sp: str, w: np.ndarray | None = None) -> float:
    """Raw balanced AUROC of a species (human: mean over ancestry groups incl. unknown)."""
    return species_result(d, sp, w)["auroc"]


def headline(df: pl.DataFrame, w: np.ndarray | None = None, species: list[str] | None = None,
             raw: bool = False) -> tuple[float, dict]:
    """Mean species score over `species` (default: the headline species): normalized scores, or raw
    balanced AUROCs with raw=True."""
    parts = {}
    for sp in species if species is not None else species_tiers()[0]:
        m = (df["species"] == sp).to_numpy()
        if m.any():
            parts[sp] = species_result(df.filter(pl.Series(m)), sp, w[m] if w is not None else None)[
                "auroc" if raw else "normalized"]
    vals = [v for v in parts.values() if not np.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), parts


def ceilings(df: pl.DataFrame, species: list[str]) -> dict:
    """Noise ceiling (raw-AUROC scale) per species, averaged over the same groups as the species score.
    Needs labels and balance weights only, not predictions."""
    out = {}
    for sp in species:
        d = df.filter(pl.col("species") == sp)
        vals = [_ceiling(g) for g in _groups(d, sp)] if d.height else []
        out[sp] = float(np.mean(vals)) if vals else float("nan")
    return out


def _row(name: str, d: pl.DataFrame) -> dict:
    y = d["label"].to_numpy()
    wg, wn = _within_gene_auc(d)
    prev = y.mean() if len(y) else float("nan")
    return {
        "stratum": name, "n": d.height, "pos": int(y.sum()), "slb_auroc": _auc(d)[0], "ceiling": _ceiling(d),
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
    if split.startswith("test"):
        _log_test_eval(split, pa)
        _log_test_eval(split, pb)
    gold, ids = load_gold(split), input_ids(split)
    da, _ = validated_join(gold, pa, inputs=ids)
    db, _ = validated_join(gold, pb, inputs=ids)
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


def input_ids(split: str) -> pl.Series:
    """Every example_id a submission for this split must score."""
    return pl.read_parquet(inputs_path(split), columns=["example_id"])["example_id"]


def _log_test_eval(split: str, preds: pl.DataFrame) -> None:
    """Append every test-split evaluation to results/test_evals.jsonl: test readouts are auditable."""
    import datetime
    import getpass
    import sys

    log = Path("results/test_evals.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(preds.sort("example_id").write_ipc(None).getvalue()).hexdigest()
    with log.open("a") as f:
        f.write(json.dumps({"time": datetime.datetime.now().isoformat(timespec="seconds"), "user": getpass.getuser(),
                            "split": split, "bench": str(BENCH), "predictions_sha256": digest,
                            "argv": sys.argv}) + "\n")


def evaluate(preds: pl.DataFrame, split: str, boot: int = 0, allow_missing: bool = False, log: bool = True) -> dict:
    """log=False only for maintainer re-verification of already-recorded results (leaderboard, refresh)."""
    if split.startswith("test") and not (BENCH / "hidden" / f"{split}_labels.parquet").exists():
        raise SystemExit("test labels are private: tune on dev and submit test predictions (README, Submitting test predictions)")
    if split.startswith("test") and log:
        _log_test_eval(split, preds)
    df, missing = validated_join(load_gold(split), preds, allow_missing, inputs=input_ids(split))
    score, parts = headline(df)
    raw, raw_parts = headline(df, raw=True)
    head, auxsp = species_tiers()
    aux = headline(df, species=auxsp)[1]
    res = {"benchmark": bench_id(), "manifest_sha256": file_sha256(BENCH / "manifest.json"),
           "scorer_version": SCORER_VERSION, "split": split, "slb_score": score, "species_scores": parts,
           "auxiliary_species_scores": aux, "slb_auroc": raw, "species_auroc": raw_parts,
           "auxiliary_species_auroc": headline(df, species=auxsp, raw=True)[1],
           "ceilings": ceilings(df, head + auxsp), "n": df.height, "missing_filled": int(missing), "strata": []}
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
    lines.append(f"SLB score (normalized: 0 = chance, 1 = independent re-measurement): {res['slb_score']:.4f}"
                 + (f"  (95% CI {ci[0]:.4f}–{ci[1]:.4f})" if ci else ""))
    lines.append("  " + "  ".join(f"{SPECIES_NAMES[k]}={v:.4f}" for k, v in res["species_scores"].items()))
    if "slb_auroc" in res:
        lines.append(f"raw balanced AUROC: {res['slb_auroc']:.4f}  " + "  ".join(
            f"{SPECIES_NAMES[k]}={v:.4f} (ceiling {res['ceilings'].get(k, float('nan')):.3f})"
            for k, v in res["species_auroc"].items()))
    if res.get("auxiliary_species_scores"):
        lines.append("auxiliary (not in SLB score): " + "  ".join(
            f"{SPECIES_NAMES[k]}=" + ("n/a (<20 pos)" if np.isnan(v) else f"{v:.4f}")
            for k, v in res["auxiliary_species_scores"].items()))
    if res["missing_filled"]:
        lines.append(f"  WARNING: {res['missing_filled']:,} missing scores filled with the median")
    lines.append("\nper stratum: SLB = fitness-balanced AUROC (what the score uses); wg = within-gene SLB;"
                 " unadj = AUROC without fitness balancing (diagnostic)")
    lines.append(f"{'stratum':44s} {'n':>9s} {'pos':>7s} {'SLB':>7s} {'ceil':>6s} {'wg':>7s} {'unadj':>7s} {'AP×':>6s}")
    for r in res["strata"]:
        lines.append(f"{r['stratum'][:44]:44s} {r['n']:>9,d} {r['pos']:>7,d} {r['slb_auroc']:>7.4f} "
                     f"{r.get('ceiling', float('nan')):>6.3f} {r['within_gene']:>7.4f} {r['unadjusted_auroc']:>7.4f} "
                     f"{r['ap_lift']:>6.2f}")
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
