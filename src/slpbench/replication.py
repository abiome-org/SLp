"""Label reproducibility checks.

1. Within-study, replicate-split (SLKB raw counts): score every biological replicate
   independently with one uniform additive model, then ask whether replicates agree.
     LFC   = log2 normalised end counts - log2 normalised start counts, centred on control x control
     f_g   = median LFC of gene x control guide pairs
     GI_ab = median over guide pairs of (LFC - f_a - f_b)
2. Cross-study (built measurements): for pairs measured by two studies in the same context,
   AUROC of study B's within-study score percentile for study A's labels.
"""

from __future__ import annotations

import sqlite3
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

INTERIM = Path("data/interim")


def _counts(study: str, cell: str) -> pl.DataFrame:
    con = sqlite3.connect(INTERIM / "slkb/slkb.sqlite")
    q = """select sgRNA_target_name_g1 g1, sgRNA_target_name_g2 g2, T0_counts t0, TEnd_counts te, target_type tt
           from joined_counts where study_origin=? and cell_line_origin=?"""
    return pl.read_database(q, con, execute_options={"parameters": [study, cell]}, infer_schema_length=None)


def replicate_gi(study: str, cell: str) -> tuple[pl.DataFrame, list[str]]:
    """Per-replicate gene-pair GI for one SLKB study/cell line. Returns (df[gene_a, gene_b, gi_0..], rep names)."""
    d = _counts(study, cell)
    t0 = np.array([[float(x) for x in s.split(";")] for s in d["t0"]])
    te = np.array([[float(x) for x in s.split(";")] for s in d["te"]])
    t0 = t0.mean(1)
    base = np.log2((t0 + 1) / (t0 + 1).sum())
    keep = t0 >= np.quantile(t0, 0.02)  # drop guide pairs essentially absent at the start
    reps = [f"rep{i}" for i in range(te.shape[1])]
    g1 = d["g1"].to_numpy()
    g2 = d["g2"].to_numpy()
    tt = d["tt"].to_numpy()
    out = None
    for i, r in enumerate(reps):
        lfc = np.log2((te[:, i] + 1) / (te[:, i] + 1).sum()) - base
        ctrl = keep & (tt == "Control")
        lfc = lfc - (np.median(lfc[ctrl]) if ctrl.any() else np.median(lfc[keep]))
        single = pl.DataFrame({"g1": g1, "g2": g2, "lfc": lfc, "tt": tt, "keep": keep}).filter(
            pl.col("keep") & (pl.col("tt") == "Single"))
        single = pl.concat([
            single.filter(pl.col("g2").str.to_uppercase().str.contains("CONTROL|NONTARGET|NTC|SAFE")).select(pl.col("g1").alias("g"), "lfc"),
            single.filter(pl.col("g1").str.to_uppercase().str.contains("CONTROL|NONTARGET|NTC|SAFE")).select(pl.col("g2").alias("g"), "lfc"),
        ]).group_by("g").agg(pl.col("lfc").median().alias("f"))
        f = dict(zip(single["g"], single["f"]))
        dual = pl.DataFrame({"g1": g1, "g2": g2, "lfc": lfc, "tt": tt, "keep": keep}).filter(
            pl.col("keep") & (pl.col("tt") == "Dual") & (pl.col("g1") != pl.col("g2")))
        fa = np.array([f.get(x, np.nan) for x in dual["g1"]])
        fb = np.array([f.get(x, np.nan) for x in dual["g2"]])
        dual = dual.with_columns(pl.Series("gi", dual["lfc"].to_numpy() - fa - fb)).drop_nans("gi").with_columns(
            pl.min_horizontal("g1", "g2").alias("gene_a"), pl.max_horizontal("g1", "g2").alias("gene_b"))
        agg = dual.group_by("gene_a", "gene_b").agg(pl.col("gi").median().alias(f"gi_{r}"))
        out = agg if out is None else out.join(agg, on=["gene_a", "gene_b"], how="inner")
    return out, reps


def replicate_agreement(gi: pl.DataFrame, reps: list[str]) -> dict:
    """Mean pairwise Spearman across replicates, and tail reproducibility: of the 2% most negative
    pairs in one replicate, the fraction in the 10% most negative of another."""
    cols = [f"gi_{r}" for r in reps]
    x = gi.select(cols).to_numpy()
    rho, tail = [], []
    for i, j in combinations(range(len(cols)), 2):
        rho.append(spearmanr(x[:, i], x[:, j]).statistic)
        for a, b in ((i, j), (j, i)):
            top = x[:, a] < np.quantile(x[:, a], 0.02)
            tail.append(float(np.mean(x[top, b] < np.quantile(x[:, b], 0.10))))
    return {"pairs": gi.height, "replicates": len(cols), "spearman": float(np.mean(rho)), "tail_2pct_in_10pct": float(np.mean(tail))}


def label_vs_replicates(gi: pl.DataFrame, reps: list[str], labels: pl.DataFrame) -> dict:
    """AUROC of each single replicate's GI for the benchmark labels (labels come from the pooled
    published analysis, so this is optimistic; a label set that single replicates can't recover is noise)."""
    j = gi.join(labels, on=["gene_a", "gene_b"]).filter(pl.col("label").is_not_null())
    y = j["label"].to_numpy()
    if y.sum() < 5 or (1 - y).sum() < 5:
        return {"labelled": len(y), "pos": int(y.sum()), "auroc_single_replicate": float("nan")}
    aucs = [roc_auc_score(y, -j[f"gi_{r}"].to_numpy()) for r in reps]
    return {"labelled": len(y), "pos": int(y.sum()), "auroc_single_replicate": float(np.mean(aucs))}


def cross_study(measurements: pl.DataFrame) -> pl.DataFrame:
    """Per source: AUROC of other studies' within-study score percentile for this source's labels,
    over pairs measured by both in the same context. measurements needs a context_id column."""
    m = measurements.with_columns(
        (pl.col("score").rank("average").over("source", "context_id") / pl.col("score").count().over("source", "context_id")).alias("pct"))
    k = ["species", "context_id", "gene_a", "gene_b"]
    a = m.select(*k, "source", "pct", "label")
    j = a.join(a, on=k, suffix="_2").filter((pl.col("source") != pl.col("source_2")) & pl.col("label").is_not_null())
    rows = []
    for (s,), d in j.group_by(["source"]):
        y = d["label"].to_numpy()
        ok = y.sum() >= 5 and (1 - y).sum() >= 5
        rows.append({"source": s, "overlap_labelled": len(y), "overlap_pos": int(y.sum()),
                     "cross_study_auroc": float(roc_auc_score(y, -d["pct_2"].to_numpy())) if ok else None,
                     "partners": ",".join(sorted(d["source_2"].unique()))})
    return pl.DataFrame(rows).sort("source")


# --------------------------------------------------------------------------------------------
# Per-source evidence used by `slpbench audit` (writes REPLICATION.md)

SLKB_JOBS = [  # (pubmed, cell line, source key, measurement file)
    ("28319113", "A549", "shen2017", "slkb"), ("28319113", "HELA", "shen2017", "slkb"), ("28319113", "293T", "shen2017", "slkb"),
    ("29452643", "A549", "zhao2018", "slkb"), ("29452643", "HELA", "zhao2018", "slkb"),
    ("30033366", "K562", "horlbeck2018", "slkb"), ("30033366", "JURKAT", "horlbeck2018", "slkb"),
    ("33637726", "A375", "thompson2021", "ryanlab_zdlfc"), ("33637726", "MEWO", "thompson2021", "ryanlab_zdlfc"),
    ("33637726", "RPE1", "thompson2021", "ryanlab_zdlfc"),
    ("34469736", "HELA", "parrish2021", "ryanlab_zdlfc"), ("34469736", "PC9", "parrish2021", "ryanlab_zdlfc"),
    ("34857952", "A549", "ito2021", "ryanlab_zdlfc"), ("34857952", "MEWO", "ito2021", "ryanlab_zdlfc"),
    ("34857952", "GI1", "ito2021", "ryanlab_zdlfc"), ("34857952", "HSC5", "ito2021", "ryanlab_zdlfc"),
    ("34857952", "PK1", "ito2021", "ryanlab_zdlfc"),
]


def _norm_cell(x: str) -> str:
    import re
    return re.sub(r"[^A-Z0-9]", "", x.upper())


def slkb_job(job: tuple) -> dict:
    study, cell, src, mfile = job
    try:
        gi, reps = replicate_gi(study, cell)
        res = {"source": src, "context": cell, **replicate_agreement(gi, reps)}
        m = pl.read_parquet(INTERIM / f"measurements/{mfile}.parquet").filter(pl.col("source") == src)
        m = m.filter(pl.col("context").map_elements(_norm_cell, return_dtype=pl.String) == _norm_cell(cell))
        res.update(label_vs_replicates(gi, reps, m.select("gene_a", "gene_b", "label")))
        return res
    except Exception as e:  # report, don't abort the audit
        return {"source": src, "context": cell, "error": repr(e)[:100]}


def split_rule_auroc(x1: np.ndarray, x2: np.ndarray, pos_q: float | None = None, pos_thr: float | None = None) -> float:
    """Define labels from measurement 1 (strongest-negative tail vs |x| below median), score with measurement 2."""
    pos = x1 < (np.quantile(x1, pos_q) if pos_q is not None else pos_thr)
    neg = np.abs(x1) < np.quantile(np.abs(x1), 0.5)
    m = pos | neg
    return float(roc_auc_score(pos[m], -x2[m]))


def costanzo_orientation(thresholds=(-0.12, -0.2, -0.3)) -> list[dict]:
    parts = []
    for f in ["SGA_ExE.txt", "SGA_NxN.txt", "SGA_ExN_NxE.txt", "SGA_DAmP.txt"]:
        parts.append(pl.scan_csv(INTERIM / f"costanzo/S1/{f}", separator="\t", quote_char=None,
                                 schema_overrides={"P-value": pl.Float64}).select(
            pl.col("Query Strain ID").str.split("_").list.first().alias("q"),
            pl.col("Array Strain ID").str.split("_").list.first().alias("a"),
            pl.col("Genetic interaction score (ε)").cast(pl.Float64).alias("e"), pl.col("P-value").alias("p")).collect())
    c = pl.concat(parts).drop_nulls()
    med = c["e"].abs().median()
    c1 = c.group_by("q", "a").agg(pl.col("e").mean(), pl.col("p").min())
    rev = c1.join(c1.rename({"q": "a", "a": "q", "e": "e2", "p": "p2"}), on=["q", "a"])
    neg = rev.filter((pl.col("p") > 0.25) & (pl.col("e").abs() < med))
    out = []
    for t in thresholds:
        pos = rev.filter((pl.col("e") < t) & (pl.col("p") < 0.05))
        y = np.r_[np.ones(pos.height), np.zeros(neg.height)]
        out.append({"source": "costanzo2016", "check": f"reverse orientation, positive eps<{t}", "slb_rule": t == -0.2,
                    "pos": pos.height, "auroc": float(roc_auc_score(y, -np.r_[pos["e2"].to_numpy(), neg["e2"].to_numpy()]))})
    return out


def ryan_alleles(thresholds=(-2.3, -3.0)) -> list[dict]:
    import re
    import zipfile

    from slpbench import ids

    with zipfile.ZipFile("data/raw/ryan2012_spombe/mmc4_unaveraged.zip") as z:
        t = z.read(z.namelist()[0]).decode("latin-1").replace("\r\n", "\n").replace("\r", "\n").split("\n")

    def gid(s):
        m = re.match(r"\s*\"?(SP[A-Z0-9.]+)", s)
        return ids.spom()(m.group(1)) if m else None

    cols = [gid(h) for h in t[0].split("\t")[1:]]
    rows = []
    for line in t[1:]:
        p = line.split("\t")
        q = gid(p[0]) if len(p) > 1 else None
        if not q:
            continue
        for a, v in zip(cols, p[1:]):
            if a and a != q and v.strip():
                try:
                    rows.append((min(q, a), max(q, a), float(v)))
                except ValueError:
                    pass
    g = pl.DataFrame(rows, schema=["a", "b", "S"], orient="row").group_by("a", "b").agg(pl.col("S")) \
        .filter(pl.col("S").list.len() >= 2)
    s1, s2 = g["S"].list.get(0).to_numpy(), g["S"].list.get(1).to_numpy()
    out = []
    for t_ in thresholds:
        pos = s1 < t_
        neg = np.abs(s1) < 1
        m = pos | neg
        out.append({"source": "ryan2012", "check": f"independent allele/orientation, positive S<{t_}", "slb_rule": t_ == -3.0,
                    "pos": int(pos.sum()), "auroc": float(roc_auc_score(pos[m], -s2[m]))})
    return out


def spne_replicates() -> list[dict]:
    c = pl.read_csv("data/raw/dualcrispri2025_spneumo/mmc3_counts.csv")
    g = pl.read_csv("data/raw/dualcrispri2025_spneumo/mmc4.csv", infer_schema_length=0, null_values=["NA", ""])
    reps = {}
    for r in (1, 2, 3):
        no = sum(c[k].to_numpy() for k in c.columns if f"NoIPTG_{r}_" in k).astype(float)
        wi = sum(c[k].to_numpy() for k in c.columns if f"WithIPTG_{r}_" in k).astype(float)
        reps[r] = np.log2((wi + 1) / (wi + 1).sum()) - np.log2((no + 1) / (no + 1).sum())
    d = c.select("SG1", "SG2").with_columns([pl.Series(f"l{r}", v) for r, v in reps.items()])
    same = d.filter(pl.col("SG1") == pl.col("SG2"))
    x = d.filter(pl.col("SG1") != pl.col("SG2"))
    for r in reps:
        f = dict(zip(same["SG1"], same[f"l{r}"]))
        x = x.with_columns(pl.Series(f"gi{r}", x[f"l{r}"].to_numpy() - np.array([f.get(v, np.nan) for v in x["SG1"]])
                                     - np.array([f.get(v, np.nan) for v in x["SG2"]])))
    lab = g.select("SG1", "SG2", "pairs", pl.when(pl.col("interactionSum") == "Negative").then(1)
                   .when(pl.col("interactionSum") == "Neutral").then(0).otherwise(None).alias("label"))
    x = x.join(lab, on=["SG1", "SG2"]).filter(pl.col("pairs") != "E-E").drop_nans().drop_nulls()
    y = x["label"].to_numpy()
    return [
        {"source": "dualcrispri2025", "check": "author labels vs single replicate GI (mean of 3)", "slb_rule": True, "pos": int(y.sum()),
         "auroc": float(np.mean([roc_auc_score(y, -x[f"gi{r}"].to_numpy()) for r in reps]))},
        {"source": "dualcrispri2025", "check": "labels from rep 1 (2% tail) scored by reps 2+3", "slb_rule": False, "pos": None,
         "auroc": split_rule_auroc(x["gi1"].to_numpy(), ((x["gi2"] + x["gi3"]) / 2).to_numpy(), pos_q=0.02)},
    ]


def spidr_checks() -> list[dict]:
    from slpbench.sources.human import spidr_replicate_gi

    gi = spidr_replicate_gi()
    gem = pl.read_csv("data/raw/spidr2025/MOESM5_pairs.csv").with_columns(
        pl.col("gene_combination").str.split_exact(";", 1).struct.rename_fields(["x", "y"])).unnest("gene_combination") \
        .with_columns(pl.min_horizontal("x", "y").alias("gene_a"), pl.max_horizontal("x", "y").alias("gene_b"),
                      pl.when(pl.col("sens.score") <= -1).then(1).when(pl.col("sens.score") == 0).then(0).alias("label")) \
        .filter(~pl.col("x").str.contains("_mis") & ~pl.col("y").str.contains("_mis"))
    j = gi.join(gem.select("gene_a", "gene_b", "label"), on=["gene_a", "gene_b"]).drop_nulls("label")
    y = j["label"].to_numpy()
    x1, x2 = gi["gi_rep1"].to_numpy(), gi["gi_rep2"].to_numpy()
    z1 = (x1 - x1.mean()) / x1.std()
    return [
        {"source": "spidr2025", "check": "published GEMINI calls vs single replicate additive GI", "slb_rule": False, "pos": int(y.sum()),
         "auroc": float(np.mean([roc_auc_score(y, -j[c].to_numpy()) for c in ("gi_rep1", "gi_rep2")]))},
        {"source": "spidr2025", "check": "SLB rule (z<=-3) on rep 1, scored by rep 2", "slb_rule": True, "pos": int((z1 <= -3).sum()),
         "auroc": split_rule_auroc(x1, x2, pos_thr=x1.mean() - 3 * x1.std())},
    ]


def heigwer_split_half() -> list[dict]:
    lf = pl.scan_csv("data/raw/heigwer2023_dmel/interactions_stat_tested_bias_corrected.csv.gz",
                     infer_schema_length=10000, null_values=["NA"])
    d = lf.filter(pl.col("feature") == "cells").select([pl.col(c).cast(pl.Float64) for c in "1234"]).collect() \
        .drop_nulls().drop_nans().to_numpy()
    return [{"source": "heigwer2023", "check": "labels from reps 1-2 (1% tail) scored by reps 3-4", "slb_rule": True, "pos": None,
             "auroc": split_rule_auroc(d[:, :2].mean(1), d[:, 2:].mean(1), pos_q=0.01)}]


def fitness_diagnostic() -> pl.DataFrame:
    from slpbench.fitness import gene_effects

    f = {(s, g): v for s, g, v in gene_effects().iter_rows()}
    rows = []
    for p in sorted((INTERIM / "measurements").glob("*.parquet")):
        m = pl.read_parquet(p).filter(pl.col("label").is_not_null())
        if m.height > 3_000_000:
            m = m.sample(3_000_000, seed=0)
        for (src,), d in m.group_by(["source"]):
            sp = d["species"][0]
            s = -(np.array([f.get((sp, a), np.nan) for a in d["gene_a"]]) + np.array([f.get((sp, b), np.nan) for b in d["gene_b"]]))
            ok = ~np.isnan(s)
            y = d["label"].to_numpy()[ok]
            if ok.sum() > 50 and 0 < y.sum() < len(y):
                rows.append({"source": src, "fitness_only_auroc": float(roc_auc_score(y, s[ok]))})
    return pl.DataFrame(rows)
