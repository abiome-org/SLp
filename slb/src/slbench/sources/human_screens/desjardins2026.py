"""Desjardins, Bowlan et al. 2026 Cell Rep, doi 10.1016/j.celrep.2026.117961 (Repare Therapeutics).

Genome-wide TKOv3 screens in isogenic pairs (parental vs altered clone), T0 and two technical replicates
(T18A, T18B) per arm, raw sgRNA read counts from Mendeley Data 10.17632/k6wm46g4tw v1 (CC BY 4.0).

Only loss-of-function query KNOCKOUTS (clone annotated "-/-") are gene x gene interactions and are kept:
  RPE1 (hTERT, Cas9, TP53-/-):  FBXW7, CDK12, STK11, TET2
  BEAS-2B:                      ARID1A, KMT2D
Excluded (not a pairwise gene knockout, or not drug-free):
  CCNE1 FT282 (overexpression); IDH1 p.R132H RPE1, DNMT3A p.R882H K562, SF3B1 p.K666N / SRSF2 p.P95H /
  U2AF1 p.Q157R K562 (hotspot knock-ins); Chr18q RPE1 and Chr13q FT282 (arm losses);
  KEAP1 A549 (A549 carries a LoF KEAP1 point mutation; the "altered" arm is a KEAP1 *rescue* clone, i.e.
  re-expression, not a knockout); PKMYT1i COL-hTERT and DLD-1 base-editor files (drug screens).

Scoring (uniform for desjardins2026 / feng2022 / deweirdt2020; `query_screen` below):
  per arm and replicate: guide LFC = log2(RPM+1) end - log2(RPM+1) T0 (the arm's own T0), guides with
  < 30 T0 reads or in the bottom 2% of T0 dropped; gene LFC = mean over guides (>= 2 guides).
  GI of query q with library gene g = LFC_g(q-KO) - smooth_g(LFC_g(WT)), where smooth is a running median
  of the KO LFC over genes ordered by WT LFC (absorbs clone growth-rate differences and the essential-gene
  floor); z = residual / running MAD (local scale, same ordering), then re-standardised to median 0, MAD-SD 1
  over genes; the local scale is floored at the global residual MAD-SD. Each arm's gene LFCs are centred
  on their median gene. Per replicate (KO rep A vs WT rep A, B vs B) -> gi_rep1, gi_rep2; pooled score from the
  replicate-mean LFCs of each arm. The query's own library row is dropped.

Label rule (desjardins2026): positive = pooled z <= -4, z < 0 in both replicates, and the library gene depletes in the
query-KO background (median-centred LFC_KO < -0.5; without this, tumour-suppressor genes whose knockout is a
growth GAIN in WT but neutral in the KO clone, e.g. TSC2/NF1/PTEN, come out as "negative GI" although the
double mutant is not sick), and the library gene is not a positive for >= 3 distinct queries of the study
(IER3IP1, MARK2, LARP7: clone-vs-parental generic); negative = |pooled z| < 1; else null.
Why -4 and not the usual -3: scoring one RPE1/BEAS-2B parental arm against another screen's parental arm
(a null with no query) gives 56% of the z <= -3 hit rate of the real query screens but ~30% at z <= -4.
The two replicates are technical (same clone, same T0), so split-half agreement is an upper bound.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl
import polars as pl
from scipy.ndimage import median_filter

from slbench.sources import finalize

RAW = Path("data/raw/desjardins2026")
KEY = "desjardins2026"
# file stem -> (query gene, SLB context)
KEEP = {
    "FBXW7_RPE1": ("FBXW7", "RPE1"),
    "CDK12_RPE1": ("CDK12", "RPE1"),
    "STK11_RPE1": ("STK11", "RPE1"),
    "TET2_RPE1": ("TET2", "RPE1"),
    "ARID1A_BEAS2B": ("ARID1A", "BEAS-2B"),
    "KMT2D_BEAS2B": ("KMT2D", "BEAS-2B"),
}
EXCLUDED = {
    "CCNE1_FT282": "overexpression", "IDH1_RPE1": "point mutation (R132H knock-in)",
    "DNMT3A_K562": "point mutation (R882H)", "SF3B1_K562": "point mutation (K666N)",
    "SRSF2_K562": "point mutation (P95H)", "U2AF1_K562": "point mutation (Q157R)",
    "Chr18q_RPE1": "chromosome-arm loss", "Chr13q_FT282": "chromosome-arm loss",
    "KEAP1_A549": "KEAP1 rescue (re-expression) in KEAP1-mutant A549, not a knockout",
    "PKMYT1i_COLhTERT": "drug screen", "PKMYT1i_DLD1_FNLS_FBXW7": "drug screen / base editor",
}
Z_POS, Z_NEG = -3.0, 1.0  # engine defaults; desjardins2026 uses Z_POS_DESJARDINS
Z_POS_DESJARDINS = -4.0  # calibrated on parental-vs-parental null screens (see notes): FDR ~0.3 at -4 vs ~0.56 at -3
MAX_QUERIES = 3
LFC_KO_MAX = -0.5  # a positive must actually deplete in the query-KO background (excludes masking of a WT gain)
MIN_T0 = 30


# ---------------------------------------------------------------- shared scoring engine
def _lrpm(c: np.ndarray) -> np.ndarray:
    c = c.astype(float)
    return np.log2(c / c.sum() * 1e6 + 1)


def gene_lfc(genes: np.ndarray, t0: np.ndarray, end: np.ndarray, keep: np.ndarray) -> dict[str, float]:
    lfc = _lrpm(end) - _lrpm(t0)
    d = pl.DataFrame({"g": genes[keep], "l": lfc[keep]}).group_by("g").agg(pl.col("l").mean(), pl.len().alias("n"))
    d = d.filter(pl.col("n") >= 2)
    return dict(zip(d["g"], d["l"]))


def residual_z(ko: np.ndarray, wt: np.ndarray, window: int | None = None) -> np.ndarray:
    """z of KO LFC relative to a running median over genes ordered by WT LFC, scaled by a running MAD."""
    n = len(ko)
    w = window or max(101, (n // 40) | 1)
    o = np.argsort(wt, kind="stable")
    fit = median_filter(ko[o], size=w, mode="nearest")
    r = ko[o] - fit
    s = 1.4826 * median_filter(np.abs(r - median_filter(r, size=w, mode="nearest")), size=w, mode="nearest")
    s = np.maximum(s, 1.4826 * np.median(np.abs(r - np.median(r))))  # floor: the saturated essential tail has ~0 local MAD
    z = np.empty(n)
    z[o] = r / np.maximum(s, 1e-6)
    return (z - np.median(z)) / (1.4826 * np.median(np.abs(z - np.median(z))))


def query_screen(genes: np.ndarray, wt_t0, wt_ends: list, ko_t0, ko_ends: list, drop: set[str] = frozenset()) -> pl.DataFrame:
    """Gene-level GI for one query-KO arm vs its WT arm. wt_ends/ko_ends: per-replicate end counts (paired
    by index). Returns gene, lfc_wt, lfc_ko, z (pooled), z_rep1.. (per replicate)."""
    k_wt = (wt_t0 >= MIN_T0) & (wt_t0 >= np.quantile(wt_t0, 0.02))
    k_ko = (ko_t0 >= MIN_T0) & (ko_t0 >= np.quantile(ko_t0, 0.02))
    keep = k_wt & k_ko
    wl = [gene_lfc(genes, wt_t0, e, keep) for e in wt_ends]
    kl = [gene_lfc(genes, ko_t0, e, keep) for e in ko_ends]
    common = sorted(set.intersection(*[set(d) for d in wl + kl]) - set(drop))
    W = np.array([[d[g] for g in common] for d in wl])
    K = np.array([[d[g] for g in common] for d in kl])
    W = W - np.median(W, axis=1, keepdims=True)  # centre each replicate on the median gene
    K = K - np.median(K, axis=1, keepdims=True)
    out = {"gene": common, "lfc_wt": W.mean(0), "lfc_ko": K.mean(0), "z": residual_z(K.mean(0), W.mean(0))}
    for i in range(len(ko_ends)):
        out[f"z_rep{i + 1}"] = residual_z(K[i], W[i % len(wl)])
    return pl.DataFrame(out)


def to_measurements(screens: pl.DataFrame, key: str, mechanism: str = "CRISPR-KO", z_pos: float = Z_POS,
                    unlabelled: frozenset = frozenset(), max_queries: int = MAX_QUERIES) -> pl.DataFrame:
    """screens: query, context, gene, z, z_rep1, z_rep2 -> MEASUREMENT_SCHEMA via finalize.

    positive = z <= z_pos, z < 0 in every replicate, lfc_ko < LFC_KO_MAX, and the library gene is positive
    with fewer than `max_queries` distinct query genes of this source (a gene "hit" by many unrelated
    queries reflects the clone/anchor-vs-control comparison itself, not the query); negative = |z| < Z_NEG;
    screens in `unlabelled` ((query, context) that fail their own split-half check) get no labels."""
    from scipy.stats import norm

    rows = []
    for (q, cl), d in screens.group_by(["query", "context"], maintain_order=True):
        p = 2 * norm.sf(np.abs(d["z"].to_numpy()))
        o = np.argsort(p)
        qv = np.empty_like(p)
        qv[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
        rows.append(d.with_columns(pl.Series("q", np.minimum(qv, 1.0))))
    d = pl.concat(rows)
    both_neg = pl.all_horizontal([pl.col(c) < 0 for c in d.columns if c.startswith("z_rep")])
    hit = (pl.col("z") <= z_pos) & both_neg & (pl.col("lfc_ko") < LFC_KO_MAX)
    d = d.with_columns(hit.alias("_hit"))
    promiscuous = d.filter("_hit").group_by("gene").agg(pl.col("query").n_unique().alias("nq")).filter(pl.col("nq") >= max_queries)["gene"]
    skip = pl.struct("query", "context").map_elements(lambda r: (r["query"], r["context"]) in unlabelled, return_dtype=pl.Boolean)
    d = d.with_columns(skip.alias("_skip"), pl.col("gene").is_in(promiscuous.implode()).alias("_prom"))
    df = d.select(
        pl.lit("human").alias("species"), pl.lit(key).alias("source"), "context", pl.lit(mechanism).alias("mechanism"),
        pl.col("query").alias("gene_a"), pl.col("gene").alias("gene_b"),
        pl.col("z").alias("score"), pl.lit("zGI_query_vs_WT").alias("score_name"),
        pl.col("q").alias("signif"), pl.lit("BH_q_from_z").alias("signif_name"),
        pl.when(pl.col("_skip")).then(None)
        .when(pl.col("_hit") & ~pl.col("_prom")).then(1).when(pl.col("z").abs() < Z_NEG).then(0)
        .otherwise(None).cast(pl.Int8).alias("label"),
    )
    return finalize(df, "human")


def to_replicates(screens: pl.DataFrame) -> pl.DataFrame:
    from slbench import ids

    reps = sorted(c for c in screens.columns if c.startswith("z_rep") or c.startswith("z_anchor"))
    d = screens.with_columns(
        ids.resolve("human", screens["query"]).alias("qa"), ids.resolve("human", screens["gene"]).alias("gb"),
    ).filter(pl.col("qa").is_not_null() & pl.col("gb").is_not_null() & (pl.col("qa") != pl.col("gb")))
    d = d.select(
        pl.min_horizontal("qa", "gb").alias("gene_a"), pl.max_horizontal("qa", "gb").alias("gene_b"), "context", "query",
        *[pl.col(c).alias(c.replace("z_", "gi_")) for c in reps],
    )
    return d.group_by("gene_a", "gene_b", "context").agg(pl.col("query").first(), *[pl.col(c.replace("z_", "gi_")).mean() for c in reps])


# ---------------------------------------------------------------- Desjardins parsing
def _read(stem: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    wb = openpyxl.load_workbook(RAW / f"{stem}_sgRNA_readcounts.xlsx", read_only=True)
    rows = list(wb.worksheets[0].iter_rows(values_only=True))
    h = list(rows[0])
    body = [r for r in rows[1:] if r[0]]
    genes = np.array([str(r[1]) for r in body])
    counts = np.array([[float(x or 0) for x in r[2:]] for r in body])
    return h[2:], genes, counts


def screens() -> pl.DataFrame:
    out = []
    for stem, (q, cl) in KEEP.items():
        h, genes, c = _read(stem)

        def col(arm: str, tp: str) -> np.ndarray:
            m = [i for i, x in enumerate(h) if (("parental" in x) == (arm == "wt")) and ("-/-" in x or arm == "wt") and x.endswith(tp)]
            assert len(m) == 1, (stem, arm, tp, h)
            return c[:, m[0]]

        d = query_screen(genes, col("wt", "T0"), [col("wt", "T18A"), col("wt", "T18B")],
                         col("ko", "T0"), [col("ko", "T18A"), col("ko", "T18B")], drop={q})
        out.append(d.with_columns(pl.lit(q).alias("query"), pl.lit(cl).alias("context")))
    return pl.concat(out)


LABEL_KW = {"z_pos": Z_POS_DESJARDINS}


def load() -> pl.DataFrame:
    return to_measurements(screens(), KEY, **LABEL_KW)


def replicates() -> pl.DataFrame:
    """gene_a, gene_b, context, query, gi_rep1 (KO T18A vs WT T18A), gi_rep2 (T18B vs T18B); z-scaled GI."""
    return to_replicates(screens())


# ---------------------------------------------------------------- SLB inclusion-gate checks (shared)
MEAS_FILES = ["slkb", "ryanlab_zdlfc", "chou2025", "spidr2025", "harle2025", "flister2025"]


def _auc(y: np.ndarray, s: np.ndarray) -> float | None:
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if y.sum() < 5 or (1 - y).sum() < 5:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def _cello(ctx: str) -> str | None:
    from slbench import contexts
    lab, _, dep = ctx.partition("|")
    r = contexts.lookup(lab, dep or None)
    return r["cellosaurus_name"] if r else None


def checks(key: str, scr: pl.DataFrame, extra_partners: dict[str, pl.DataFrame] | None = None, **label_kw) -> dict:
    """Own replication, cross-study, fitness confounding and counts for one staged source."""
    from slbench import ids
    from slbench.replication import replicate_agreement, split_rule_auroc

    meas = to_measurements(scr, key, **label_kw)
    zp = label_kw.get("z_pos", Z_POS)
    reps = to_replicates(scr)
    res: dict = {}
    # symbol resolution
    syms = pl.Series(sorted(set(scr["gene"]) | set(scr["query"])))
    res["unresolved_symbols"] = int(ids.resolve("human", syms).is_null().sum())
    res["symbols"] = len(syms)
    # counts
    n_pos, n_neg = int((meas["label"] == 1).sum()), int((meas["label"] == 0).sum())
    res.update(n_pairs=meas.height, n_pos=n_pos, n_neg=n_neg, pos_rate=n_pos / meas.height,
               unique_genes=len(set(meas["gene_a"]) | set(meas["gene_b"])))
    par = pl.read_parquet("data/slb/features/paralogs.parquet").filter(pl.col("species") == "human")
    pp = set(zip(par["a"], par["b"])) | set(zip(par["b"], par["a"]))
    isp = np.array([(a, b) in pp for a, b in zip(meas["gene_a"], meas["gene_b"])])
    res["paralog_fraction"] = float(isp.mean())
    res["paralog_pairs"] = int(isp.sum())
    res["paralog_pos"] = int((isp & (meas["label"] == 1).fill_null(False).to_numpy()).sum())
    # per query: resolve query symbol, then per (query, context)
    qres = dict(zip(syms, ids.resolve("human", syms)))
    pq = []
    for (q, cl), d in scr.group_by(["query", "context"], maintain_order=True):
        qq = qres[q]
        m = meas.filter((pl.col("context") == cl) & ((pl.col("gene_a") == qq) | (pl.col("gene_b") == qq)))
        z1, z2 = d["z_rep1"].to_numpy(), d["z_rep2"].to_numpy()
        row = {"query": q, "context": cl, "pairs": m.height, "pos": int((m["label"] == 1).sum()),
               "neg": int((m["label"] == 0).sum()), "pos_rate": float((m["label"] == 1).sum() / max(m.height, 1)), "labelled": bool(m["label"].null_count() < m.height),
               "rep_spearman": float(pl.DataFrame({"a": z1, "b": z2}).select(pl.corr("a", "b", method="spearman")).item()),
               "split_auroc_1to2": split_rule_auroc(z1, z2, pos_thr=zp), "split_auroc_2to1": split_rule_auroc(z2, z1, pos_thr=zp)}
        if "z_anchor1" in d.columns and d["z_anchor1"].null_count() < d.height:
            a1, a2 = d["z_anchor1"].to_numpy(), d["z_anchor2"].to_numpy()
            ok = ~(np.isnan(a1) | np.isnan(a2))
            row["anchor_split_auroc_1to2"] = split_rule_auroc(a1[ok], a2[ok], pos_thr=zp)
            row["anchor_split_auroc_2to1"] = split_rule_auroc(a2[ok], a1[ok], pos_thr=zp)
        top = meas.filter((pl.col("context") == cl) & ((pl.col("gene_a") == qq) | (pl.col("gene_b") == qq)) & (pl.col("label") == 1)) \
            .sort("score").head(8)
        row["top_hits"] = [b if a == qq else a for a, b in zip(top["gene_a"], top["gene_b"])]
        pq.append(row)
    res["per_query"] = pq
    # own replication, pooled over all screens (z is standardised per screen)
    z1, z2 = scr["z_rep1"].to_numpy(), scr["z_rep2"].to_numpy()
    own = {"split_rule_auroc_rep1_to_rep2": split_rule_auroc(z1, z2, pos_thr=zp),
           "split_rule_auroc_rep2_to_rep1": split_rule_auroc(z2, z1, pos_thr=zp),
           "split_rule_n_pos_rep1": int((z1 <= zp).sum()), "split_rule_n_pos_rep2": int((z2 <= zp).sum()),
           "replicate_agreement": replicate_agreement(reps.drop_nulls(["gi_rep1", "gi_rep2"]), ["rep1", "rep2"])}
    j = reps.join(meas.select("gene_a", "gene_b", "context", "label"), on=["gene_a", "gene_b", "context"]).drop_nulls("label")
    y = j["label"].to_numpy()
    own["final_labels_auroc_rep1"] = _auc(y, -j["gi_rep1"].to_numpy())
    own["final_labels_auroc_rep2"] = _auc(y, -j["gi_rep2"].to_numpy())
    own["note_final_labels"] = "final labels require z<0 in both replicates, so single-replicate AUROCs for them are optimistic"
    if "z_anchor1" in scr.columns:
        a = scr.drop_nulls(["z_anchor1", "z_anchor2"])
        if a.height:
            own["anchor_guide_split_auroc_1to2"] = split_rule_auroc(a["z_anchor1"].to_numpy(), a["z_anchor2"].to_numpy(), pos_thr=zp)
            own["anchor_guide_split_auroc_2to1"] = split_rule_auroc(a["z_anchor2"].to_numpy(), a["z_anchor1"].to_numpy(), pos_thr=zp)
    res["own_replication"] = own
    # fitness confounding
    fe = pl.read_parquet("data/slb/gene_single_effects.parquet").filter(pl.col("species") == "human")
    f = {g: (np.nan if v is None else v) for g, v in zip(fe["gene"], fe["single_effect"])}
    lab = meas.drop_nulls("label")
    s = -(np.array([f.get(g, np.nan) for g in lab["gene_a"]], dtype=float) + np.array([f.get(g, np.nan) for g in lab["gene_b"]], dtype=float))
    res["fitness_auroc"] = _auc(lab["label"].to_numpy(), s)
    # the screen's own single-gene effect: WT-arm LFC of the library gene
    lj = _label_join(scr, meas, qres)
    res["fitness_auroc_own_wt_lfc"] = _auc(lj["label"].to_numpy(), -lj["lfc_wt"].to_numpy())
    # cross-study
    res["cross_study"] = cross_study(meas, extra_partners or {})
    return res


def _label_join(scr: pl.DataFrame, meas: pl.DataFrame, qres: dict) -> pl.DataFrame:
    from slbench import ids
    d = scr.with_columns(pl.col("query").replace_strict(qres).alias("qa"), ids.resolve("human", scr["gene"]).alias("gb")) \
        .drop_nulls(["qa", "gb"]).with_columns(pl.min_horizontal("qa", "gb").alias("gene_a"), pl.max_horizontal("qa", "gb").alias("gene_b"))
    return d.join(meas.select("gene_a", "gene_b", "context", "label"), on=["gene_a", "gene_b", "context"]).drop_nulls("label") \
        .unique(["gene_a", "gene_b", "context"])


def cross_study(meas: pl.DataFrame, extra: dict[str, pl.DataFrame]) -> dict:
    parts = [pl.read_parquet(f"data/interim/measurements/{f}.parquet").filter(pl.col("species") == "human") for f in MEAS_FILES]
    parts += list(extra.values())
    other = pl.concat([p.select("source", "context", "gene_a", "gene_b", "score", "label") for p in parts])
    cmap = {c: _cello(c) for c in set(other["context"]) | set(meas["context"])}
    other = other.with_columns(pl.col("context").replace_strict(cmap, default=None).alias("cello"))
    mine = meas.with_columns(pl.col("context").replace_strict(cmap, default=None).alias("cello"))
    j = mine.select("cello", "gene_a", "gene_b", "score", "label").join(
        other.select("source", "cello", "gene_a", "gene_b", pl.col("score").alias("score_o"), pl.col("label").alias("label_o")),
        on=["gene_a", "gene_b"], suffix="_o")
    out = {"same_line": [], "different_line": []}
    for same, name in ((True, "same_line"), (False, "different_line")):
        jj = j.filter((pl.col("cello") == pl.col("cello_o")) if same else (pl.col("cello") != pl.col("cello_o")))
        for (src,), d in jj.group_by(["source"]):
            # one row per pair for the different-line comparison (average scores over line combinations)
            lines = sorted({f"{x}~{y}" for x, y in zip(d["cello"], d["cello_o"])})
            if not same:
                d = d.group_by("gene_a", "gene_b").agg(pl.col("score").mean(), pl.col("score_o").mean(),
                                                       pl.col("label").max(), pl.col("label_o").max())
            a = d.drop_nulls("label")
            b = d.drop_nulls("label_o")
            out[name].append({
                "partner": src, "overlap_pairs": d.height,
                "their_score_for_our_labels_auroc": _auc(a["label"].to_numpy(), -a["score_o"].to_numpy()) if a.height else None,
                "our_pos": int((a["label"] == 1).sum()), "our_labelled": a.height,
                "our_score_for_their_labels_auroc": _auc(b["label_o"].to_numpy(), -b["score"].to_numpy()) if b.height else None,
                "their_pos": int((b["label_o"] == 1).sum()), "their_labelled": b.height,
                "lines": lines[:12],
            })
    return out
