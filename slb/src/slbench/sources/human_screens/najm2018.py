"""Najm et al. 2018 Nat Biotechnol 36:179 "Big Papi" (doi 10.1038/nbt.4048; journal supplementary terms, no open
licence stated): orthogonal SpCas9 (U6 promoter, 20-nt guides) + SaCas9 (H1 promoter, 21-nt guides) combinatorial
knockout, day 21 vs plasmid DNA. Re-scored here from the per-construct read counts in MOESM27 (zip).

Two drug-free libraries are loaded (both are KO; the TsgOnco CRISPRa library in HA1E, Supp. Table 9, is a
proliferation gain screen and is not used, nor are the Venetoclax-family drug arms of Supp. Table 6):
  SynLet (Supp. Tables 2-4): 25 genes (+ CD81 / HPRT-intron cutting controls, 6T, EEF2), 3 guides per gene and
      enzyme, both orientations (A on Sp x B on Sa and the reverse) -> 18 constructs per gene pair; 786-O, A375,
      OVCAR-8 (3 replicates) and A549, HT-29, MEL-JUSO (2 replicates), day 21.
  Apoptosis (Supp. Tables 5-7): 32 BCL2-family / caspase genes (+ same controls), 4 guides per gene and enzyme
      -> 32 constructs per pair; MEL-JUSO (3 replicates) and OVCAR-8 (1 replicate), no-drug arm.
  Six BCL2-family genes are in both libraries; finalize() merges those pairs (mean score, consensus label).

Scoring (spidr_replicate_gi recipe, guide-level singles because guide efficacy varies strongly):
  per replicate LFC = log2 norm(reads+1) - log2 norm(pDNA+1), constructs in the bottom 2% of pDNA dropped,
  centred on the median of cutting-control x cutting-control constructs (CD81, HPRT intron);
  f(guide) = median LFC of that guide x cutting controls (separately for the U6 and H1 positions);
  GI(construct) = LFC - f(U6 guide) - f(H1 guide); gene-pair GI = median over constructs.
  score = gene-pair GI on the replicate-mean LFC (negative = synthetic sick/lethal).
  6T (non-cutting) and EEF2 (essential control) constructs are dropped.

Label rule (SLB): the authors' per-cell-line "SynLet q-value" (Supp. Tables 4 and 7, per-line rows only; not
the pooled "All 6"/"Both" or leave-one-out rows) is used as the call.
  positive: SynLet q < 0.05 and score < 0
  negative: SynLet q > 0.25, buffering q > 0.25 and |score| below that line/library's median |score|
  else null; 786-O is measurements-only (label null): its replicates and orientations do not reproduce each
  other's tails at all and the authors call nothing there.
signif = SynLet q-value.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import polars as pl

from slbench.sources import finalize

RAW = Path("data/raw/najm2018")
ZIP = RAW / "41587_2018_BFnbt4048_MOESM27_ESM.zip"
LINES = {"786O": "786-O", "A375": "A375", "A549": "A549", "HT29": "HT-29", "Meljuso": "MEL-JUSO", "OVCAR8": "OVCAR-8"}
CONTROL = {"CD81", "HPRT intron"}
DROP = {"6T", "EEF2"}
POS_Q, NEG_Q = 0.05, 0.25
# 786-O: no author call and no reproducible tail (replicate/orientation split-rule AUROC 0.29-0.59), so its
# pairs are kept as measurements with label null rather than 300 unverifiable negatives.
NO_LABEL = {"786-O"}


def _table(n: int) -> list[list[str]]:
    with zipfile.ZipFile(ZIP) as z:
        name = next(x for x in z.namelist() if x.startswith(f"Supplementary Tables/Supplementary Table {n} "))
        txt = z.read(name).decode("latin-1").replace("\r\n", "\n").replace("\r", "\n")
    return [r.split("\t") for r in txt.split("\n") if r.strip()]


def constructs() -> pl.DataFrame:
    """Long per-construct reads: library, line, u6_gene, u6_guide, h1_gene, h1_guide, pdna, rep, reads."""
    out = []
    t = _table(3)
    h = t[0]
    ix = {c: h.index(c) for c in h}
    for r in t[1:]:
        if r[ix["Time Point"]] != "Day 21":
            continue
        u6, h1 = r[ix["U6 Sequence;H1 Sequence"]].split(";")
        for rep in ("A", "B", "C"):
            v = r[ix[f"Rep {rep} Reads"]].strip()
            if v:
                out.append(("SynLet", r[ix["Cell Line"]], r[ix["U6 gene"]], u6, r[ix["H1 gene"]], h1,
                            float(r[ix["pDNA Reads"]]), rep, float(v)))
    t = _table(6)
    pdna = {r[0]: float(r[7]) for r in t[3:] if r[6] == "pDNA pool"}
    for r in t[3:]:
        if r[6] not in ("Meljuso", "OVCAR8"):
            continue
        u6, h1 = r[0].split(";")
        for rep, col in zip("ABC", (8, 9, 10)):
            v = r[col].strip()
            if v:
                out.append(("Apoptosis", r[6], r[2], u6, r[4], h1, pdna[r[0]], rep, float(v)))
    return pl.DataFrame(out, schema=["library", "line", "u6_gene", "u6_guide", "h1_gene", "h1_guide", "pdna", "rep", "reads"],
                        orient="row")


def _gi_one(d: pl.DataFrame, lfc: np.ndarray) -> pl.DataFrame:
    """Construct-level additive GI for one library x line x replicate (or replicate mean)."""
    u6g, h1g = d["u6_gene"].to_numpy(), d["h1_gene"].to_numpy()
    cu, ch = np.isin(u6g, list(CONTROL)), np.isin(h1g, list(CONTROL))
    lfc = lfc - np.median(lfc[cu & ch])
    x = d.select("u6_gene", "u6_guide", "h1_gene", "h1_guide").with_columns(pl.Series("lfc", lfc))
    fu = x.filter(pl.Series(ch)).group_by("u6_guide").agg(pl.col("lfc").median().alias("fu"))
    fh = x.filter(pl.Series(cu)).group_by("h1_guide").agg(pl.col("lfc").median().alias("fh"))
    x = x.filter(~pl.Series(cu | ch) & ~pl.col("u6_gene").is_in(DROP) & ~pl.col("h1_gene").is_in(DROP)
                 & (pl.col("u6_gene") != pl.col("h1_gene")))
    x = x.join(fu, on="u6_guide").join(fh, on="h1_guide")
    return x.select("u6_gene", "h1_gene", (pl.col("lfc") - pl.col("fu") - pl.col("fh")).alias("gi"))


def construct_gi() -> pl.DataFrame:
    """Construct GI per library, line and replicate ('mean' = replicate-mean LFC)."""
    c = constructs()
    out = []
    for (lib, line), d in c.group_by(["library", "line"], maintain_order=True):
        w = d.pivot(on="rep", index=["u6_gene", "u6_guide", "h1_gene", "h1_guide", "pdna"], values="reads")
        w = w.filter(~pl.col("u6_gene").is_in({"6T"}) & ~pl.col("h1_gene").is_in({"6T"}))
        reps = [r for r in "ABC" if r in w.columns]
        keep = w["pdna"].to_numpy() >= np.quantile(w["pdna"].to_numpy(), 0.02)
        w = w.filter(pl.Series(keep))
        p = w["pdna"].to_numpy()
        base = np.log2((p + 1) / (p + 1).sum())
        lfcs = {}
        for r in reps:
            e = w[r].to_numpy()
            lfcs[r] = np.log2((e + 1) / (e + 1).sum()) - base
        lfcs["mean"] = np.mean([lfcs[r] for r in reps], axis=0)
        for r, l in lfcs.items():
            out.append(_gi_one(w, l).with_columns(pl.lit(lib).alias("library"), pl.lit(LINES[line]).alias("context"),
                                                  pl.lit(r).alias("rep")))
    g = pl.concat(out)
    return g.with_columns(pl.min_horizontal("u6_gene", "h1_gene").alias("gene_a"),
                          pl.max_horizontal("u6_gene", "h1_gene").alias("gene_b"),
                          (pl.col("u6_gene") < pl.col("h1_gene")).alias("a_on_u6"))


def qvalues() -> pl.DataFrame:
    """Authors' per-line SynLet / buffering q-values: library, context, gene_a, gene_b, q_sl, q_buf."""
    rows = []
    t = _table(4)
    for r in t[1:]:
        if r[4] in LINES:
            rows.append(("SynLet", LINES[r[4]], r[0], r[1], float(r[5]), float(r[6])))
    t = _table(7)
    h = t[0]
    for r in t[1:]:
        if r[4] == "no drug" and r[5] in LINES:
            rows.append(("Apoptosis", LINES[r[5]], r[0], r[1], float(r[h.index("SynLet q-value")]),
                         float(r[h.index("Buffering q-value")])))
    q = pl.DataFrame(rows, schema=["library", "context", "x", "y", "q_sl", "q_buf"], orient="row")
    return q.with_columns(pl.min_horizontal("x", "y").alias("gene_a"), pl.max_horizontal("x", "y").alias("gene_b")) \
        .group_by("library", "context", "gene_a", "gene_b").agg(pl.col("q_sl").min(), pl.col("q_buf").min())


def pair_table() -> pl.DataFrame:
    """Gene-pair GI (replicate-mean LFC) with q-values and SLB labels, per library and line (raw symbols)."""
    g = construct_gi().filter(pl.col("rep") == "mean")
    s = g.group_by("library", "context", "gene_a", "gene_b").agg(pl.col("gi").median().alias("score"), pl.len().alias("n_constructs"))
    s = s.join(qvalues(), on=["library", "context", "gene_a", "gene_b"], how="left")
    s = s.with_columns(pl.col("score").abs().median().over("library", "context").alias("_med"))
    return s.with_columns(
        pl.when(pl.col("context").is_in(NO_LABEL)).then(None)
        .when((pl.col("q_sl") < POS_Q) & (pl.col("score") < 0)).then(1)
        .when((pl.col("q_sl") > NEG_Q) & (pl.col("q_buf") > NEG_Q) & (pl.col("score").abs() < pl.col("_med"))).then(0)
        .otherwise(None).cast(pl.Int8).alias("label")).drop("_med")


def load() -> pl.DataFrame:
    s = pair_table()
    df = s.select(
        pl.lit("human").alias("species"), pl.lit("najm2018").alias("source"), "context",
        pl.lit("CRISPR-KO").alias("mechanism"), "gene_a", "gene_b", "score", pl.lit("gi_additive").alias("score_name"),
        pl.col("q_sl").alias("signif"), pl.lit("synlet_q").alias("signif_name"), "label")
    return finalize(df, "human")


def replicates() -> pl.DataFrame:
    """Per-replicate gene-pair GI (gi_rep1..3 = replicates A..C; C missing for A549, HT-29, MEL-JUSO SynLet and all
    but rep A for OVCAR-8 apoptosis) and per-orientation GI on the replicate-mean LFC (gi_ori1: gene_a on
    SpCas9/U6 and gene_b on SaCas9/H1; gi_ori2: the reverse). Keyed by library, context, gene_a, gene_b."""
    g = construct_gi()
    k = ["library", "context", "gene_a", "gene_b"]
    reps = g.filter(pl.col("rep") != "mean").group_by(*k, "rep").agg(pl.col("gi").median()) \
        .with_columns(pl.col("rep").replace_strict({"A": "gi_rep1", "B": "gi_rep2", "C": "gi_rep3"})) \
        .pivot(on="rep", index=k, values="gi")
    ori = g.filter(pl.col("rep") == "mean").with_columns(
        pl.when(pl.col("a_on_u6")).then(pl.lit("gi_ori1")).otherwise(pl.lit("gi_ori2")).alias("o")) \
        .group_by(*k, "o").agg(pl.col("gi").median()).pivot(on="o", index=k, values="gi")
    r = reps.join(ori, on=k, how="left").sort(k)
    return r.select(*k, *[c for c in ("gi_rep1", "gi_rep2", "gi_rep3", "gi_ori1", "gi_ori2") if c in r.columns])


# ------------------------------------------------------------------------------------------------------
# Checks for data/interim/new_human/najm2018.json


def _auc(y, s) -> float | None:
    from sklearn.metrics import roc_auc_score

    y, s = np.asarray(y), np.asarray(s, float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if y.sum() < 3 or (1 - y).sum() < 3:
        return None
    return float(roc_auc_score(y, s))


def own_replication() -> dict:
    from itertools import permutations

    from slbench.replication import replicate_agreement, split_rule_auroc

    r = replicates()
    p = pair_table().select("library", "context", "gene_a", "gene_b", "label", "q_sl")
    r = r.join(p, on=["library", "context", "gene_a", "gene_b"])
    out: dict = {"per_library_line": {}}
    rep_aucs, ori_aucs, split_rep, split_ori = [], [], [], []
    pooled_y, pooled_s = {}, {}
    for (lib, ctx), d in r.group_by(["library", "context"], maintain_order=True):
        reps = [c for c in ("gi_rep1", "gi_rep2", "gi_rep3") if d[c].null_count() < d.height] if "gi_rep3" in d.columns \
            else [c for c in ("gi_rep1", "gi_rep2") if c in d.columns]
        res = {"pairs": d.height, "replicates": len(reps), "n_pos": int((d["label"] == 1).sum()), "n_neg": int((d["label"] == 0).sum())}
        lab = d.filter(pl.col("label").is_not_null())
        y = lab["label"].to_numpy()
        if len(reps) >= 2:
            dd = d.select(reps).drop_nulls()
            res["replicate_agreement"] = replicate_agreement(dd.rename({c: f"gi_{c}" for c in reps}), reps)
            res["split_rule_2pct_rep_to_rep"] = [split_rule_auroc(dd[a].to_numpy(), dd[b].to_numpy(), pos_q=0.02)
                                                 for a, b in permutations(reps, 2)]
            split_rep += res["split_rule_2pct_rep_to_rep"]
        res["label_auroc_single_replicates"] = {c: _auc(y, -lab[c].to_numpy()) for c in reps}
        o = d.select("gi_ori1", "gi_ori2").drop_nulls()
        res["orientation_spearman"] = float(o.select(pl.corr("gi_ori1", "gi_ori2", method="spearman")).item())
        res["split_rule_2pct_ori"] = [split_rule_auroc(o["gi_ori1"].to_numpy(), o["gi_ori2"].to_numpy(), pos_q=0.02),
                                      split_rule_auroc(o["gi_ori2"].to_numpy(), o["gi_ori1"].to_numpy(), pos_q=0.02)]
        split_ori += res["split_rule_2pct_ori"]
        res["label_auroc_single_orientations"] = {c: _auc(y, -lab[c].to_numpy()) for c in ("gi_ori1", "gi_ori2")}
        for c in reps:
            pooled_y.setdefault("rep", []).append(y)
            pooled_s.setdefault("rep", []).append(-lab[c].to_numpy())
        for c in ("gi_ori1", "gi_ori2"):
            pooled_y.setdefault("ori", []).append(y)
            pooled_s.setdefault("ori", []).append(-lab[c].to_numpy())
        out["per_library_line"][f"{lib}|{ctx}"] = res
    # labels are from the authors' q (computed on all replicates), so single-replicate AUROCs are optimistic;
    # the SLB-rule transfer between orientations is the independent check (disjoint constructs and guides).
    out["pooled_label_auroc_single_replicate"] = _auc(np.concatenate(pooled_y["rep"]), np.concatenate(pooled_s["rep"]))
    out["pooled_label_auroc_single_orientation"] = _auc(np.concatenate(pooled_y["ori"]), np.concatenate(pooled_s["ori"]))
    out["mean_split_rule_2pct_rep_to_rep"] = float(np.mean(split_rep))
    out["mean_split_rule_2pct_orientation"] = float(np.mean(split_ori))
    # orientation-held-out label rule: positives = pooled z <= -3 of one orientation (z over all pairs in the
    # library x line), negatives = |z| < 1; scored by the other orientation, pooled over library x line
    ys, ss = [], []
    for (_lib, _ctx), d in r.group_by(["library", "context"]):
        o = d.select("gi_ori1", "gi_ori2").drop_nulls()
        for a, b in (("gi_ori1", "gi_ori2"), ("gi_ori2", "gi_ori1")):
            x1 = o[a].to_numpy()
            z = (x1 - np.median(x1)) / (1.4826 * np.median(np.abs(x1 - np.median(x1))))
            m = (z <= -3) | (np.abs(z) < 1)
            ys.append((z[m] <= -3).astype(int))
            ss.append(-o[b].to_numpy()[m])
    y = np.concatenate(ys)
    out["orientation_robust_z3_rule_pooled"] = {"n_pos": int(y.sum()), "auroc": _auc(y, np.concatenate(ss))}
    return out


def checks() -> dict:
    from slbench import contexts, features
    from slbench.sources.human_screens.najm2023 import cross_study

    m = load()
    lab = m.filter(pl.col("label").is_not_null())
    out: dict = {"own_replication": own_replication()}
    genes = set(m["gene_a"]) | set(m["gene_b"])
    par = features.read(Path("data/slb"), "paralogs").filter(pl.col("species") == "human")
    pset = {tuple(sorted(p)) for p in par.select("a", "b").iter_rows()}
    is_par = np.array([(a, b) in pset for a, b in m.select("gene_a", "gene_b").iter_rows()])
    lp = (m["label"] == 1).fill_null(False).to_numpy()
    out["counts"] = {"n_pairs": m.height, "n_unique_pairs": m.select("gene_a", "gene_b").unique().height,
                     "n_pos": int(lp.sum()), "n_neg": int((m["label"] == 0).sum()), "unique_genes": len(genes),
                     "paralog_fraction": float(is_par.mean()), "paralog_pos": int((is_par & lp).sum()),
                     "positives": sorted(f"{c}:{a}-{b}" for c, a, b in m.filter(pl.col("label") == 1).select("context", "gene_a", "gene_b").iter_rows()),
                     "per_context": {c: {"pairs": g.height, "pos": int((g["label"] == 1).sum()), "neg": int((g["label"] == 0).sum())}
                                     for (c,), g in m.group_by(["context"])}}
    se = pl.read_parquet("data/slb/gene_single_effects.parquet").filter(pl.col("species") == "human")
    f = {g: v for g, v in zip(se["gene"], se["single_effect"]) if v is not None}
    le = pl.read_parquet("data/slb/features/line_effects.parquet")
    fl = {}
    for ctx in m["context"].unique():
        dep = contexts.lookup(ctx)["depmap_id"]
        for g, v in le.filter(pl.col("depmap_id") == dep).select("gene", "effect").iter_rows():
            if v is not None:
                fl[(ctx, g)] = v
    # screen's own singles: gene-level median over guides of the guide x control LFC (both positions)
    from slbench import ids
    cc = constructs().filter(pl.col("library") == "SynLet")
    ss = {}
    for (line,), d in cc.group_by(["line"]):
        w = d.group_by("u6_gene", "h1_gene", "u6_guide", "h1_guide", "pdna").agg(pl.col("reads").mean())
        p, e = w["pdna"].to_numpy(), w["reads"].to_numpy()
        lfc = np.log2((e + 1) / (e + 1).sum()) - np.log2((p + 1) / (p + 1).sum())
        u6, h1 = w["u6_gene"].to_numpy(), w["h1_gene"].to_numpy()
        cu, ch = np.isin(u6, list(CONTROL)), np.isin(h1, list(CONTROL))
        lfc = lfc - np.median(lfc[cu & ch])
        for gname in set(u6) - CONTROL - DROP:
            ss[(LINES[line], ids.resolve("human", pl.Series([gname]))[0])] = float(np.median(lfc[((u6 == gname) & ch) | ((h1 == gname) & cu)]))
    y = lab["label"].to_numpy()
    rows = list(lab.select("context", "gene_a", "gene_b").iter_rows())
    out["fitness"] = {
        "depmap_mean_sum": _auc(y, -np.array([f.get(a, np.nan) + f.get(b, np.nan) for _, a, b in rows])),
        "depmap_line_sum": _auc(y, -np.array([fl.get((c, a), np.nan) + fl.get((c, b), np.nan) for c, a, b in rows])),
        "screen_singles_sum_synlet": _auc(y, -np.array([ss.get((c, a), np.nan) + ss.get((c, b), np.nan) for c, a, b in rows])),
        "paralog_flag": _auc(y, is_par[m["label"].is_not_null().to_numpy()].astype(float)),
    }
    out["cross_study"] = cross_study(m)
    return out
