"""Burgold et al. 2025 Nat Commun (doi 10.1038/s41467-025-67256-9, CC BY 4.0): Sanger ENCORE "COLO1".

Design: tRNA-spaced dual-guide Cas9 library in HT-29 (Cas9-expressing), 40 anchor genes x 404 library
genes (2 guides per gene -> 4 guide pairs per gene pair), 40 x 40 anchor-anchor pairs (both
orientations), a small set of GI-control pairs (known paralogs / BioGRID pairs), singletons (each gene's
guides paired with guides against 3 non-expressed cutting-control genes ADAD1, CYLC2, KLK12) and
control pairs (non-essential x non-essential, intergenic x intergenic, non-targeting, essential x control).

Raw counts: GitHub ibarrioh/DualGuide_COLO1 Analysis/ENCORE/input/COLO1_RUNMERGED_EXACT_ANNOTATED.txt
  lib.COLO.1                      plasmid library
  SIDM00136_CPID2437/2440/2443    HT-29 Cas9-negative parental, 3 reps (no essential-gene depletion)
  SIDM00136_CPID1020/1023/1026    HT-29 Cas9, 3 biological reps (essential x control depleted ~ -1.4 log2)
The column roles were identified from the data (essential-control depletion), consistent with the repo
readme ("colo1_c91.rds: log fold changes of HT29 vs parental (cas9 negative)").

The authors publish no stringent GI call table (GEMINI scores mentioned in the text are not released; the
supplementary data are the pilot library and source data), so the screen is re-scored from counts with
the SLB additive recipe (spidr_replicate_gi):
  reference ("start") = mean normalised counts of the 3 Cas9-negative parental replicates: same cells,
    same transduction and culture time, so it removes library skew and any Cas9-independent vector
    effects; only Cas9 cutting differs. (ref="plasmid" is available for a sensitivity check.)
  per Cas9 replicate: LFC = log2 norm(Cas9) - log2 norm(ref), guide pairs with ref counts in the bottom
    2% dropped; centred on cutting control x control pairs (non-essential x non-essential and intergenic
    x intergenic, i.e. two cuts, matching the two cuts of every gene x gene and gene x control construct);
    f_g = median LFC over gene x cutting-control guide pairs; GI = LFC - f_a - f_b; median over guide
    pairs (anchor x anchor pairs pool both orientations).
  per-anchor normalisation (added to the recipe): the raw additive GI has anchor-specific offsets
    (median GI over partners -0.45 for WEE1 ... +0.20 for CHEK1) and spreads (MAD 2x larger for strongly
    essential anchors), which made WEE1/PRMT1 hubs of 56/80 raw positives and let single-gene fitness
    predict labels (DepMap AUROC 0.79). Each replicate's GI is therefore centred on the anchor's median
    and divided by its robust SD (see _anchor_normalise); this also raised independent-reference
    replication and same-line agreement with Flister 2025 HT-29 (see burgold2025.md).
Measurement score = z-score over gene pairs of the mean of the 3 replicate GIs (negative = SL).
Label: positive z <= -3 and GI < 0 in all three replicates; negative |z| < 1; else null.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from slbench.sources import finalize

RAW = Path("data/raw/burgold2025")
COUNTS = RAW / "COLO1_RUNMERGED_EXACT_ANNOTATED.txt"
PLASMID = "lib.COLO.1"
PARENTAL = ["SIDM00136_CPID2437", "SIDM00136_CPID2440", "SIDM00136_CPID2443"]
CAS9 = ["SIDM00136_CPID1020", "SIDM00136_CPID1023", "SIDM00136_CPID1026"]
CUT_CTRL_GENES = {"ADAD1", "CYLC2", "KLK12"}  # partners of every singleton construct
CTRL_CTRL = {"NONESSENTIAL-NONESSENTIAL", "INTERGENIC-INTERGENIC"}
PAIR_CLASSES = {"AnchorCombinations", "LibraryCombinations", "GIControlsCombinations"}
SINGLE_CLASSES = {"AnchorSingletons", "LibrarySingletons", "GIControlsSingletons"}
CONTEXT = "HT-29"


def counts() -> pl.DataFrame:
    d = pl.read_csv(COUNTS, separator="\t", infer_schema_length=0)
    return d.with_columns([pl.col(c).str.strip_chars().cast(pl.Float64) for c in [PLASMID, *PARENTAL, *CAS9]])


def _norm(x: np.ndarray) -> np.ndarray:
    return np.log2((x + 1) / (x + 1).sum())


def _ref(d: pl.DataFrame, ref: str) -> np.ndarray:
    if ref == "parental":  # depth-normalised mean of the 3 parental replicates
        r = d.select(PARENTAL).to_numpy()
        return (r / r.sum(0) * r.sum(0).mean()).mean(1)
    if ref == "plasmid":
        return d[PLASMID].to_numpy()
    raise ValueError(ref)


def replicate_gi(ref: str = "parental", paired: bool = False, normalise: bool = True) -> pl.DataFrame:
    """Per-Cas9-replicate additive GI per unordered gene pair: gene_a, gene_b, gi_rep1..3 (raw symbols).

    paired=True scores Cas9 replicate i against parental replicate i alone (CPID order), so the three
    replicate GIs share no reference counts; with a shared reference the reference's own sampling noise
    is common to all replicates and inflates their agreement (Spearman 0.23 shared vs 0.09 paired).
    normalise=True applies the per-anchor robust standardisation (_anchor_normalise) to each replicate.
    """
    d = counts()
    if paired:
        outs = [_gi_one(d, d[p].to_numpy(), c, i) for i, (p, c) in enumerate(zip(PARENTAL, CAS9), start=1)]
    else:
        r = _ref(d, ref)
        outs = [_gi_one(d, r, c, i) for i, c in enumerate(CAS9, start=1)]
    out = outs[0]
    for o in outs[1:]:
        out = out.join(o, on=["gene_a", "gene_b"])
    out = out.sort("gene_a", "gene_b")
    if normalise:
        out = out.with_columns([pl.Series(c, _anchor_normalise(out["gene_a"].to_numpy(), out["gene_b"].to_numpy(),
                                                                out[c].to_numpy())) for c in ("gi_rep1", "gi_rep2", "gi_rep3")])
    return out


def anchors() -> set[str]:
    d = pl.read_csv(COUNTS, separator="\t", infer_schema_length=0, columns=["Note", "Gene1"])
    return set(d.filter(pl.col("Note") == "AnchorCombinations")["Gene1"])


def _anchor_normalise(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per-anchor robust standardisation of additive GI (each anchor is effectively its own screen of
    ~440 partners): subtract the anchor's median GI over its partners, divide by 1.4826 x its MAD.
    Anchor x anchor pairs use the mean of the two medians and the geometric mean of the two scales;
    the few GI-control pairs without an anchor are only divided by the median anchor scale."""
    anc = anchors()
    st = pl.DataFrame({"g": np.r_[a, b], "x": np.r_[x, x]}).filter(pl.col("g").is_in(list(anc))) \
        .group_by("g").agg(pl.col("x").median().alias("med"),
                           ((pl.col("x") - pl.col("x").median()).abs().median() * 1.4826).alias("sc"))
    med, sc = dict(zip(st["g"], st["med"])), dict(zip(st["g"], st["sc"]))
    ma = np.array([med.get(v, np.nan) for v in a])
    mb = np.array([med.get(v, np.nan) for v in b])
    sa = np.array([sc.get(v, np.nan) for v in a])
    sb = np.array([sc.get(v, np.nan) for v in b])
    both = np.c_[ma, mb]
    n = (~np.isnan(both)).sum(1)
    centre = np.where(n > 0, np.nansum(both, 1) / np.maximum(n, 1), 0.0)
    scale = np.where(np.isnan(sa), sb, np.where(np.isnan(sb), sa, np.sqrt(sa * sb)))
    scale = np.where(np.isnan(scale), np.median(list(sc.values())), scale)
    return (x - centre) / scale


def _gi_one(d: pl.DataFrame, r: np.ndarray, col: str, i: int) -> pl.DataFrame:
    keep = r >= np.quantile(r, 0.02)
    base = _norm(r)
    note = d["MyNote"].to_numpy()
    g1, g2 = d["Gene1"].to_numpy(), d["Gene2"].to_numpy()
    cc = np.isin(note, list(CTRL_CTRL)) & keep
    is_single = np.isin(note, list(SINGLE_CLASSES)) & keep
    s1 = is_single & np.isin(g2, list(CUT_CTRL_GENES)) & ~np.isin(g1, list(CUT_CTRL_GENES))
    s2 = is_single & np.isin(g1, list(CUT_CTRL_GENES)) & ~np.isin(g2, list(CUT_CTRL_GENES))
    pair = np.isin(note, list(PAIR_CLASSES)) & keep & (g1 != g2) \
        & ~np.isin(g1, list(CUT_CTRL_GENES)) & ~np.isin(g2, list(CUT_CTRL_GENES))
    lfc = _norm(d[col].to_numpy()) - base
    lfc = lfc - np.median(lfc[cc])
    single = pl.DataFrame({"g": np.r_[g1[s1], g2[s2]], "lfc": np.r_[lfc[s1], lfc[s2]]}) \
        .group_by("g").agg(pl.col("lfc").median().alias("f"))
    f = dict(zip(single["g"], single["f"]))
    fa = np.array([f.get(v, np.nan) for v in g1[pair]])
    fb = np.array([f.get(v, np.nan) for v in g2[pair]])
    du = pl.DataFrame({"g1": g1[pair], "g2": g2[pair], "gi": lfc[pair] - fa - fb}).drop_nans("gi").with_columns(
        pl.min_horizontal("g1", "g2").alias("gene_a"), pl.max_horizontal("g1", "g2").alias("gene_b"))
    return du.group_by("gene_a", "gene_b").agg(pl.col("gi").median().alias(f"gi_rep{i}"))


def singles(ref: str = "parental") -> pl.DataFrame:
    """Screen's own single-gene effects (mean over Cas9 reps of median gene x cutting-control LFC)."""
    d = counts()
    r = _ref(d, ref)
    base = _norm(r)
    keep = r >= np.quantile(r, 0.02)
    note, g1, g2 = d["MyNote"].to_numpy(), d["Gene1"].to_numpy(), d["Gene2"].to_numpy()
    cc = np.isin(note, list(CTRL_CTRL)) & keep
    is_single = np.isin(note, list(SINGLE_CLASSES)) & keep
    s1 = is_single & np.isin(g2, list(CUT_CTRL_GENES)) & ~np.isin(g1, list(CUT_CTRL_GENES))
    s2 = is_single & np.isin(g1, list(CUT_CTRL_GENES)) & ~np.isin(g2, list(CUT_CTRL_GENES))
    fs = []
    for col in CAS9:
        lfc = _norm(d[col].to_numpy()) - base
        lfc = lfc - np.median(lfc[cc])
        fs.append(pl.DataFrame({"g": np.r_[g1[s1], g2[s2]], "lfc": np.r_[lfc[s1], lfc[s2]]})
                  .group_by("g").agg(pl.col("lfc").median()))
    x = fs[0]
    for k, y in enumerate(fs[1:], start=2):
        x = x.join(y, on="g", suffix=f"_{k}")
    return x.select(pl.col("g").alias("gene"), pl.mean_horizontal(pl.exclude("g")).alias("f"))


def _scored(ref: str = "parental") -> pl.DataFrame:
    gi = replicate_gi(ref)
    pooled = gi.select("gi_rep1", "gi_rep2", "gi_rep3").mean_horizontal()
    z = (pooled - pooled.mean()) / pooled.std()
    allneg = (pl.col("gi_rep1") < 0) & (pl.col("gi_rep2") < 0) & (pl.col("gi_rep3") < 0)
    return gi.with_columns(z.alias("score")).with_columns(
        pl.when((pl.col("score") <= -3) & allneg).then(1).when(pl.col("score").abs() < 1).then(0)
        .otherwise(None).cast(pl.Int8).alias("label"))


def load() -> pl.DataFrame:
    df = _scored().select(
        pl.lit("human").alias("species"), pl.lit("burgold2025").alias("source"), pl.lit(CONTEXT).alias("context"),
        pl.lit("CRISPR-KO").alias("mechanism"), "gene_a", "gene_b", "score",
        pl.lit("zGI_additive").alias("score_name"),
        pl.lit(None, pl.Float64).alias("signif"), pl.lit(None, pl.String).alias("signif_name"), "label",
    )
    return finalize(df, "human")


def replicates() -> pl.DataFrame:
    """Per-Cas9-replicate additive GI (3 biological replicates), HGNC-resolved and ordered.

    Each Cas9 replicate is scored against its own parental replicate (paired=True) so the columns are
    fully independent measurements; the measurement itself (load) uses the mean parental reference.
    """
    from slbench import ids

    gi = replicate_gi(paired=True)
    gi = gi.with_columns(ids.resolve("human", gi["gene_a"]).alias("a"), ids.resolve("human", gi["gene_b"]).alias("b")) \
        .filter(pl.col("a").is_not_null() & pl.col("b").is_not_null() & (pl.col("a") != pl.col("b")))
    gi = gi.with_columns(pl.min_horizontal("a", "b").alias("gene_a"), pl.max_horizontal("a", "b").alias("gene_b"))
    return gi.group_by("gene_a", "gene_b").agg(pl.col("gi_rep1", "gi_rep2", "gi_rep3").mean()) \
        .with_columns(pl.lit(CONTEXT).alias("context")) \
        .select("gene_a", "gene_b", "context", "gi_rep1", "gi_rep2", "gi_rep3").sort("gene_a", "gene_b")
