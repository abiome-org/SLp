"""Lenoir, DeWeirdt et al. 2021 Nat Commun (doi 10.1038/s41467-021-26867-8, CC BY): enCas12a lipid-metabolism
GI screen in two AML lines (MOLM-13, NOMO-1).

Design ("Experiment_Main" in the library): 8 anchor genes (ACACA, C12orf49/SPRING1, FASN, GPX4, PSTK, PTEN,
SREBF1, TP53) x 99 library genes (incl. the anchors), 3 x 3 guides per pair = 756 distinct non-self gene pairs.
Each gene is also paired with 15 non-expressed "control" genes (singles). The authors' gene_residuals.csv
(github.com/PeterDeWeirdt/FASTS, 1,658 rows per condition) also contains the 891 gene x control-gene rows and
self pairs; those are dropped here. Two replicate infections (A, B) and two timepoints (day 14, day 21);
the authors' residuals are computed on the A/B mean.

Measurement: day 21 (the later timepoint), score = authors' combined pair_z_score (gnt linear anchor model,
negative = synergistic/SL), signif = BH FDR within condition.

Label rule (SLB):
  positive: pair_z_score <= -3 and fdr_bh < 0.05 (day 21)
  negative: |pair_z_score| < 1
  else null.

replicates(): per-replicate (A, B) additive GI recomputed from the published lognorm table (day 21), following
the SLB spidr recipe: LFC = lognorm - pDNA (pDNA |z|<3 filter, as the authors), centred on non-essential gene x
control-gene constructs, f_g = median LFC of gene x control-gene constructs, GI = LFC - f_a - f_b, median over
guide pairs.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import polars as pl

from slbench.sources import finalize

RAW = Path("data/raw/lenoir2021")
GENE = RAW / "gene_residuals.csv"
LOGNORM = RAW / "lognorm-FASN_MOLM13_NOMO-1_AKS_ALG_v2.txt"
LIB = RAW / "Cas12_FASN_GI_12k_library.csv"

CONTROL_GENES = ["CXorf66", "GPX6", "BMP15", "RXFP3", "PRSS37", "TSSK1B", "SLC36A3", "C3orf30",
                 "NDST4", "RNF17", "OPN5", "RAX", "GSX2", "GLRA2", "PPP3R2"]
LINES = {"MOLM13": "MOLM-13", "NOMO-1": "NOMO-1"}
DAY = "21"
POS_Z, POS_FDR, NEG_Z = -3.0, 0.05, 1.0


def pair_table(day: str = DAY) -> pl.DataFrame:
    """Authors' gene-level residuals for real gene pairs (no controls, no self pairs) at one timepoint."""
    g = pl.read_csv(GENE)
    g = g.with_columns(pl.col("condition").str.split("_").list.first().alias("line"),
                       pl.col("condition").str.split("_").list.last().alias("day"))
    g = g.filter((pl.col("day") == day) & (pl.col("gene_a") != pl.col("gene_b"))
                 & ~pl.col("gene_a").is_in(CONTROL_GENES) & ~pl.col("gene_b").is_in(CONTROL_GENES))
    return g.with_columns(pl.col("line").replace_strict(LINES).alias("context"))


def _label(z: pl.Expr, fdr: pl.Expr) -> pl.Expr:
    return (pl.when((z <= POS_Z) & (fdr < POS_FDR)).then(1)
            .when(z.abs() < NEG_Z).then(0).otherwise(None).cast(pl.Int8))


def load() -> pl.DataFrame:
    d = pair_table().with_columns(_label(pl.col("pair_z_score"), pl.col("fdr_bh")).alias("label"))
    df = d.select(
        pl.lit("human").alias("species"), pl.lit("lenoir2021").alias("source"), "context",
        pl.lit("CRISPR-Cas12a").alias("mechanism"), "gene_a", "gene_b",
        pl.col("pair_z_score").alias("score"), pl.lit("gnt_pair_z_d21").alias("score_name"),
        pl.col("fdr_bh").alias("signif"), pl.lit("fdr_bh").alias("signif_name"), "label",
    )
    return finalize(df, "human")


def _library() -> pl.DataFrame:
    t = LIB.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    return pl.read_csv(io.StringIO(t)).select("GENE_1", "GENE_2", "Type", "SGRNA_1", "SGRNA_2")


def construct_lfc() -> pl.DataFrame:
    """Long table: construct (g1, g2, sg1, sg2, Type) x (line, rep, day) LFC vs pDNA."""
    ln = pl.read_csv(LOGNORM, separator="\t").drop(";;;;;;")
    pd_col = "pDNA;FASN;RDA_052;;;;"
    p = ln[pd_col].to_numpy()
    ln = ln.filter(np.abs((p - p.mean()) / p.std(ddof=1)) < 3)  # authors' pDNA filter
    ln = ln.with_columns(pl.col("Construct IDs").str.split(";").list.get(1).str.split(":").alias("_s")).with_columns(
        pl.col("_s").list.get(0).alias("SGRNA_1"), pl.col("_s").list.get(2).alias("SGRNA_2"))
    samples = [c for c in ln.columns if c.count(";") >= 5 and not c.startswith("pDNA")]
    long = ln.select("SGRNA_1", "SGRNA_2", *[(pl.col(c) - pl.col(pd_col)).alias(c) for c in samples]).unpivot(
        index=["SGRNA_1", "SGRNA_2"], variable_name="sample", value_name="lfc")
    parts = pl.col("sample").str.split(";")
    long = long.with_columns(parts.list.get(0).alias("line"), parts.list.get(3).alias("rep"), parts.list.get(4).alias("day"))
    return long.join(_library(), on=["SGRNA_1", "SGRNA_2"], how="inner")


def replicate_gi(day: str = DAY) -> pl.DataFrame:
    """Additive GI per (line, rep) for Experiment_Main pairs, wide: gene_a, gene_b, context, gi_rep1 (A), gi_rep2 (B)."""
    d = construct_lfc().filter(pl.col("day") == day)
    ctrl = pl.col("Type") == "Overall_Control_Non_Ess"
    d = d.with_columns((pl.col("lfc") - pl.col("lfc").filter(ctrl).median().over("line", "rep")).alias("lfc"))
    # singles: gene x control gene, either orientation
    s1 = d.filter(pl.col("GENE_2").is_in(CONTROL_GENES) & ~pl.col("GENE_1").is_in(CONTROL_GENES)).select(
        "line", "rep", pl.col("GENE_1").alias("g"), "lfc")
    s2 = d.filter(pl.col("GENE_1").is_in(CONTROL_GENES) & ~pl.col("GENE_2").is_in(CONTROL_GENES)).select(
        "line", "rep", pl.col("GENE_2").alias("g"), "lfc")
    f = pl.concat([s1, s2]).group_by("line", "rep", "g").agg(pl.col("lfc").median().alias("f"))
    x = d.filter((pl.col("Type") == "Experiment_Main") & (pl.col("GENE_1") != pl.col("GENE_2")))
    x = x.join(f.rename({"g": "GENE_1", "f": "fa"}), on=["line", "rep", "GENE_1"]) \
         .join(f.rename({"g": "GENE_2", "f": "fb"}), on=["line", "rep", "GENE_2"])
    x = x.with_columns((pl.col("lfc") - pl.col("fa") - pl.col("fb")).alias("gi"),
                       pl.min_horizontal("GENE_1", "GENE_2").alias("gene_a"),
                       pl.max_horizontal("GENE_1", "GENE_2").alias("gene_b"))
    agg = x.group_by("line", "rep", "gene_a", "gene_b").agg(pl.col("gi").median())
    w = agg.with_columns(pl.col("rep").replace_strict({"A": "gi_rep1", "B": "gi_rep2"}).alias("r")).pivot(
        on="r", index=["line", "gene_a", "gene_b"], values="gi")
    return w.with_columns(pl.col("line").replace_strict(LINES).alias("context")).select(
        "gene_a", "gene_b", "context", "gi_rep1", "gi_rep2").drop_nulls()


def replicates() -> pl.DataFrame:
    """Independent replicate infections A (gi_rep1) and B (gi_rep2), day 21, symbols HGNC-resolved and ordered."""
    from slbench import ids

    w = replicate_gi()
    w = w.with_columns(ids.resolve("human", w["gene_a"]).alias("gene_a"), ids.resolve("human", w["gene_b"]).alias("gene_b"))
    return w.with_columns(pl.min_horizontal("gene_a", "gene_b").alias("gene_a"),
                          pl.max_horizontal("gene_a", "gene_b").alias("gene_b")).sort("context", "gene_a", "gene_b")
