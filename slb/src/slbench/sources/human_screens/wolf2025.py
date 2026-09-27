"""Wolf, Leippe et al. 2025 Mol Syst Biol: genetic interaction map of the human SLC superfamily (HCT 116).

The same ~33k SLC x SLC gene-pair design was screened with enCas12a (4 sublibraries, 3 replicates) and
SpCas9 (1 library, 2 replicates at the scoring time point) in HCT 116 in standard high-glucose medium
(every SLC x SLC sample is untreated). The SLC x enzyme arm (glucose/galactose/antimycin/hypoxia) and
the Cas12a metal pilot (Dataset EV1) are not used. The two nucleases are separate measurement rows
(mechanism CRISPR-Cas12a vs CRISPR-KO, context "HCT 116"); the SLB build collapses them per pair.

Measurements (load): the authors' own scoring of each nuclease (Dataset EV2 = Cas12a, EV3 = Cas9,
sheet "scoring of genetic interactions"): score = week-3 dLFC = LFC_obs - LFC_exp (negative = synthetic
lethal), signif = week-3 padj (Cas12a: rep-pooled padj at week 3; Cas9: padj over reps and weeks 2-4).

Labels: see LABEL_RULE. The authors' call is stringent (Cas12a: padj < 0.1 in each of 3 replicates,
dLFC < -0.3, plus saturation and week-1->3 trend filters; Cas9: pooled padj < 0.005, dLFC < -0.3, same
filters). Re-scoring from counts with the additive z <= -3 recipe was tried and rejected: those calls
are dominated by the handful of essential SLCs (fitness-only AUROC 0.90-0.92) and the Cas12a ones do
not replicate in Cas9 (0.52); see data/interim/new_human/wolf2025.md.

Replicates (replicates): re-scored from GEO GSE269905 counts with the spidr_replicate_gi recipe at week 3,
each replicate against its OWN plasmid-input replicate (a shared input makes replicates share the
input-sampling noise and inflates their agreement; for Cas9 that inflation is most of the apparent
agreement).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl
import polars as pl

from slbench.sources import finalize

RAW = Path("data/raw/wolf2025")
CONTEXT = "HCT 116"

EV = {  # mechanism: (file, dLFC week-3 column index, padj column index, call column index) in the scoring sheet
    "CRISPR-Cas12a": ("44320_2025_105_MOESM5_ESM.xlsx", 22, 26, 32),
    "CRISPR-KO": ("44320_2025_105_MOESM6_ESM.xlsx", 22, 24, 25),
}

# Arms whose labels pass the SLB gate. Cas12a: own replicates 0.71-0.89, recovered by Cas9 at 0.73-0.77.
# Cas9 (2 replicates, low input depth) fails both gates: independent-input replicates 0.59-0.65 and
# recovered by Cas12a at only 0.61-0.63, so its rows are kept as measurements with label = null.
LABELLED_ARMS = ("CRISPR-Cas12a",)

LABEL_RULE = (
    "Per nuclease, authors' call at week 3. Positive: interaction == 'synthetic_lethal'. Negative: "
    "interaction == 'neutral', |dLFC| below that screen's median |dLFC| and padj > 0.1. Otherwise null "
    "(incl. synthetic_viable and pairs passing padj/dLFC but removed by the authors' filters)."
)

# Counts: (file, week-3 columns per replicate, matching plasmid-input columns)
CAS12A = [
    ("GSE269905_Cas12a-SLCxSLC_sublib1_counts.txt.gz", ["week3_v0_1", "week3_v0_2", "week3_v0_3"],
     ["input_v0_1", "input_v0_2", "input_v0_3"]),
    ("GSE269905_Cas12a-SLCxSLC_sublib2_counts.txt.gz", ["week3_1", "week3_2", "week3_3"], ["input_1", "input_2", "input_3"]),
    ("GSE269905_Cas12a-SLCxSLC_sublib3_4_counts.txt.gz", ["week3_1", "week3_2", "week3_3"], ["input_1", "input_2", "input_3"]),
]
CAS9 = [("GSE269905_Cas9-SLCxSLC_counts.txt.gz", ["week3_1", "week3_2"], ["input_1", "input_2"])]
ARMS = {"CRISPR-Cas12a": CAS12A, "CRISPR-KO": CAS9}


def authors(mechanism: str) -> pl.DataFrame:
    """Authors' per-pair week-3 dLFC, padj and call (one row per sheet row; control pairs repeat per sublibrary)."""
    fname, i_d, i_p, i_c = EV[mechanism]
    wb = openpyxl.load_workbook(RAW / fname, read_only=True)
    rows = []
    for r in wb["scoring of genetic interactions"].iter_rows(min_row=3, values_only=True):
        if r[1] and r[2] and isinstance(r[i_d], (int, float)):
            rows.append((r[1], r[2], float(r[i_d]), float(r[i_p]) if isinstance(r[i_p], (int, float)) else None, r[i_c]))
    wb.close()
    return pl.DataFrame(rows, schema={"gene_a": pl.String, "gene_b": pl.String, "dlfc": pl.Float64,
                                      "padj": pl.Float64, "call": pl.String}, orient="row")


def load(labelled_arms: tuple[str, ...] = LABELLED_ARMS) -> pl.DataFrame:
    parts = []
    for mech in EV:
        a = authors(mech)
        med = a["dlfc"].abs().median()
        parts.append(a.select(
            pl.lit("human").alias("species"), pl.lit("wolf2025").alias("source"), pl.lit(CONTEXT).alias("context"),
            pl.lit(mech).alias("mechanism"), "gene_a", "gene_b", pl.col("dlfc").alias("score"),
            pl.lit("dLFC_week3").alias("score_name"), pl.col("padj").alias("signif"), pl.lit("padj").alias("signif_name"),
            pl.when(pl.col("call") == "synthetic_lethal").then(1)
            .when((pl.col("call") == "neutral") & (pl.col("dlfc").abs() < med) & (pl.col("padj") > 0.1)).then(0)
            .otherwise(None).cast(pl.Int8).alias("label"),
        ).with_columns(pl.col("label") if mech in labelled_arms else pl.lit(None, pl.Int8).alias("label")))
    return finalize(pl.concat(parts), "human")


# ---------------------------------------------------------------------------------------------
# Per-replicate GI from raw counts

def _gene(col: str) -> pl.Expr:
    g = pl.col(col)  # cutting controls: olfactory-receptor guides (Cas9) / "CTRL" (Cas12a)
    return pl.when(g.str.contains(r"^OR\d") | (g == "CTRL")).then(pl.lit("CONTROL")).otherwise(g)


def _screen_gi(fname: str, reps: list[str], inputs: list[str], shared_input: bool) -> tuple[pl.DataFrame, pl.DataFrame]:
    d = pl.read_csv(RAW / fname, separator="\t", infer_schema_length=10000)
    if "library" in d.columns:  # Cas9: drop the spiked-in non-SLC control sublibrary (10x input depth, own scale)
        d = d.filter(pl.col("library") != "Cas9-Control")
    pair = "gene pair" if "gene pair" in d.columns else "Gene"
    d = d.with_columns(pl.col(pair).str.split_exact("-", 1).struct.rename_fields(["x", "y"])).unnest(pair)
    d = d.with_columns(_gene("x").alias("g1"), _gene("y").alias("g2"))
    t0_all = d.select(inputs).mean_horizontal().to_numpy().astype(float)
    keep = t0_all >= np.quantile(t0_all, 0.02)
    g1, g2 = d["g1"].to_numpy(), d["g2"].to_numpy()
    gis, sfs = [], []
    for i, (col, inp) in enumerate(zip(reps, inputs), start=1):
        t0 = t0_all if shared_input else d[inp].to_numpy().astype(float)
        te = d[col].to_numpy().astype(float)
        lfc = np.log2((te + 1) / (te + 1).sum()) - np.log2((t0 + 1) / (t0 + 1).sum())
        lfc = lfc - np.median(lfc[keep & (g1 == "CONTROL") & (g2 == "CONTROL")])
        x = pl.DataFrame({"g1": g1, "g2": g2, "lfc": lfc}).filter(pl.Series(keep))
        single = pl.concat([
            x.filter(pl.col("g2") == "CONTROL").select(pl.col("g1").alias("g"), "lfc"),
            x.filter(pl.col("g1") == "CONTROL").select(pl.col("g2").alias("g"), "lfc"),
        ]).filter(pl.col("g") != "CONTROL").group_by("g").agg(pl.col("lfc").median().alias("f"))
        f = dict(zip(single["g"], single["f"]))
        du = x.filter((pl.col("g1") != "CONTROL") & (pl.col("g2") != "CONTROL") & (pl.col("g1") != pl.col("g2")))
        fa = np.array([f.get(v, np.nan) for v in du["g1"]])
        fb = np.array([f.get(v, np.nan) for v in du["g2"]])
        du = du.with_columns(pl.Series("gi", du["lfc"].to_numpy() - fa - fb)).drop_nans("gi").with_columns(
            pl.min_horizontal("g1", "g2").alias("gene_a"), pl.max_horizontal("g1", "g2").alias("gene_b"))
        gis.append(du.group_by("gene_a", "gene_b").agg(pl.col("gi").median().alias(f"gi_rep{i}")))
        sfs.append(single.rename({"g": "gene", "f": f"f_rep{i}"}))
    gi, sf = gis[0], sfs[0]
    for g, s in zip(gis[1:], sfs[1:]):
        gi, sf = gi.join(g, on=["gene_a", "gene_b"]), sf.join(s, on="gene")
    return gi, sf


def arm_gi(mechanism: str, shared_input: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-replicate GI for one nuclease (sublibraries concatenated; pairs in >1 sublibrary averaged) and
    per-replicate single-gene effects f_g."""
    out = [_screen_gi(f, r, i, shared_input) for f, r, i in ARMS[mechanism]]
    gi = pl.concat([o[0] for o in out]).group_by("gene_a", "gene_b").agg(pl.col("^gi_rep\\d$").mean())
    sf = pl.concat([o[1] for o in out]).group_by("gene").agg(pl.col("^f_rep\\d$").mean())
    return gi.sort("gene_a", "gene_b"), sf


def replicates() -> pl.DataFrame:
    """Per-replicate additive GI at week 3 (each replicate vs its own input replicate), one row per
    (mechanism, pair): Cas12a gi_rep1..3, Cas9 gi_rep1..2 (gi_rep3 null). Gene symbols as in the library.
    Join the two mechanisms on (gene_a, gene_b) for the Cas9-vs-Cas12a check."""
    parts = []
    for mech in ARMS:
        gi, _ = arm_gi(mech)
        parts.append(gi.with_columns(pl.lit(CONTEXT).alias("context"), pl.lit(mech).alias("mechanism")))
    return pl.concat(parts, how="diagonal").select("gene_a", "gene_b", "context", "mechanism", "gi_rep1", "gi_rep2", "gi_rep3")
