"""Hayward, Vaitsiankova et al. 2026 bioRxiv (Ciccia lab with Hart lab), doi 10.64898/2026.06.07.728858.

enCas12a in4mer all-by-all of 233 DNA-damage-response genes in MCF10A and hTERT-RPE1 TP53-/-; three
arrays (guide sets) per gene pair, three replicates, T0 / T9 / T17 (MCF10A) or T18 (RPE1).

Scores: the authors' per-timepoint GI z-score and FDR_negative from the Ciccia-lab data portal
(full_dataset_for_download.zip). Only the final timepoint is used (MCF10A T17, sheet "3-2 MCF10A_T17";
RPE1 T18, sheet "3-5 RPE-1_T18"). The screen is drug-free.

Replicates: re-scored from GEO GSE343308 guide-array counts with the SPIDR additive recipe, per
replicate, final timepoint vs T0 (T0 pooled over replicates): LFC centred on the median of the
non-essential-gene control arrays (the library has no non-targeting x non-targeting arrays; 50
non-essential genes such as CXorf66 carry single-gene arrays), f_g = median LFC of gene g's single-gene
arrays, GI = LFC - f_a - f_b, median over the (up to 3) pair arrays.

Label rule (chosen by replication, see data/interim/new_human/hayward2026.md):
  positive = GI_z <= Z_POS and FDR_negative < 0.05 at the final timepoint
  negative = |GI_z| < 1 and FDR_negative > 0.25
  otherwise null.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl
import polars as pl

from slbench.sources import finalize

RAW = Path("data/raw/hayward2026")
XLSX = {
    "MCF10A": ("full_dataset_for_download/GIs_MCF10A_dataset.xlsx", {"T9": "3-1 MCF10A_T9", "T17": "3-2 MCF10A_T17"}, "T17"),
    "RPE1": ("full_dataset_for_download/GIs_RPE-1_dataset.xlsx", {"T9": "3-4 RPE-1_T9", "T18": "3-5 RPE-1_T18"}, "T18"),
}
COUNTS = {
    "MCF10A": ("GSE343308_Cas12a_IN4MER_MCF10A_gRNA_counts_GEO.tsv.gz", "T17"),
    "RPE1": ("GSE343308_Cas12a_IN4MER_RPE1_gRNA_counts_GEO.tsv.gz", "T18"),
}
Z_POS = -5.0
FDR_POS = 0.05
Z_NEG, FDR_NEG = 1.0, 0.25


def published(context: str, timepoint: str | None = None) -> pl.DataFrame:
    """Authors' GI table for one line/timepoint: gene_a, gene_b (as given), smf_a, smf_b, gi_raw, z, fdr."""
    f, sheets, final = XLSX[context]
    if not (RAW / f).exists():  # the portal ships one zip; unpack it on first use
        import zipfile
        with zipfile.ZipFile(RAW / "full_dataset_for_download.zip") as z:
            z.extractall(RAW)
    wb = openpyxl.load_workbook(RAW / f, read_only=True)
    rows = list(wb[sheets[timepoint or final]].iter_rows(values_only=True))
    h = rows[0]
    df = pl.DataFrame([dict(zip(h, r)) for r in rows[1:] if r[0]], infer_schema_length=None)
    return df.select(
        pl.col("GENE_PAIR").str.split_exact("_", 1).struct.rename_fields(["gene_a", "gene_b"]).alias("p"),
        pl.col("Gene1_SMF").cast(pl.Float64).alias("smf_a"), pl.col("Gene2_SMF").cast(pl.Float64).alias("smf_b"),
        pl.col("GI_raw").cast(pl.Float64).alias("gi_raw"),
        pl.col("GI_z-score").cast(pl.Float64).alias("z"), pl.col("FDR_negative").cast(pl.Float64).alias("fdr"),
    ).unnest("p")


def label_expr(z_pos: float = Z_POS, fdr_pos: float = FDR_POS) -> pl.Expr:
    return (pl.when((pl.col("z") <= z_pos) & (pl.col("fdr") < fdr_pos)).then(1)
            .when((pl.col("z").abs() < Z_NEG) & (pl.col("fdr") > FDR_NEG)).then(0)
            .otherwise(None).cast(pl.Int8))


def load() -> pl.DataFrame:
    out = []
    for cl in XLSX:
        d = published(cl)
        out.append(d.select(
            pl.lit("human").alias("species"), pl.lit("hayward2026").alias("source"), pl.lit(cl).alias("context"),
            pl.lit("CRISPR-Cas12a").alias("mechanism"), "gene_a", "gene_b",
            pl.col("z").alias("score"), pl.lit(f"GI_z_{XLSX[cl][2]}").alias("score_name"),
            pl.col("fdr").alias("signif"), pl.lit("FDR_negative").alias("signif_name"),
            label_expr().alias("label"),
        ))
    return finalize(pl.concat(out), "human")


def _counts(context: str) -> tuple[pl.DataFrame, str]:
    f, tend = COUNTS[context]
    d = pl.read_csv(RAW / f, separator="\t")
    parts = d["guide"].str.split("_")
    d = d.with_columns(
        parts.list.len().alias("n"),
        parts.list.get(0).alias("g1"),
        pl.when(parts.list.len() == 3).then(parts.list.get(1)).otherwise(None).alias("g2"),
    )
    return d, tend


def nonessential_controls(d: pl.DataFrame) -> set[str]:
    singles = set(d.filter(pl.col("n") == 2)["g1"])
    paired = set(d.filter(pl.col("n") == 3)["g1"]) | set(d.filter(pl.col("n") == 3)["g2"])
    return singles - paired


def replicates(context: str | None = None, tend: str | None = None) -> pl.DataFrame:
    """Per-replicate additive GI (gene_a, gene_b, gi_rep1..3, f_sum_rep1 = f_a + f_b in rep1, context) at the final timepoint.

    Covers all arrayed pairs incl. the 49 essential-gene controls; join to load() to restrict to DDR x DDR.
    Raw additive GI is dominated by pairs of strongly depleted genes (counts at the floor); for replicate
    checks, z-score it within bins of f_a + f_b (as the authors' local-std z does).
    """
    if context is None:
        return pl.concat([replicates(c) for c in COUNTS])
    d, final = _counts(context)
    tend = tend or final
    ctrl_genes = nonessential_controls(d)
    t0 = d.select(pl.col("^T0_rep.*$")).to_numpy().mean(1)
    keep = t0 >= np.quantile(t0, 0.02)
    base = np.log2((t0 + 1) / (t0 + 1).sum())
    g1, g2, n = d["g1"].to_numpy(), d["g2"].to_numpy(), d["n"].to_numpy()
    is_ctrl = (n == 2) & np.isin(g1, list(ctrl_genes))
    out = None
    for i in (1, 2, 3):
        te = d[f"{tend}_rep{i}"].to_numpy().astype(float)
        lfc = np.log2((te + 1) / (te + 1).sum()) - base
        lfc = lfc - np.median(lfc[keep & is_ctrl])
        x = pl.DataFrame({"g1": d["g1"], "g2": d["g2"], "n": d["n"], "lfc": lfc}).filter(pl.Series(keep))
        single = x.filter(pl.col("n") == 2).group_by("g1").agg(pl.col("lfc").median().alias("f"))
        f = dict(zip(single["g1"], single["f"]))
        du = x.filter((pl.col("n") == 3) & (pl.col("g1") != pl.col("g2")))
        fa = np.array([f.get(v, np.nan) for v in du["g1"]])
        fb = np.array([f.get(v, np.nan) for v in du["g2"]])
        du = du.with_columns(pl.Series("gi", du["lfc"].to_numpy() - fa - fb), pl.Series("fs", fa + fb)).drop_nans("gi").with_columns(
            pl.min_horizontal("g1", "g2").alias("gene_a"), pl.max_horizontal("g1", "g2").alias("gene_b"))
        agg = du.group_by("gene_a", "gene_b").agg(pl.col("gi").median().alias(f"gi_rep{i}"), pl.col("fs").first().alias(f"fs{i}"))
        out = agg if out is None else out.join(agg.drop(f"fs{i}"), on=["gene_a", "gene_b"])
    return out.rename({"fs1": "f_sum_rep1"}).with_columns(pl.lit(context).alias("context"))
