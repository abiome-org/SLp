"""Yeast genetic-interaction maps: S. cerevisiae SGA (Costanzo 2016) and S. pombe E-MAP (Ryan 2012)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import polars as pl

from . import finalize

RAW = Path("data/raw")
INTERIM = Path("data/interim")

COSTANZO_POS = -0.2
RYAN_POS = -3.0
COSTANZO_FILES = ["SGA_ExE.txt", "SGA_NxN.txt", "SGA_ExN_NxE.txt", "SGA_DAmP.txt"]


def costanzo2016() -> pl.DataFrame:
    """Costanzo et al. 2016 Science global SGA network (thecellmap.org release).

    Strains are collapsed to ORFs (temperature-sensitive, DAmP and deletion alleles alike).
    Positive: epsilon < -0.2 and p < 0.05. The authors' stringent cut (-0.12) replicates across
    the two orientations of the same ORF pair at AUROC 0.68; -0.2 raises that to 0.74
    (REPLICATION.md), which is why SLB uses the stricter cut.
    Negative: p > 0.25 and |epsilon| below the file's median |epsilon|.
    Conflicting calls across alleles/orientations of the same ORF pair become ambiguous.
    """
    d = INTERIM / "costanzo/S1"
    if not d.exists():
        raise FileNotFoundError("unzip data/raw/costanzo2016_scer/pairwise.zip into data/interim/costanzo/S1")
    out = []
    for f in COSTANZO_FILES:
        lf = pl.scan_csv(d / f, separator="\t", quote_char=None, schema_overrides={"P-value": pl.Float64})
        lf = lf.select(
            pl.col("Query Strain ID").str.split("_").list.first().alias("gene_a"),
            pl.col("Array Strain ID").str.split("_").list.first().alias("gene_b"),
            pl.col("Genetic interaction score (ε)").cast(pl.Float64).alias("score"),
            pl.col("P-value").alias("signif"),
        ).drop_nulls(["score", "signif"])
        med = lf.select(pl.col("score").abs().median()).collect().item()
        out.append(lf.with_columns(
            pl.when((pl.col("score") < COSTANZO_POS) & (pl.col("signif") < 0.05)).then(1)
            .when((pl.col("signif") > 0.25) & (pl.col("score").abs() < med)).then(0)
            .otherwise(None).cast(pl.Int8).alias("label")
        ).collect())
    df = pl.concat(out).with_columns(
        pl.lit("scer").alias("species"), pl.lit("costanzo2016").alias("source"), pl.lit("S288C").alias("context"),
        pl.lit("SGA").alias("mechanism"), pl.lit("epsilon").alias("score_name"), pl.lit("p").alias("signif_name"),
    )
    return finalize(df, "scer")


def ryan2012() -> pl.DataFrame:
    """Ryan et al. 2012 Mol Cell S. pombe E-MAP, Dataset S2 (averaged, one allele per gene).

    Gene x gene S-score matrix with CR line endings; blank = not measured.
    Positive: S < -3 (the paper used -2.3; independent allele/orientation measurements replicate
    at AUROC 0.72 at -2.3 and 0.76 at -3, see REPLICATION.md). Negative: |S| < 1.
    """
    with zipfile.ZipFile(RAW / "ryan2012_spombe/mmc5_averaged.zip") as z:
        name = max(z.namelist(), key=lambda n: z.getinfo(n).file_size)
        text = z.read(name).decode("latin-1").replace("\r\n", "\n").replace("\r", "\n")
    lines = [l.split("\t") for l in text.split("\n") if l.strip()]
    header = lines[0]
    cols = [_pombe_id(h) for h in header[1:]]
    rows = []
    for l in lines[1:]:
        a = _pombe_id(l[0])
        for b, v in zip(cols, l[1:]):
            if v.strip() and a and b:
                try:
                    rows.append((a, b, float(v)))
                except ValueError:
                    pass
    df = pl.DataFrame(rows, schema=["gene_a", "gene_b", "score"], orient="row").with_columns(
        pl.lit("spom").alias("species"), pl.lit("ryan2012").alias("source"), pl.lit("972h-").alias("context"),
        pl.lit("E-MAP").alias("mechanism"), pl.lit("S").alias("score_name"),
        pl.lit(None, pl.Float64).alias("signif"), pl.lit(None, pl.String).alias("signif_name"),
        pl.when(pl.col("score") < RYAN_POS).then(1).when(pl.col("score").abs() < 1).then(0)
        .otherwise(None).cast(pl.Int8).alias("label"),
    )
    return finalize(df, "spom")


def _pombe_id(label: str) -> str | None:
    """'SPAC26F1.14C(aif1AIF1)' or 'SPCC1672.04C' -> 'SPAC26F1.14C' (resolved to PomBase case later)."""
    s = label.strip().strip('"').split("(")[0].split(" ")[0]
    return s or None
