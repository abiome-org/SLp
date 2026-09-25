"""Single-gene loss effects (negative = sicker) and the SL propensity they imply.

Reference single-loss effect per species:
  human   DepMap 24Q4 Chronos gene effect: in the example's cell line when DepMap screened it,
          plus the pan-line mean
  scer    Costanzo 2016 SGA single-mutant fitness - 1 (0 = wild type)
  spom    PomBase deletion viability: inviable -1, slow/decreased growth -0.5, viable 0
  spne    dual CRISPRi-seq single-sgRNA log2FC
  dmel    single-dsRNA main effect on cell count (Heigwer 2023)
  bsub    CRISPRi single-knockdown fitness (sources.bacteria_extra.bsub_single)
  cele    WormBase RNAi/allele phenotypes: lethal/arrest -1, sterile/slow -0.5, else 0 (eukaryotes_extra.cele_single)
  mmus    DepMap pan-line mean of the one-to-one human ortholog (eukaryotes_extra.mmus_single)
These are single-gene measurements, so they are permitted model inputs.

`propensity()` estimates P(SL | the two genes' single-loss effects, screen) per species within an
evaluation split. SLB uses it to weight pairs so that SL and non-SL pairs have the same single-gene
fitness profile (see evaluate.py): predicting which genes are sick earns nothing.
"""

from __future__ import annotations

import functools
import gzip
from pathlib import Path

import numpy as np
import polars as pl

RAW = Path("data/raw")
INTERIM = Path("data/interim")
COVARIATES = ["f_lo", "f_hi", "pan_lo", "pan_hi"]


@functools.cache
def depmap_long(depmap_ids: tuple[str, ...]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(per-line effects for the given DepMap IDs [depmap_id, gene, effect], pan-line mean [gene, effect])."""
    from slbench import ids

    df = pl.read_csv(RAW / "depmap/CRISPRGeneEffect_24Q4.csv")
    idc = df.columns[0]
    h = ids.human()
    cols, seen = {}, set()
    for c in df.columns[1:]:
        sym = h(c.split(" (")[0]) or h(c.split("(")[-1].rstrip(")"))
        if sym and sym not in seen:
            cols[c] = sym
            seen.add(sym)
    x = df.select(idc, *cols).rename({idc: "depmap_id", **cols})
    pan = x.drop("depmap_id").mean().transpose(include_header=True, header_name="gene", column_names=["effect"])
    line = x.filter(pl.col("depmap_id").is_in(list(depmap_ids))).unpivot(
        index="depmap_id", variable_name="gene", value_name="effect").drop_nulls()
    return line, pan


@functools.cache
def yeast_smf() -> pl.DataFrame:
    import openpyxl

    wb = openpyxl.load_workbook(INTERIM / "costanzo/S1/strain_ids_and_single_mutant_fitness.xlsx", read_only=True)
    acc: dict[str, list[float]] = {}
    for r in list(wb.worksheets[0].iter_rows(values_only=True))[1:]:
        if r[1] and isinstance(r[5], (int, float)):
            acc.setdefault(r[1], []).append(float(r[5]))
    return pl.DataFrame({"gene": list(acc), "effect": [float(np.mean(v)) - 1 for v in acc.values()]})


INVIABLE, SLOW = {"FYPO:0002061"}, {"FYPO:0001234", "FYPO:0001355", "FYPO:0000046"}


@functools.cache
def pombe_viability() -> pl.DataFrame:
    from slbench import ids

    worst: dict[str, float] = {}
    with gzip.open(RAW / "pombase/phenotype_annotations.phaf.gz", "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 12 or p[11] != "deletion":
                continue
            g = ids.spom()(p[1])
            if not g:
                continue
            v = -1.0 if p[2] in INVIABLE else -0.5 if p[2] in SLOW else 0.0 if p[2] == "FYPO:0002060" else None
            if v is not None:
                worst[g] = min(worst.get(g, 0.0), v)
    return pl.DataFrame({"gene": list(worst), "effect": list(worst.values())})


@functools.cache
def spne_single() -> pl.DataFrame:
    d = pl.read_csv(RAW / "dualcrispri2025_spneumo/mmc4.csv", infer_schema_length=0, null_values=["NA", ""])
    parts = [d.select(pl.col(f"SG{i}.targets").alias("gene"), pl.col(f"SG{i}.refLog2FC").cast(pl.Float64).alias("effect"))
             for i in (1, 2)]
    return pl.concat(parts).filter(~pl.col("gene").str.contains(",")).drop_nulls().group_by("gene").agg(
        pl.col("effect").median())


@functools.cache
def fly_main() -> pl.DataFrame:
    """Single-dsRNA main effect on cell count (Heigwer 2023), mean per FBgn; negative = fewer cells."""
    from slbench import ids

    lf = pl.scan_csv(RAW / "heigwer2023_dmel/interactions_stat_tested_bias_corrected.csv.gz", infer_schema_length=10000)
    d = lf.filter(pl.col("feature") == "cells").select("fbgn", "query_name", "query_main", "target_main").collect()
    out = dict(d.group_by("fbgn").agg(pl.col("target_main").mean()).iter_rows())
    q = d.group_by("query_name").agg(pl.col("query_main").mean())
    for g, v in zip(ids.resolve("dmel", q["query_name"]), q["query_main"]):
        if g and g not in out:
            out[g] = v
    return pl.DataFrame({"gene": list(out), "effect": list(out.values())})


def gene_effects() -> pl.DataFrame:
    """[species, gene, effect]: the pan-context reference single-loss effect."""
    _, pan = depmap_long(())
    parts = [pan.with_columns(pl.lit("human").alias("species")),
             yeast_smf().with_columns(pl.lit("scer").alias("species")),
             pombe_viability().with_columns(pl.lit("spom").alias("species")),
             spne_single().with_columns(pl.lit("spne").alias("species")),
             fly_main().with_columns(pl.lit("dmel").alias("species"))]
    from slbench.sources import bacteria_extra, eukaryotes_extra

    for sp, fn in (("bsub", bacteria_extra.bsub_single), ("cele", eukaryotes_extra.cele_single),
                   ("mmus", eukaryotes_extra.mmus_single)):
        parts.append(fn().select("gene", "effect").with_columns(pl.lit(sp).alias("species")))
    return pl.concat([p.select("species", "gene", pl.col("effect").cast(pl.Float64)) for p in parts])


def covariates(df: pl.DataFrame, contexts: pl.DataFrame) -> pl.DataFrame:
    """Add f_lo/f_hi (context-specific where available, else reference) and pan_lo/pan_hi."""
    ctx = contexts.select("context_id", "depmap_id")
    ids_ = tuple(sorted(i for i in ctx["depmap_id"].drop_nulls().unique().to_list()))
    line, _ = depmap_long(ids_)
    ref = gene_effects()
    out = df.join(ctx, on="context_id", how="left")
    for g in ("a", "b"):
        out = out.join(ref.rename({"gene": f"gene_{g}", "effect": f"pan_{g}"}), on=["species", f"gene_{g}"], how="left")
        out = out.join(line.rename({"gene": f"gene_{g}", "effect": f"ctx_{g}"}), on=["depmap_id", f"gene_{g}"], how="left")
        out = out.with_columns(pl.coalesce(f"ctx_{g}", f"pan_{g}").alias(f"f_{g}"))
    return out.with_columns(
        pl.min_horizontal("f_a", "f_b").alias("f_lo"), pl.max_horizontal("f_a", "f_b").alias("f_hi"),
        pl.min_horizontal("pan_a", "pan_b").alias("pan_lo"), pl.max_horizontal("pan_a", "pan_b").alias("pan_hi"),
    ).drop("depmap_id", "pan_a", "pan_b", "ctx_a", "ctx_b", "f_a", "f_b")


def _design(d: pl.DataFrame) -> np.ndarray:
    """Propensity design: cubic splines of each covariate (indicators for categorical ones such as
    S. pombe viability) + missingness flags + f_lo*f_hi + screen and context intercepts + screen x
    linear covariates. Standardised columns."""
    from sklearn.preprocessing import SplineTransformer

    cols = []
    for c in COVARIATES:
        v = d[c].cast(pl.Float64).fill_null(np.nan).to_numpy()
        miss = np.isnan(v)
        if miss.all():
            continue
        v = np.where(miss, np.nanmedian(v), v)
        levels = np.unique(v)
        if len(levels) <= 6:
            cols += [(v == u).astype(float)[:, None] for u in levels[1:]]
        else:
            cols.append(SplineTransformer(n_knots=5, degree=3, knots="quantile", include_bias=False).fit_transform(v[:, None]))
        if miss.any():
            cols.append(miss.astype(float)[:, None])
    lin = np.column_stack([d[c].cast(pl.Float64).fill_null(0).to_numpy() for c in COVARIATES])
    cols.append((lin[:, 0] * lin[:, 1])[:, None])
    for key in ("sources", "context_id"):
        k = d[key].to_numpy()
        levels = np.unique(k)
        if len(levels) > 1:
            dummies = np.column_stack([(k == u).astype(float) for u in levels[1:]])
            cols.append(dummies)
            if key == "sources":
                cols += [dummies[:, [i]] * lin for i in range(dummies.shape[1])]
    x = np.column_stack(cols)
    return (x - x.mean(0)) / (x.std(0) + 1e-9)


def propensity(ex: pl.DataFrame, contexts: pl.DataFrame) -> pl.Series:
    """P(label=1 | the two genes' single-loss effects, screen, context), per species.

    ex needs species, context_id, gene_a, gene_b, sources, label. A lightly penalised logistic
    regression on a smooth, low-dimensional design (_design) fitted on the evaluation split itself:
    the fitness-SL relation differs between held-out family sets, so a model fitted elsewhere does
    not balance this split. Logistic propensity + overlap weights balance every design column in the
    mean (Li, Morgan & Zaslavsky 2018), and the smooth design cannot memorise individual genes. The
    build stores it under hidden/: evaluation machinery, never a model input.
    """
    from sklearn.linear_model import LogisticRegression

    x = covariates(ex.with_row_index("_i"), contexts)
    out = np.full(ex.height, np.nan)
    for _, d in x.group_by(["species"]):
        y = d["label"].to_numpy()
        if y.min() == y.max():  # one class only (tiny species in a small split): no AUROC to balance
            out[d["_i"].to_numpy()] = float(y.mean())
            continue
        m = LogisticRegression(C=1.0, max_iter=5000).fit(_design(d), y)
        out[d["_i"].to_numpy()] = m.predict_proba(_design(d))[:, 1]
    return pl.Series("propensity", out)
