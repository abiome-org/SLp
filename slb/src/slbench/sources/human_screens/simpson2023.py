"""Simpson, Ling, Jing & Adamson 2023 bioRxiv (doi 10.1101/2023.08.19.553986, CC BY; preprint): K562 CRISPRi
all-by-all dual-sgRNA GI map over 543 genes (147,153 gene pairs), both sgRNA orders, two independent screens
(2022, 2023), 2 replicates each, +/- niraparib. Raw counts from github.com/simpsondl/parpi-manuscript
(manuscript_data/counts/screen{2022,2023}_raw_counts.zip).

Only the untreated arm is used: gamma = T0 -> DMSO (the NIRAP columns are never read).
The authors' pipeline (workflow/scripts/*.R, config/config.yaml) is re-implemented from the counts:
  filters: sgRNAs whose mean-over-replicates median T0 count at either position is <= 35; constructs with any
  T0 replicate count <= 50; sgRNAs whose combination phenotypes correlate < 0.25 with their partners' single
  phenotypes (gamma correlation filter).
  gamma = log2 of (DMSO+10)/(T0+10) count fractions, minus the NT+NT median, divided by doublings
  (2022: 7.19, 7.73; 2023: 8.04, 7.98). Orientation-independent (OI) phenotype = mean of the AB and BA
  constructs. Single sgRNA phenotype = mean over constructs pairing it with a non-targeting sgRNA.
  For every query sgRNA: gamma_OI(partner, query) ~ 1 + single + single^2 (free intercept, lm); GI.z = residual
  / SD of residuals over non-targeting partners. Gene GI = mean GI.z over the gene pair's sgRNA combinations
  (both query directions, identical sgRNAs excluded). Genes rather than the authors' "pseudogenes" (one gene
  has two) are the unit, which reproduces the paper's 147,153 pairs.
Score = mean of the two screens' gene GI (replicate-averaged gamma), pairs measured in both screens.

The authors' hit call (discriminant = -log10 Wilcoxon p x |GI|, top 0.5% of control-pair discriminants)
is sign-agnostic and not reproduced; SLB uses the same rule as horlbeck2018/herken2026 (same pipeline family,
same GI units):
  positive: mean GI <= -3 and GI < 0 in both screens
  negative: |mean GI| below its median over all pairs (the horlbeck2018 SLB rule; |GI| < 1 would
            label 72-87% of pairs negative here, too loose on these compressed gene-level scales)
  otherwise null.
"""

from __future__ import annotations

import functools
import io
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

from slbench.sources import finalize
from .herken2026 import gene_level, quad_gi, sg_gene

RAW = Path("data/raw/simpson2023")
KEY = "simpson2023"
CONTEXT = "K562"
SCREENS = ("screen2022", "screen2023")
DOUBLINGS = {"screen2022": (7.19, 7.73), "screen2023": (8.04, 7.98)}  # config.yaml, DMSO R1/R2
COLS = ["FirstPosition", "SecondPosition", "Category", "Identical", "Orientation", "GuideCombinationID",
        "T0.R1", "T0.R2", "DMSO.R1", "DMSO.R2"]
POS = -3.0


@functools.cache
def _counts(screen: str) -> pl.DataFrame:
    with zipfile.ZipFile(RAW / f"{screen}_raw_counts.zip") as z:
        data = z.read(f"{screen}_raw_counts.tsv")
    return pl.read_csv(io.BytesIO(data), separator="\t", columns=COLS)


@functools.cache
def _screen(screen: str):
    """Returns (sgRNA names, P per replicate {'R1','R2','Avg'} as first x second matrices, orientation masks)."""
    d = _counts(screen)
    # filt_low_representation: mean over T0 replicates of the per-sgRNA median, at each position
    m1 = d.group_by("FirstPosition").agg(pl.col("T0.R1").median(), pl.col("T0.R2").median())
    m2 = d.group_by("SecondPosition").agg(pl.col("T0.R1").median(), pl.col("T0.R2").median())
    bad = set(m1.filter((pl.col("T0.R1") + pl.col("T0.R2")) / 2 <= 35)["FirstPosition"]) | \
        set(m2.filter((pl.col("T0.R1") + pl.col("T0.R2")) / 2 <= 35)["SecondPosition"])
    # filt_combinations: any T0 replicate <= 50
    d = d.filter(~pl.col("FirstPosition").is_in(list(bad)) & ~pl.col("SecondPosition").is_in(list(bad))
                 & (pl.col("T0.R1") > 50) & (pl.col("T0.R2") > 50))
    ph = {}
    nn = (d["Category"] == "NT+NT").to_numpy()
    for k, r in enumerate(("R1", "R2")):
        t0 = d[f"T0.{r}"].to_numpy() + 10.0
        te = d[f"DMSO.{r}"].to_numpy() + 10.0
        g = np.log2((te / te.sum()) / (t0 / t0.sum()))
        ph[r] = (g - np.median(g[nn])) / DOUBLINGS[screen][k]
    ph["Avg"] = (ph["R1"] + ph["R2"]) / 2
    sgs = np.array(sorted(set(d["FirstPosition"]) | set(d["SecondPosition"])))
    idx = {s: i for i, s in enumerate(sgs)}
    ii = np.array([idx[x] for x in d["FirstPosition"]])
    jj = np.array([idx[x] for x in d["SecondPosition"]])
    ab = (d["Orientation"] == "AB").to_numpy()
    n = len(sgs)
    P = {}
    for r, v in ph.items():
        M = np.full((n, n), np.nan)
        M[ii, jj] = v
        P[r] = M
    AB = np.zeros((n, n), bool)
    AB[ii[ab], jj[ab]] = True
    return sgs, P, AB


def _singles(P: np.ndarray, is_nt: np.ndarray) -> np.ndarray:
    """calculate_single_sgrna_phenotypes: mean over constructs pairing the sgRNA with a non-targeting sgRNA."""
    return np.nanmean(np.concatenate([P[:, is_nt], P[is_nt, :].T], axis=1), axis=1)


def _oi(P: np.ndarray) -> np.ndarray:
    return np.nanmean(np.stack([P, P.T]), axis=0)


@functools.cache
def _keep(screen: str) -> np.ndarray:
    """Correlation filter on gamma (Avg): sgRNAs whose partners' OI combination phenotype correlates < 0.25
    with those partners' single phenotypes are dropped."""
    sgs, P, _ = _screen(screen)
    is_nt = np.array([s.startswith("non-targeting") for s in sgs])
    s = _singles(P["Avg"], is_nt)
    O = _oi(P["Avg"])
    keep = np.ones(len(sgs), bool)
    for i in range(len(sgs)):
        y = O[:, i]  # constructs with SecondPosition == i (OI value)
        ok = ~np.isnan(P["Avg"][:, i]) & ~np.isnan(s)
        keep[i] = np.corrcoef(y[ok], s[ok])[0, 1] >= 0.25
    return keep


def _gi(screen: str, rep: str, orientation: str | None = None) -> pl.DataFrame:
    sgs, P, AB = _screen(screen)
    keep = _keep(screen)
    M = P[rep].copy()
    if orientation == "AB":
        M[~AB] = np.nan
    elif orientation == "BA":
        M[AB] = np.nan
    M = M[np.ix_(keep, keep)]
    s_ = sgs[keep]
    is_nt = np.array([s.startswith("non-targeting") for s in s_])
    single = _singles(M, is_nt)
    O = _oi(M)
    G = quad_gi(O, is_nt, single, fixed_intercept=False)
    return gene_level(G, sg_gene(s_))


@functools.cache
def all_scores() -> pl.DataFrame:
    out = None
    for sc in SCREENS:
        tag = sc[-4:]
        parts = {f"gi_{tag}": _gi(sc, "Avg"), f"gi_{tag}_R1": _gi(sc, "R1"), f"gi_{tag}_R2": _gi(sc, "R2"),
                 f"gi_{tag}_AB": _gi(sc, "Avg", "AB"), f"gi_{tag}_BA": _gi(sc, "Avg", "BA")}
        for name, g in parts.items():
            g = g.select("gene_a", "gene_b", pl.col("gi").alias(name))
            out = g if out is None else out.join(g, on=["gene_a", "gene_b"], how="full", coalesce=True)
    return out


REPLICATE_MAP = {"gi_rep1": "gi_2022", "gi_rep2": "gi_2023",
                 "gi_rep3": "gi_2022_R1", "gi_rep4": "gi_2022_R2", "gi_rep5": "gi_2023_R1", "gi_rep6": "gi_2023_R2",
                 "gi_rep7": "gi_2022_AB", "gi_rep8": "gi_2022_BA", "gi_rep9": "gi_2023_AB", "gi_rep10": "gi_2023_BA"}
REPLICATE_GROUPS = {"screens": ["gi_rep1", "gi_rep2"],
                    "bio_replicates": ["gi_rep3", "gi_rep4", "gi_rep5", "gi_rep6"],
                    "orientation_2022": ["gi_rep7", "gi_rep8"], "orientation_2023": ["gi_rep9", "gi_rep10"]}
CROSS_EXTRA = ["herken2026"]


def replicates() -> pl.DataFrame:
    """gi_rep1/2 = screen 2022 / 2023 (replicate-averaged); gi_rep3..6 = 2022 R1, R2, 2023 R1, R2;
    gi_rep7..10 = orientation-specific (AB only / BA only) GI for 2022 and 2023."""
    a = all_scores()
    return a.select("gene_a", "gene_b", pl.lit(CONTEXT).alias("context"),
                    *[pl.col(v).alias(k) for k, v in REPLICATE_MAP.items()])


def singles() -> pl.DataFrame:
    """Screen's own single-gene gamma (mean over sgRNAs and screens)."""
    rows = []
    for sc in SCREENS:
        sgs, P, _ = _screen(sc)
        is_nt = np.array([s.startswith("non-targeting") for s in sgs])
        rows.append(pl.DataFrame({"gene": sg_gene(sgs), "single": _singles(P["Avg"], is_nt)}))
    return pl.concat(rows).filter(pl.col("gene") != "CONTROL").group_by("gene").agg(pl.col("single").mean())


def raw_symbols() -> list[str]:
    return sorted(set(sg_gene(_counts("screen2022")["FirstPosition"].unique().to_numpy())) - {"CONTROL"})


def load() -> pl.DataFrame:
    m = all_scores().drop_nulls(["gi_2022", "gi_2023"]).drop_nans(["gi_2022", "gi_2023"])
    df = m.with_columns(((pl.col("gi_2022") + pl.col("gi_2023")) / 2).alias("score")).select(
        pl.lit("human").alias("species"), pl.lit(KEY).alias("source"), pl.lit(CONTEXT).alias("context"),
        pl.lit("CRISPRi").alias("mechanism"), "gene_a", "gene_b", "score",
        pl.lit("GI_z_mean2screens").alias("score_name"),
        pl.lit(None, pl.Float64).alias("signif"), pl.lit(None, pl.String).alias("signif_name"),
        pl.when((pl.col("score") <= POS) & (pl.col("gi_2022") < 0) & (pl.col("gi_2023") < 0)).then(1)
        .when(pl.col("score").abs() < pl.col("score").abs().median()).then(0).otherwise(None).cast(pl.Int8).alias("label"),
    )
    return finalize(df, "human")
