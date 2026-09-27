"""Herken, Norman & Gilbert 2026 Mol Cell (doi 10.1016/j.molcel.2026.01.025, CC BY): K562 CRISPRi all-by-all
dual-sgRNA GI maps over ~310 DNA-damage-response / essential genes (GEO GSE312636).

Two independent drug-free maps, each with 2 replicates:
  expt1: Rep{1,2}_T0 -> Rep{1,2}_UT   (the DRUG arm, ATR inhibitor, is never used)
  expt2: t0_r{1,2}   -> dmso_r{1,2}   (etoposide / ketoconazole arms never used)

No processed GI table is public (Zenodo 10.5281/zenodo.17822522 is code only), so the authors' own pipeline
(`call_genetic_interactions.ipynb`, Horlbeck et al. 2018 method, same as SLKB's horlbeck2018 scores) is
re-implemented here from the counts:
  per replicate: drop sgRNAs whose median end count at either position is < 35; log2((end+10)/(T0+10)) with
  library-size ratio, minus the non-targeting x non-targeting median, divided by the replicate's doublings;
  the two replicates are averaged (sgRNAs passing in both); the sgRNA x sgRNA matrix is ABBA-averaged;
  single phenotype = mean over non-targeting partners; for every query sgRNA a quadratic in the partner's
  single phenotype is fitted with intercept fixed at the query's single phenotype; GI = residual divided by
  the SD of residuals over non-targeting partners; the matrix is symmetrised; gene GI = mean over the
  gene x gene sgRNA block.
Score = mean of the two maps' gene GI (pairs measured in both maps).

Label (same units and cut-off as horlbeck2018 in SLB):
  positive: mean GI <= -3 and GI < 0 in both maps
  negative: |mean GI| below its median over all pairs (the horlbeck2018 SLB rule; |GI| < 1 would
            label 72-87% of pairs negative here, too loose on these compressed gene-level scales)
  otherwise null.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from slbench.sources import finalize

RAW = Path("data/raw/herken2026")
KEY = "herken2026"
CONTEXT = "K562"

# doublings per replicate (growthScores in call_genetic_interactions.ipynb)
DOUBLINGS = {("expt1", 1): 9.632926829, ("expt1", 2): 9.559756098,
             ("expt2", 1): 5.761137265, ("expt2", 2): 6.324829553}
COLS = {("expt1", 1): ("Rep1_T0", "Rep1_UT"), ("expt1", 2): ("Rep2_T0", "Rep2_UT"),
        ("expt2", 1): ("t0_r1", "dmso_r1"), ("expt2", 2): ("t0_r2", "dmso_r2")}
FILES = {"expt1": ("GSE312636_GI_expt1_counts.txt.gz", "\t"), "expt2": ("GSE312636_GI_expt2_counts.txt.gz", ",")}


# ---------------------------------------------------------------------------------------------
# Shared machinery (also used by simpson2023)

def sg_gene(names: np.ndarray) -> np.ndarray:
    return np.array(["CONTROL" if n.startswith("non-targeting") else n.split("_")[0] for n in names])


def quad_gi(P: np.ndarray, is_nt: np.ndarray, single: np.ndarray, fixed_intercept: bool) -> np.ndarray:
    """Guide-level GI matrix (Horlbeck 2018 / Simpson 2023 model) from a symmetric sgRNA x sgRNA phenotype
    matrix P (NaN = missing). For each query q: fit P[q, i] ~ quadratic(single[i]) over partners i, residual
    scaled by the SD of residuals over non-targeting partners. Returns the symmetrised z-matrix."""
    n = P.shape[0]
    E = np.full((n, n), np.nan)
    for q in range(n):
        y = P[q]
        ok = ~np.isnan(y) & ~np.isnan(single)
        x = single[ok]
        if fixed_intercept:
            if np.isnan(single[q]):
                continue
            X = np.c_[x ** 2, x]
            coef, *_ = np.linalg.lstsq(X, y[ok] - single[q], rcond=None)
            fit = X @ coef + single[q]
        else:
            X = np.c_[np.ones_like(x), x, x ** 2]
            coef, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
            fit = X @ coef
        r = np.full(n, np.nan)
        r[ok] = y[ok] - fit
        sd = np.nanstd(r[is_nt], ddof=1)
        E[q] = r / sd
    return np.nanmean(np.stack([E, E.T]), axis=0)


def gene_level(G: np.ndarray, genes: np.ndarray, exclude_identical: bool = True) -> pl.DataFrame:
    """Mean guide-level GI over each (gene_a, gene_b) sgRNA block; controls and same-gene pairs dropped."""
    iu = np.triu_indices(len(genes), k=1)
    d = pl.DataFrame({"g1": genes[iu[0]], "g2": genes[iu[1]], "gi": G[iu]})
    d = d.filter((pl.col("g1") != "CONTROL") & (pl.col("g2") != "CONTROL") & (pl.col("g1") != pl.col("g2")))
    d = d.drop_nans("gi").drop_nulls("gi").with_columns(
        pl.min_horizontal("g1", "g2").alias("gene_a"), pl.max_horizontal("g1", "g2").alias("gene_b"))
    return d.group_by("gene_a", "gene_b").agg(pl.col("gi").mean(), pl.len().alias("n_guide_pairs"))


# ---------------------------------------------------------------------------------------------

def _counts(expt: str) -> pl.DataFrame:
    f, sep = FILES[expt]
    d = pl.read_csv(RAW / f, separator=sep)
    return d.with_columns(pl.col("double").str.split_exact(":", 1).struct.rename_fields(["a", "b"])).unnest("double")


def _rep_phenotype(d: pl.DataFrame, t0c: str, endc: str, doublings: float) -> tuple[dict, set]:
    """calcLog2e_cycledonly_corrected: returns {(a, b): phenotype} over sgRNAs passing the filter."""
    pc, thr = 10.0, 35.0
    d = d.drop_nulls([t0c, endc])  # expt2 has a few hundred empty count fields
    med_a = d.group_by("a").agg(pl.col(endc).median())
    med_b = d.group_by("b").agg(pl.col(endc).median())
    bad = set(med_a.filter(pl.col(endc) < thr)["a"]) | set(med_b.filter(pl.col(endc) < thr)["b"])
    f = d.filter(~pl.col("a").is_in(list(bad)) & ~pl.col("b").is_in(list(bad)))
    t0 = f[t0c].to_numpy().astype(float) + pc
    te = f[endc].to_numpy().astype(float) + pc
    lfc = np.log2(te / t0 * (t0.sum() / te.sum()))
    nn = (f["a"].str.starts_with("non-targeting") & f["b"].str.starts_with("non-targeting")).to_numpy()
    lfc = (lfc - np.median(lfc[nn])) / doublings
    sgs = set(f["a"]) & set(f["b"])
    return pl.DataFrame({"a": f["a"], "b": f["b"], "p": lfc}), sgs


def _matrix(ph: pl.DataFrame, sgs: list[str]) -> np.ndarray:
    idx = {s: i for i, s in enumerate(sgs)}
    ph = ph.filter(pl.col("a").is_in(sgs) & pl.col("b").is_in(sgs))
    P = np.full((len(sgs), len(sgs)), np.nan)
    P[[idx[x] for x in ph["a"]], [idx[x] for x in ph["b"]]] = ph["p"].to_numpy()
    return P


def _map_gi(P: np.ndarray, sgs: np.ndarray) -> pl.DataFrame:
    P = (P + P.T) / 2  # ABBA
    is_nt = np.array([s.startswith("non-targeting") for s in sgs])
    single = np.nanmean(P[is_nt], axis=0)
    G = quad_gi(P, is_nt, single, fixed_intercept=True)
    return gene_level(G, sg_gene(sgs))


def map_scores() -> pl.DataFrame:
    """Gene-level GI per map (replicate-averaged, the authors' map) and per replicate.
    Columns gene_a, gene_b, gi_expt1, gi_expt2, gi_expt1_rep1, gi_expt1_rep2, gi_expt2_rep1, gi_expt2_rep2."""
    out = None
    for expt in ("expt1", "expt2"):
        d = _counts(expt)
        reps = {}
        for r in (1, 2):
            t0c, endc = COLS[(expt, r)]
            reps[r] = _rep_phenotype(d, t0c, endc, DOUBLINGS[(expt, r)])
        sgs = np.array(sorted(reps[1][1] & reps[2][1]))
        Ps = {r: _matrix(reps[r][0], list(sgs)) for r in (1, 2)}
        parts = {f"gi_{expt}": _map_gi((Ps[1] + Ps[2]) / 2, sgs)}
        for r in (1, 2):
            parts[f"gi_{expt}_rep{r}"] = _map_gi(Ps[r], sgs)
        for name, g in parts.items():
            g = g.select("gene_a", "gene_b", pl.col("gi").alias(name))
            out = g if out is None else out.join(g, on=["gene_a", "gene_b"], how="full", coalesce=True)
    return out


def singles() -> pl.DataFrame:
    """Screen's own single-gene CRISPRi growth phenotype (gamma per doubling, mean over sgRNAs and the two maps)."""
    rows = []
    for expt in ("expt1", "expt2"):
        d = _counts(expt)
        for r in (1, 2):
            t0c, endc = COLS[(expt, r)]
            ph, _ = _rep_phenotype(d, t0c, endc, DOUBLINGS[(expt, r)])
            ph = ph.filter(pl.col("b").str.starts_with("non-targeting") & ~pl.col("a").str.starts_with("non-targeting"))
            rows.append(ph.with_columns(pl.col("a").str.split("_").list.first().alias("gene")).select("gene", "p"))
    return pl.concat(rows).group_by("gene").agg(pl.col("p").mean().alias("single"))


def replicates() -> pl.DataFrame:
    """Independent replicate columns: gi_rep1 = map 1 (expt1), gi_rep2 = map 2 (expt2), plus the four
    within-map biological replicates (gi_rep3..gi_rep6 = expt1 r1, expt1 r2, expt2 r1, expt2 r2)."""
    m = map_scores().drop_nulls()
    return m.select(
        "gene_a", "gene_b", pl.lit(CONTEXT).alias("context"),
        pl.col("gi_expt1").alias("gi_rep1"), pl.col("gi_expt2").alias("gi_rep2"),
        pl.col("gi_expt1_rep1").alias("gi_rep3"), pl.col("gi_expt1_rep2").alias("gi_rep4"),
        pl.col("gi_expt2_rep1").alias("gi_rep5"), pl.col("gi_expt2_rep2").alias("gi_rep6"),
    )


POS = -3.0
# replicate structure for the checks: independent maps, and the four biological replicates
REPLICATE_GROUPS = {"maps": ["gi_rep1", "gi_rep2"], "bio_replicates": ["gi_rep3", "gi_rep4", "gi_rep5", "gi_rep6"]}


def raw_symbols() -> list[str]:
    d = _counts("expt1")
    return sorted({s.split("_")[0] for s in d["a"].unique() if not s.startswith("non-targeting")})


def load() -> pl.DataFrame:
    m = map_scores().drop_nulls(["gi_expt1", "gi_expt2"])
    s = (m["gi_expt1"] + m["gi_expt2"]) / 2
    df = m.with_columns(s.alias("score")).select(
        pl.lit("human").alias("species"), pl.lit(KEY).alias("source"), pl.lit(CONTEXT).alias("context"),
        pl.lit("CRISPRi").alias("mechanism"), "gene_a", "gene_b", "score",
        pl.lit("GI_horlbeck_mean2maps").alias("score_name"),
        pl.lit(None, pl.Float64).alias("signif"), pl.lit(None, pl.String).alias("signif_name"),
        pl.when((pl.col("score") <= POS) & (pl.col("gi_expt1") < 0) & (pl.col("gi_expt2") < 0)).then(1)
        .when(pl.col("score").abs() < pl.col("score").abs().median()).then(0).otherwise(None).cast(pl.Int8).alias("label"),
    )
    return finalize(df, "human")


CROSS_EXTRA = ["simpson2023"]  # used by the inclusion checks only
