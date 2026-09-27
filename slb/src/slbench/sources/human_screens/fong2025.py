"""Fong, Kuenzi et al. 2025 Nat Genet (SCHEMATIC): dual-guide Cas9, 176 x 67 gene panel, 7 cell lines.

Files (data/raw/fong2025/, Springer ESM, journal terms):
  41588_2024_1971_MOESM4_ESM.xlsx  per-line GI scores (sheet cell-line-scores, z-standardised per line)
                                   and FDRs (cell-line-fdrs); multi-lineage sheets are NOT used (SLB is per context)
  41588_2024_1971_MOESM3_ESM.txt   construct read counts: plasmid + <LINE>_T<day>_<rep>, 2 replicates x 4 timepoints

Score: the authors' per-line genetic-interaction score (negative = synthetic sick/lethal), signif = their FDR.
Label (see data/interim/new_human/fong2025.md for how it was chosen):
  positive = score <= -3 and FDR < 0.05 (strong and significant negative GI)
  negative = |score| < 1 and FDR > 0.25
  otherwise null.
Only the contexts in INCLUDED_LINES get labels; the others keep measurements with label null.

replicates(): per-replicate additive GI from the counts (the spidr_replicate_gi recipe): per biological
replicate, LFC = log2 normalised abundance at the last timepoint minus the plasmid (no per-line T0 exists;
'first_last' and 'slope' modes are available and replicate no better), centred on control x control
constructs (AAVS / non-targeting); single effect f_g = median LFC of gene x AAVS constructs (either position); GI = LFC - f_a - f_b, median over the 9 guide pairs.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import openpyxl
import polars as pl

from .. import finalize

RAW = Path("data/raw/fong2025")
COUNTS = RAW / "41588_2024_1971_MOESM3_ESM.txt"
SCORES = RAW / "41588_2024_1971_MOESM4_ESM.xlsx"
LINES = {"MCF7": "MCF7", "MDAMB231": "MDA-MB-231", "MCF10A": "MCF10A", "CAL27": "CAL27",
         "CAL33": "CAL33", "A549": "A549", "A427": "A427"}
CTRL = ("AAVS", "nontargeting")
POS_Z, POS_FDR, NEG_Z, NEG_FDR = -3.0, 0.05, 1.0, 0.25
# No line passes the own-replication gate (biological-replicate and guide-split AUROC 0.39-0.72, mostly ~0.5)
# and there is no same-line cross-study support, so no line is labelled: measurements only.
INCLUDED_LINES: tuple[str, ...] = ()


def _sheet(name: str) -> pl.DataFrame:
    wb = openpyxl.load_workbook(SCORES, read_only=True)
    rows = list(wb[name].iter_rows(values_only=True))
    return pl.DataFrame([dict(zip(rows[0], r)) for r in rows[1:] if r[0]], infer_schema_length=None)


def scores_long() -> pl.DataFrame:
    """Authors' per-line score + FDR, long format (target_a_id, target_b_id, line, score, fdr); controls dropped."""
    s = _sheet("cell-line-scores").unpivot(index=["target_a_id", "target_b_id"], variable_name="line", value_name="score")
    f = _sheet("cell-line-fdrs").unpivot(index=["target_a_id", "target_b_id"], variable_name="line", value_name="fdr")
    d = s.join(f, on=["target_a_id", "target_b_id", "line"]).with_columns(
        pl.col("score").cast(pl.Float64), pl.col("fdr").cast(pl.Float64))
    return d.filter(~pl.col("target_a_id").is_in(CTRL) & ~pl.col("target_b_id").is_in(CTRL))


def load(labelled: tuple[str, ...] | None = None) -> pl.DataFrame:
    """labelled: raw line names (keys of LINES) that receive labels; default INCLUDED_LINES."""
    labelled = INCLUDED_LINES if labelled is None else labelled
    d = scores_long()
    lab = (pl.when((pl.col("score") <= POS_Z) & (pl.col("fdr") < POS_FDR)).then(1)
           .when((pl.col("score").abs() < NEG_Z) & (pl.col("fdr") > NEG_FDR)).then(0)
           .otherwise(None))
    df = d.select(
        pl.lit("human").alias("species"), pl.lit("fong2025").alias("source"),
        pl.col("line").replace_strict(LINES).alias("context"), pl.lit("CRISPR-KO").alias("mechanism"),
        pl.col("target_a_id").alias("gene_a"), pl.col("target_b_id").alias("gene_b"),
        "score", pl.lit("GI_z").alias("score_name"), pl.col("fdr").alias("signif"), pl.lit("fdr").alias("signif_name"),
        pl.when(pl.col("line").is_in(list(labelled))).then(lab).otherwise(None).cast(pl.Int8).alias("label"),
    )
    return finalize(df, "human")


# ---------------------------------------------------------------------------------------------- counts

def _counts() -> pl.DataFrame:
    return pl.read_csv(COUNTS)


def _timepoints(cols, line: str) -> dict[int, list[str]]:
    """{rep: [columns sorted by day]} for one line."""
    out: dict[int, list[tuple[int, str]]] = {}
    for c in cols:
        m = re.fullmatch(rf"{line}_T(\d+)_(\d)", c)
        if m:
            out.setdefault(int(m.group(2)), []).append((int(m.group(1)), c))
    return {r: [c for _, c in sorted(v)] for r, v in sorted(out.items())}


def _lab(x: np.ndarray) -> np.ndarray:
    return np.log2((x + 1) / (x + 1).sum())


def construct_lfc(d: pl.DataFrame, line: str, mode: str = "first_last") -> dict[str, np.ndarray]:
    """Per-replicate construct fitness for one line, centred on control x control. mode: first_last | plasmid_last | slope."""
    tps = _timepoints(d.columns, line)
    ctrl = (d["target_a_id"].is_in(CTRL) & d["target_b_id"].is_in(CTRL)).to_numpy()
    out = {}
    for r, cs in tps.items():
        if mode == "first_last":
            lfc = _lab(d[cs[-1]].to_numpy().astype(float)) - _lab(d[cs[0]].to_numpy().astype(float))
        elif mode == "plasmid_last":
            lfc = _lab(d[cs[-1]].to_numpy().astype(float)) - _lab(d["plasmid"].to_numpy().astype(float))
        elif mode == "slope":
            days = np.array([int(re.search(r"_T(\d+)_", c).group(1)) for c in cs], float)
            y = np.stack([_lab(d[c].to_numpy().astype(float)) for c in cs], 1)
            dc = days - days.mean()
            lfc = (y - y.mean(1, keepdims=True)) @ dc / (dc @ dc) * (days[-1] - days[0])
        else:
            raise ValueError(mode)
        out[f"rep{r}"] = lfc - np.median(lfc[ctrl])
    return out


def pair_gi(d: pl.DataFrame, lfc: np.ndarray, mask: np.ndarray | None = None, guide_single: bool = False) -> pl.DataFrame:
    """Gene-pair additive GI from construct LFCs (optionally only constructs in mask). Columns gene_a, gene_b, gi, n."""
    x = d.select("target_a_id", "probe_a_id", "target_b_id", "probe_b_id").with_columns(pl.Series("lfc", lfc))
    if mask is not None:
        x = x.filter(pl.Series(mask))
    ka, kb = ("probe_a_id", "probe_b_id") if guide_single else ("target_a_id", "target_b_id")
    single = pl.concat([
        x.filter((pl.col("target_b_id") == "AAVS") & ~pl.col("target_a_id").is_in(CTRL)).select(pl.col(ka).alias("g"), "lfc"),
        x.filter((pl.col("target_a_id") == "AAVS") & ~pl.col("target_b_id").is_in(CTRL)).select(pl.col(kb).alias("g"), "lfc"),
    ]).group_by("g").agg(pl.col("lfc").median().alias("f"))
    f = dict(zip(single["g"], single["f"]))
    du = x.filter(~pl.col("target_a_id").is_in(CTRL) & ~pl.col("target_b_id").is_in(CTRL)
                  & (pl.col("target_a_id") != pl.col("target_b_id")))
    gi = du["lfc"].to_numpy() - np.array([f.get(v, np.nan) for v in du[ka]]) - np.array([f.get(v, np.nan) for v in du[kb]])
    du = du.with_columns(pl.Series("gi", gi)).drop_nans("gi").with_columns(
        pl.min_horizontal("target_a_id", "target_b_id").alias("gene_a"),
        pl.max_horizontal("target_a_id", "target_b_id").alias("gene_b"))
    return du.group_by("gene_a", "gene_b").agg(pl.col("gi").median(), pl.len().alias("n"))


def replicates(mode: str = "plasmid_last") -> pl.DataFrame:
    """Per-replicate additive GI per line: gene_a, gene_b, context, gi_rep1, gi_rep2 (raw target symbols)."""
    d = _counts()
    keep = (d["plasmid"] >= np.quantile(d["plasmid"].to_numpy(), 0.02)).to_numpy()
    d = d.filter(pl.Series(keep))
    parts = []
    for line, ctx in LINES.items():
        lfc = construct_lfc(d, line, mode)
        out = None
        for r, v in lfc.items():
            g = pair_gi(d, v).select("gene_a", "gene_b", pl.col("gi").alias(f"gi_{r}"))
            out = g if out is None else out.join(g, on=["gene_a", "gene_b"])
        parts.append(out.with_columns(pl.lit(ctx).alias("context")))
    return pl.concat(parts).select("gene_a", "gene_b", "context", "gi_rep1", "gi_rep2")
