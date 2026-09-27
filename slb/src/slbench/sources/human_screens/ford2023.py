"""Ford et al. 2023 Sci Rep (doi 10.1038/s41598-023-33329-2, CC BY): all-by-all CDK-family dual-sgRNA Cas9 screen.

Design: 26 genes (CDK1-20, CDK11A/B, AR, EZH2, PARP1, PRMT5, TGFBR1) + non-targeting (`ntc`) and AAVS1-cutting
(`ntc-AAV`) controls, 4 sgRNAs each, all-by-all in both orientations (16 guide pairs per ordered gene pair,
32 per unordered pair). Lines MDA-MB-231, MDA-MB-468, Hs 578T; 2 replicates x 4 timepoints (first T3, last T28).
Supplementary 41598_2023_33329_MOESM1_ESM.xlsx: per-line count sheets and `Interaction_Scores` (authors'
pi score `new_pi` per line, incl. self and control rows). The authors' "Genetic interaction score" in the
Physical_interaction_evidence sheet is the z-score of new_pi within line (their highlighted |z| >~ 2 set is
not stringent), so SLB re-scores.

Measurement: score = authors' new_pi (negative = synthetic sick/lethal), no p-values published.
Label rule (per line, over the 325 gene x gene pairs, self and control rows excluded):
  z = (new_pi - mean) / sd within line
  positive: z <= -3 AND own additive GI (replicates()) < 0 in both replicates
  negative: |z| < 1
  else null.

replicates(): spidr_replicate_gi recipe per line and replicate: LFC = log2 norm counts T28_rep vs mean T3,
guide pairs with T3 below the 2% quantile dropped, centred on control x control (ntc + AAVS1), f_g = median LFC
of gene x control, GI = LFC - f_a - f_b, median over the 32 guide pairs (both orientations).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from slbench.sources import finalize

RAW = Path("data/raw/ford2023")
XLSX = RAW / "41598_2023_33329_MOESM1_ESM.xlsx"
CACHE = RAW / "_parsed"

# sheet, context, (start rep1, start rep2), (end rep1, end rep2), Interaction_Scores tag
LINES = {
    "MDA-MB-231": ("MDA-MB-231_Dual_CRISPR_Counts", ("MDA_T3_1", "MDA_T3_2"), ("MDA_T28_1", "MDA_T28_2"), "mda231"),
    "MDA-MB-468": ("MDA-MB-468_Dual_CRISPR_Counts", ("MDA-MB-468_T3_1", "MDA-MB-468_T3_2"),
                   ("MDA-MB-468_T28_1", "MDA-MB-468_T28_2"), "mda468"),
    "Hs 578T": ("Hs578T_Dual_CRISPR_Counts", ("CDK-HS478_T3_1", "CDK-HS478_T3_2"),
                ("CDK-HS478_T28_1", "CDK-HS478_T28_2"), "hs578t"),
}
CONTROLS = ("ntc", "ntc-AAV")


def _sheet(name: str) -> pl.DataFrame:
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{name}.parquet"
    if not p.exists():
        import openpyxl

        wb = openpyxl.load_workbook(XLSX, read_only=True)
        rows = [r for r in wb[name].iter_rows(values_only=True) if r and r[0] is not None]
        h = [c for c in rows[0] if c is not None]
        pl.DataFrame([r[:len(h)] for r in rows[1:]], schema=h, orient="row", infer_schema_length=None).write_parquet(p)
    return pl.read_parquet(p)


def pi_scores() -> pl.DataFrame:
    """Authors' pi per line for gene x gene pairs (no self/control rows), with within-line z."""
    s = _sheet("Interaction_Scores").rename({"cell_line": "tag", "target_id_1": "a", "target_id_2": "b"})
    ctx = {v[3]: k for k, v in LINES.items()}
    s = s.filter((pl.col("a") != pl.col("b")) & ~pl.col("a").is_in(CONTROLS) & ~pl.col("b").is_in(CONTROLS))
    return s.with_columns(
        pl.col("tag").replace_strict(ctx).alias("context"),
        pl.min_horizontal("a", "b").alias("gene_a"), pl.max_horizontal("a", "b").alias("gene_b"),
        ((pl.col("new_pi") - pl.col("new_pi").mean().over("tag")) / pl.col("new_pi").std().over("tag")).alias("z"),
    ).select("context", "gene_a", "gene_b", "new_pi", "z")


def _replicate_gi_line(ctx: str, controls: tuple[str, ...] = CONTROLS) -> pl.DataFrame:
    sheet, start, end, _ = LINES[ctx]
    d = _sheet(sheet)
    g1 = np.array(["CONTROL" if x in controls else x for x in d["target_a_id"]])
    g2 = np.array(["CONTROL" if x in controls else x for x in d["target_b_id"]])
    t0 = sum(d[c].cast(pl.Float64).fill_null(0).to_numpy() for c in start) / len(start)
    keep = t0 >= np.quantile(t0, 0.02)
    base = np.log2((t0 + 1) / (t0 + 1).sum())
    out = None
    for i, col in enumerate(end, start=1):
        te = d[col].cast(pl.Float64).fill_null(0).to_numpy()
        lfc = np.log2((te + 1) / (te + 1).sum()) - base
        lfc = lfc - np.median(lfc[keep & (g1 == "CONTROL") & (g2 == "CONTROL")])
        x = pl.DataFrame({"g1": g1, "g2": g2, "lfc": lfc}).filter(pl.Series(keep))
        # guide pairs whose other arm is a control not in `controls` (e.g. ntc when only AAVS is used) are dropped
        x = x.filter(~pl.col("g1").is_in(CONTROLS) & ~pl.col("g2").is_in(CONTROLS))
        single = pl.concat([
            x.filter(pl.col("g2") == "CONTROL").select(pl.col("g1").alias("g"), "lfc"),
            x.filter(pl.col("g1") == "CONTROL").select(pl.col("g2").alias("g"), "lfc"),
        ]).filter(pl.col("g") != "CONTROL").group_by("g").agg(pl.col("lfc").median().alias("f"))
        f = dict(zip(single["g"], single["f"]))
        du = x.filter((pl.col("g1") != "CONTROL") & (pl.col("g2") != "CONTROL") & (pl.col("g1") != pl.col("g2")))
        gi = du["lfc"].to_numpy() - np.array([f.get(v, np.nan) for v in du["g1"]]) - np.array([f.get(v, np.nan) for v in du["g2"]])
        du = du.with_columns(pl.Series("gi", gi)).drop_nans("gi").with_columns(
            pl.min_horizontal("g1", "g2").alias("gene_a"), pl.max_horizontal("g1", "g2").alias("gene_b"))
        agg = du.group_by("gene_a", "gene_b").agg(pl.col("gi").median().alias(f"gi_rep{i}"))
        out = agg if out is None else out.join(agg, on=["gene_a", "gene_b"])
    return out.with_columns(pl.lit(ctx).alias("context")).select("gene_a", "gene_b", "context", "gi_rep1", "gi_rep2")


def replicates(controls: tuple[str, ...] = CONTROLS) -> pl.DataFrame:
    """Per-replicate additive GI (T28 vs T3) for every line: gene_a, gene_b, context, gi_rep1, gi_rep2."""
    return pl.concat([_replicate_gi_line(c, controls) for c in LINES]).sort("context", "gene_a", "gene_b")


def load() -> pl.DataFrame:
    s = pi_scores().join(replicates(), on=["context", "gene_a", "gene_b"], how="left")
    df = s.select(
        pl.lit("human").alias("species"), pl.lit("ford2023").alias("source"), "context",
        pl.lit("CRISPR-KO").alias("mechanism"), "gene_a", "gene_b",
        pl.col("new_pi").alias("score"), pl.lit("new_pi").alias("score_name"),
        pl.lit(None, pl.Float64).alias("signif"), pl.lit(None, pl.String).alias("signif_name"),
        pl.when((pl.col("z") <= -3) & (pl.col("gi_rep1") < 0) & (pl.col("gi_rep2") < 0)).then(1)
        .when(pl.col("z").abs() < 1).then(0).otherwise(None).cast(pl.Int8).alias("label"),
    )
    return finalize(df, "human")
