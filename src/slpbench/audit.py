"""`slpbench audit`: re-run every label-reproducibility check and write REPLICATION.md.

The build's source decisions (build.EXCLUDED_SOURCES) are justified by this report; if a rebuild
changes a number enough to flip a decision, the report shows it next to the decision.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

from slpbench import replication as R
from slpbench.build import EXCLUDED_SOURCES, INTERIM, _context_table

MIN_AUROC = 0.65

# Checks for sources added in SLB-1.3 (implemented next to their parsers).
NEW_CHECKS = [
    "eukaryotes_extra.kuzmin2018_checks", "eukaryotes_extra.kuzmin2020_checks", "eukaryotes_extra.costanzo2021_checks",
    "eukaryotes_extra.scer_emaps_checks", "eukaryotes_extra.frost2012_checks", "eukaryotes_extra.horn2011_checks",
    "eukaryotes_extra.billmann2016_checks", "eukaryotes_extra.byrne2007_checks", "eukaryotes_extra.roguev2013_checks",
    "eukaryotes_extra.gier2020_checks", "bacteria_extra.koo2025_checks", "bacteria_extra.dualtnseq2025_checks",
    "bacteria_extra.crisprtnseq2024_checks", "bacteria_extra.spne_cross_checks", "bacteria_extra.ecoli_array_checks",
]
# Which of those rows summarise each source in the decisions table: {source: {kind: regex on `check`}}
_EC = {s: {"within": rf"^within-study .*: {s} ", "cross": rf"^cross-study: {s} labels"}
       for s in ("babu2011", "gagarinova2016", "kumar2016", "cote2016")}
SUMMARY = {
    "kuzmin2018": {"cross": r"^kuzmin2018 labels scored by costanzo2016"},
    "kuzmin2020": {"cross": r"^kuzmin2020 labels scored by (costanzo2016|kuzmin2018)"},
    "costanzo2021": {"within": r"^costanzo2021 reference epsilon", "cross": r"^costanzo2021 reference labels scored"},
    "scer_emaps": {"within": r"^hoppins2011 individual crosses.*S<-3\.0", "cross": r"^scer_emaps labels scored"},
    "frost2012": {"within": r"orientation/allele, pos S<-4\.0", "cross": r"^frost2012 labels \(S<-4\.0\)"},
    "horn2011": {"within": r"^horn2011 replicate screen .*2% tail", "cross": r"^horn2011 labels scored by heigwer2023"},
    "billmann2016": {"cross": r"^billmann2016 labels scored"},
    "byrne2007": {"within": r"^byrne2007 duplicate", "cross": r"^(byrne2007 labels scored by lehner|lehner2006 labels scored by byrne)"},
    "lehner2006": {"cross": r"^lehner2006 labels scored by byrne2007"},
    "roguev2013": {"within": r"^roguev2013 orientation split, pos S<-3\.0"},
    "gier2020": {"within": r"^AUROC of diff"},
    "koo2025": {"within": r"mean GI <= -1\.5\)"},
    "dualtnseq2025": {"within": r"^within-study: labels from run", "cross": r"^cross-study: dualtnseq2025 labels"},
    "crisprtnseq2024": {"within": r"median over", "cross": r"^cross-study: crisprtnseq2024 labels scored by dualcrispri"},
    "dualcrispri2025": {"cross": r"^cross-study: dualcrispri2025 labels"},
    "costanzo2016": {"cross": r"^costanzo2016 labels scored by"},
    "ryan2012": {"cross": r"^ryan2012 labels scored by frost2012"},
    "heigwer2023": {"cross": r"^heigwer2023 labels scored by (horn2011|billmann2016)"},
    "fischer2015": {"cross": r"^fischer2015 labels scored by (horn2011|heigwer2023)"},
    **_EC,
}


def _new_check(name: str) -> list[dict]:
    import importlib

    mod, fn = name.split(".")
    rows = getattr(importlib.import_module(f"slpbench.sources.{mod}"), fn)()
    return [{"function": fn, "check": r.get("check"), "auroc": r.get("auroc"), "pos": r.get("pos"),
             "ci95": "–".join(f"{x:.2f}" for x in r["ci95"]) if r.get("ci95") else None} for r in rows]


def _summarise(new: pl.DataFrame) -> dict[tuple[str, str], str]:
    out = {}
    for src, kinds in SUMMARY.items():
        for kind, rx in kinds.items():
            v = new.filter(pl.col("check").str.contains(rx) & pl.col("auroc").is_not_null())["auroc"].unique()
            if v.len():
                out[(src, kind)] = f"{v.min():.2f}" if v.len() == 1 else f"{v.min():.2f}–{v.max():.2f}"
    return out


def _md(df: pl.DataFrame) -> str:
    cols = df.columns
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.iter_rows():
        out.append("| " + " | ".join(
            "" if v is None else (f"{v:.3f}" if isinstance(v, float) else (f"{v:,}" if isinstance(v, int) else str(v)))
            for v in r) + " |")
    return "\n".join(out)


def run() -> dict:
    m = pl.concat([pl.read_parquet(p) for p in sorted((INTERIM / "measurements").glob("*.parquet"))])
    ctx = _context_table(m)
    m = m.join(ctx.select("species", "source", "context", "context_id"), on=["species", "source", "context"])
    hm = m.filter(pl.col("species").is_in(["human"]))
    cross = R.cross_study(hm)
    inc = R.cross_study(hm.filter(~pl.col("source").is_in(list(EXCLUDED_SOURCES)))).select(
        "source", pl.col("overlap_pos").alias("pos_vs_included"), pl.col("cross_study_auroc").alias("auroc_vs_included"))
    cross = cross.join(inc, on="source", how="left")

    with ProcessPoolExecutor(8, mp_context=mp.get_context("spawn")) as ex:
        slkb = pl.DataFrame(list(ex.map(R.slkb_job, R.SLKB_JOBS)), infer_schema_length=None)
        new = pl.DataFrame([r for rows in ex.map(_new_check, NEW_CHECKS) for r in rows], infer_schema_length=None,
                           schema_overrides={"auroc": pl.Float64, "pos": pl.Int64})
    within = pl.DataFrame(
        R.costanzo_orientation() + R.ryan_alleles() + R.spne_replicates() + R.spidr_checks() + R.heigwer_split_half(),
        infer_schema_length=None)
    fit = R.fitness_diagnostic()
    return {"cross": cross, "slkb": slkb, "within": within, "fitness": fit, "new": new,
            "sources": pl.DataFrame({"source": sorted(m["source"].unique().to_list())})}


def decisions(res: dict) -> pl.DataFrame:
    srcs = res["sources"]["source"].to_list()
    cross = {r["source"]: r["cross_study_auroc"] for r in res["cross"].iter_rows(named=True)}
    cross_inc = {r["source"]: r["auroc_vs_included"] for r in res["cross"].iter_rows(named=True)}
    slkb = res["slkb"].filter(pl.col("auroc_single_replicate").is_not_null() & pl.col("auroc_single_replicate").is_not_nan()) \
        .group_by("source").agg(pl.col("auroc_single_replicate").min().alias("lo"), pl.col("auroc_single_replicate").max().alias("hi"))
    rep = {s: f"{lo:.2f}–{hi:.2f}" for s, lo, hi in slkb.iter_rows()}
    for r in res["within"].filter(pl.col("slb_rule")).iter_rows(named=True):
        rep[r["source"]] = f"{r['auroc']:.2f}"
    fit = dict(res["fitness"].iter_rows())
    summ = _summarise(res["new"])
    rows = []
    for s in srcs:
        cs = cross.get(s)
        rows.append({"source": s, "cross-study AUROC": f"{cs:.3f}" if cs is not None else summ.get((s, "cross")),
                     "vs included studies": cross_inc.get(s),
                     "within-study AUROC": rep.get(s) or summ.get((s, "within")),
                     "fitness-only AUROC": fit.get(s),
                     "in benchmark": "no" if s in EXCLUDED_SOURCES else "yes",
                     "reason": EXCLUDED_SOURCES.get(s, "")})
    return pl.DataFrame(rows, infer_schema_length=None)


def write(res: dict) -> None:
    dec = decisions(res)
    slkb = res["slkb"].select([c for c in ["source", "context", "pairs", "replicates", "spearman", "tail_2pct_in_10pct",
                                           "labelled", "pos", "auroc_single_replicate", "error"] if c in res["slkb"].columns])
    text = f"""# Label reproducibility (SLB-1.3)

Generated by `slpbench audit`. Do not edit by hand.

A benchmark label is only as good as its chance of coming out the same way when the experiment is
repeated. Every source is checked against an independent re-measurement, using the strongest test
available for it:

- **Cross-study** (strongest): another published screen measured the same pair in the same cell line.
  AUROC = how well study B's score (its within-study percentile) recovers study A's labels, against
  all other studies and against the included studies only (the benchmark's consensus).
- **Within-study**: the source's own biological replicates, alleles or query/array orientations,
  each re-scored independently with one uniform additive model (LFC − f_a − f_b). This shows the
  measurement is reproducible; it cannot catch a protocol-specific artifact shared by all replicates.
- **Fitness-only AUROC**: how well −(single-loss effect of A + B) alone predicts the labels. High values
  mean the calls track single-gene sickness; that is partly real biology, but combined with failed
  replication it points to artifact.

**Rule:** a source's labels enter the benchmark if independent re-measurement recovers them at
AUROC ≥ {MIN_AUROC}, they are not contradicted by the cross-study consensus, and the hit rate is plausible.

## Decisions

{_md(dec)}

Also excluded before this audit: Diehl 2021 (63% of tested pairs called SL) and Tang 2022 (22%).

## Cross-study replication (human cell lines)

{_md(res['cross'])}

## Within-study replicates: SLKB human screens

Each replicate re-scored independently. `tail_2pct_in_10pct` = of one replicate's 2% most negative
pairs, the fraction in another replicate's 10% most negative. `auroc_single_replicate` = mean over
replicates of one replicate's GI recovering the benchmark labels (optimistic: labels were derived
from pooled replicates).

{_md(slkb)}

## Within-study checks: other sources

{_md(res['within'].select('source', 'check', 'slb_rule', 'pos', 'auroc'))}

## Checks for sources added in SLB-1.3

Each function lives next to its parser (`sources/eukaryotes_extra.py`, `sources/bacteria_extra.py`);
details and rationale per source are in `notes/data/<source>.md`. "X labels scored by Y" = AUROC of
study Y's score for study X's labels on the pairs both measured.

{_md(res['new'].select('function', 'check', 'pos', 'auroc', 'ci95'))}

Stronger positive thresholds replicate better in both yeasts, which is why SLB uses ε < −0.2 (not the
authors' −0.12) for *S. cerevisiae* and S < −3 (not −2.3) for *S. pombe*. SPIDR's published GEMINI
calls are not recovered by its own replicates, while a uniform additive re-scoring of its raw counts
is (see the SPIDR rows); SLB therefore labels SPIDR from counts.
"""
    Path("REPLICATION.md").write_text(text)
    (INTERIM / "replication.json").write_text(json.dumps({k: v.to_dicts() for k, v in res.items()}, indent=1, default=str))
    print(_md(dec))


def main() -> None:
    write(run())
