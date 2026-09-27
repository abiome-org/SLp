"""Feng, Tang, Dede et al. 2022 Sci Adv 8:eabm6638, doi 10.1126/sciadv.abm6638 (CC BY-NC).

Genome-wide TKOv3 screens in HEK293A isogenic tumour-suppressor knockouts, raw counts from figshare
10.6084/m9.figshare.19398332 (Table_S1_Raw_read_counts_master_Xu_Feng_Tm_sup_screens.txt). Each arm has a
T0 and two replicates (A, B) at T20-T25. Screens were run in batches, each with its own WT 293A arm:
  XF498: WT, STK11 (LKB1), PTEN, VHL        XF646: WT, CDH1, NF2, BAP1
  XF804: WT, ARID1A, PBRM1                  XF821: WT, KEAP1, NF1, RB1, TP53
  XF443: TP53BP1 only (no WT arm in its batch; scored against the mean of the four WT arms, rep A with the
         WT A mean and rep B with the WT B mean; flagged in the notes).
The authors' own calls (Table S2) are binary BAGEL essentiality calls per screen, not a KO-vs-WT
interaction score, so the screens are re-scored from counts with the shared engine in
`human_screens.desjardins2026` (residual z of KO LFC vs WT LFC; see that module for the recipe and label rule:
positive = z <= -4, z < 0 in both replicates, LFC_KO < -0.5, not a positive for >= 3 distinct queries
(PRDX1, DDX19A, PRKRA, TYMS recur across unrelated queries and batches); negative = |z| < 1).
Scoring one batch's WT arm against another's (no query) gives ~80% of the z <= -3 and ~50% of the z <= -4
hit rate of the real query screens, so these labels are mostly noise: verdict measurements-only. SETD2 appears in the
authors' BF matrix but not in the count table; it is not scored. Context: HEK293-A (Cellosaurus CVCL_6910).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from .desjardins2026 import query_screen, to_measurements, to_replicates

RAW = Path("data/raw/feng2022")
KEY = "feng2022"
CONTEXT = "HEK293-A"
FILE = "Table_S1_Raw_read_counts_master_Xu_Feng_Tm_sup_screens.txt"
CONTROLS = {"LacZ", "luciferase", "EGFP"}
# column prefix -> query symbol
QUERY = {"LKB1": "STK11", "PTEN": "PTEN", "VHL": "VHL", "CDH1": "CDH1", "NF2": "NF2", "BAP1": "BAP1",
         "ARID1A": "ARID1A", "PBRM1": "PBRM1", "KEAP1": "KEAP1", "NF1": "NF1", "RB1": "RB1", "TP53": "TP53",
         "TP53BP1": "TP53BP1"}


def _arms() -> tuple[np.ndarray, dict]:
    d = pl.read_csv(RAW / FILE, separator="\t")
    d = d.filter(~pl.col("GENE").is_in(list(CONTROLS)))
    cols = d.columns[2:]
    assert len(cols) % 3 == 0
    arms = {}
    for i in range(0, len(cols), 3):
        t0, a, b = cols[i:i + 3]
        name = t0.split("_")[1]
        batch = a.split("_")[-1]  # the CDH1 T0 column lacks the batch tag
        arms[(name, batch)] = tuple(d[c].cast(pl.Float64).fill_null(0).to_numpy() for c in (t0, a, b))
    return d["GENE"].to_numpy(), arms


def screens() -> pl.DataFrame:
    genes, arms = _arms()
    wt = {b: v for (n, b), v in arms.items() if n == "WT"}
    out = []
    for (n, b), (t0, a, bb) in arms.items():
        if n == "WT":
            continue
        q = QUERY[n]
        if b in wt:
            w0, wa, wb = wt[b]
        else:  # TP53BP1: no matched WT; average the WT arms (counts normalised per arm first)
            norm = [[x / x.sum() * 1e7 for x in v] for v in wt.values()]
            w0, wa, wb = (np.mean([v[k] for v in norm], axis=0) for k in range(3))
        d = query_screen(genes, w0, [wa, wb], t0, [a, bb], drop={q, n})
        out.append(d.with_columns(pl.lit(q).alias("query"), pl.lit(CONTEXT).alias("context"),
                                  pl.lit(b in wt).alias("matched_wt")))
    return pl.concat(out)


LABEL_KW = {"z_pos": -4.0}


def load() -> pl.DataFrame:
    return to_measurements(screens(), KEY, **LABEL_KW)


def replicates() -> pl.DataFrame:
    """gene_a, gene_b, context, query, gi_rep1 (KO rep A vs WT rep A), gi_rep2 (B vs B); z-scaled GI."""
    return to_replicates(screens())
