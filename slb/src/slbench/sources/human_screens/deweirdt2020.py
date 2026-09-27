"""DeWeirdt, Sangree et al. 2020 Nat Commun 11:752, doi 10.1038/s41467-020-14620-6 (CC BY 4.0).

Anchor screens: a SaCas9 "anchor" guide (BCL2L1, MCL1 or PARP1; or a control guide) is combined with the
genome-wide SpCas9 Brunello library (4 guides/gene), dropout (no drug) arms only, two replicates (A, B) per
arm, counts vs the plasmid (pDNA) in Supplementary Data 3 (MOESM4, sheet "Brunello"). Genetic anchors kept:
  MEL-JUSO: MCL1, BCL2L1                         (control-guide arm = WT)
  OVCAR-8:  MCL1, BCL2L1, PARP1 (Sa-guides 1, 2)
  A375:     PARP1 (Sa-guides 1, 2)
  HAP1:     PARP1 single-cell knockout clone vs parental (sheet "HAP1 single-cell KO Brunello")
Drug arms (A-1331852, S63845, olaparib, talazoparib), the Gattinara library copies of the drug screens, and
the secondary (targeted-library, SpCas9-anchor) A375 MCL1 screen are not used.

Scoring: shared engine in `human_screens.desjardins2026` (residual z of anchor-arm LFC vs control-arm LFC,
running median/MAD over genes ordered by control LFC; T0 = pDNA for both arms). This is the same quantity
as the authors' published anchor z-scores (MOESM5: residual from a control-vs-anchor fit, z-scored); ours
is recomputed so per-replicate values exist, and agreement with MOESM5 is reported in the notes.
For PARP1 in OVCAR-8 / A375 the two independent anchor guides are pooled (mean arm LFC over both guides x
both replicates); gi_anchor1 / gi_anchor2 give each anchor guide separately (independent perturbations),
gi_rep1 / gi_rep2 the technical replicates (A, B; pooled over anchor guides where there are two).
Label rule: positive = z <= -3 (the authors' z convention; our z correlates 0.90-0.96 with MOESM5),
z < 0 in both replicates, LFC_anchor < -0.5, and not a positive for >= 3 distinct anchors (OTUD5, H2BC13);
negative = |z| < 1; else null. MCL1 in MEL-JUSO and PARP1 in A375 fail their own split-half check
(AUROC ~0.59-0.60) and carry no labels (UNLABELLED).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl
import polars as pl

from .desjardins2026 import query_screen, to_measurements, to_replicates

RAW = Path("data/raw/deweirdt2020")
KEY = "deweirdt2020"
# (sheet, context, query) -> (control cols [A, B], list of anchor-guide col pairs [[A, B], ...]); 0-based
# column indices in the sheet (col 0 = barcode, col 1 = pDNA)
SCREENS = {
    ("Brunello", "MEL-JUSO", "MCL1"): ([2, 3], [[8, 9]]),
    ("Brunello", "MEL-JUSO", "BCL2L1"): ([2, 3], [[10, 11]]),
    ("Brunello", "OVCAR-8", "MCL1"): ([12, 13], [[20, 21]]),
    ("Brunello", "OVCAR-8", "BCL2L1"): ([12, 13], [[22, 23]]),
    ("Brunello", "OVCAR-8", "PARP1"): ([12, 13], [[24, 25], [26, 27]]),
    ("Brunello", "A375", "PARP1"): ([28, 29], [[32, 33], [34, 35]]),
    ("HAP1 single-cell KO Brunello", "HAP1", "PARP1"): ([2, 3], [[4, 5]]),
}
HEADER_CHECK = {  # expected (cell line, perturbation, small molecule) per used column
    "Brunello": {2: ("Meljuso", "Control guide", "Dropout"), 8: ("Meljuso", "MCL1 Saur-guide", "Dropout"),
                 10: ("Meljuso", "BCL2L1 Saur-guide", "Dropout"), 12: ("OVCAR8", "Control guide", "Dropout"),
                 20: ("OVCAR8", "MCL1 Saur-guide", "Dropout"), 22: ("OVCAR8", "BCL2L1 Saur-guide", "Dropout"),
                 24: ("OVCAR8", "PARP1 Saur-guide 1", "Dropout"), 26: ("OVCAR8", "PARP1 Saur-guide 2", "Dropout"),
                 28: ("A375", "Control guide", "Dropout"), 32: ("A375", "PARP1 Saur-guide 1", "Dropout"),
                 34: ("A375", "PARP1 Saur-guide 2", "Dropout")},
    "HAP1 single-cell KO Brunello": {2: ("HAP1", "Parental", "Dropout"), 4: ("HAP1", "PARP1 SSC", "Dropout")},
}


def _sheet(name: str) -> tuple[np.ndarray, np.ndarray]:
    wb = openpyxl.load_workbook(RAW / "MOESM4.xlsx", read_only=True)
    ref = {}
    for r in wb["Brunello reference"].iter_rows(min_row=2, values_only=True):
        if r[0]:
            ref.setdefault(r[0], set()).add(str(r[1]))
    rows = list(wb[name].iter_rows(values_only=True))
    for j, exp in HEADER_CHECK[name].items():
        got = tuple(str(rows[i][j]).strip() for i in range(3))
        assert got == exp, (name, j, got, exp)
    body = [r for r in rows[4:] if r[0]]
    genes = []
    for r in body:  # guides mapping to exactly one gene; controls / multi-mapping guides dropped
        g = ref.get(r[0], set())
        genes.append(next(iter(g)) if len(g) == 1 and not next(iter(g)).startswith(("NO_", "ONE_", "POTENTIALLY")) else "")
    counts = np.array([[float(x or 0) for x in r[1:]] for r in body])  # col 0 here = pDNA
    return np.array(genes), counts


def screens() -> pl.DataFrame:
    cache: dict = {}
    out = []
    for (sheet, cl, q), (ctrl, anchors) in SCREENS.items():
        if sheet not in cache:
            cache[sheet] = _sheet(sheet)
        genes, c = cache[sheet]
        ok = genes != ""
        g, c = genes[ok], c[ok]
        pdna = c[:, 0]
        col = lambda j: c[:, j - 1]  # noqa: E731  (sheet col j -> counts col j-1)
        wt = [col(j) for j in ctrl]
        # pooled over anchor guides: replicate A = all anchor guides' rep A (mean of normalised counts)
        norm = lambda x: x / x.sum() * 1e7  # noqa: E731
        ko = [np.mean([norm(col(pair[k])) for pair in anchors], axis=0) for k in range(2)]
        d = query_screen(g, pdna, wt, pdna, ko, drop={q})
        if len(anchors) == 2:
            for i, pair in enumerate(anchors, start=1):
                a = query_screen(g, pdna, wt, pdna, [col(pair[0]), col(pair[1])], drop={q})
                # anchor-guide score: pooled over its two replicates
                d = d.join(a.select("gene", pl.col("z").alias(f"z_anchor{i}")), on="gene", how="left")
        out.append(d.with_columns(pl.lit(q).alias("query"), pl.lit(cl).alias("context")))
    return pl.concat(out, how="diagonal")


# Screens whose own replicates do not agree (split-half AUROC < 0.65 at z <= -3): measured, not labelled.
UNLABELLED = frozenset({("MCL1", "MEL-JUSO"), ("PARP1", "A375")})
LABEL_KW = {"z_pos": -3.0, "unlabelled": UNLABELLED}


def load() -> pl.DataFrame:
    return to_measurements(screens(), KEY, **LABEL_KW)


def replicates() -> pl.DataFrame:
    """gene_a, gene_b, context, query, gi_rep1/gi_rep2 (technical A/B), gi_anchor1/gi_anchor2 (PARP1 in
    OVCAR-8 and A375 only: independent SaCas9 anchor guides); z-scaled GI."""
    return to_replicates(screens())
