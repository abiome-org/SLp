# roguev2013 (mouse esiRNA E-MAP of chromatin factors, Roguev et al. 2013)

**Citation.** Roguev A, Talbot D, Negri GL, et al. Quantitative genetic-interaction mapping in
mammalian cells. *Nat Methods* 10:432-437 (2013). doi:10.1038/nmeth.2398. PMC3641890.

**Files** (`data/raw/roguev2013_mmus/`; `roguev2013_mmus.fetch.tsv`; URLs re-verified by sha256 2026-09-24)

| file | sha256 |
|---|---|
| MOESM185.zip (S_scoresAvg.txt symmetric 130x130; S_scores.txt unaveraged 124 x 130; TreeView files) | 8794b434b6b1881056ec858dfe2c89cee91d6123974adeb4bb8a79d599a351f9 |
| MOESM182.xlsx (1,540 pairs with abs(S) >= 2) | 495f88b22340f9cc37435c17ce08da32b1f19055c10e20a9ce705eb55ff7f834 |
Also downloaded, not used: MOESM181.pdf (suppl. figures), MOESM183.xlsx (esiRNA primers),
MOESM184.zip (raw colony/cell-count .dat files per batch). License: Springer Nature ESM.

**Design.** ~11,000 pairwise esiRNA double knockdowns of 130 chromatin regulators in mouse
fibroblasts (abstract: "mouse fibroblasts"; exact line not stated in the supplement), cell-number
read-out, E-MAP S-scores. Species `mmus` (MGI symbol via `ids_extra.mmus`; the file uses upper-case
symbols), context `fibroblast`, mechanism `RNAi`.

**Parser.** `eukaryotes_extra.roguev2013()` (averaged symmetric matrix; diagonal dropped).

**Counts.** 6,937 pairs over 126 genes (4 symbols unresolved); **31 positives, 4,835 negatives**,
2,071 ambiguous. Positive rate 0.6% of labelled pairs; the map is dominated by positive
(alleviating) interactions within complexes.

**Label rule.** SLB E-MAP rule: positive S < -3, negative |S| < 1.

**Replication evidence** (`roguev2013_checks()`, unaveraged matrix, the two orientations of a pair
are separate measurements)

| check | pairs measured twice | pos | AUROC | Pearson |
|---|---|---|---|---|
| orientation split, positive S < -2 | 2,903 | 148 | 0.744 | 0.50 |
| orientation split, positive S < -3 (SLB rule) | 2,903 | 34 | 0.731 | 0.50 |

No usable cross-study overlap: Gier 2020 shares 4 genes and 6 labelled pairs (Gier calls 5 of them SL, Roguev calls none; Gier is excluded for artefactual calls, see gier2020_mmus.md).
Fitness-only AUROC with `mmus_single()`: 0.41 (positives are not on sicker genes).

**Recommendation: INCLUDE (small).** Passes within-study (0.73) with a plausible hit rate and no
fitness confound, but has only 31 positives over 126 genes: too small for a scored mouse species on
its own; useful as training data and as a (tiny) dev/test stratum if the lead wants a mouse context.

**Single-gene effect for `mmus`:** `eukaryotes_extra.mmus_single()` = DepMap 24Q4 Chronos pan-line
mean of the 1:1 human ortholog (Alliance combined orthology, IsBestScore and IsBestRevScore = Yes,
exactly one human gene), 17,051 mouse genes. There is no genome-wide mouse DepMap; this proxy uses
only permitted single-gene data. A context-specific alternative would need a genome-wide mouse
fibroblast knockout screen, which I did not find.
