# *E. coli* K-12 colony-array double-mutant GI screens — four sources, all recommended EXCLUDE

Species code `ecol` (canonical gene ID = b-number, `ids_extra.ecol`; `bacteria_extra.ecol_map` also
resolves JW/Keio ids and gene names). Parsers in `slpbench.sources.bacteria_extra`:
`babu2011()`, `gagarinova2016()`, `kumar2016()`, `cote2016()`. Single-gene fitness:
`ecol_single()`.

**Headline: these four screens do not reproduce one another.** Every cross-study label-recovery
AUROC among them is 0.46–0.51, on 400–1,600 overlapping labelled pairs, while each screen agrees
with *itself* across growth conditions at 0.60–0.91. Two of them also call an implausible fraction
of pairs (Babu 2011 28% of labelled pairs, Kumar 2016 24%). Under the SLB rule none of them can
supply labels. They are parsed so they can be kept as training measurements.

## Sources

| key | citation | design | pairs (mapped) | conditions |
|---|---|---|---|---|
| `babu2011` | Babu M *et al.*, **PLoS Genet** 7(11):e1002377 (2011), doi:10.1371/journal.pgen.1002377 | eSGA: 821 cell-envelope query deletions × Keio array | 227,050 | rich (RM), minimal (MM) |
| `babu2014` | Babu M *et al.*, **PLoS Genet** 10(2):e1004120 (2014) | eSGA, genome-scale E. coli GI map | 42,705 **high-confidence GIs only** | rich |
| `gagarinova2016` | Gagarinova A *et al.*, **Cell Rep** 17(3):904–916 (2016), doi:10.1016/j.celrep.2016.09.040 | all pairs among ~338 translation/ribosome genes | 42,545 | RM, MM, 23 °C (LT), 42 °C (HT) |
| `kumar2016` | Kumar A *et al.*, **Cell Rep** 14(3):648–661 (2016), doi:10.1016/j.celrep.2015.12.060 | genome-integrity GI map, ± MMS | 100,170 | untreated (UT), MMS |
| `cote2016` | Côté J-P *et al.*, **mBio** 7(6):e01714-16 (2016), doi:10.1128/mBio.01714-16 (plus Author Correction mBio 7(6):e02138-16) | 82 (+2) nutrient-stress query deletions × whole Keio collection, 315,400 double mutants | 303,113 | LB (and M9 for gdhA) |

**Babu 2014 supplies only its called interactions** (Table S2 = 42,705 pairs with GI scores, all
hits). With no measured non-interactions it cannot provide negatives, so it is not parsed as a
source; its curated-literature comparison table (S3) is useful for sanity checks only.

## Files and provenance

URL lists are in `notes/data/<key>.fetch.tsv`; every URL there was verified by re-downloading and
matching sha256 against the local file, except where noted. Full sha256 values:
`(cd data/raw/<dir> && sha256sum *)`. First 16 hex digits:

`babu2011_ecoli` (PLoS supplement ids s009–s030): TableS3_GI_scores.xlsx `49abf52b26719400`
(= .s022, the parsed file), TableS1_targets.xlsx `44e5d2defb08af82` (.s020), TableS4_SL.xlsx
`7d04960ee7c5f5ca` (.s023), TableS9_differential.xlsx `eeb461f57780d914` (.s028),
TableS11_strains.xlsx `5dab0d16e5f1642c` (.s030), ProtocolS2/3/4 pdf (.s009/.s010/.s011).

`babu2014_ecoli`: TableS2_hiconf_GIs.xls `0789563ada0db3e3` (.s023), TableS1_queries.xls
`7506c2edc9975d0e` (.s022), TableS3_curated.xls `35c799584d4c23ac` (.s024), TableS9_correlations
`de90da1e8416ef86` (.s030). TableS16_strains.xls `51733f265f62ea76` — supplement id not found in
s001–s030.

`gagarinova2016_ecoli` (Elsevier PII S2211124716312797): mmc3.xlsx `08dc6fb0027d115c` is the parsed
GI table; mmc2/4/5/6 and the two PDFs all matched their `mmcN` URLs.

`kumar2016_ecoli`: `cellrep_mmc2.zip` … `cellrep_mmc7.xlsx` matched Elsevier PII S2211124715015016.
**The six `Table_S*.xlsx` files, including `Table_S2.xlsx` (`4fc8d472d1c15f08`) that the parser
reads, have no verified URL:** they were fetched by a helper agent whose log was lost in the
machine's power cycle, they are not the contents of `cellrep_mmc2.zip` (which holds a single
`.xlsb`), and Kumar 2016 has no PMC record. Their content matches the paper (Table S2 = "List of
gene pairs with GI scores in static (UT and MMS) networks", 107,147 pairs), but treat the
provenance as unverified until someone re-locates the file.

`cote2016_ecoli`: file names match the mBio supplement pattern
`https://journals.asm.org/doi/suppl/10.1128/mBio.01714-16/suppl_file/mbo006163075stN.xlsx`;
journals.asm.org now returns 403 to this machine, so the URLs could not be re-verified by hash.
The PMC copy (`pmc.ncbi.nlm.nih.gov/articles/instance/5120140/bin/…`) serves different, much
smaller files. Parsed file: `mbo006163075st2.xlsx` `233dbfb9b66f67d2`, whose main sheet is exactly
the published 3,803 Keio rows × 82 nutrient-stress query columns (plus `panF` and `gdhA (M9)`
sheets) — i.e. the 315,400 double mutants of the paper.

`keio2006` (for `ecol_single`): Baba T *et al.*, **Mol Syst Biol** 2:2006.0008 (2006), PMC1681482,
supplementary tables via the EuropePMC supplementaryFiles API
(`https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1681482/supplementaryFiles`, zip
`16815f59cf5283ef`). `S3_keio_mutants.xls` `a1a9c30b6cd703ff` (= msb4100050-s5.xls: ECK/JW/b-number
plus OD600 growth), `S6_essential_candidates.xls` `ded72d1f890d47b0` (= -s8.xls),
`S2.xls`, `S7.xls` kept for the JW↔b mapping and COG/essentiality flags.

## Label rules used

Shared helper `_ecoli_finalize`: positive = score ≤ threshold and p < 0.05; negative = |score|
below that screen's median |score| and p > 0.25. Scores are the authors' own, with negative =
aggravating.

Final counts (after routing IDs through `ids_extra`, lead decision 2026-09-24):

| source (context) | positive | negative | rows | pos | neg | ambiguous | genes | hit rate | fitness-only AUROC |
|---|---|---|---|---|---|---|---|---|---|
| `babu2011` (BW25113_RM) | E ≤ −2.5, p < 0.05 | \|E\| < median, p > 0.25 | 232,772 | 42,705 | 111,268 | 78,799 | 817 | **28%** | 0.50 |
| `babu2011` (BW25113_MM) | same | same | 128,084 | 32,066 | 60,431 | 35,587 | 788 | **35%** | — |
| `gagarinova2016` (RM) | GI ≤ −0.2, p < 0.05 | \|GI\| < median, p > 0.25 | 43,166 | 920 | 21,533 | 20,713 | 312 | 4.1% | 0.64 |
| `gagarinova2016` (MM / LT / HT) | same | same | 42,860 / 43,065 / 43,065 | 1,828 / 1,556 / 923 | 19,813 / 20,616 / 17,945 | — | 312 | 4–8% | — |
| `kumar2016` (UT) | S ≤ −2.5, p < 0.05 | \|S\| < median, p > 0.25 | 101,175 | 9,345 | 29,839 | 61,991 | 546 | **24%** | 0.57 |
| `kumar2016` (MMS) | same | same | 75,189 | 13,785 | 15,404 | 46,000 | 538 | **47%** | — |
| `cote2016` (BW25113_LB) | SIV ≥ 2.5 SD below the mean (authors' rule) | 0.9 < SIV < 1.1 | 309,424 | 292 | 140,747 | 168,385 | 3,770 | **0.2%** | 0.71 |

(The cross-study AUROCs below were computed before the ID switch, on the same pairs; the mapping
change moves a few hundred pairs and does not affect any figure to two decimals.)

Babu 2011's E-score has a floor of −20 that the authors use to mean "synthetic lethal, no colony";
25,866 pairs (11% of all pairs) sit exactly on it, which is why any reasonable threshold yields a
double-digit hit rate.

## Replication evidence

**Cross-study** (labels by the rule above from A, scored by B's score on the shared pairs; primary
condition of each source; recomputed by `<source>_checks()` with 500-resample pair bootstraps):

| labels from | scored by | n | positives | AUROC | 95% CI |
|---|---|---|---|---|---|
| babu2011 | gagarinova2016 | 744 | 205 | 0.483 | 0.439–0.529 |
| babu2011 | kumar2016 | 1,474 | 422 | 0.459 | 0.426–0.489 |
| babu2011 | cote2016 | 1,618 | 421 | 0.468 | 0.441–0.499 |
| gagarinova2016 | babu2011 | 509 | 52 | 0.510 | 0.422–0.598 |
| gagarinova2016 | kumar2016 | 485 | 29 | 0.456 | 0.320–0.587 |
| kumar2016 | babu2011 | 742 | 233 | 0.512 | 0.473–0.554 |
| kumar2016 | gagarinova2016 | 400 | 90 | 0.481 | 0.402–0.557 |
| kumar2016 | cote2016 | 540 | 83 | 0.482 | 0.417–0.552 |
| cote2016 | babu2011 | 917 | 0 | undefined | — |
| cote2016 | kumar2016 | 702 | 0 | undefined | — |
| gagarinova2016 ↔ cote2016 | — | 0 | — | no overlap | — |

Every CI that exists contains 0.5 and none reaches the 0.65 bar. The two `cote2016` directions are
undefined rather than bad: Côté has only 292 positives genome-wide and none of them lands in the
917- and 702-pair overlaps — which is itself informative, since Babu and Kumar call 24–28% of
labelled pairs positive, so their positives cover these overlaps densely while Côté's do not
include a single one. Spearman correlation of the raw scores on the same overlaps: −0.03 to +0.05.

**Within-study across conditions** (same library, different growth condition — not independent
replicates, but the strongest internal check the supplements allow):

| source | labels from | scored by | n | positives | AUROC |
|---|---|---|---|---|---|
| babu2011 | RM | MM | 83,084 | 18,005 | 0.829 |
| babu2011 | MM | RM | 92,497 | 32,066 | 0.697 |
| kumar2016 | UT | MMS | 27,056 | 5,753 | 0.703 |
| kumar2016 | MMS | UT | 27,289 | 12,647 | 0.602 |
| gagarinova2016 | RM | MM | 22,315 | 907 | 0.837 |
| gagarinova2016 | RM | LT (23 °C) | 22,420 | 909 | 0.887 |
| gagarinova2016 | RM | HT (42 °C) | 22,389 | 914 | 0.906 |
| gagarinova2016 | MM | RM | 21,641 | 1,828 | 0.742 |
| cote2016 | — | — | — | — | n/a (biological duplicates were merged before publication) |

(Raw-score Spearman on the same overlaps: babu 0.31, kumar 0.34, gagarinova 0.26–0.43.)

So: high internal consistency, no cross-study agreement. That is the signature of screen-specific
artifacts (plate/position effects, marker and suppressor effects in array crosses), the same
pattern that excluded Ito 2021 from SLB.

**Literature sanity check.** Kumar's Table S3 lists 114 literature-curated *E. coli* pairs
(95 aggravating, 19 alleviating). Kumar's own scores separate them at AUROC 0.76 (circular — the
list was curated alongside the screen); Babu 2011 overlaps only 10 of them (0.63); Côté and
Gagarinova overlap 0 and 4. Not enough overlap to adjudicate, but it confirms that `ecol_map`
joins the datasets on the intended genes.

## Recommendation

**LEAD DECISION 2026-09-24: all four are measurements only, not benchmark labels**, since every
cross-study comparison fails. `babu2011_checks()`, `gagarinova2016_checks()`, `kumar2016_checks()`
and `cote2016_checks()` (wrapper: `ecoli_array_checks()`) recompute the tables below.

**Exclude all four from benchmark labels; keep as optional training measurements** (the same status
as Ito 2021 / Thompson 2021). Under the audit rule they fail on two counts: cross-study AUROC
0.46–0.51 against each other, and (for Babu 2011 and Kumar 2016) implausible hit rates.

If the lead wants *any* E. coli labels, `cote2016` is the least bad candidate: its hit rate is 0.2%,
its rule is the authors' own 2.5 SD cut-off, and its fitness-only AUROC (0.71) is in the normal
range. But it has no usable internal replicate structure (duplicates were averaged before
publication) and it is contradicted by Babu 2011 at 0.47, so on the current evidence it does not
clear the bar either. The honest summary for BENCHMARK.md is that **no published *E. coli* GI screen
passes the SLB reproducibility rule**, which is worth stating given how often eSGA/GIANT-coli data
are used to train SL models.

Another possibility, not pursued for lack of coverage: `rachwalski2024`
(Rachwalski K *et al.*, **Cell Rep Methods** 3:100693 (2023), doi:10.1016/j.crmeth.2023.100693;
`data/raw/rachwalski2024_crispri_ecoli/`, Elsevier PII S266723752300379X, all six files
hash-verified against their `mmcN` URLs)
crossed three CRISPRi knockdowns (lolA, pssA, mreD) into the whole Keio collection at three inducer
levels with an empty-vector control arm — ~11k pairs, an orthogonal mechanism (knockdown × deletion)
and a proper control arm. Three query genes is too few for a benchmark stratum, but it would make a
good independent yardstick for whichever E. coli screen someone wants to rehabilitate.

## Single-gene fitness

`ecol_single()` — 4,161 genes. `effect` = log2(OD600 after 22 h in LB ÷ median OD600) from Baba
2006 Supplementary Table 3 (3,912 measured deletion mutants; range −4.0 to +0.6, negative =
sicker), with the 303 essential-gene candidates (no viable deletion) set to the 1st percentile,
−1.21, in the same spirit as the *S. pombe* inviable/slow/viable scheme. Coverage of the labelled
pairs above: 97–100%.
