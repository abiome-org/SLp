# horn2011 (D. melanogaster signalling GI map, Horn et al. 2011)

**Citation.** Horn T, Sandmann T, Fischer B, Axelsson E, Huber W, Boutros M. Mapping of signaling
networks through synthetic genetic interaction analysis by RNAi. *Nat Methods* 8:341-346 (2011).
doi:10.1038/nmeth.1581. Raw data: Bioconductor experiment package RNAinteractMAPK (Fischer B.).

**Files** (`data/raw/horn2011_dmel/`; list in `horn2011_dmel.fetch.tsv`)

| file | URL | sha256 |
|---|---|---|
| nmeth1581_MOESM15.xls (Suppl. Table 3, pair-level scores) | https://static-content.springer.com/esm/art%3A10.1038%2Fnmeth.1581/MediaObjects/41592_2011_BFnmeth1581_MOESM15_ESM.xls | 4fc4497319ca2948af0b172f0e393becc919def67dc874608cbd69f76d1823db |
| Dmel2PPMAPK.rda (well-level data, fetched 2026-09-24) | https://raw.githubusercontent.com/bioc/RNAinteractMAPK/devel/data/Dmel2PPMAPK.rda | b931e11dce0aba7ae072086322ab24ce24833f9985281157048953ec20d59cd9 |

**License.** Supplement: Springer Nature ESM (free access). RNAinteractMAPK: Artistic-2.0 (Bioconductor).

**Design.** 93 signalling genes (+ controls), 2 independent dsRNAs per gene, all 192 x 192 dsRNA
combinations in both template/query orientations, 2 biological replicate screens, D-Mel2 cells
(an S2 derivative). Phenotypes: cell number, nuclear area, intensity; SLB uses cell number.
Multiplicative model; pi = log2(measured / expected) per pair; q from a t-test over replicate
dsRNA combinations (Storey q-values). Species `dmel` (FBgn via ids.dmel from FBgn/CG/symbol),
context `S2` (D-Mel2 is S2-derived; same context string as the other fly parsers so cross-study
checks can pair them), mechanism `RNAi`.

**Parser.** `slpbench.sources.eukaryotes_extra.horn2011()` (supplementary sheet `nrCells`,
control dsRNA rows dropped).

**Counts.** 4,278 pairs (all pairs of 93 genes), **179 positives, 2,139 negatives**, 1,960 ambiguous.
Positive rate 7.7% of labelled pairs (4.2% of all pairs). Signalling genes were chosen to interact,
so this is plausible but high.

**Label rule.** Positive: q (t-test) < 0.05 and pi < 0 (aggravating). Negative: q > 0.25 and
|pi| < median |pi|. Same shape as the SLB fly rules for Fischer/Heigwer.

**Replication evidence** (`horn2011_checks()`; well-level pi re-aggregated from the .rda)

| check | AUROC (2% / 5% tail) | Pearson |
|---|---|---|
| replicate screen 1 -> screen 2 | 0.93 / 0.85 | 0.80 |
| replicate screen 2 -> screen 1 | 0.92 / 0.89 | 0.80 |
| orientation (A template -> A query) | 0.86 / 0.86 | 0.82 |
| independent dsRNAs: (1,1) -> (2,2) | 0.85 / 0.76 | 0.59 |
| independent dsRNAs: (2,2) -> (1,1) | 0.68 / 0.71 | 0.59 |

Cross-study (same lab, S2-lineage cells; pairs shared with the excluded fly maps):

| labels | scored by | labelled overlap | pos | AUROC | 95% bootstrap CI |
|---|---|---|---|---|---|
| horn2011 | heigwer2023 | 123 | 36 | 0.720 | 0.61-0.82 |
| heigwer2023 | horn2011 | 165 | 13 | 0.696 | 0.47-0.89 |
| horn2011 | fischer2015 | 20 | 5 | 0.627 | 0.35-0.87 |
| fischer2015 | horn2011 | 23 | 6 | 0.304 | 0.03-0.62 |
| fischer2015 | heigwer2023 | 1,818 | 106 | 0.571 | (SLB audit) |
| heigwer2023 | fischer2015 | 1,548 | 74 | 0.410 | (SLB audit) |

Fitness-only AUROC: 0.875 using Horn's own single-dsRNA main effects (log2), 0.80 on the 32% of
labelled pairs covered by the SLB fly reference (Heigwer main effects). Calls track single-gene
sickness strongly (as do Heigwer's, 0.87); SLB's fitness balancing handles this, but it means much
of the raw signal is fitness.

**Can Horn arbitrate Fischer vs Heigwer?** Only weakly. Horn and Heigwer agree in both directions
(0.72 and 0.70) and Horn disagrees with Fischer (0.30 on 6 positives), which points to Fischer 2015
as the outlier. But the overlaps are tiny (Horn shares 54 genes with Heigwer, 29 with Fischer;
13-36 positives with Heigwer, 5-6 with Fischer) and the CIs include 0.5 for three of the four
comparisons. It is not enough evidence to re-admit Heigwer 2023 on its own. If the lead wants to act
on it, the defensible reading is "Heigwer 2023 is supported by the only third map (Horn), Fischer
2015 is not", and Heigwer's own replicate split is 0.85.

**Recommendation: INCLUDE as a small fly source (training/dev value), but do not create a scored
fly species from Horn alone.** It passes the rule within-study (0.68-0.93 depending on the split,
all >= 0.65 incl. independent dsRNAs) and cross-study vs Heigwer (0.72 / 0.70), is not contradicted
by any adequately powered comparison, and has a plausible hit rate. But it has 93 genes: after the
family split a test fold would hold roughly 15-20 genes, i.e. about 150 test pairs and fewer than 10
test positives, too few for a species score that counts 1/N of the headline. A fly species score
becomes viable only if Heigwer 2023 is re-admitted (Horn is the evidence for doing so).
