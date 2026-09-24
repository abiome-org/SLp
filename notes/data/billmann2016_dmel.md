# billmann2016 (D. melanogaster cell-cycle GI map, Billmann et al. 2016)

**Citation.** Billmann M, Horn T, Fischer B, Sandmann T, Huber W, Boutros M. A genetic interaction map
of cell cycle regulators. *Mol Biol Cell* 27:1397-1407 (2016). doi:10.1091/mbc.E15-07-0467. PMC4831891.

**Files** (`data/raw/billmann2016_dmel/`, from the Europe PMC supplementary zip; `billmann2016_dmel.fetch.tsv`)

| file | sha256 |
|---|---|
| s07.xlsx (Table S6: median pi, 350 candidates x 14 queries; cellCount / mitoticIndex / nucArea) | 5225dd67022e59763a94abee6672eafa5ad863fd41b85b87530ef7a9fb24308f |
| s05.xlsx (candidate annotation, FBgn/CG/symbol) | 07a9f29aebfa852fc7c1feb8b4d695b1e8298bf451ceaabb5984d197bd9286aa |
| s06.xlsx (query annotation) | 042f666111a8e3cd4c5f6466cdc38a90ddf2f3b6af53e8ef572bd9126b8e36fa |
Also downloaded: s02-s04 (single-knockdown z-scores), s08 (correlation matrices), combined PDF.
License: MBoC (ASCB) open access, CC BY-NC-SA 3.0.

**Design.** Combinatorial RNAi in S2 cells, imaging, multiplicative model; median pi per pair. No
p-values or replicate-level values are published.

**Parser.** `eukaryotes_extra.billmann2016()`: cell-count sheet; positive pi < -3, negative |pi| < 1.
Species `dmel`, context `S2`, mechanism `RNAi`.

**Counts.** 4,890 pairs; **96 positives, 3,139 negatives**, 1,655 ambiguous.

**Replication evidence** (`billmann2016_checks()`). No replicates available. Overlap with the other
fly maps is tiny: 137 labelled pairs shared with Fischer 2015 / Heigwer 2023 with 1-2 Billmann
positives (no AUROC computable); Heigwer 2023 labels scored by Billmann: 0.65 (90 pairs, 5 positives);
Fischer 2015 labels: 3 positives, not computable; no overlap with Horn 2011.
Fitness-only AUROC 0.91 on the 33% of labelled pairs covered by the fly reference effect.

**Recommendation: EXCLUDE (unverifiable).** No replicate data and no powered cross-study overlap,
the same situation as Han 2017 / Wong 2016. Keep as optional training measurements.
