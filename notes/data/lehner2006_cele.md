# lehner2006 (C. elegans signalling GI screen, Lehner et al. 2006)

**Citation.** Lehner B, Crombie C, Tischler J, Fortunato A, Fraser AG. Systematic mapping of genetic
interactions in *Caenorhabditis elegans* identifies common modifiers of diverse signaling pathways.
*Nat Genet* 38:896-903 (2006). doi:10.1038/ng1844

**Files** (`data/raw/lehner2006_cele/`; `lehner2006_cele.fetch.tsv`)

| file | sha256 |
|---|---|
| MOESM1.xls (screened RNAi library: 1,860 clones, 1,744 genes, WBGene IDs) | 664fd974ae48e0be8ed2b0fa31f438c261482c465070d9bd340d563a6bfe0b5a |
| MOESM2.xls (377 synthetic-enhancement hits, 21 query strains) | 98bc770c3646fb2fd7dccc86cc7174c19aa25327604438cacd1961c4094fec9a |
MOESM3-7 are PDFs (not used). License: Springer Nature ESM (free access).

**Design.** Feeding RNAi of 1,744 signalling/chromatin/TF genes in ~35 query mutant strains
(~65,000 pairs), scored by eye for enhanced phenotypes (Emb/Ste/Gro/Lvl/Bmd/Rup). Binary calls,
no scores. The supplement names only the 21 query strains with >= 1 hit; queries without hits are
listed only in the paywalled main-text table and are not reconstructed here.

**Parser.** `eukaryotes_extra.lehner2006()`: the 21 queries x library; score -1 (hit) / 0.
Species `cele`, context `N2`, mechanism `RNAi x mutant`.

**Counts.** 34,740 pairs; **333 positives, 34,402 negatives** (hit rate 1.0%).

**Label rule.** Positive: listed hit. Negative: query x library pair not listed. This is "tested and
not called" without any neutral band: RNAi clones that are already lethal/sterile in wild type
cannot show enhancement and end up as negatives, so the negatives are not reliable.

**Replication evidence.** Lehner hits are recovered by Byrne 2007's quantitative strengths at AUROC
0.703 (1,127 shared labelled pairs, 26 positives). No replicate data are published.

**Recommendation: EXCLUDE as a label source** (negatives are inferred, not measured with a score;
no replicate data). Useful as the cross-study reference for Byrne 2007. Keep the measurements only
as optional training data.
