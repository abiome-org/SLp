# Non-human eukaryote GI datasets evaluated but not parsed (data-eukaryotes, 2026-09-24)

These were downloaded to `data/raw/<key>/` and inspected. None gets a parser: each fails the
inclusion rule for a clear reason, or is too small to matter. Numbers come from ad-hoc checks run
on the raw files.

| key | species | study | design | why not |
|---|---|---|---|---|
| heigwer2018_dmel | dmel | Heigwer et al. 2018 eLife 7:e40174 (MODIFI pilot; figshare 10.6084/m9.figshare.6819557, MODIFIdata_0.1.0.tar.gz sha256 9b43a0b37354dfc0af29ec69ad3fb629d7fd1bfa068ded05aeaa023e5d6f2a8b) | 168 targets x 76 queries of signalling genes, S2 cells, +/- MEK inhibitor, time course | Only 3-4 of 12,768 pairs reach limma FDR < 0.1 on cell number at 96 h, so there are no usable positives. Its per-pair pi is also ambiguous about which treatment arm is DMSO. As an arbiter of the fly contradiction it is weak and inconsistent: it scores Heigwer 2023 labels at 0.63, Fischer 2015 labels at 0.60 and Horn 2011 labels at 0.64 (DMSO-arm estimate; the other arm gives 0.36 / 0.80 / 0.67). |
| billmann2018_dmel | dmel | Billmann et al. 2018 Cell Syst 6:52 (Wnt SGI; github boutroslab/Supplemental-Material Billmann_2017) | Wnt luciferase reporter, 336 x 72 genes | The read-out is a pathway reporter, not fitness. Rluc is only a viability proxy. GI scores would have to be recomputed from the raw plate arrays. Not attempted. |
| maia2015_cele | cele | Maia et al. 2015 Sci Data 2:150020 (figshare 10.6084/m9.figshare.1314604) | genome-wide RNAi x bmk-1(ok391), progeny counts, 2 replicates | Replicates do not agree on the interaction ratio: Pearson 0.06 (12,493 clones), and one replicate's 2% tail is recovered by the other at AUROC 0.54-0.56. It also calls 3x more suppressors than enhancers. It has a single query gene. |
| lin44_2015_cele | cele | Hartin et al. 2015 PLoS One 10:e0121397 | ~3,650 RNAi clones scored categorically in WT, ptp-3, sdn-1 | Noise-dominated. For ptp-3, "normal in WT, abnormal in mutant" (518 clones) exactly equals "abnormal in WT, normal in mutant" (518). For sdn-1, apparent suppression (854) exceeds enhancement (289). There are no replicates. |
| lehner2006_cele | cele | see lehner2006_cele.md | | Parsed (as the cross-study reference for Byrne 2007), but not recommended as a label source. |
| norris2017_cele | cele | Norris et al. 2017 eLife 6:e28129 | CRISPR double mutants of 14 RNA-binding-protein genes, competition fitness | 66 pairs. 24 are significant (p < 0.05) in either direction. Too small, and cannot be verified. |
| tischler2006_cele | cele | Tischler et al. 2006 Genome Biol 7:R69 | RNAi of ~143 duplicate-gene pairs | 16 positives. Tested pairs sit in a Word table (S2.doc) and the positives in the main text. Paralog-relevant but tiny, and no replicate data. Could be hand-curated later. |
| baugh2005_cele | cele | Baugh et al. 2005 Genome Biol 6:R45 | 13 RNAi x 15 strains, % embryonic lethality with p | About 195 pairs in a Word table. Too small. |
| chow2019_mmus | mmus | Chow et al. 2019 Nat Methods 16:405 (MCAP Cpf1) | 325 pairs of 26 tumour suppressors, in vivo metastasis | The phenotype is pro-metastatic enrichment (cooperation), not lethality. |
| gier2020_mmus | mmus | see gier2020_mmus.md | | Parsed, but calls cannot be told apart from pairs with non-expressed control genes. Exclude. |
| shapiro2018_calb | calb | Shapiro et al. 2018 Nat Microbiol 3:73 (C. albicans gene drive) | 44 efflux-pump double mutants x 25 conditions; 66 adhesin pairs x 3 biofilm substrates | Untreated (NT) fitness has 1 negative GI and 25 positive GIs among 44 pairs. The adhesin read-outs are biofilm formation, not fitness. There is no usable SL signal. |
| diezmann2012_calb | calb | Diezmann et al. 2012 PLoS Genet 8:e1002562 | Hsp90 network | Hits only, no measured negatives. |
| cusack2021_athal | athal | Cusack et al. 2021 Mol Biol Evol 38:3397 (zenodo 3987384) | 161/300 Arabidopsis paralog pairs labelled redundant or not | Labels were curated from the double-mutant literature, not measured in one screen. SLB excludes literature-mined labels. |

**Searched with no systematic pairwise GI dataset found:** zebrafish, Dictyostelium, Neurospora,
Aspergillus, Tetrahymena, Chlamydomonas, Trypanosoma, Plasmodium (the 2012 "SEE" data are QTL
epistasis), Toxoplasma, Cryptococcus. Also none in non-human mammalian lines (CHO, dog) and no mouse
in-vivo SL screens: the Winslow-lab Tuba-seq pairwise knockouts measure tumour growth. Fortunato 2009
(efl-1 x chromosome III RNAi) has no obtainable supplement. Tischler 2008 (C. briggsae) is
paywalled. Jin 2025 (enAsCas12a mice) has single-gene libraries only.

## Yeast: evaluated or unobtainable

| key | study | status |
|---|---|---|
| ryan2012_scer_merged | Ryan et al. 2012 Dataset S6, S. cerevisiae merged data (4.76M pairs) | Not independent. 4.36M of its pairs are in Costanzo 2016, and it recovers Costanzo's labels at AUROC 0.93 (its own labels are recovered by Costanzo at 0.92). It is the Costanzo 2010 SGA data merged with the E-MAPs. Do not use. |
| costanzo2010_scer | Costanzo et al. 2010 Science SGA | Its screens were re-scored and included in Costanzo 2016, so it is not independent. Not parsed. |
| braberg2013_scer | Braberg et al. 2013 Cell, RNA pol II point-mutant pE-MAP | Point mutants, not loss of function. Not parsed. |
| collins2006_esp_scer | Collins et al. 2006 Genome Biol | Raw colony sizes of the Schuldiner 2005 ESP screen. Its S-scores are parsed as schuldiner2005. |
| kuzmin2020 Table S5 | paralog-paralog digenic interactions (195 pairs) | Almost all p ~ 0, so no neutral band. 10 positives. Not parsed (see kuzmin2020_scer.md). |
| dixon2008_spombe | Dixon et al. 2008 PNAS, S. pombe SGA (PDF tables ST3-ST6 + SI, from web.archive.org copies of pnas.org; ST4.pdf sha256 145dd8508ce17e2e90f640ef93870189c7fdeed26bb0b5e0ee32ec8974918a80; all URLs in dixon2008_spombe.fetch.tsv) | High-confidence hits only (SGA score cut-off). No scores for non-interacting pairs, so no measured negatives. Not parsed. |
| roguev2008 (S. pombe) | Roguev et al. 2008 Science chromosome-function E-MAP | Supplement behind Cloudflare. Its data are reported to be part of Ryan 2012's S. pombe map (already in SLB). |
| bandyopadhyay2010 (S. cerevisiae) | Bandyopadhyay et al. 2010 Science DNA-damage dE-MAP | Supplement behind Cloudflare, and the PMC copy is methods only. Guenole 2013 (same field, includes an untreated map) is parsed instead, as part of scer_emaps. |
| roguev2007 | Roguev et al. 2007 Nat Methods (pombe PEM) | Only a PDF supplement, no data table. |
| yeast CRISPR double-KO screens | none found | none |
