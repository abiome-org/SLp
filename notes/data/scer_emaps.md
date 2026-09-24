# scer_emaps (nine S. cerevisiae E-MAPs as one source)

**Studies** (each has its own parser in `eukaryotes_extra`; `scer_emaps()` combines them):

| key | citation | file (data/raw/<key>_scer/) | sha256 |
|---|---|---|---|
| schuldiner2005 | Schuldiner M et al. 2005 Cell 123:507 (early secretory pathway, ESP) | schuldiner_esp_Sscores.txt (long format) | 1b544fb0f32062eb6ae6605da3a5f14fd903ea42aeaa5dc889346ff98cd53bc1 |
| collins2007 | Collins SR et al. 2007 Nature 446:806 (chromosome biology) | MOESM264_zip1.zip -> "Chromosome biology EMAP data.txt" | 93768fe7146f9b1516d043450e732f758394d9499213f55fb4b6ef45592336f0 |
| wilmes2008 | Wilmes GM et al. 2008 Mol Cell 32:735 (RNA processing) | mmc3_Sscore_matrix.xls | 2f287bea6010cc22c39dc34394bfacd483e42728860ea0fd00130cfb69968ff5 |
| fiedler2009 | Fiedler D et al. 2009 Cell 136:952 (signalling/phosphorylation) | mmc3_Sscore_matrix.txt | 98f52bb9718a682689f910844bffadbaac1a258bbefdfac0e4ef52c1a508d49a |
| zheng2010 | Zheng J et al. 2010 Mol Syst Biol 6:420 (transcription factors) | msb201077-s2_Sscore_matrix.txt | b5b7f6fbc187da17b538a4476f8c26cd449c1f6f7d1757cd69214d4c6d3f00b9 |
| aguilar2010 | Aguilar PS et al. 2010 Nat Struct Mol Biol 17:901 (plasma membrane) | MOESM19_ESM_treeview_PM_EMAP.zip (.cdt) | 3382e6c1ed8cfa65d0151ef88aba84239efde8c356d049262d58e088ff677307 |
| hoppins2011 | Hoppins S et al. 2011 J Cell Biol 195:323 (MITO-MAP) | TableS1.xls (final), TableS4.txt (individual crosses) | b12b63fb8097ec15931b7791f3e6a65c61babdcd3cb7d92089780cdfea71bee2 / f08adf1c9529bb383b024ed0f18532d8b09138d83c2206c80825eaf65b7dc4a0 |
| guenole2013 | Guenole A et al. 2013 Mol Cell 49:346 (DNA damage; untreated/DMSO map only) | mmc3_Sscores_all_pairs.xlsx | 20409db0e1c7225af76501bc6fff15e5a8c41ed8ec485b620902707273265d90 |
| surma2013 | Surma MA et al. 2013 Mol Cell 51:519 (lipid) | mmc7_Sscores.zip -> "S-Scores lipid E-MAP.xlsx" | 46848896fdbb440eb40233c2034e890ef0eb957e6b578c21ac90cd11b672ede0 |
URLs: `scer_emaps.fetch.tsv`. Licenses: journal supplementary material (Cell/Elsevier, Nature, MSB
CC BY-NC-SA, JCB CC BY-NC-SA, NSMB). Derived labels only.

**Design.** E-MAP (epistatic miniarray profile): SGA-like crosses of NAT-marked queries with
KAN-marked arrays, colony size, S-score (Collins et al. 2006). Krogan/Weissman/Walter labs. They are
independent of the Boone-lab SGA (Costanzo 2016). Species `scer`, context `S288C` (BY4741
background, the same context string as Costanzo 2016), mechanism `E-MAP`.

**Parsing.** Only loss-of-function alleles are kept (deletion, DAmP, temperature-sensitive).
Over-expression (OEX) and point-mutant strains are dropped. Names are resolved with `ids.scer`.
Averaged/final score files are used.

**Why one source.** Later E-MAPs re-used earlier maps' screens for shared genes. Pairs shared by two
E-MAPs mostly agree at AUROC 0.94-1.00 (e.g. Fiedler vs Wilmes 0.9997, Schuldiner vs Hoppins 0.995),
which means largely the same measurements (range 0.81-1.00; the lower values involve Surma 2013,
Zheng 2010 and Guenole 2013, whose screens are partly new). Counted as separate sources, they would look like
independent agreeing studies. `scer_emaps()` concatenates them and merges repeats (mean S;
conflicting labels -> null).

**Do not use** `data/raw/ryan2012_scer_merged` (Ryan 2012 Dataset S6, "Cerevisiae merged data").
It is 4.76M pairs, 4.36M of them shared with Costanzo 2016, and its labels are recovered by
Costanzo 2016 at AUROC 0.92-0.93. That is the Costanzo 2010 SGA data re-expressed as S-scores, not an
independent measurement. Also not parsed: braberg2013 (histone point mutants, not loss of
function), collins2006_esp (raw colony sizes of the ESP screen, already covered by schuldiner2005's
S-scores), costanzo2010 (its screens are re-scored inside Costanzo 2016).

**Label rule.** SLB E-MAP rule (same as Ryan 2012): positive S < -3; negative |S| < 1.

**Counts and cross-study replication vs Costanzo 2016** (SGA, independent lab and method)

| key | pairs | genes | pos | neg | E-MAP labels scored by Costanzo eps: labelled / pos | AUROC | Costanzo labels scored by E-MAP S: pos | AUROC | fitness-only |
|---|---|---|---|---|---|---|---|---|---|
| schuldiner2005 | 83,118 | 424 | 2,169 | 64,122 | 55,969 / 1,790 | 0.872 | 965 | 0.839 | 0.755 |
| collins2007 | 179,028 | 734 | 8,894 | 123,456 | 111,202 / 7,143 | 0.788 | 2,139 | 0.873 | 0.755 |
| wilmes2008 | 106,484 | 550 | 2,884 | 81,099 | 66,604 / 2,206 | 0.751 | 900 | 0.751 | 0.710 |
| fiedler2009 | 101,624 | 483 | 2,000 | 83,594 | 75,105 / 1,629 | 0.799 | 519 | 0.816 | 0.794 |
| zheng2010 | 47,814 | 321 | 2,638 | 33,741 | 31,234 / 2,210 | 0.709 | 280 | 0.865 | 0.814 |
| aguilar2010 | 57,349 | 372 | 1,046 | 49,406 | 44,310 / 907 | 0.771 | 331 | 0.781 | 0.784 |
| hoppins2011 | 568,393 | 1,483 | 11,522 | 501,598 | 421,513 / 8,735 | 0.781 | 2,811 | 0.801 | 0.821 |
| guenole2013 | 97,203 | 2,020 | 4,346 | 68,010 | 59,489 / 3,719 | 0.759 | 864 | 0.743 | 0.722 |
| surma2013 | 250,792 | 741 | 4,871 | 221,714 | 193,638 / 4,374 | 0.813 | 1,644 | 0.779 | 0.756 |
| **scer_emaps (combined)** | **1,320,083** | **3,749** | **26,737** | **1,082,846** | 920,159 / 20,630 | **0.769** | 7,863 | **0.786** | 0.795 |

The combined source adds 6,107 positives and 183,317 negatives on pairs not in Costanzo 2016. Where
both label a pair (274,028 pairs), 3,320 of the 8,281 pairs called by either are called by both
(40%; the human cross-study figure in BENCHMARK.md is 43%). Guenole 2013 is only partly shared with
the older maps (its labels are recovered by Collins 2007 / Hoppins 2011 at 0.84).

**Within-study** (`hoppins2011_checks()`, Table S4 = every individual cross: replicate crosses and
both orientations): 106,727 pairs measured at least twice. Labels from one cross, scored by another:
AUROC 0.909 (S < -2.5) and **0.926** (S < -3). Pearson 0.72.

**Recommendation: INCLUDE `scer_emaps` as one source** (not the nine separately). Every map passes
cross-study in both directions (0.71-0.87). The hit rate is plausible (2.4% of labelled pairs), and
fitness confounding is similar to Costanzo 2016 (0.80 vs 0.73). It is also independent cross-study evidence for Costanzo 2016's own labels (0.79 by E-MAP scores,
vs 0.74 from Costanzo's orientation check).
