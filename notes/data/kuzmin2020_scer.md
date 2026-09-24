# kuzmin2020 (S. cerevisiae digenic SGA screens from the paralog trigenic study, Kuzmin et al. 2020)

**Citation.** Kuzmin E, VanderSluis B, Nguyen Ba AN, et al. Exploring whole-genome duplicate gene
retention with complex genetic interaction analysis. *Science* 368:eaaz5667 (2020).
doi:10.1126/science.aaz5667. Data: Dryad doi:10.5061/dryad.g79cnp5m9 (CC0).

**Files** (`data/raw/kuzmin2020_scer/`, fetched 2026-09-24 through Dryad's Anubis gate with
`scripts/dryad_anubis_fetch.py`; `kuzmin2020_scer.fetch.tsv`)

| file | sha256 |
|---|---|
| TableS1.tsv (raw digenic + trigenic, main screens; first line is a title) | 2213866487547648fe3546fac033d6df9ac5e353bd169dfd5b6f527e1951b087 |
| TableS3.tsv (same format, pilot screens) | f90d3a1683797b4bf8d514029919b36f7757e0bcfd7a4fc195fc8e999df3951e |
| TableS2.tsv (thresholded calls; not used) | 619305e51259c0f0258ad232e3f28b107f5a6ce48c70726d1ad7ad3bf75f598b |
| TableS4.xlsx (query strains) | f9927efe5eca0ad368257910c95e35ff55eaa8e00688686162807685d5dca069 |
| TableS5.xlsx (fitness standards; paralog-paralog "query interactions") | 632405396de8e78807c6e94023780344ff53b2bfe4579a2a7609a4309693c6c0 |
Tables S6-S13 kept arriving truncated through the gate (4 retries each) and are not needed.

**Design.** Same as Kuzmin 2018: each paralog of ~240 whole-genome-duplicate pairs was screened as
a "paralog + hoΔ" query against the diagnostic array. Digenic rows are independent SGA
re-measurements. Species `scer`, context `S288C`, mechanism `SGA`.

**Parser.** `eukaryotes_extra.kuzmin2020()` (shared helper `_kuzmin_digenic`). Label rule is SLB's
Costanzo 2016 rule (eps < -0.2 and p < 0.05 / p > 0.25 and |eps| < median).

**Counts.** 595,001 pairs; **1,739 positives, 290,062 negatives**, 303,200 ambiguous. 39,330 pairs
are not in Costanzo 2016.

**Replication evidence** (`kuzmin2020_checks()`)

| check | labelled overlap | pos | AUROC |
|---|---|---|---|
| Kuzmin 2020 labels scored by Costanzo 2016 | 272,256 | 1,675 | **0.798** |
| Costanzo 2016 labels scored by Kuzmin 2020 | 168,037 | 1,037 | **0.738** |
| Kuzmin 2020 labels scored by Kuzmin 2018 | 7,276 | 79 | 0.887 |
| Kuzmin 2018 labels scored by Kuzmin 2020 | 8,738 | 88 | 0.824 |
Pearson of epsilon with Costanzo 2016 is 0.19 on shared pairs (0.47 with Kuzmin 2018), so these are
independent screens. Fitness-only AUROC 0.77.

**The paralog-pair interactions themselves** (Table S5, "Query interactions": 195 scored paralog
pairs, epsilon + p) are **not** parsed. Almost every p-value is ~0 (179/195 < 0.05, 8 > 0.25), so no
neutral band can be defined. With the SLB cut-off there would be 10 positives, and 176 of the 195
pairs are already in Costanzo 2016.

**Recommendation: INCLUDE.** It passes cross-study in both directions (0.74-0.80), with a plausible
but low hit rate (0.6% of labelled pairs). It adds 39k new pairs and a third independent SGA
measurement of Costanzo 2016 pairs.
