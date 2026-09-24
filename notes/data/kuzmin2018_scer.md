# kuzmin2018 (S. cerevisiae digenic SGA screens from the trigenic study, Kuzmin et al. 2018)

**Citation.** Kuzmin E, VanderSluis B, Wang W, et al. Systematic analysis of complex genetic
interactions. *Science* 360:eaao1729 (2018). doi:10.1126/science.aao1729

**Files** (`data/raw/kuzmin2018_scer/`; URLs in `kuzmin2018_scer.fetch.tsv`)

| file | sha256 |
|---|---|
| DataFileS1_Raw_genetic_interaction_dataset.tsv (used) | c82b60f98590071fcb07b74b672958a5863b58e03b78631e6235cb664ee82afb |
| DataFileS2_Digenic_and_adjusted_trigenic.txt (UTF-16; not used) | 3c6a6f2dcea490e013dfdd2d7f3e8bd5290d4ed694f4afdb81ac2fe82741efd3 |
| DataFileS3_Query_strains.xlsx | bf81bfb62f56847878171ec13cebcde513c7544521345dd048724eecac1d3907 |
| DataFileS5_Diagnostic_array_strains.xlsx | 03a18539f5c13bf6fec41a79ce3355ee4b818d814bf6dda29e59edd891863531 |
License: Science supplementary material (AAAS). Redistributing derived labels only.

**Design.** SGA. To measure trigenic interactions, every single-mutant query was also screened as a
"geneX + hoΔ" control strain (HO/YDL227C is a neutral locus) against the ~1,200-strain diagnostic
array. These "digenic" rows are new, independent measurements of pairs that Costanzo 2016 also
screened (same lab, same scoring, different screens; Pearson of epsilon on 382,756 shared pairs 0.42).
Trigenic rows are ignored. Species `scer`, context `S288C`, mechanism `SGA`.

**Parser.** `eukaryotes_extra.kuzmin2018()`: rows with type `digenic`; query ORF = the non-HO gene of
the query strain; strain IDs collapsed to ORFs as in `yeast.costanzo2016`.

**Counts.** 395,365 pairs, 364 query strains x ~1,200 arrays; **4,077 positives, 193,345 negatives**,
197,943 ambiguous. Only 12,609 pairs are absent from Costanzo 2016.

**Label rule.** The SLB Costanzo 2016 rule: positive eps < -0.2 and p < 0.05; negative p > 0.25 and
|eps| < median |eps|.

**Replication evidence** (`kuzmin2018_checks()`)

| check | labelled overlap | pos | AUROC |
|---|---|---|---|
| Kuzmin labels scored by Costanzo 2016 epsilon | 191,349 | 4,007 | **0.893** |
| Costanzo 2016 labels scored by Kuzmin epsilon | 96,072 | 1,780 | **0.877** |
Where both label a pair (52,301): 1,012 of the 1,351 pairs called by either are called by both (75%).
Fitness-only AUROC (Costanzo single-mutant fitness): 0.81 (Costanzo 2016: 0.73).

**Recommendation: INCLUDE.** It passes cross-study at 0.88-0.89 with a plausible hit rate (2.1% of
labelled pairs). It adds few new pairs, but it is the first cross-study replication of Costanzo 2016's
labels (0.88, stronger than the 0.74 orientation check now in REPLICATION.md). When merged, agreeing
labels reinforce and conflicts are dropped.
