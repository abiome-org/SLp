# costanzo2021 (S. cerevisiae SGA across 14 environments, Costanzo et al. 2021)

**Citation.** Costanzo M, Hou J, Messier V, et al. Environmental robustness of the global yeast genetic
interaction network. *Science* 372:eabf8424 (2021). doi:10.1126/science.abf8424

**Files** (`data/raw/costanzo2021_scer/`; URLs in `costanzo2021_scer.fetch.tsv`)

| file | sha256 |
|---|---|
| DataFileS3_Raw_interaction_dataset.xlsx (sheets "Diagnostic array_complete" 363k rows, "Genome-scale_Benomyl" 100k rows) | 114683d4127b6c5eb89357dfd5487e452ccd0a06ebd617e490d755a25635c1b4 |
| DataFileS1_Conditions_Strains_Fitness.xlsx (conditions, query list, single-mutant fitness per condition) | f6c313de416ce8cc6ae87e2020b4389bd4adeb07cdb6a438aecaf1e45e6228ad |
License: Science supplementary material (AAAS).

**Design.** SGA, 26 query mutants x ~1,000-strain diagnostic array in 14 conditions (actinomycin D,
benomyl, bortezomib, caspofungin, concanamycin A, cycloheximide, fluconazole, galactose,
geldanamycin, MMS, monensin, rapamycin, sorbitol, tunicamycin), plus a genome-scale benomyl screen.
Each condition screen ran alongside a matched reference (standard condition) screen. Two biological
replicates each; epsilon, p per condition and reference. Species `scer`, mechanism `SGA`.
Contexts: `S288C` (matched reference, the same context as Costanzo 2016) and `S288C+<condition>`
(14 new condition contexts).

**Parser.** `eukaryotes_extra.costanzo2021()` (openpyxl, ~3 min). Reference measurements of the same
pair from different runs are merged by `finalize` (mean epsilon, min p, conflicting labels -> null).
The file's own copy of the Costanzo 2016 epsilon is not used.

**Counts.** 526,880 (context, pair) rows. Reference `S288C`: 97,490 pairs over 4,336 genes,
**610 positives / 39,980 negatives**. Conditions: 20-97k pairs each, 349-735 positives and 9-43k
negatives per condition (S288C+Benomyl 96,890 pairs, 573 pos, 43,029 neg).

**Label rule.** SLB Costanzo 2016 rule on the replicate-mean epsilon and its p-value: positive
eps < -0.2 and p < 0.05; negative p > 0.25 and |eps| < the context's median |eps|.

**Replication evidence** (`costanzo2021_checks()`)

| check | pairs | pos | AUROC |
|---|---|---|---|
| within: reference epsilon, replicate 1 (eps < -0.2) scored by replicate 2 | 463,438 | 12,353 | 0.943 |
| within: condition epsilon, replicate 1 scored by replicate 2 | 463,438 | 10,411 | 0.943 |
| cross-study: 2021 reference labels scored by Costanzo 2016 | 39,776 | 594 | **0.949** |
| cross-study: Costanzo 2016 labels scored by 2021 reference | 21,566 | 557 | **0.930** |
| condition labels scored by Costanzo 2016 (standard condition), 14 conditions | 9-43k each | 347-735 | 0.909-0.970 |

Replicate Pearson 0.66 (reference) and 0.62 (condition). The 2021 reference epsilon is an independent
screen: Pearson 0.51 with the 2016 epsilon on 95,417 shared pairs, and <0.2% identical values.
Where both label a pair, 238 of the 275 pairs called by either are called by both. Most SL pairs
persist across conditions, as the paper reports ("environmental robustness"), so the condition
contexts mostly repeat standard-condition labels. The condition-specific differences are the new
information.

**Recommendation: INCLUDE** the reference context (`S288C`, merged with Costanzo 2016/Kuzmin 2018).
**Include the 14 condition contexts as separate strata if the lead wants condition contexts.** Each
passes replication (0.94 within, 0.91-0.97 vs standard) with a plausible hit rate (2-5% of labelled
pairs). But a pair that is SL in every condition would appear 15 times, once per context, so they
inflate the effective yeast sample. The simplest option is to take the reference only. Condition
contexts are most valuable if the benchmark later scores context-specific SL.
