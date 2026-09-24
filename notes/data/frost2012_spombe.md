# frost2012 (S. pombe E-MAP, Frost et al. 2012)

**Citation.** Frost A, Elgort MG, Brandman O, et al. Functional repurposing revealed by comparing
S. pombe and S. cerevisiae genetic interactions. *Cell* 149:1339-1352 (2012). doi:10.1016/j.cell.2012.04.028

**Files** (`data/raw/frost2012_spombe/`, fetched 2026-09-22 by the SLB fetch step; list in `frost2012_spombe.fetch.tsv`)

| file | URL | sha256 |
|---|---|---|
| mmc2_averaged.zip | https://ars.els-cdn.com/content/image/1-s2.0-S0092867412005739-mmc2.zip | a46290ca8527532285b2bd3bb6a1c0e8c91af11a2c59e6c1070656d46fdd2b44 |
| mmc3_unaveraged.zip | https://ars.els-cdn.com/content/image/1-s2.0-S0092867412005739-mmc3.zip | 479e4679f08b68027475738cdd65cfd0e42e5ceeeb6007e1d89bb702620365fb |

Each zip holds a gzipped tar with a Java TreeView clustered matrix (`.cdt`, CR/LF line endings):
columns = 597 query strains, rows = 1,298 array strains, cell = E-MAP S-score, blank = not measured.
Strain labels embed the PomBase systematic ID (`(cdc12BNR1,BNI1)_SPAC1F5.04C__cdc12__...`), plus
`_DAMP` / `_DEGRON-DAMP` hypomorph suffixes for essential genes. The "averaged" file merges the two
orientations of a pair where a gene is both query and array; the "unaveraged" file keeps them.

**License.** Elsevier supplementary material (Cell); free to download, redistribution terms not stated.
SLB redistributes derived labels only, as for Ryan 2012.

**Design.** Pairwise E-MAP (PEM/SGA-like mating, colony size), 972h- background (the Krogan/Roguev
lab system used by Ryan 2012), deletion alleles and DAmP/degron hypomorphs; focus on membrane
trafficking, cytoskeleton, signalling, lipid metabolism. Species `spom`, context `972h-` (same as
Ryan 2012, so the two merge and are cross-checked in the same stratum), mechanism `E-MAP`.

**Parser.** `slpbench.sources.eukaryotes_extra.frost2012()` (averaged file; alleles collapsed to the
gene, duplicates merged by `finalize`).

**Counts** (after ID resolution and merging): 588,043 gene pairs over 1,452 genes;
**9,664 positives, 462,880 negatives**, 115,499 ambiguous. Positive rate 2.0% of labelled pairs.
185,653 pairs are also in Ryan 2012 (1,047 shared genes); 402,390 pairs are new
(6,764 pos / 316,274 neg).

**Label rule.** Positive S < -4; negative |S| < 1; otherwise ambiguous.
Rationale: Ryan 2012 (same lab, same scoring) uses S < -3. Frost S-scores are wider (sd 1.23 vs
1.04 on shared pairs; 3.7% of all Frost pairs are < -3, versus 1.3% in Ryan), and Frost's S < -3
calls are recovered by Ryan's independent measurements at only AUROC 0.645 (below the 0.65 bar).
At S < -4 the cross-study AUROC is 0.670 and the positive rate matches Ryan's. This is the same
"stricter cut replicates better" adjustment SLB already makes for both yeasts; the cut was chosen
after looking at the cross-study numbers below, so it is data-informed (reported openly).

**Replication evidence** (`frost2012_checks()`)

| check | pairs | pos | AUROC |
|---|---|---|---|
| within-study: unaveraged file, independent orientation / allele, S < -2.5 | 80,219 | 3,745 | 0.765 |
| same, S < -3 | 80,219 | 2,693 | 0.781 |
| same, S < -4 | 80,219 | 1,503 | 0.802 |
| cross-study: Frost labels (S < -3) scored by Ryan 2012 S | 154,671 | 5,518 | 0.644 |
| **cross-study: Frost labels (S < -4, SLB rule) scored by Ryan 2012 S** | 152,131 | 2,978 | **0.670** |
| cross-study: Frost labels (S < -5) scored by Ryan 2012 S | 150,780 | 1,627 | 0.691 |
| cross-study: Ryan 2012 labels (SLB rule) scored by Frost S | 156,523 | 3,153 | **0.793** |

Pearson r of S-scores on the 185,653 shared pairs is 0.24 (independent measurements; no shared
raw data). Orientation/allele Pearson within Frost: 0.45.
Fitness-only AUROC (PomBase deletion viability, -(f_a+f_b)): 0.513 (Ryan 2012: 0.593), so the
calls do not track single-gene sickness.
Label concordance where both studies label a pair (126,308 pairs): 3,142 conflicts (dropped at
merge); of 3,712 pairs called SL by either, 570 (15%) are called by both.

**Recommendation: INCLUDE** (with S < -4). Passes the rule in both directions of the cross-study
test (0.67 and 0.79) and within-study (0.80); plausible hit rate (2.0%); not fitness-driven. It adds
~400k new labelled S. pombe pairs and, for the first time, a cross-study replication for Ryan 2012
(Ryan labels recovered by Frost at 0.79). Caveat: the positive-positive concordance between the two
maps is low (15%), typical of colony-size GI maps; merge conflicts are dropped by the build.
