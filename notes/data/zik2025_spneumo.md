# dualtnseq2025 — *S. pneumoniae* D39 Dual Tn-seq (Zik et al. 2025)

Source key: `dualtnseq2025` · raw dir: `data/raw/zik2025_spneumo/` · parser:
`slpbench.sources.bacteria_extra.dualtnseq2025()` · species code `spne`

## Citation

Zik JJ, Price MN, Arkin AP, Deutschbauer AM, Sham L-T.
**Dual transposon sequencing profiles the genetic interaction landscape in bacteria.**
*Science* (2025), doi:10.1126/science.adt7685. Preprint: bioRxiv 2024.09.24.614635.

Data: figshare doi:10.6084/m9.figshare.29382974 (**CC BY 4.0**) and Dryad
doi:10.5061/dryad.7d7wm3840. Both are cited in the paper as the source of the processed data.

## Files

Downloaded to `data/raw/zik2025_spneumo/` (URLs in `zik2025_spneumo.fetch.tsv`):

| file | bytes | sha256 |
|---|---|---|
| small.zip (figshare; processed tables + code + annotation) | 101,999,168 | 51db1f90312b71d3d250f1437acb3f6d84b465135eec6817a45a307ee37ec47a |
| feba.tar.gz (figshare; RB-TnSeq code) | 4,708,771 | f51da55f514529d312e6965c95bf398a804c866707613400a32943ff37c587e9 |
| 1vsall.image (figshare; one-vs-all RB-TnSeq, R image) | 40,119,098 | 38ebd37ce877f92cd5a0fe6aa95b59773dd371c8b0200e734a76926c6043b991 |
| README_dryad.md | 6,205 | f48c536fdebad60a3f1daf0633a1a7332e6aded91a3f80763e59d9b2e9827fa2 |
| tableS1.tsv.gz (Dryad; both 10-90% and 0-100% scorings) | 67,302,255 | ff0bcb5c35b5cc5fe9b7020f31a4a18d2cdfd52fc02c692ff25a340453ad78dc |
| run1_genepairs_min6.tsv.gz | 29,595,923 | 83b68bcacc3655b8202e1d946f93cf6c00e185ab97e7f9269d786ffea09c6103 |
| run2_genepairs_min3.tsv.gz | 32,175,920 | de978a8b473534833718fd8ffe1b19e98def4baad96240286113abae1cd1dc9d |
| run3_genepairs_min3.tsv.gz | 32,976,281 | 1e94a0236baa6ca42b7c63d47d4c3da86cf58d55d77af8fddac225eb3d90d065 |
| run4_genepairs_min4.tsv.gz | 32,083,665 | dc4b65e066586513cf01b31ad1cdfd1abcec237872e043a93b5e42daec6e8ffa |
| run5_genepairs_min4.tsv.gz | 32,454,663 | 4ce48783ac0d17712f01edbba347e5fb14b8fc2f9fb679aeca46683cf45227e0 |

The figshare md5s match the API values (e95c874e…, a5240307…, c5c62bb7…), and the Dryad README's
sha256 matches the Dryad API digest. The parser reads `small/genepair_stats.tsv.gz`,
`small/genes.tab` and `small/esstable` (extract with `unzip small.zip -d small`).

Not downloaded: `run*_filtered.tsv.gz`, the per-barcode-pair tables, 21 GB in total; they are only
needed to recompute the gene-pair counts from scratch. `table_S1_dual_tnseq_dataset_full.xlsx`
(170 MB) is skipped because `tableS1.tsv.gz` has the same content.

**Fetch note.** Dryad is behind an Anubis proof-of-work gate, so plain `curl` gets an HTML
challenge page and the JSON API path returns 401. The challenge is a sha256 partial pre-image
(`sha256(randomData + nonce)` with `difficulty` leading zero hex digits) redeemed at
`/.within.website/x/cmd/anubis/api/pass-challenge`; a ~30-line Python solver fetches the files. All
files except the per-run tables and tableS1 are also on figshare, which needs no gate.

## Design

Two RB-TnSeq libraries carrying different markers and *lox* sites (RBloxSpec, RBloxErm; 15.2 M and
8.8 M barcodes) were combined by transformation into *S. pneumoniae* D39 and selected on rich
media; Cre was induced to bring the two barcodes into proximity, and barcode pairs were sequenced.
About 1.4 billion double mutants were sampled across 5 big Dual Tn-seq runs (ML1×ML2, ML3×ML2
library combinations). Per gene pair the readout is the number of distinct double-mutant strains
(`nStrains`) and their reads, compared with the expectation from each gene's marginal strain count
after adjusting for chromosomal-position bias (30×30 position bins, median ratio per bin):

- `zStrains` = (nStrains − expectStrainsAdj) / sqrt(expectStrainsAdj) — negative = double mutants
  are missing = synthetic sick/lethal.
- `readRatio` = nReads / expectReadsAdj.

Authors' calls: medium-confidence GI = `zStrains ≤ −3 and readRatio ≤ 0.2`; strong = medium plus
(`zStrains ≤ −4` or `readRatio ≤ 0.05`). Coverage: 894,694 unordered gene pairs over 1,504 genes
(insertions in the central 10-90% of each gene). Only non-essential genes are assayable, so this
screen and `crisprtnseq2024` (essential × non-essential) have **zero** overlapping pairs.

Gene IDs: D39 `SPD_` locus tags → benchmark D39V IDs via `bacteria_extra.spne_map` (canonical IDs from `ids_extra.spne`, lead decision 2026-09-24). 113 of the
1,504 SPD genes have no D39V counterpart in the CP027540.1 cross-references (unnamed short ORFs
dropped in the D39V re-annotation); a coordinate-based rescue recovered only 6 of them
unambiguously, so all 113 are dropped instead.

## Parser and label rule

`dualtnseq2025()`:

- **Chromosomal-proximity filter:** gene pairs whose midpoints are < 6 kb apart are dropped
  entirely. Cre recombination between two nearby *lox* sites excises the intervening chromosome
  segment, so nearby double mutants are lost for physical reasons. This is the authors' own
  `notNearby(minDist = 6000)` threshold. It matters a lot: 836 medium-confidence calls become 317,
  i.e. **62% of the published calls are between genes < 6 kb apart** and are not usable as labels.
- **Positive:** `zStrains ≤ −3 and readRatio ≤ 0.2` (authors' medium confidence).
- **Negative:** `|zStrains| < 1` and `0.8 < readRatio < 1.25` and `expectStrainsAdj ≥ 10`, so the
  pair had the coverage to detect a depletion. The expectation floor is the analogue of the other
  parsers' "clearly non-significant" requirement; the source publishes no p-value, so `signif` is
  null.
- **Ambiguous (null):** everything else, including positive/alleviating deviations.

Counts after `finalize` (ID mapping, dedup):

| rows | positives | negatives | ambiguous |
|---|---|---|---|
| 764,838 | 292 | 326,696 | 437,850 |

Hit rate among labelled pairs 292/326,988 = **0.09%** — the same order as Costanzo SGA and much
lower than any human screen, as expected for an unbiased genome-wide double-knockout screen.

## Replication evidence

**Within-study, across the 5 independent Dual Tn-seq runs.** Each run's gene-pair counts were
re-scored from scratch with the authors' recipe reimplemented in Python (marginal-product
expectation, then the 30×30 chromosomal-bin median adjustment; `expectStrainsAdj ≥ 5`), giving 5
independent scorings of 0.37–0.64 M pairs. Labels defined from **one** run (same SLB rule) and
scored by the mean `zStrains` of the **other four**, with the 6 kb filter applied:

| labels from | pairs | positives | AUROC (other 4 runs) |
|---|---|---|---|
| run1 | 373,065 | 20 | 0.794 |
| run2 | 373,065 | 62 | 0.923 |
| run3 | 373,065 | 73 | 0.959 |
| run4 | 373,065 | 141 | 0.974 |
| run5 | 373,065 | 99 | 0.973 |

The authors' pooled labels are recovered by each single run at 0.97–0.99 (optimistic: the labels
were derived from all runs including that one). Run-to-run Spearman over *all* pairs is only
0.08–0.15, which is expected when 99.9% of pairs are noise around zero; the signal is in the tail.

**Within-study, insertion-subset variant.** The 10-90% labels are recovered by the independent
0-100% (all-insertion) scoring at AUROC 0.999, Spearman 0.81 over all 889k pairs. Weak evidence
(the two scorings share most insertions) but consistent.

**Cross-study vs `dualcrispri2025` (dual CRISPRi-seq, same strain, the source currently supplying
SLB's spne labels).** 39,065 pairs are measured by both:

| direction | n labelled | positives | AUROC | 95% CI (pair bootstrap) |
|---|---|---|---|---|
| Dual Tn-seq labels recovered by dual CRISPRi εSum | 16,316 | 26 | 0.607 | 0.489–0.718 |
| dual CRISPRi labels recovered by Dual Tn-seq zStrains | 27,749 | 69 | 0.511 | 0.429–0.589 |

Both directions, and the same comparisons against `crisprtnseq2024`, are recomputed by
`spne_cross_checks()`; the within-study and one-vs-all rows above by `dualtnseq2025_checks()`.

Spearman between the two interaction scores on shared pairs: 0.002. Restricting to stronger Dual
Tn-seq positives does not help (z ≤ −4: 0.517; z ≤ −6: 0.564 with 22 positives).

Again this is not an ID or annotation artifact: the two studies' **single-gene** measures
(Dual Tn-seq insertion tolerance vs dual CRISPRi single-sgRNA log2FC) correlate at Spearman
**0.69** over the 477 shared genes.

**Fitness-only AUROC** (how much of the label is explained by the two genes' single-loss effects,
using `spne_single()` below): **0.708**. Comparable to Costanzo 2016 (0.73) and Dede 2020 (0.71),
well below dual CRISPRi-seq's 0.85.

**Independent assay: one-versus-all RB-TnSeq** (`1vsall.image`, readable with
`uv run --with pyreadr`). Here the ML2/ML3 transposon library was transferred into individual
deletion backgrounds and gene fitness measured the ordinary RB-TnSeq way — a different experiment
from the barcode-pair counting, with different strains and analysis. Its significant-hit tables
(`ML2hits`, `ML3hits`: 798 hits, 451 with `fitnorm < 0`, i.e. the knockout is sicker in that
deletion background) give 438 mapped aggravating pairs.

Restricted to the 72,437 SLB-labelled Dual Tn-seq pairs that involve one of the 1-vs-all background
genes:

| statistic | value |
|---|---|
| Dual Tn-seq positives that are also 1-vs-all aggravating hits | 17 / 192 = **8.9%** |
| Dual Tn-seq negatives that are also 1-vs-all aggravating hits | 41 / 72,245 = **0.057%** |
| enrichment | **156-fold** |
| AUROC of `zStrains` recovering the 1-vs-all aggravating hits | **0.706** |
| AUROC of the SLB label recovering them | 0.645 |

This is the direction the audit cares about read backwards (the independent assay publishes only
its hits, so it cannot score every pair and a label-recovery AUROC in the usual direction is not
computable), but a 156-fold enrichment of an independently measured aggravating effect among the
positives is strong support for the labels. Only `ML2hits`/`ML3hits` are readable: the full
per-background fitness tables (`ML2fit`, `ML3fit`) are R lists of data frames, which `pyreadr`
cannot convert.

## Recommendation

**LEAD DECISION 2026-09-24: measurements only, not benchmark labels** — together with the other two
pneumococcal screens, because they contradict each other (see `spne_cross_checks()`). My own
recommendation had been to include it as its own stratum; the reasons are kept below because they
bear on how much weight the measurements deserve as training data, and because they are the argument
to revisit if a fourth pneumococcal screen ever adjudicates the disagreement.

My recommendation was: **include, as its own screen (stratum), with the 6 kb proximity filter and
the label rule above.** Reasons:

1. Its own five independent runs recover its labels at 0.79–0.97, above the benchmark's 0.65 bar,
   with the leave-one-run-out design the audit asks for, and an independent assay (one-vs-all
   RB-TnSeq in deletion backgrounds) is enriched 156-fold among its positives (AUROC 0.71).
2. Hit rate 0.09% is plausible; fitness-only AUROC 0.71 is in line with included sources.
3. It adds ~327k measured labels and 1,348 genes in a species that currently has one screen and is
   the benchmark's noisiest, and it is a different mechanism (double knockout, not knockdown).

**Caveat that the lead must weigh:** cross-study concordance with `dualcrispri2025` is at chance
(0.51–0.61, CIs spanning 0.5). Rule 1 prefers cross-study evidence where a same-context study
exists, and by that reading this source (and `dualcrispri2025`) both fail. Two things argue that
the overlap test is weak rather than decisive here:

- Power: 26 and 69 overlapping positives; the 95% CIs span 0.5, so a real 0.7 concordance could not
  be distinguished from chance.
- Mechanism mismatch: Dual Tn-seq only covers genes where transposon insertion is viable, while
  dual CRISPRi-seq's informative pairs concentrate on essential genes and operon-level knockdowns,
  and knockdown ≠ knockout. The two screens' informative regimes barely intersect: the overlap is
  1.8% of dual CRISPRi's labelled pairs and 5% of Dual Tn-seq's.

If the lead wants one uniform decision for *S. pneumoniae*, the honest summary is: **three
independent pneumococcal screens each replicate themselves strongly (0.79–0.99) and none of them
reproduces another's interaction calls** (dual CRISPRi vs Dual Tn-seq 0.51/0.61; dual CRISPRi vs
CRISPRi-TnSeq 0.50/0.66; Dual Tn-seq vs CRISPRi-TnSeq: no overlapping pairs at all). That is worth
stating in BENCHMARK.md's limitations regardless of which sources are kept, and it argues for
keeping the three as separate strata rather than merging their labels.

## Single-gene fitness

- `spne_single_dualtnseq()` — per gene, log2 of the insertion-tolerance ratio `normreads` from
  `small/esstable` (reads per nt ÷ genome median), clipped at −4. Negative = sicker. 1,743 genes,
  covering 1,348 of the 1,377 genes in this screen (the benchmark's current
  `fitness.spne_single()` covers only 318 of them).
- `spne_single()` in `bacteria_extra` — merged table for the species: dual CRISPRi-seq log2FC where
  available, else the Tn-seq effect rescaled to the log2FC median/IQR on the 318 shared genes.
  1,796 genes, 98% coverage of this source's labelled pairs, fitness-only AUROC 0.708. **This is
  the table to point `fitness.gene_effects()` at if either new spne source is added**, otherwise
  95% of Dual Tn-seq pairs would have no single-gene covariate and could not be fitness-balanced.
