# Held-out human SL performance by cell-line ancestry

The [Estimated Cell Line Ancestry study by Dutil et al. (2019)](https://pubmed.ncbi.nlm.nih.gov/30894373/) estimated ancestry from genome-wide genotypes. All 47 genotype-annotated SLB cell lines cite that paper in the checksum-pinned Cellosaurus file. This audit uses those **existing** annotations to evaluate frozen SLB-1.3 test predictions; the paper supplies no SL labels.

The measure is SLB's fitness-balanced AUROC **within each cell line and screen**. CIs use 400 shared gene-family bootstrap resamples; they describe pair/family sampling uncertainty, not variation across future donors. This audit follows the frozen test readout and does not select new model weights.

## Test support

| Cell-line genetic ancestry | Cell lines | Measured test pairs | SL pairs | Assignment |
|---|---:|---:|---:|---|
| AFR | 3 | 768 | 32 | 3 genotype |
| EAS | 7 | 2,472 | 92 | 6 genotype, 1 self-reported |
| EUR | 38 | 12,681 | 442 | 38 genotype |
| unknown | 2 | 4,289 | 9 | 0 genotype |

The SLB human headline averages AFR, EAS and EUR AUROCs equally, so the 38 EUR lines cannot dominate it by row count. `unknown` is reported for coverage but excluded. One EAS cell line (PC-9) has a reported population but no genotype estimate.

## Representative frozen models

| Model | AFR (95% CI) | EAS (95% CI) | EUR (95% CI) | AFR − EUR (95% CI) |
|---|---:|---:|---:|---:|
| Dev Rank Ensemble (loss) | 0.716 (0.560–0.878) | 0.687 (0.553–0.816) | 0.649 (0.566–0.725) | +0.067 (-0.066 to +0.228) |
| SL-Predict 2026 (MAE branch) | 0.757 (0.625–0.872) | 0.749 (0.656–0.835) | 0.667 (0.562–0.743) | +0.090 (-0.036 to +0.238) |
| Ryan 2026 (full clean refit) | 0.768 (0.652–0.882) | 0.728 (0.601–0.852) | 0.641 (0.560–0.721) | +0.127 (+0.020 to +0.245) |
| Ontotype | 0.677 (0.540–0.784) | 0.603 (0.487–0.701) | 0.595 (0.510–0.669) | +0.083 (-0.036 to +0.192) |

[Figure 4](figures/04_human_ancestry.svg) plots these and two additional reference methods with the same uncertainty intervals. The point estimates do **not** show worse performance on AFR-annotated lines. For the dev-selected ensemble, AFR − EUR is +0.067, with an interval from -0.066 to +0.228. The groups contain different cell lines, cancer types, screens and pair sets, so the contrast is not an isolated effect of donor ancestry.

### African-ancestry cell-line sensitivity

| Cell line omitted | Remaining SL pairs | Dev Rank Ensemble | SL-Predict MAE | Ryan full refit |
|---|---:|---:|---:|---:|
| HeLa | 23 | 0.712 | 0.753 | 0.791 |
| NCI-H23 | 28 | 0.716 | 0.758 | 0.768 |
| RKO | 13 | 0.757 | 0.786 | 0.522 |

Ryan's AFR score falls from 0.768 to 0.522 when RKO is removed. The 3 AFR lines contribute only 32 held-out SL pairs, so even an apparently separated family-bootstrap interval cannot establish a population-level advantage or disadvantage.

### Same-screen, same-pair panel check

Only pairs measured in both an AFR-annotated and a EUR-annotated cell line under the same screen are retained. This leaves 642 AFR examples (23 SL) and 7,336 EUR examples (194 SL). It equalizes the gene-pair panel and screen name, but different cell lines and cancer types still determine the labels.

| Model | AFR matched AUROC | EUR matched AUROC |
|---|---:|---:|
| Dev Rank Ensemble (loss) | 0.712 | 0.657 |
| SL-Predict 2026 (MAE branch) | 0.753 | 0.732 |
| Ryan 2026 (full clean refit) | 0.791 | 0.711 |
| Ontotype | 0.685 | 0.587 |

This is a panel-matching diagnostic, not a donor-matched estimate of an ancestry effect.

## All ranked models

Every clean, ranked model is included below. Full precision, intervals, input hashes and leave-one-line-out values are in [`reference/slb1.3_ancestry_audit.json`](reference/slb1.3_ancestry_audit.json).

| Model | AFR | EAS | EUR | Human mean |
|---|---:|---:|---:|---:|
| SL-Predict 2026 (MAE branch) | 0.757 | 0.749 | 0.667 | 0.724 |
| Ryan 2026 (full clean refit) | 0.768 | 0.728 | 0.641 | 0.712 |
| De Kegel 2021 (all-species) | 0.717 | 0.764 | 0.617 | 0.699 |
| Dev Rank Ensemble (loss) | 0.716 | 0.687 | 0.649 | 0.684 |
| Ryan 2026 (context clean refit) | 0.754 | 0.669 | 0.623 | 0.682 |
| Dev Rank Ensemble (core) | 0.689 | 0.672 | 0.642 | 0.668 |
| GO/PPI GBM | 0.708 | 0.623 | 0.658 | 0.663 |
| GO/PPI GBM (pooled) | 0.710 | 0.614 | 0.655 | 0.660 |
| paralog_identity | 0.666 | 0.726 | 0.566 | 0.653 |
| DepMap OLS (loss) | 0.710 | 0.663 | 0.578 | 0.650 |
| MuSL (human) | 0.718 | 0.684 | 0.537 | 0.646 |
| Ontotype (pooled) | 0.690 | 0.608 | 0.620 | 0.639 |
| MuSL (all-species) | 0.711 | 0.662 | 0.534 | 0.636 |
| Ontotype | 0.677 | 0.603 | 0.595 | 0.625 |
| lgbm | 0.642 | 0.617 | 0.572 | 0.610 |
| Cilantro-SL 2026 (Geneformer branch) | 0.609 | 0.606 | 0.582 | 0.599 |
| PAGAN 2026 (genes-to-pairs) | 0.685 | 0.566 | 0.523 | 0.591 |
| codependency | 0.543 | 0.590 | 0.579 | 0.571 |
| GiGCN 2026 (binary GO adaptation) | 0.571 | 0.546 | 0.589 | 0.569 |
| SynLeaF (human) | 0.561 | 0.520 | 0.559 | 0.546 |
| SynLeaF (all-species) | 0.537 | 0.544 | 0.530 | 0.537 |
| fitness_lgbm | 0.551 | 0.498 | 0.515 | 0.521 |
| random | 0.522 | 0.461 | 0.498 | 0.494 |
| fitness | 0.514 | 0.444 | 0.511 | 0.490 |

## What the next equity test needs

A credible disparity estimate needs more independent AFR and other underrepresented-ancestry donors, with overlapping cancer lineages, perturbation libraries and gene-pair panels across groups. The next study should hold out donors, report within-line ranking and calibration or precision at a fixed experimental budget, and quantify the worst-group gap with donor-level uncertainty. It should also audit guide-target germline variants and verify hits with independent guides or assays: [Misek et al. (2024)](https://doi.org/10.1038/s41467-024-48957-z) found ancestry-associated CRISPR false negatives from guide mismatches. Cell-line genetic ancestry is not a patient's race or a clinical outcome; these SLB results are a screen-level coverage audit.
