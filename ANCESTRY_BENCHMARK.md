# SLB-ANC-1.0: matched cell-line ancestry benchmark

**Question.** Do frozen synthetic-lethality predictors rank *measured* gene pairs as well in cancer cell lines of African or East Asian genetic ancestry as in European-ancestry lines, on the same cancer-site, screen and pair panel?

**Result.** The matched African–European comparison contains 2 AFR lines (23 SL pairs) and 14 EUR lines (76 SL pairs) across two scorable site × screen blocks. Each AFR block has **one** cell line. Frozen SLp Fusion's equal-cell-line, equal-block AUROC is 0.705 versus 0.790 (gap -0.085). The predeclared adequacy gate fails, so the benchmark issues **no population-level disparity or parity verdict**. The East Asian comparison also fails the gate; Fusion scores 0.838 versus 0.698 on its own matched blocks. These are retrospective diagnostic scores.

The ancestry annotations are genome-wide genotype estimates from [Dutil et al. 2019](https://pubmed.ncbi.nlm.nih.gov/30894373/) through the checksum-pinned Cellosaurus file. Dutil's resource supplies ancestry metadata, **not** SL labels. The SL labels come from the dual-perturbation screens described in [BENCHMARK.md](BENCHMARK.md#label-rules); all included sources passed [REPLICATION.md](REPLICATION.md). Cell-line ancestry is not patient race or clinical treatment response.

The scorable matched rows come from [Flister et al. 2025](https://doi.org/10.1016/j.celrep.2025.116512) (Cas12a paralog screens) and [Harle et al. 2025](https://doi.org/10.1186/s13059-025-03737-w) (dual-guide knockout screens); exact source files and label thresholds are pinned in the base benchmark.

## Fixed evaluation contract

- Include only genotype-inferred majority groups (fraction >0.50); exclude the self-reported PC-9 annotation. The exact genotype fractions remain visible below.
- Match **cancer site + exact merged source set + exact ordered gene pair** across each target group and EUR: each retained pair occurs in at least one line of each group. A complete-case sensitivity uses only pairs with usable labels in every line of a block. Cancer site is a coarse mapping of the pinned Cellosaurus disease; histology and molecular subtype are not matched.
- Recompute the existing SLB fitness-overlap weights after panel restriction. Compute AUROC separately per cell line × screen, average cell lines equally in each site × screen block, then average scorable blocks equally. The secondary metric is tie-aware precision among the top 10 pairs per line.
- Treat a cell line as the independent sampling unit. The only primary model for a future disparity decision is the previously dev-selected SLp Fusion; every other clean ranked model is descriptive. The matching rule uses assay availability rather than model scores or SL calls.
- The adequacy gate requires at least 20 distinct lines per group, two shared blocks, five qualified lines per group in every block (each with ≥100 pairs and ≥10 SL pairs), ≥100 complete-case pairs per block, donor-disjoint training, and variant-aware guide QC. It then requires a donor-bootstrap gap interval half-width ≤0.05. Two ancestry comparisons use Bonferroni-adjusted 97.5% intervals; a gap below −0.05 is the practical-disadvantage margin. The exact machine-readable contract is [`reference/slb_ancestry_protocol_v1.json`](reference/slb_ancestry_protocol_v1.json).

This protocol was written **after** the SLB-1.3 test readout; it is a retrospective diagnostic and a fixed template for newly collected donors, not a claim of prospective preregistration. Original SLB test genes are family-held-out; cell lines were present in SLB training and therefore do not meet donor-disjoint training for supervised entries.

## Panel support

The full genotype-only test has AFR 3 lines / 768 pairs / 32 SL, EAS 6 lines / 2,357 pairs / 87 SL, EUR 38 lines / 12,681 pairs / 442 SL. The matched panel is selected solely from measured pair availability; no model score determines inclusion. No test line is assigned to AMR, SAS, admixed, so those groups have no scorecard.

### AFR versus EUR

| Cancer site × screen | Group-shared pairs | In every line | AFR lines / pairs / SL | EUR lines / pairs / SL |
|---|---:|---:|---:|---:|
| colorectal × `flister2025` | 599 | 165 | 1 / 599 / 19 | 7 / 3,307 / 61 |
| lung × `harle2025` | 27 | 0 | 1 / 27 / 4 | 7 / 93 / 15 |

HeLa has no European-ancestry cervical line on the same source/pair panel, so it contributes no strict matched score. RKO supplies the colorectal block; NCI-H23 supplies the lung block. The unequal number of measured rows cannot turn either single line into more donors. Group-shared pairs can be absent or ambiguously labelled in individual lines; the `In every line` column exposes that attrition.

### EAS versus EUR

| Cancer site × screen | Group-shared pairs | In every line | EAS lines / pairs / SL | EUR lines / pairs / SL |
|---|---:|---:|---:|---:|
| head_neck × `flister2025` | 499 | 499 | 1 / 499 / 7 | 1 / 499 / 13 |
| lung × `flister2025` | 511 | 189 | 1 / 511 / 4 | 4 / 1,565 / 15 |
| lung × `flister2025,harle2025` | 3 | 1 | 1 / 3 / 0 | 2 / 4 / 0 |
| lung × `harle2025` | 34 | 0 | 2 / 47 / 15 | 7 / 117 / 25 |
| pancreas × `harle2025` | 22 | 1 | 1 / 22 / 8 | 8 / 81 / 37 |

The three-row `flister2025,harle2025` lung block has no EAS positive and is recorded as unscorable. EAS gastric lines have no European gastric comparator in these sources. The two target comparisons use **different** matched blocks, so their score gaps are not directly interchangeable.

## Frozen model battery on the matched panels

AUROC values are equal-cell-line and equal-block means. `Δ` is target minus EUR within that target's matched panel. A positive Δ is a point estimate, not a population-level advantage. Every row is blocked from a fairness verdict by the same data-adequacy gate. Full line-level scores, top-10 precision, hashes, and gate reasons are in [`reference/slb1.3_ancestry_benchmark.json`](reference/slb1.3_ancestry_benchmark.json).

| Clean ranked model | AFR | EUR matched to AFR | Δ AFR−EUR | EAS | EUR matched to EAS | Δ EAS−EUR |
|---|---:|---:|---:|---:|---:|---:|
| SL-Predict 2026 (MAE branch) | 0.686 | 0.775 | -0.088 | 0.808 | 0.747 | +0.061 |
| Ryan 2026 (full clean refit) | 0.768 | 0.838 | -0.070 | 0.842 | 0.781 | +0.062 |
| De Kegel 2021 (all-species) | 0.693 | 0.680 | +0.013 | 0.818 | 0.680 | +0.138 |
| SLP Fusion (loss) | 0.705 | 0.790 | -0.085 | 0.838 | 0.698 | +0.140 |
| Ryan 2026 (context clean refit) | 0.805 | 0.861 | -0.056 | 0.864 | 0.776 | +0.088 |
| SLP Fusion (core) | 0.698 | 0.772 | -0.074 | 0.807 | 0.679 | +0.128 |
| GO/PPI GBM | 0.693 | 0.773 | -0.079 | 0.743 | 0.651 | +0.092 |
| GO/PPI GBM (pooled) | 0.652 | 0.683 | -0.031 | 0.777 | 0.658 | +0.119 |
| paralog_identity | 0.619 | 0.681 | -0.062 | 0.805 | 0.644 | +0.161 |
| DepMap OLS (loss) | 0.608 | 0.748 | -0.141 | 0.781 | 0.697 | +0.084 |
| MuSL (human) | 0.646 | 0.688 | -0.042 | 0.699 | 0.592 | +0.107 |
| Ontotype (pooled) | 0.722 | 0.793 | -0.071 | 0.722 | 0.699 | +0.023 |
| MuSL (all-species) | 0.632 | 0.590 | +0.042 | 0.678 | 0.535 | +0.143 |
| Ontotype | 0.643 | 0.706 | -0.063 | 0.663 | 0.686 | -0.023 |
| lgbm | 0.511 | 0.632 | -0.121 | 0.747 | 0.604 | +0.143 |
| Cilantro-SL 2026 (Geneformer branch) | 0.616 | 0.669 | -0.053 | 0.760 | 0.674 | +0.086 |
| PAGAN 2026 (genes-to-pairs) | 0.622 | 0.621 | +0.001 | 0.521 | 0.567 | -0.046 |
| codependency | 0.687 | 0.716 | -0.030 | 0.633 | 0.665 | -0.033 |
| GiGCN 2026 (binary GO adaptation) | 0.671 | 0.651 | +0.021 | 0.629 | 0.571 | +0.058 |
| SynLeaF (human) | 0.622 | 0.740 | -0.119 | 0.627 | 0.619 | +0.008 |
| SynLeaF (all-species) | 0.614 | 0.660 | -0.046 | 0.684 | 0.604 | +0.080 |
| fitness_lgbm | 0.430 | 0.453 | -0.023 | 0.629 | 0.501 | +0.128 |
| random | 0.700 | 0.435 | +0.265 | 0.517 | 0.455 | +0.063 |
| fitness | 0.350 | 0.487 | -0.137 | 0.572 | 0.554 | +0.018 |

The table is ordered by each model's original human SLB score, not by the ancestry gap. The complete-case sensitivity below is restricted to pairs labelled in **every** line of a block. It leaves only colorectal × Flister for AFR–EUR and head/neck × Flister plus lung × Flister for EAS–EUR; Harle's all-line overlap is zero and pancreas has one pair with no scorable class contrast.

| Model | AFR complete-case | EUR | Δ | EAS complete-case | EUR | Δ |
|---|---:|---:|---:|---:|---:|---:|
| SLP Fusion (loss) | 0.418 | 0.788 | -0.370 | 0.931 | 0.571 | +0.360 |
| SL-Predict 2026 (MAE branch) | 0.419 | 0.901 | -0.482 | 0.778 | 0.695 | +0.083 |
| Ryan 2026 (full clean refit) | 0.737 | 0.789 | -0.052 | 0.937 | 0.520 | +0.417 |
| SynLeaF (human) | 0.488 | 0.786 | -0.298 | 0.824 | 0.485 | +0.339 |

The AFR complete-case score is RKO on 165 pairs with only **2 SL labels**. The EAS lung complete-case block has one SL label in LK-2. These large score swings demonstrate how little stable information remains after enforcing identical panels; they do not provide stronger evidence of an ancestry effect.

### Fixed experimental budget

Tie-aware precision@10 counts hits among up to ten proposed measurements per line in each shared panel; if fewer than ten pairs remain, all are used. A random ranker achieves the local SL prevalence. It is a useful lab-budget diagnostic but these panels have very different base rates.

| Model | AFR P@10 | EUR P@10 (AFR panel) | EAS P@10 | EUR P@10 (EAS panel) |
|---|---:|---:|---:|---:|
| SLP Fusion (loss) | 0.400 | 0.245 | 0.463 | 0.321 |
| SL-Predict 2026 (MAE branch) | 0.250 | 0.254 | 0.363 | 0.289 |
| Ryan 2026 (full clean refit) | 0.550 | 0.273 | 0.500 | 0.348 |
| SynLeaF (human) | 0.200 | 0.123 | 0.300 | 0.231 |

[Figure 5](figures/05_ancestry_support.svg) displays the matched support and the donor count. [Figure 6](figures/06_ancestry_line_scores.svg) displays individual-line scores and block means for the primary model. Both plots read the pinned JSON above.

The ancestry-only prior scores exactly 0.500 in both groups because each cell line is scored internally. A single frozen random-score control yields an AFR–EUR point gap of +0.265 on this tiny donor panel. That large accidental contrast is a direct warning against interpreting a model's point gap as evidence of ancestry-related performance.

## Adequacy and assay-quality checks

The actual gate results for the primary model are:

| Comparison | Scorable blocks | Target / EUR lines | Gate |
|---|---:|---:|---|
| AFR–EUR | 2 | 2 / 14 | fail |
| EAS–EUR | 4 | 4 / 18 | fail |

Both comparisons fail the donor-count, within-block pair/positive-count, complete-case pair, donor-disjoint training and variant-aware guide-QC requirements. The JSON lists every exact gate failure. No population interval is emitted from one AFR donor per block.

For collection planning, the two EUR comparator blocks have donor-level Fusion AUROC SDs of 0.135 (colorectal) and 0.112 (lung). A normal approximation for two balanced blocks and a familywise 97.5% half-width of 0.05 suggests roughly 31 independent lines **per ancestry, per block** if the SD is similar. The observed donor-bootstrap precision gate determines the actual requirement; this 13-line EUR pilot is only a planning estimate.

The SLB training split contains labelled rows from every AFR test cell line. For example, HeLa has 804 train pairs, NCI-H23 99, and RKO 2,441. This does not prove that every method used those labels, but the current frozen battery is not certified donor-disjoint. A new evaluation must hold donor lines out *before* fitting or feature selection and publish per-model training-context provenance.

The [Harle et al. 2025](https://doi.org/10.1186/s13059-025-03737-w) supplementary Table S4 includes guide-pair and replicate counts. With the same pinned HGNC symbol resolver used in SLB, all 27 NCI-H23 pair records on the AFR strict panel link to ≥16 guide pairs and ≥2 replicates. Those counts check design coverage, **not** whether a guide matches that donor's genome. The pinned Flister RKO label table is gene-level. No per-donor guide-target variant overlap or independent confirmation is available in this benchmark, so the variant-aware QC gate fails. [Misek et al. 2024](https://doi.org/10.1038/s41467-024-48957-z) measured ancestry-dependent false negatives from germline variants in CRISPR guide targets; this is a concrete reason to require that QC.

### Ancestry-definition sensitivity

| Minimum genotype fraction | AFR lines / test SL | EAS lines / test SL | EUR lines / test SL |
|---|---:|---:|---:|
| >0.5 | 3 / 32 | 6 / 87 | 38 / 442 |
| >0.75 | 1 / 19 | 6 / 87 | 38 / 442 |
| >0.9 | 0 / 0 | 6 / 87 | 28 / 302 |

AFR fractions are HeLa 0.6474, NCI-H23 0.6786 and RKO 0.8143. At a >0.75 threshold, the strict matched comparison retains only RKO; at >0.90 there is no AFR line. These sensitivity checks make the group definition visible rather than silently treating admixed lines as homogeneous.

## Next data collection and decision rule

Run the *same* dual-perturbation library with matched cancer-site blocks across independently derived, genotype-annotated donor lines. Recruit enough underrepresented-ancestry lines to clear the counts in the fixed protocol, and include variants in every guide target plus orthogonal guide or assay confirmation. Freeze donor-disjoint training, scoring code and the assay panel before new labels are read. When the gate passes, use the donor-bootstrap interval: upper bound below −0.05 means a material target-group disadvantage; lower bound above −0.05 supports performance within the margin; otherwise the comparison is inconclusive. The unit remains the donor line, not the number of gene pairs. This experiment would evaluate screen generalization, not clinical effectiveness in patients.

A [West African patient-derived breast organoid screen](https://doi.org/10.1158/0008-5472.CAN-24-0775) shows a route to more representative models, but it perturbs single kinases and drug treatments rather than the same dual-gene panel. Its outcomes are therefore not relabelled as SLB pair truth.

## Rebuild

```bash
SLB_BENCH=data/bench/slb1.3 uv run python scripts/benchmark_ancestry.py
uv run python figures/make_figures.py
```

All 24 clean ranked model predictions and result JSON are checked against `leaderboard.yaml`; benchmark, annotation, protocol, guide supplement and scoring-code SHA-256 hashes are in the result JSON.
