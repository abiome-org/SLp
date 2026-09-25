# SLB: synthetic-lethality benchmark

**Task.** Given a genetic context (species + cell line or strain) and a gene pair, score how likely
losing both genes is synthetically lethal or sick: a strong negative genetic interaction beyond the
two single-loss effects.

**Data.** Every example is a measured outcome from a published combinatorial screen. Positives and
negatives were both tested. There are no "unknown = negative" pairs, no literature-mined labels and no
synthetic data. Every source that supplies labels passed a reproducibility check.

**Held out.** Genes are grouped into families: paralogs with ≥ 30% protein identity, plus orthologs
across 12 proteomes (human, mouse, worm, fly, both yeasts, *C. albicans*, *S. pneumoniae*, *E. coli*,
*B. subtilis*, *M. tuberculosis*, *S. aureus*). Orthologs come from the Alliance (reciprocal best hits,
or ones supported by at least 3 methods), PomBase curated orthologs, and reciprocal best DIAMOND hits
between every pair of proteomes. Whole families are hashed into train, dev or test. In a test pair,
neither gene has a paralog above 30% identity or any ortholog in train, in any species.

There is only one evaluation: both genes held out. Splits that hold out a pair but let its genes
appear in training (CV1/CV2) are not offered: models can memorise which genes have many SL partners,
so those splits measure training-set lookup, not prediction.

There is one benchmark, built into `data/slb/`. It is updated in place. `manifest.json` records its
revision (currently `slb1.3`), and results are pinned to the manifest's SHA-256 and the scorer version
(`1.3.2`).

## Species

- **Headline** (averaged into the SLB score), each with cross-study replication and hundreds of test
  positives:
  - **Human:** 50 cell lines from 8 studies, annotated with genetic ancestry.
  - ***S. cerevisiae***: Costanzo 2016, Kuzmin 2018 and 2020, Costanzo 2021, 9 merged E-MAPs.
  - ***S. pombe***: Ryan 2012, Frost 2012.
- **Auxiliary** (scored and reported the same way, not averaged in): *B. subtilis* (Koo 2025),
  *C. elegans* (Byrne 2007), *D. melanogaster* (Horn 2011), *M. musculus* (Roguev 2013). A species is
  scored only when the split has at least 20 positives; on test that covers *B. subtilis* (55) and
  *C. elegans* (38).

*S. pneumoniae* and *E. coli* are measured but supply no labels: their screens contradict each other.

## Label quality

A source's labels are used only if all three hold:
1. An independent re-measurement recovers them at AUROC ≥ 0.65: another study in the same cell line
   where one exists, otherwise the source's own replicates, alleles or query/array orientations.
2. They are not contradicted by the cross-study consensus.
3. The hit rate is plausible.

`slbench audit` re-runs every check and writes `results/reports/replication.md`.

| Included source | Evidence | AUROC |
|---|---|---|
| Dede 2020 | Cross-study, vs the other included studies | 0.93 |
| Chou 2025 | Cross-study | 0.81 |
| Harle 2025 | Cross-study | 0.80 |
| Flister 2025 | Cross-study | 0.76 |
| Horlbeck 2018 | Own replicates | 0.71–0.81 |
| Parrish 2021 | Own replicates | 0.90–0.95 |
| Zhao 2018 | Own replicates | 0.69–0.72 |
| SPIDR 2025 | Own replicates, after re-scoring | 0.86 |
| Costanzo 2016 (*S. cerevisiae*) | Cross-study, 4 independent re-measurements; own orientation 0.74 | 0.74–0.93 |
| Kuzmin 2018 (*S. cerevisiae*) | Cross-study vs Costanzo 2016 | 0.89 |
| Kuzmin 2020 (*S. cerevisiae*) | Cross-study vs Costanzo 2016 and Kuzmin 2018 | 0.80–0.89 |
| Costanzo 2021, reference condition (*S. cerevisiae*) | Cross-study vs Costanzo 2016; own replicates 0.94 | 0.95 |
| 9 *S. cerevisiae* E-MAPs, merged as one source¹ | Cross-study vs Costanzo 2016 (each map 0.71–0.87) | 0.77 |
| Ryan 2012 (*S. pombe*) | Cross-study vs Frost 2012; own alleles/orientation 0.76 | 0.79 |
| Frost 2012 (*S. pombe*) | Cross-study vs Ryan 2012²; own orientation/alleles 0.80 | 0.67 |
| Koo 2025 (*B. subtilis*, auxiliary) | Own sgRNA orientation swap, at the SLB cut-off | 0.87–0.89 |
| Byrne 2007 (*C. elegans*, auxiliary) | Pairs measured twice; Lehner 2006 hits recovered at 0.70 | 0.87–0.93 |
| Horn 2011 (*D. melanogaster*, auxiliary) | Replicate screens; vs Heigwer 2023 0.72 | 0.92–0.93 |
| Roguev 2013 (*M. musculus*, auxiliary) | Own orientations | 0.73 |

¹ Schuldiner 2005, Collins 2007, Wilmes 2008, Fiedler 2009, Zheng 2010, Aguilar 2010, Hoppins 2011,
Guénolé 2013, Surma 2013. They share many measurements, so listed separately they would look like
independent confirmations of each other.

² Frost's positive cut-off (S < −4) was tightened after its S < −3 calls replicated in Ryan 2012 at
only 0.64, so it was chosen with the cross-study number in view.

**Excluded sources** keep their measurements in `data/interim/measurements` as optional training data
but supply no labels:

| Source | Reason |
|---|---|
| Ito 2021 | Agrees with itself (0.91–1.00), but contradicted by 4 concordant studies (0.55) |
| Thompson 2021 | Cross-study 0.60; its calls are largely predictable from single-gene fitness (0.87) |
| Shen 2017 | Its own replicates don't recover its labels (0.56–0.72) |
| Han 2017, Wong 2016, CHyMErA 2020 | Unverifiable: no usable replicate counts and no overlap with other studies |
| Fischer 2015, Heigwer 2023 (fly) | Same lab and cell line, but they contradict each other (0.57 / 0.41) |
| Billmann 2016 (fly) | No replicates and ≤ 5 positives shared with other maps |
| Dual CRISPRi-seq 2025, Dual Tn-seq 2025 (Zik), CRISPRi-TnSeq 2024 (*S. pneumoniae*) | Each replicates internally (0.79–0.99), but none reproduces another (0.50–0.51) |
| Babu 2011, Gagarinova 2016, Kumar 2016, Côté 2016 (*E. coli*) | Every cross-study comparison sits at 0.46–0.51 |
| Lehner 2006 (*C. elegans*) | Hit list only: negatives would be inferred, not measured |
| Gier 2020 (mouse) | 56% of labelled pairs called SL; indistinguishable from non-expressed controls (0.47) |
| Diehl 2021, Tang 2022 | Implausible hit rates: 63% and 22% of tested pairs called SL |

**Where published calls were replaced:** the yeast positive cut-offs are stricter than the authors'
(ε < −0.2 instead of −0.12; S < −3 instead of −2.3), because stronger interactions replicate better.
SPIDR's published GEMINI calls are recovered by its own replicates at only 0.62, so its raw counts are
re-scored with the same additive zdLFC recipe as the other paralog screens.

## Label rules

- **Negative:** a tested pair that was not called, and whose score falls in that screen's neutral band.
- **Ambiguous:** anything between positive and negative. It is dropped.
- **Merging:** the same (species, context, pair) measured by several included sources is merged;
  conflicting labels are dropped.

| Source | Positive | Negative |
|---|---|---|
| Horlbeck 2018, Zhao 2018 (SLKB original calls) | Author SL call | Not called, and \|score\| < median for that screen |
| Dede 2020, Parrish 2021 (Ryan-lab uniform zdLFC) | zdLFC ≤ −3 | \|zdLFC\| < 1 |
| SPIDR 2025 (RPE1, CRISPRi; re-scored from counts) | z(additive GI) ≤ −3 and GI < 0 in both replicates | \|z\| < 1 |
| Chou 2025 | ZdLFC < −2 (authors) | \|ZdLFC\| < 1 |
| Flister 2025 | Author "Lethal" call; diff_z ≤ −2 for lines with no call | \|diff_z\| < 1 |
| Harle 2025 | Author binary hit matrix | Not a hit, FDR > 0.25, \|GI\| < median for that line |
| Costanzo 2016 SGA | ε < −0.2 and p < 0.05 | p > 0.25 and \|ε\| < median |
| Ryan 2012 *S. pombe* E-MAP | S < −3 | \|S\| < 1 |
| Kuzmin 2018/2020, Costanzo 2021 (reference condition) SGA | ε < −0.2 and p < 0.05 | p > 0.25 and \|ε\| < median |
| *S. cerevisiae* E-MAPs (merged) | S < −3 | \|S\| < 1 |
| Frost 2012 *S. pombe* E-MAP | S < −4 | \|S\| < 1 |
| Koo 2025 *B. subtilis* dual CRISPRi | GI ≤ −1.5 | see `bacteria_extra.koo2025` |
| Byrne 2007 *C. elegans* | In the authors' SGI network, strength ≥ 2 | Not in the network, weak |
| Horn 2011 fly | q < 0.05 and π < 0 | q > 0.25, small \|π\| |
| Roguev 2013 mouse E-MAP | S < −3 | \|S\| < 1 |

Per-source rationale is in the parser docstrings under `src/slbench/sources/`.

## Metric

**SLB score** is the one number to hill-climb. It is fitness-balanced by construction.

1. **Compare within a stratum.** A stratum is one context (cell line or strain) × one screen. SL pairs
   are compared by AUROC only with non-SL pairs from the same stratum, so knowing which cell lines or
   libraries have high hit rates earns nothing: a library-prior baseline scores exactly 0.500.
2. **Balance single-gene fitness.** SL calls concentrate on genes that are already sick on their own.
   Each pair gets a propensity *e* = P(SL | both genes' single-loss effects, screen, context), fitted
   on the evaluation split itself. SL pairs are weighted 1 − *e* and non-SL pairs *e* (overlap weights;
   Li, Morgan & Zaslavsky 2018), rescaled per stratum and class. The fitted fitness design columns
   balance in their weighted means; the `fitness_lgbm` control measures any residual nonlinear signal.
3. **Human species score:** the mean over genetic-ancestry groups with at least 20 positives in the
   split. Ancestry is the donor's genotype-inferred majority super-population (> 50%) from Cellosaurus
   ([Dutil et al. 2019](https://doi.org/10.1158/0008-5472.CAN-18-2747)). Lines with no estimate
   (hTERT-RPE1, C092) are reported but not averaged in.
4. **SLB score** = the mean of the headline species' scores, each counting equally. The tiers are
   recorded in `manifest.json`.

`eval` also reports every species, ancestry group, cell line and paralog/non-paralog stratum:
**within-gene** balanced AUROC (stratified additionally by gene), **unadj** (plain stratified AUROC; the
gap to SLB AUROC is how much of a ranking is single-gene fitness) and **AP lift** (average precision ÷
prevalence). `--boot N` adds a gene-family cluster-bootstrap 95% CI; `slbench compare A B` runs a paired
bootstrap of the difference.

**Single-loss effect per gene** (`gene_single_effects.parquet`, a permitted model input):

| Species | Single-loss effect |
|---|---|
| Human | DepMap 24Q4 Chronos effect in that cell line where screened, plus the pan-line mean |
| *S. cerevisiae* | SGA single-mutant fitness − 1 (Costanzo 2016) |
| *S. pombe* | PomBase deletion viability: inviable −1, slow growth −0.5, viable 0 |
| *B. subtilis* | CRISPRi single-knockdown fitness (Koo 2025) |
| *C. elegans* | WormBase WS298 phenotypes: lethal or larval arrest −1, sterile or slow −0.5, else 0 |
| *D. melanogaster* | Single-dsRNA main effect on cell count (Heigwer 2023) |
| *M. musculus* | DepMap pan-line mean of the one-to-one human ortholog |

The propensity model (`fitness.propensity`) is a lightly penalised logistic regression per species on
cubic splines of each gene's context and pan-context effect, their product, screen and context
intercepts, and screen × effect interactions. It is fitted on the evaluation split because the
fitness→SL relation differs between held-out family sets, and it is logistic so the smooth design
cannot memorise individual genes. Within-stratum standardised mean differences of the fitness
covariates fall from up to 1.4 to below 0.03 (asserted in `tests/test_benchmark.py`). On test,
−(f_a + f_b) scores 0.49–0.50 and `fitness_lgbm` 0.52–0.53 on the headline species; on the yeasts
`fitness_lgbm` keeps a small residual (0.505–0.526 across split seeds), so compare yeast gains against
it, not against 0.5. Propensities live under `hidden/` and are never a model input.

**Protocol.** Hill-climb on `dev`. Evaluate on `test` only at milestones and record every test
evaluation in `leaderboard.yaml`.

## Leakage rules

A model scored on `test` must not have been fitted on any record involving a gene from a test family,
in any species:

- **Not allowed:** any combinatorial or multi-gene perturbation readout; any SL or genetic-interaction
  label or edge from any source (SynLethDB, BioGRID, SLKB, knowledge graphs with SL or GI edges,
  papers), including the excluded sources above.
- **Allowed, but declare it:** single-gene data (DepMap and other single-KO screens, expression,
  sequence, GO, PPI), and pretrained models whose training data follows the same rule.

`slbench check-leakage pairs.parquet` (columns `species, gene_a, gene_b`) flags records touching
held-out families, including genes absent from the benchmark that are paralogs or orthologs of
held-out genes. Add `--allow-dev` for final models trained on train + dev. Models fitted on public SL
databases without this filter are marked `leaky` and not ranked. What holding out leaves behind:
distant paralogs below 30% identity, and orthologs too distant to be reciprocal best hits.

## Leaderboard

Test split, fitness-balanced AUROC with family-bootstrap 95% CI. Regenerated from `leaderboard.yaml` by
`slbench leaderboard`, which re-checks every result hash and re-scores every prediction file. Italic
columns are auxiliary species; n/a = fewer than 20 test positives. Leaky, possibly leaky and
exploratory entries are unranked (–).

<!-- leaderboard:start -->
| # | model | SLB score (95% CI) | human | scer | spom | *bsub* | *cele* | *dmel* | *mmus* | trained on | leaky |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Dev Rank Ensemble (loss)** — core blend plus human DepMap OLS | 0.645 (0.595–0.684) | 0.684 | 0.581 | 0.669 | 0.561 | 0.546 | n/a | n/a | SLB train; blend selected on dev | no |
| 2 | **Ontotype** — ontology-based model retrained by species | 0.640 (0.597–0.673) | 0.625 | 0.589 | 0.708 | 0.514 | 0.618 | n/a | n/a | SLB train + filtered GO | no |
| 3 | **Dev Rank Ensemble (core)** — dev-selected rank blend of GO/PPI, Ontotype, SynLeaF and De Kegel | 0.639 (0.588–0.680) | 0.668 | 0.581 | 0.669 | 0.561 | 0.546 | n/a | n/a | SLB train; blend selected on dev | no |
| 4 | **Ontotype (pooled)** — one ontology model across species | 0.632 (0.591–0.666) | 0.639 | 0.581 | 0.674 | 0.469 | 0.542 | n/a | n/a | SLB train + filtered GO | no |
| 5 | **GO/PPI GBM** — multi-network feature model, trained separately by species | 0.625 (0.576–0.660) | 0.663 | 0.590 | 0.621 | 0.565 | 0.537 | n/a | n/a | SLB train + filtered GO/PPI/fitness | no |
| 6 | **GO/PPI GBM (pooled)** — one multi-species model on filtered networks | 0.624 (0.581–0.659) | 0.660 | 0.589 | 0.622 | 0.570 | 0.570 | n/a | n/a | SLB train + filtered GO/PPI/fitness | no |
| 7 | **De Kegel 2021 (all-species)** — paralog feature RF adapted across species | 0.603 (0.551–0.647) | 0.699 | 0.552 | 0.557 | 0.535 | 0.617 | n/a | n/a | SLB train + filtered GO/PPI/ESM2 | no |
| 8 | **MuSL (all-species)** — GNN branch retrained across species | 0.593 (0.549–0.634) | 0.636 | 0.553 | 0.592 | 0.545 | 0.562 | n/a | n/a | SLB train + filtered PPI/ESM2 | no |
| 9 | **SL-Predict 2026 (MAE branch)** — released frozen DepMap 26Q1 MAE gene encoder plus SLB-trained symmetric LightGBM; human only, 99% native coverage | 0.575 (0.547–0.595) | 0.724 | 0.500 | 0.500 | 0.500 | 0.500 | n/a | n/a | single-gene DepMap 26Q1 unlabeled profiles; SLB train pair labels | no |
| 10 | **Ryan 2026 (full clean refit)** — context-specific paralog random forest refitted on SLB; human only, 77% native coverage | 0.571 (0.542–0.597) | 0.712 | 0.500 | 0.500 | 0.500 | 0.500 | n/a | n/a | SLB train; filtered GO/PPI and DepMap single-gene data | no |
| 11 | **Ryan 2026 (context clean refit)** — context-only paralog random forest refitted on SLB; human only, 77% native coverage | 0.561 (0.530–0.591) | 0.682 | 0.500 | 0.500 | 0.500 | 0.500 | n/a | n/a | SLB train; filtered GO/PPI and DepMap single-gene data | no |
| 12 | **SynLeaF (all-species)** — KG/RGCN branch retrained on the filtered multi-species bundle | 0.554 (0.508–0.605) | 0.537 | 0.523 | 0.602 | 0.517 | 0.364 | n/a | n/a | SLB train + filtered GO/PPI | no |
| 13 | **paralog_identity** — Ensembl 116 paralog protein identity | 0.552 (0.503–0.584) | 0.653 | 0.503 | 0.501 | 0.500 | 0.500 | n/a | n/a | none | no |
| 14 | **lgbm** — gradient boosting on single-gene fitness, paralog identity, DepMap co-dependency | 0.551 (0.509–0.581) | 0.610 | 0.518 | 0.526 | 0.575 | 0.503 | n/a | n/a | SLB train | no |
| 15 | **DepMap OLS (loss)** — human single-gene dependency scan; other species tied | 0.550 (0.516–0.582) | 0.650 | 0.500 | 0.500 | 0.500 | 0.500 | n/a | n/a | DepMap single-gene/omics only | no |
| 16 | **MuSL (human)** — full multimodal model; other species tied | 0.549 (0.524–0.576) | 0.646 | 0.500 | 0.500 | 0.500 | 0.500 | n/a | n/a | SLB train + BioGRID physical/ESM2/TCGA | no |
| 17 | **Cilantro-SL 2026 (Geneformer branch)** — Geneformer in-silico-knockout, viability FiLM and five-fold SLNet refit; human only, 55% native coverage | 0.533 (0.503–0.563) | 0.599 | 0.500 | 0.500 | 0.500 | 0.500 | n/a | n/a | Geneformer/Gene2vec unlabeled pretraining; DepMap single-gene effects; SLB train pairs | no |
| 18 | **codependency** — DepMap gene-effect profile correlation (human only) | 0.524 (0.489–0.552) | 0.571 | 0.500 | 0.500 | 0.500 | 0.500 | n/a | n/a | none (single-gene data) | no |
| 19 | **GiGCN 2026 (binary GO adaptation)** — signed factor graph network refitted on SLB train; binary SL versus neutral with filtered GO features, human only | 0.523 (0.492–0.549) | 0.569 | 0.500 | 0.500 | 0.500 | 0.500 | n/a | n/a | SLB train pairs; filtered GO annotations, no external GI graph | no |
| 20 | **fitness_lgbm** — gradient boosting on the single-loss covariates only (probe of residual fitness signal) | 0.521 (0.485–0.559) | 0.521 | 0.515 | 0.526 | 0.571 | 0.503 | n/a | n/a | SLB train | no |
| 21 | **PAGAN 2026 (genes-to-pairs)** — essentiality-trained genes-to-pairs GraphSAGE on filtered SLB graph; human and budding yeast, fission yeast tied | 0.520 (0.490–0.547) | 0.591 | 0.469 | 0.500 | 0.500 | 0.500 | n/a | n/a | single-gene essentiality; filtered GO/PPI/paralogs; no SL pair labels | no |
| 22 | **SynLeaF (human)** — dual-stage omics and KG model; other species tied | 0.515 (0.482–0.550) | 0.546 | 0.500 | 0.500 | 0.500 | 0.500 | n/a | n/a | SLB train + filtered KG/TCGA omics | no |
| 23 | **fitness** — sickness of the two single mutants, -(f_a + f_b) | 0.498 (0.468–0.533) | 0.490 | 0.501 | 0.502 | 0.527 | 0.503 | n/a | n/a | none (single-gene data) | no |
| 24 | **random** — uniform noise | 0.492 (0.471–0.515) | 0.494 | 0.490 | 0.492 | 0.450 | 0.578 | n/a | n/a | none | no |
| – | **SL-Predict MAE vectors only (exploratory test ablation)** — feature-group ablation run after full-branch test inspection; human only | 0.563 (0.533–0.585) | 0.688 | 0.500 | 0.500 | 0.500 | 0.500 | n/a | n/a | DepMap 26Q1 single-gene profiles; SLB train pairs | no |
| – | **SL-Predict coessentiality only (exploratory test ablation)** — feature-group ablation run after full-branch test inspection; human only | 0.514 (0.479–0.542) | 0.543 | 0.500 | 0.500 | 0.500 | 0.500 | n/a | n/a | DepMap 26Q1 single-gene profiles; SLB train pairs | no |
| – | **Ryan 2026 (released external-GEMINI weights, leakage diagnostic)** — released classifier trained on external GEMINI SL screens overlapping held-out families | 0.535 (0.507–0.564) | 0.606 | 0.500 | 0.500 | 0.500 | 0.500 | n/a | n/a | external GEMINI labels, including source screens used by SLB | yes |
| – | **SLp-1.1 (decoder)** — prior abiome world model + LightGBM SL decoders (10 decoders from its MuSL gene-held-out folds, averaged); human only, 39% of human test pairs in vocabulary (rest tied) | 0.526 (0.491–0.555) | 0.578 | 0.500 | 0.500 | 0.500 | 0.500 | n/a | n/a | MuSL/SynLethDB SL labels; DepMap; Costanzo yeast (25% sample) | yes |
| – | **SLxGO 2026 (GO-PCA branch, input provenance unresolved)** — released GO-PCA embedding branch, refitted on SLB train; GO evidence provenance unavailable | 0.519 (0.492–0.545) | 0.556 | 0.500 | 0.500 | 0.500 | 0.500 | n/a | n/a | authors' GO-PCA vectors; SLB train pairs | possible |
| – | **SLp-1.1 (label-free)** — prior abiome world model, fixed excess-fitness-loss readout (no SL labels); human only, 39% coverage | 0.503 (0.465–0.537) | 0.510 | 0.500 | 0.500 | 0.500 | 0.500 | n/a | n/a | DepMap; Costanzo yeast (25% sample); Perturb-seq | yes |
<!-- leaderboard:end -->

**Human ancestry.** `scripts/audit_ancestry.py` scores frozen test predictions per cell-line ancestry
(AFR 3 lines / 32 test SL pairs, EAS 7 / 92, EUR 38 / 442). `scripts/benchmark_ancestry.py` runs a
stricter cancer-site/screen/pair-matched comparison under `reference/ancestry_protocol.json`; its
donor-adequacy gate fails on current public screens (2 matched AFR lines), so it issues no
population-level verdict. Both write JSON to `reference/` and a report to `results/reports/`.

## Files (`data/slb/`)

| File | Contents |
|---|---|
| `train.parquet` | Labelled. Both genes in train families. |
| `dev.parquet`, `dev_semi.parquet` | Labelled. Hill-climbing and model selection. `*_semi`: one gene held out. |
| `test_inputs.parquet`, `test_semi_inputs.parquet` | No labels. |
| `hidden/test*_labels.parquet` | Test labels. Only `slbench eval --split test` reads them. |
| `hidden/*_propensity.parquet` | Per-example fitness propensity for the balance weights. Not a model input. |
| `contexts.parquet` | Per context: Cellosaurus accession, DepMap ID, disease, sex, ancestry group and fractions. |
| `held_out_families.parquet` | Gene → family → bucket. Used by the leakage checker. |
| `gene_single_effects.parquet` | Reference single-loss effect per gene. A permitted input. |
| `manifest.json` | Revision, build parameters, species tiers, excluded sources, row counts, sha256 of every file. |

Example columns: `example_id, species, context_id, ancestry_group, gene_a, gene_b, same_family,
sources, label`. A model reads `{split}.parquet` (or `test_inputs.parquet`) and writes one `score` per
`example_id`; higher means more likely SL. Gene IDs: HGNC symbol (human), SGD systematic ORF
(*S. cerevisiae*), PomBase systematic ID (*S. pombe*), BSU locus tag (*B. subtilis*), WBGene
(*C. elegans*), FBgn (fly), MGI symbol (mouse). `slbench.ids.resolve` and `slbench.ids_extra.resolve`
map other names.

## Known limitations

- **Ancestry coverage is thin:** European 38 lines, East Asian 7, African 3. New screens in
  non-European lines are the highest-value additions.
- **Few non-human metazoan labels.** Fly and mouse only add training data; *C. elegans* and
  *B. subtilis* are scored but noisy.
- **No bacterium in the headline.** Every bacterium with more than one pairwise screen has screens
  that contradict each other; *B. subtilis* has one screen, verified only against itself.
- **Human screens mostly test paralog pairs,** so same-family pairs are overrepresented.
- **Inclusion is binary.** Labels are not weighted by source reliability (Dede 0.93 versus Zhao 0.69).

## Layout

```
src/slbench/        package: build, evaluate, audit, leaderboard, release, ancestry, leakage, sources/
scripts/            ancestry, robustness, release packaging, test-result refresh, grader probe
scripts/models/     one adapter per published SL model (see scripts/models/README.md)
tests/              pytest suite
figures/            publication figures and make_figures.py (see figures/README.md)
reference/          checked-in pins: raw_sha256sums.txt, lock.json, public.sha256, fetch/ URLs, ancestry JSON
battery.yaml        which model prediction files `slbench battery` scores
leaderboard.yaml    hash-pinned test results
data/               (local) raw/, interim/, slb/ (the benchmark), release/, robustness/
external/           (local) cloned model repos, their envs and work caches
results/            (local) predictions, result JSON, reports/ (generated markdown), logs/
```

## Commands

Run everything from this directory.

```bash
uv sync
uv run python -m slbench.fetch        # raw sources -> data/raw; manual ones are listed in reference/fetch/
uv run slbench build                  # -> data/slb
uv run slbench verify --raw           # check built artifacts and every pinned raw file
uv run slbench audit                  # -> results/reports/replication.md
uv run slbench card                   # -> results/reports/data_card.md
uv run slbench baseline lgbm --split dev --out results/slb/lgbm_dev.parquet
uv run slbench eval results/slb/lgbm_dev.parquet --split dev
uv run slbench compare results/slb/fitness_dev.parquet results/slb/lgbm_dev.parquet --split dev
uv run slbench check-leakage my_training_pairs.parquet
bash scripts/models/run_battery.sh    # re-run every adapter, then `slbench battery` -> results/reports/models_dev.md
uv run python scripts/refresh_test_results.py   # re-score pinned test results after a rebuild
uv run slbench leaderboard            # verify leaderboard.yaml, update the table above
uv run slbench export-public data/release/slb && bash scripts/package_public.sh
uv run pytest -q
```

The public bundle (`data/release/slb`, archived as `slb-public.tar.gz` and pinned by
`reference/public.sha256`) holds train/dev labels, dev propensity weights, test inputs, context and
single-gene data and the family map, but no test labels or test propensities.
`SLB_BENCH=data/release/slb uv run slbench verify` checks it.

**Updating the benchmark.** Change the build, bump `VERSION` in `src/slbench/build.py`, rebuild into
`data/slb`, then refresh the pinned test results and the leaderboard. The split depends only on
`SALT`, the family graph and the bucket fractions (all in `manifest.json`); `scripts/robustness.sh`
rebuilds with 4 alternative salts to check the ranking holds.
