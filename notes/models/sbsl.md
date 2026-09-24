# sbsl: selection-bias-resilient SL prediction (Seale, Tepeli, Goncalves, Bioinformatics 2022)

battery: sbsl__en, sbsl__rf; species: human only (DepMap / TCGA / GTEx features; other species get the shared constant fill); needs: SLB train/dev(+contexts), DepMap 24Q4 (CRISPRGeneEffect, OmicsSomaticMutations, Model), DEMETER2 (data/raw/demeter2/D2_gene_effect.csv), TCGA PanCanAtlas Xena (EB++, GISTIC2, MC3, TCGA-CDR), GTEx v8, MSigDB v7.0 c2.cp KEGG/Reactome/PID; the ELISL adapter's python env + omics caches (scripts/models/elisl); docker image slb/sbsl (R 4.3.3)

- Paper: Seale C, Tepeli Y, Goncalves JP. Overcoming selection bias in synthetic lethality prediction. Bioinformatics 38(18):4360 (2022). doi:10.1093/bioinformatics/btac523
- Repo: https://github.com/joanagoncalveslab/SBSL, commit 6f4a9349e78160a69dd53f18bbfa46a3e6b64f37 (external/models/sbsl), R code. License: GPL-3.0.
- Weights: none released (models are refit in every experiment script). Processed feature tables for their labels are in r/data/*.RData (ISLE / DiscoverSL pairs only).
- Original training data: ~8,000 SL / non-SL pairs from ISLE and DiscoverSL for BRCA, COAD, LUAD, OV. Not used.

## Leakage
Retrained only on SLB train labels (human rows of `train.parquet`) => **not leaky**. Single-gene sources: DepMap
24Q4 CRISPR gene effect + somatic mutations, DEMETER2 RNAi, TCGA PanCanAtlas expression / GISTIC / MC3 mutations /
TCGA-CDR survival, GTEx v8 expression, MSigDB v7.0 pathway gene sets. **No PPI / STRING / BioGRID features**
(SBSL's only network-like feature is pathway co-membership from MSigDB), so the STRING genetic-interaction issue
does not arise.

## What was rebuilt / changed (scripts/models/sbsl/)
features.py re-implements the features of r/features/*.R + r/experiments/generate_features.R per
(gene_a, gene_b, cancer type), using the same context -> TCGA-type mapping as ELISL (scripts/models/elisl/common.py,
incl. PANCAN for hTERT-RPE1):
- copathway_participation (hypergeometric, MSigDB v7.0 KEGG+Reactome+PID as in SBSL); dsl_mutex_{amp,del,mut},
  dsl_mutex (Fisher/sumlog), mutex_alt (hypergeometric mutual exclusivity on GISTIC +-2 / MC3 non-silent);
  discover_mutex: DISCOVER (R package from github NKI-CCB/DISCOVER a46d99f, v0.9.4) background model fitted in R
  (`discover.matrix`, strata = TCGA type for PANCAN); the pairwise one-sided test is computed in Python
  (exact Poisson-binomial DP, identical to `pairwise.discover.test` p-values on test data; refined normal
  approximation when the observed co-occurrence > 30, max abs deviation 0.0017 on the toy check) because calling
  pairwise.discover.test once per SLB pair ran at ~1 pair/s.
- exp_corr / exp_corr_normal (+p): Pearson of TCGA tumour / normal expression (EB++ linear scale; SBSL: Firehose
  RSEM); gtex_corr (+p) with ELISL's type -> GTEx tissue map.
- diff_exp_logfc / _pvalue: SBSL ran edgeR exactTest on GSE62944 raw counts; PanCanAtlas has no raw counts, so we use
  the log2 EB++ mean difference + Welch t-test (only when > 3 tumours carry a gene_a mutation, else 0 / 1 as SBSL).
- avana_* / d2_*: co-dependency (Pearson), mean dependency, Wilcoxon of gene_b effect in gene_a-mutated (>= 3) vs
  other lines of the cancer type, with SBSL's error/warning defaults (ties -> p 1, W 0). SBSL: Avana 19Q3 + DEMETER2;
  here DepMap 24Q4 Chronos gene effect + DEMETER2.
- mut_logrank.pval / mrna_logrank.pval: log-rank of patients with both genes mutated / both under-expressed
  (bottom 5%) vs the rest (lifelines instead of survminer).
- MUTEX (Babur et al., Java tool, per-cancer ranked groups) is not reproduced: generate_feature_dataset.R loads a
  `mutex` column, but its generator (generate_MUTEX_scores) defaults to a constant 2 whenever the precomputed MUTEX
  file lacks the pair. We use the remaining 27 features (listed in prep_csv.py).
- LAML: PanCanAtlas has no LAML samples with both GISTIC and MC3 calls -> mutex tests set to SBSL's
  no-alteration value (0), DISCOVER p = 1 (affects Jurkat / K-562 rows).
- Model (train.R): SBSL's caret recipes (utils/train-model.R): elastic net (`glmnet`, 10-fold CV, alpha 20 x lambda 50
  grid, metric ROC) and random forest (`rf`, mtry 4:8); `na.omit` on training rows; class balancing by undersampling
  (seed 124) done per SLB context (SBSL balanced per cancer type with one global n, which would be 0 here);
  preProcess center/scale/nzv (3 features dropped as near-zero variance). Scored rows with NA features get the
  training median (SBSL had no NA test rows). RRF / L0Learn / MUVR variants not run (SBSL-EN is the paper's main
  linear model; RF is the standard caret forest, not RRF).
- Versions: R 4.3.3, caret 6.0-94, glmnet 4.1-8, randomForest 4.7-1.1 (slb/sbsl, external/models/sbsl/slb_docker).

## Status
acquired / env built (docker slb/sbsl + ELISL venv) / original R feature code not runnable as is (hard-coded
~/repos paths, Firehose/GSE62944 raw files, per-label precomputed RData) -> re-implemented / adapted / dev-scored on
slb1.2 and slb1.3

## Dev results
Native coverage (.coverage.json): human 15,048/15,048 (slb1.2), 19,149/19,149 (slb1.3); other species 0 scored
(constant fill). Balanced training set after na.omit: 5,776 rows (slb1.2), 5,820 (slb1.3).

| bench | variant | SLB | H. sapiens | S. cerevisiae | S. pombe | human AFR | EAS | EUR |
|---|---|---|---|---|---|---|---|---|
| slb1.2 dev | sbsl__en | 0.5159 | 0.5635 | 0.5000 (fill) | 0.5000 (fill) | 0.6545 | 0.5702 | 0.4657 |
| slb1.2 dev | sbsl__rf | 0.5230 | 0.5922 | 0.5000 (fill) | 0.5000 (fill) | 0.6272 | 0.6013 | 0.5482 |
| slb1.3 dev | sbsl__en | 0.5093 | 0.5280 | 0.5000 (fill) | 0.5000 (fill) | 0.5728 | 0.5417 | 0.4695 |
| slb1.3 dev | sbsl__rf | 0.5127 | 0.5382 | 0.5000 (fill) | 0.5000 (fill) | 0.5003 | 0.5893 | 0.5249 |
slb1.3 auxiliary species n/a (<20 positives). caret CV ROC on the balanced train set: EN 0.63-0.64, RF 0.80-0.81
(the RF CV is optimistic: the same pair recurs across contexts, so CV folds share pairs). Largest EN weights:
copathway_participation (-), avana_avg (-), gtex_corr (+), d2_avg (-), avana_codep (+).

## Runtime
Features ~25 min for slb1.2 train (PANCAN group dominant) + DISCOVER ~10 min; incremental for new versions
(slb1.3: ~10 min); caret training ~5 min (EN) + ~5 min (RF) per run, 8 CPUs.

## Run
`SLB_BENCH=data/bench/slb1.3 SLB_SPLIT=dev scripts/models/sbsl/run.sh` -> results/models/[<bench>/]sbsl__{en,rf}_<split>.parquet
(+ .coverage.json, .txt/.json on dev). Work files: external/models/sbsl/_slb/{cache,feat/<bench>}.
