# Shared statistical SL-inference engine (models-features)

battery: statsl__{ess_crispr,ess_rnai}_{expr,cn,mut}, statsl__{sof_expr,sof_cna,coexp_tcga,coexp_ccle,surv_expr,surv_cna,phylo_neg_dist} (diagnostic single statistics; daisy and isle have own notes); species: human; needs: DepMap 24Q4, DEMETER2, TCGA PanCanAtlas (Xena), ISLE phylogenetic profiles

PPI/KG: none.

Used by: daisy, isle, statsl__* diagnostics. Code: scripts/models/_common/omics.py (cached single-gene matrices) and
scripts/models/_common/statsl.py (vectorised pair statistics), scoring in scripts/models/statsl/score.py.
Cache: external/models/_statsl_cache/ (pairstats_<bench>.parquet covers all human train/dev/test pairs; ~70 min CPU
for 100,832 pairs, 8-16 threads).

Inputs (all single-gene, permitted): DepMap 24Q4 CRISPR (Chronos), expression, relative CN, damaging mutations
(data/raw/depmap); DEMETER2 combined RNAi (data/raw/demeter2, figshare 9170975); TCGA PanCanAtlas via UCSC Xena
(data/raw/tcga_pancan: EB++ RNA-seq [already log2(x+1)], GISTIC2 thresholded CN, MC3 non-silent gene-level
mutations, TCGA-CDR survival); Tabach et al. 2013 phylogenetic profiles + weights shipped in the ISLE repo.
All sha256 in data/raw/<key>/SOURCES.tsv.

Statistics per unordered pair (larger = more SL-like):
- ess_<crispr|rnai>_<expr|cn|mut>: functional examination, one-sided Mann-Whitney z that gene B is more essential in
  lines where gene A is inactive (bottom expression / CN tertile across lines, or damaging mutation); max over the two
  directions.
- sof_expr / sof_cna: hypergeometric under-representation of co-inactivation in TCGA primary tumours (expression:
  bottom tertile within cancer type, ISLE's mRNAq2; CNA: GISTIC <= -1).
- coexp_tcga / coexp_ccle: Spearman co-expression.
- surv_expr / surv_cna: stratified (cancer type) log-rank z for co-inactivation; positive = better survival.
- phylo_dist: ISLE feature-weighted squared distance of phylogenetic profiles.

Dev single-statistic results (human species score): ess_crispr_expr 0.671, coexp_ccle 0.659, ess_crispr_cn 0.608,
phylo 0.601, coexp_tcga 0.592, ess_crispr_mut 0.592, ess_rnai_expr 0.557, ess_rnai_cn 0.549, sof_cna 0.490,
surv_expr 0.491, sof_expr 0.447, surv_cna 0.443, ess_rnai_mut 0.434. The tumour-based statistics (SoF, survival)
carry no signal on SLB; the cell-line functional-examination and co-expression statistics do.

## SLB-1.3 dev results (results/models/slb1.3/; SLB_BENCH=data/bench/slb1.3)
Headline = mean of human/scer/spom; auxiliary columns are the per-species strata (bsub/cele/dmel have < 20 dev positives, not in the headline). Species a model does not score get a constant (AUROC 0.5); `native` lists the species actually scored (from <name>_dev.coverage.json).

| model | SLB | human | scer | spom | bsub | cele | dmel | paralog | native |
|---|---|---|---|---|---|---|---|---|---|
| statsl__coexp_ccle | 0.5387 | 0.6162 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5558 | human |
| statsl__coexp_tcga | 0.5163 | 0.5489 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5315 | human |
| statsl__ess_crispr_cn | 0.5308 | 0.5925 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.6004 | human |
| statsl__ess_crispr_expr | 0.5527 | 0.6582 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.6244 | human |
| statsl__ess_crispr_mut | 0.5258 | 0.5773 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5518 | human |
| statsl__ess_rnai_cn | 0.5023 | 0.5070 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5329 | human |
| statsl__ess_rnai_expr | 0.5119 | 0.5356 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5591 | human |
| statsl__ess_rnai_mut | 0.5095 | 0.5286 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5302 | human |
| statsl__phylo_neg_dist | 0.5323 | 0.5969 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.6106 | human |
| statsl__sof_cna | 0.5050 | 0.5149 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5186 | human |
| statsl__sof_expr | 0.4986 | 0.4959 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5056 | human |
| statsl__surv_cna | 0.4851 | 0.4554 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.4740 | human |
| statsl__surv_expr | 0.5052 | 0.5155 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | human |

Reference on SLB-1.3 dev: lgbm 0.5315 (human 0.558), paralog_identity 0.5392 (human 0.608), codependency 0.5149, random 0.4854.
