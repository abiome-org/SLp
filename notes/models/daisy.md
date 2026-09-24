# daisy: DAISY (Jerby-Arnon et al. 2014, Cell 158:1199)

battery: daisy; species: human (needs cell-line screens + TCGA); needs: see statsl_common.md

PPI/KG: none (DepMap, DEMETER2, TCGA only).

- Paper: Jerby-Arnon L, ..., Ruppin E. Predicting cancer-specific vulnerability via data-driven detection of synthetic lethality. Cell 2014. doi:10.1016/j.cell.2014.07.027
- Code: the paper's MATLAB/COBRA code (sourceforge "logictransformationofmodel") was not used: re-implemented from the paper's three inference procedures on current public data (statsl engine). No weights (unsupervised).
- Procedure: (1) survival of the fittest: co-inactivation under-represented in tumours (TCGA, expression or CNA);
  (2) functional examination: one gene is more essential when the other is inactive (DepMap CRISPR + DEMETER2 RNAi,
  inactive = low expression / low CN / damaging mutation); (3) positive co-expression (TCGA).
  Score = number of the three tests passed at p < 0.05 (ess: Bonferroni over the screens x alterations tried; SoF: min
  of the two, x2), tie-broken by the mean percentile of the three statistics. Pan-cancer (DAISY also has
  cancer-specific networks).
- Leakage: no SL labels anywhere => **not leaky**. Human only.

## Status
acquired (paper) / re-implemented / dev-scored. Deviations: modern data (DepMap 24Q4 Chronos + DEMETER2 instead of 2014 shRNA screens; TCGA PanCanAtlas instead of 2014 TCGA/CCLE); p-value thresholds fixed a priori, not tuned.

## Dev result
SLB 0.5299; H. sapiens 0.6195 (AFR 0.700, EAS 0.624, EUR 0.535); human flat 0.539; paralog stratum 0.644. Coverage: all 15,048 human dev rows scored (NaN statistics count as failed tests); 117,674 non-human rows median-filled.

## Run
`scripts/models/statsl/run.sh` (SLB_BENCH, SLB_SPLIT) -> results/models/{daisy,isle,statsl__*}_<split>.parquet

## SLB-1.3 dev results (results/models/slb1.3/; SLB_BENCH=data/bench/slb1.3)
Headline = mean of human/scer/spom; auxiliary columns are the per-species strata (bsub/cele/dmel have < 20 dev positives, not in the headline). Species a model does not score get a constant (AUROC 0.5); `native` lists the species actually scored (from <name>_dev.coverage.json).

| model | SLB | human | scer | spom | bsub | cele | dmel | paralog | native |
|---|---|---|---|---|---|---|---|---|---|
| daisy | 0.5325 | 0.5975 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.6052 | human |

Reference on SLB-1.3 dev: lgbm 0.5315 (human 0.558), paralog_identity 0.5392 (human 0.608), codependency 0.5149, random 0.4854.
