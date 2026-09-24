# isle: ISLE (Lee et al. 2018, Nat Commun 9:2546)

battery: isle; species: human; needs: see statsl_common.md, external/models/isle (phylogenetic profile + weights)

PPI/KG: none (DepMap, DEMETER2, TCGA, Tabach phylogenetic profiles).

- Paper: Lee JS, ..., Ruppin E. Harnessing synthetic lethality to predict the response to cancer treatment. Nat Commun 9:2546 (2018). doi:10.1038/s41467-018-04647-1
- Repo: https://github.com/jooslee/ISLE (commit 4e8b61d03825ad6890cee0f85b06ed9a30fa5362), external/models/isle. No license file.
- Blocker for the original R run: the required TCGA object `prob.TCGA.RData` (ftp://ftp.umiacs.umd.edu/pub/jooslee/, also
  referenced at hpc.nih.gov/~leej55) is gone (FTP 550 / HTTP 404, checked 2026-09-24). Its shRNA screen objects and
  phylogenetic profiles are in the repo. We therefore re-implemented isle.r's steps on current data (statsl engine).
- Original pipeline also seeds the candidate pool with a literature "gold standard" SL set (`sl.golden.set.RData`) -- that
  is SL-label input and was NOT used; SLB pairs themselves form the candidate pool.
- Steps (FDR 0.2 as in isle.r): (i) in-vitro functional examination (DepMap CRISPR + DEMETER2; ISLE used five 2011-2016
  shRNA/CRISPR screens); (ii) co-inactivation under-represented in TCGA for BOTH mRNA (tertile within cancer type) and
  SCNA (GISTIC <= -1; ISLE used SCNA tertiles); (iii) co-inactivation associated with better survival (stratified
  log-rank on cancer type instead of isle.r's Cox models with age/sex/race/GII covariates); (iv) phylogenetic distance
  below the median. Score = number of consecutive steps passed + 0.999 x mean percentile of the step statistics.
- Leakage: no SL labels => **not leaky**. Human only.

## Status
acquired / original blocked (data gone) / re-implemented / dev-scored

## Dev result
SLB 0.5360; H. sapiens 0.6441 (AFR 0.678, EAS 0.685, EUR 0.570); human flat 0.569; paralog stratum 0.656. All 15,048 human dev rows scored; 117,674 non-human median-filled. Runtime: pair statistics ~7 min for dev pairs (cached), scoring seconds.

## SLB-1.3 dev results (results/models/slb1.3/; SLB_BENCH=data/bench/slb1.3)
Headline = mean of human/scer/spom; auxiliary columns are the per-species strata (bsub/cele/dmel have < 20 dev positives, not in the headline). Species a model does not score get a constant (AUROC 0.5); `native` lists the species actually scored (from <name>_dev.coverage.json).

| model | SLB | human | scer | spom | bsub | cele | dmel | paralog | native |
|---|---|---|---|---|---|---|---|---|---|
| isle | 0.5390 | 0.6169 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.6186 | human |

Reference on SLB-1.3 dev: lgbm 0.5315 (human 0.558), paralog_identity 0.5392 (human 0.608), codependency 0.5149, random 0.4854.
