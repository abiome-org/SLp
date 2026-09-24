# slidr: SLIdR (Srivatsa et al. 2022, Nat Commun 13:2699)

battery: slidr__rnai, slidr__crispr, slidr__crispr_lossexpr; species: human; needs: DepMap 24Q4 (CRISPR, CN, damaging mutations, expression), DEMETER2

PPI/KG: none.

- Paper: Srivatsa S, Montazeri H, Bianco G, ..., Beerenwinkel N. Discovery of synthetic lethal interactions from large-scale pan-cancer perturbation screens. Nat Commun 13:2699 (2022). doi:10.1038/s41467-022-30446-5
- Repo: https://github.com/cbg-ethz/slidr (commit 01faebeb, R package, GPL-3.0), external/models/slidr. No weights (unsupervised).
- Re-implemented the package's test (identifySLHits/getPval, Irwin-Hall rank-sum statistic) in Python for SLB pairs, pan-cancer. Driver alteration = damaging mutation or deep deletion (log2 rel CN < 0.5). Score = -log10(mut_pvalue) with a -1000 penalty when WT_pvalue <= 0.1 (SLIdR's WT filter); max over orientations.
  - __rnai: DEMETER2 viabilities (paper used DRIVE shRNA); __crispr: Chronos; __crispr_lossexpr: extension, alteration also includes no expression (TPM log1p < 1) so paralog loss is covered.
- Leakage: no SL labels => **not leaky**. Human only.
- Status: acquired / re-implemented / dev-scored (R package not run: its per-driver loop over all genes is the same statistic; our vectorised version evaluates only SLB pairs).
- Dev (non-human median-filled; human coverage: 14,573 rnai, 14,970 crispr rows):
  - __rnai: SLB 0.4876, human 0.4502 | __crispr: 0.5132, 0.5527 | __crispr_lossexpr: 0.4989, 0.4956
- Runtime ~20 min for the three variants (8 threads).

## SLB-1.3 dev results (results/models/slb1.3/; SLB_BENCH=data/bench/slb1.3)
Headline = mean of human/scer/spom; auxiliary columns are the per-species strata (bsub/cele/dmel have < 20 dev positives, not in the headline). Species a model does not score get a constant (AUROC 0.5); `native` lists the species actually scored (from <name>_dev.coverage.json).

| model | SLB | human | scer | spom | bsub | cele | dmel | paralog | native |
|---|---|---|---|---|---|---|---|---|---|
| slidr__crispr | 0.5068 | 0.5205 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5383 | human |
| slidr__crispr_lossexpr | 0.4905 | 0.4714 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.4926 | human |
| slidr__rnai | 0.4952 | 0.4856 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5278 | human |

Reference on SLB-1.3 dev: lgbm 0.5315 (human 0.558), paralog_identity 0.5392 (human 0.608), codependency 0.5149, random 0.4854.
