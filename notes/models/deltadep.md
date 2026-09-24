# deltadep / depmap_ols: DepMap driver-loss -> paralog dependency scans (2026 GitHub tools)

battery: deltadep__mut, depmap_ols__loss; species: human; needs: DepMap 24Q4 (CRISPR, expression, CN, damaging mutations, Model lineage)

PPI/KG: none.

- deltadep__mut: paralogSL R package, "Delta Dependency" (Mo Q, Zhu T, manuscript 2026; https://github.com/tjogzt/paralogSL commit 55b63f7, MIT; companion analysis repo tjogzt/paralog-sl-predictor commit 91a8052). DD(D,P) = mean Chronos(P | D WT) - mean Chronos(P | D mutant); re-implemented (compute_dd is a two-group mean difference) with D-mutant = damaging mutation; max over orientations; needs >= 3 mutant lines.
- depmap_ols__loss: RespAsahikawaMedicalUniv/DepMap-based-in-silico-screening-of-synthetic-lethal-paralogs (commit 02ab5ed, MIT; DepMap 25Q3 OLS scanner notebooks): OLS Chronos(P) ~ loss(D) + lineage + expr(P) + CN(P); score = -t(loss). loss = damaging mutation OR deep deletion OR no expression.
- Leakage: no SL labels => **not leaky**. Human only. Status: acquired / re-implemented / dev-scored.
- Dev: deltadep__mut SLB 0.5242, human 0.5966 (AFR .665 EAS .643 EUR .481; 3,884 human rows unscored: no mutant lines) | depmap_ols__loss SLB 0.5373, human 0.6492 (AFR .739 EAS .643 EUR .566; 104 human rows unscored).
- Runtime ~10 min (per-pair OLS loop).

## SLB-1.3 dev results (results/models/slb1.3/; SLB_BENCH=data/bench/slb1.3)
Headline = mean of human/scer/spom; auxiliary columns are the per-species strata (bsub/cele/dmel have < 20 dev positives, not in the headline). Species a model does not score get a constant (AUROC 0.5); `native` lists the species actually scored (from <name>_dev.coverage.json).

| model | SLB | human | scer | spom | bsub | cele | dmel | paralog | native |
|---|---|---|---|---|---|---|---|---|---|
| deltadep__mut | 0.5317 | 0.5951 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5727 | human |
| depmap_ols__loss | 0.5577 | 0.6731 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.6327 | human |

Reference on SLB-1.3 dev: lgbm 0.5315 (human 0.558), paralog_identity 0.5392 (human 0.608), codependency 0.5149, random 0.4854.
