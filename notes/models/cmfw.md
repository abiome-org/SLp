# CMF-W (weighted collective matrix factorisation)

battery: cmfw (human, faithful inputs), cmfw__allspecies (bundle inputs, every benchmark species); species: human + all bundle species; needs: SL matrix (training pairs) jointly factorised with GO BP, GO CC and PPI matrices (TensorFlow)

- Paper: Liany et al., Bioinformatics 36:2209-2216 (2020), doi:10.1093/bioinformatics/btz893.
- Original repo: https://github.com/lianyh/CMF-W @ 8ac0ceedd82a88d8146af320f1887af517b39b7e (no license stated). Run here through the Feng et al. 2024 unified implementation (`CMFW`), see notes/models/feng_suite.md for the shared pipeline, patches and leakage audit.
- Released weights: none. Original training data: 332-gene breast-cancer matrices (SL, co-expression, mutual exclusivity, pathway, PPI, complex); Feng's version uses GO BP/CC + PPI (SynLethDB labels include SLB source screens, so no released model could be scored anyway).
- Inputs on SLB: SL matrix (training pairs) jointly factorised with GO BP, GO CC and PPI matrices (TensorFlow).
- Leakage status: **clean**. Trained only on SLB human train pairs (fit families; valid families for model selection); dev labels never read. PPI / KG sources and GI removal: PPI = BioGRID 5.0.261 physical rows only (no STRING, no genetic rows); GO similarity from NCBI gene2go with IGI and NOT annotations removed; KG (KG models only) = Feng's SynLethKG copy with the SL_GsG / SR_GsrG / NONSL_GnsG relations verified absent; its gene-gene relations are Hetionet INTERACTS_GiG (physical interactome), COVARIES_GcG (evolutionary rate covariation) and REGULATES_GrG (LINCS knock-down signatures), no STRING; residual risk: SynLethKG's gene-GO edges are not evidence-filtered (possible IGI). allspecies variants: bundle ppi.parquet (BioGRID physical + STRING non-experimental, non-textmining channels) and bundle GO (IGI/ND/NOT dropped).
- Status: acquired, env built, adapted, dev-scored (see Results).
- Adapter: scripts/models/cmfw/run.sh (SLB_BENCH, SLB_SPLIT, SLB_VARIANTS) -> scripts/models/feng_suite/run_model.sh.

## Results (rows a variant does not score get the model's median score before `slpbench eval`, i.e. what `--allow-missing` does; for human-only variants the other species therefore sit at 0.5)
### SLB1.3 dev
- cmfw: **SLB 0.5079**; H. sapiens 0.5236, S. cerevisiae 0.5000, S. pombe 0.5000, B. subtilis n/a, C. elegans n/a, D. melanogaster n/a, M. musculus n/a. Human ancestry: AFR 0.5647, EAS 0.4673, EUR 0.5387, unknown 0.6044. Rows scored by the model: 19,149 (the rest get the model's median score).
### SLB1.2 dev
- not scored on this version.
- Runtime (SLB-1.2, CPU container, 8-16 threads): 0 min.
