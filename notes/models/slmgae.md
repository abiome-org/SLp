# SLMGAE (multi-view graph auto-encoder)

battery: slmgae (human, faithful inputs), slmgae__allspecies (bundle inputs, every benchmark species); species: human + all bundle species; needs: views: GO BP kNN-45 graph, GO CC kNN-45 graph, PPI graph (support views) + training SL graph (main view)

- Paper: Hao et al., IEEE JBHI (2021), doi:10.1109/JBHI.2021.3079302.
- Original repo: https://github.com/DiNg1011/SLMGAE @ ba0c7016922fbec569f8adda5153c28a5392ec58 (no license stated). Run here through the Feng et al. 2024 unified implementation (`SLMGAE`), see notes/models/feng_suite.md for the shared pipeline, patches and leakage audit.
- Released weights: none. Original training data: SynLethDB v1 (6,375 genes) + GO BP/CC similarity + BioGRID PPI; breast-cancer set as CMF-W (SynLethDB labels include SLB source screens, so no released model could be scored anyway).
- Inputs on SLB: views: GO BP kNN-45 graph, GO CC kNN-45 graph, PPI graph (support views) + training SL graph (main view).
- Leakage status: **clean**. Trained only on SLB human train pairs (fit families; valid families for model selection); dev labels never read. PPI / KG sources and GI removal: PPI = BioGRID 5.0.261 physical rows only (no STRING, no genetic rows); GO similarity from NCBI gene2go with IGI and NOT annotations removed; KG (KG models only) = Feng's SynLethKG copy with the SL_GsG / SR_GsrG / NONSL_GnsG relations verified absent; its gene-gene relations are Hetionet INTERACTS_GiG (physical interactome), COVARIES_GcG (evolutionary rate covariation) and REGULATES_GrG (LINCS knock-down signatures), no STRING; residual risk: SynLethKG's gene-GO edges are not evidence-filtered (possible IGI). allspecies variants: bundle ppi.parquet (BioGRID physical + STRING non-experimental, non-textmining channels) and bundle GO (IGI/ND/NOT dropped).
- Status: acquired, env built, adapted, dev-scored (see Results).
- Adapter: scripts/models/slmgae/run.sh (SLB_BENCH, SLB_SPLIT, SLB_VARIANTS) -> scripts/models/feng_suite/run_model.sh.
- Notes: Best overall model in Feng et al. 2024.

## Results (rows a variant does not score get the model's median score before `slpbench eval`, i.e. what `--allow-missing` does; for human-only variants the other species therefore sit at 0.5)
### SLB1.3 dev
- slmgae: **SLB 0.4867**; H. sapiens 0.4600, S. cerevisiae 0.5000, S. pombe 0.5000, B. subtilis n/a, C. elegans n/a, D. melanogaster n/a, M. musculus n/a. Human ancestry: AFR 0.3890, EAS 0.4505, EUR 0.5405, unknown 0.3338. Rows scored by the model: 19,149 (the rest get the model's median score).
- slmgae__allspecies: **SLB 0.5615**; H. sapiens 0.5043, S. cerevisiae 0.5761, S. pombe 0.6041, B. subtilis n/a, C. elegans n/a, D. melanogaster n/a, M. musculus n/a. Human ancestry: AFR 0.4518, EAS 0.5031, EUR 0.5581, unknown 0.3824. Rows scored by the model: 196,523 (the rest get the model's median score).
### SLB1.2 dev
- slmgae: **SLB 0.4981**; H. sapiens 0.4926, S. cerevisiae 0.5000, S. pombe 0.5000, S. pneumoniae 0.5000. Human ancestry: AFR 0.5156, EAS 0.4413, EUR 0.5207, unknown 0.3586. Rows scored by the model: 15,048 (the rest get the model's median score).
- Runtime (SLB-1.2, CPU container, 8-16 threads): 33 min.
