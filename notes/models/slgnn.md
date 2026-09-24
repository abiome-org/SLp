# SLGNN (factor-aware knowledge-graph neural network)

battery: slgnn (human, faithful inputs); species: human; needs: SynLethKG without SL relations (factor-aware aggregation, 4 factors, 3 hops) + training SL graph

- Paper: Zhu et al., Bioinformatics (2023), doi:10.1093/bioinformatics/btad015.
- Original repo: https://github.com/zy972014452/SLGNN @ c56e23fbd20e825c5571ef3e94377944c669ee80 (MIT). Run here through the Feng et al. 2024 unified implementation (`SLGNN`), see notes/models/feng_suite.md for the shared pipeline, patches and leakage audit.
- Released weights: none. Original training data: SynLethDB 2.0 + SynLethKG (24 relation types, no SL) (SynLethDB labels include SLB source screens, so no released model could be scored anyway).
- Inputs on SLB: SynLethKG without SL relations (factor-aware aggregation, 4 factors, 3 hops) + training SL graph.
- Leakage status: **clean**. Trained only on SLB human train pairs (fit families; valid families for model selection); dev labels never read. PPI / KG sources and GI removal: PPI = BioGRID 5.0.261 physical rows only (no STRING, no genetic rows); GO similarity from NCBI gene2go with IGI and NOT annotations removed; KG (KG models only) = Feng's SynLethKG copy with the SL_GsG / SR_GsrG / NONSL_GnsG relations verified absent; its gene-gene relations are Hetionet INTERACTS_GiG (physical interactome), COVARIES_GcG (evolutionary rate covariation) and REGULATES_GrG (LINCS knock-down signatures), no STRING; residual risk: SynLethKG's gene-GO edges are not evidence-filtered (possible IGI). allspecies variants: bundle ppi.parquet (BioGRID physical + STRING non-experimental, non-textmining channels) and bundle GO (IGI/ND/NOT dropped).
- Status: acquired, env built, adapted, dev-scored (see Results).
- Adapter: scripts/models/slgnn/run.sh (SLB_BENCH, SLB_SPLIT, SLB_VARIANTS) -> scripts/models/feng_suite/run_model.sh.
- Notes: Human-only (SynLethKG).

## Results (rows a variant does not score get the model's median score before `slpbench eval`, i.e. what `--allow-missing` does; for human-only variants the other species therefore sit at 0.5)
### SLB1.3 dev
- slgnn: **SLB 0.4695**; H. sapiens 0.4085, S. cerevisiae 0.5000, S. pombe 0.5000, B. subtilis n/a, C. elegans n/a, D. melanogaster n/a, M. musculus n/a. Human ancestry: AFR 0.3473, EAS 0.4128, EUR 0.4653, unknown 0.4388. Rows scored by the model: 19,149 (the rest get the model's median score).
### SLB1.2 dev
- slgnn: **SLB 0.5050**; H. sapiens 0.5198, S. cerevisiae 0.5000, S. pombe 0.5000, S. pneumoniae 0.5000. Human ancestry: AFR 0.5636, EAS 0.4657, EUR 0.5302, unknown 0.5123. Rows scored by the model: 15,048 (the rest get the model's median score).
- Runtime (SLB-1.2, CPU container, 8-16 threads): 53 min.
- Comment: The Feng implementation's final step scores all N(N-1)/2 = 78.5 M gene pairs (~4 h on 8 CPU threads); patched to score only the evaluated pairs (fit/valid train pairs + dev/test-input pairs, scripts/models/feng_suite/make_score_pairs.py), which gives identical scores for those pairs. (A GPU run was not possible: the dgl-cu117 wheel needs CUDA-11 libraries missing from the CUDA-12 NGC image.)
