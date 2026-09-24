# GCATSL (graph contextualised attention network)

battery: gcatsl (human, faithful inputs), gcatsl__allspecies (bundle inputs, every benchmark species); species: human + all bundle species; needs: PPI, GO BP, GO CC as node features (PCA-128) + local (training SL graph) and global (random-walk-with-restart on it) neighbourhoods, node- and feature-level attention

- Paper: Long et al., Bioinformatics (2021), doi:10.1093/bioinformatics/btab110.
- Original repo: https://github.com/lichenbiostat/GCATSL @ 1ad960e03a0b9bc4f0c3f35db66b5b45d53bad3b (MIT; Zenodo 10.5281/zenodo.4522679). Run here through the Feng et al. 2024 unified implementation (`GCATSL`), see notes/models/feng_suite.md for the shared pipeline, patches and leakage audit.
- Released weights: none. Original training data: SynLethDB v1 (6,375 genes) with BioGRID PPI, GO BP, GO CC feature graphs (SynLethDB labels include SLB source screens, so no released model could be scored anyway).
- Inputs on SLB: PPI, GO BP, GO CC as node features (PCA-128) + local (training SL graph) and global (random-walk-with-restart on it) neighbourhoods, node- and feature-level attention.
- Leakage status: **clean**. Trained only on SLB human train pairs (fit families; valid families for model selection); dev labels never read. PPI / KG sources and GI removal: PPI = BioGRID 5.0.261 physical rows only (no STRING, no genetic rows); GO similarity from NCBI gene2go with IGI and NOT annotations removed; KG (KG models only) = Feng's SynLethKG copy with the SL_GsG / SR_GsrG / NONSL_GnsG relations verified absent; its gene-gene relations are Hetionet INTERACTS_GiG (physical interactome), COVARIES_GcG (evolutionary rate covariation) and REGULATES_GrG (LINCS knock-down signatures), no STRING; residual risk: SynLethKG's gene-GO edges are not evidence-filtered (possible IGI). allspecies variants: bundle ppi.parquet (BioGRID physical + STRING non-experimental, non-textmining channels) and bundle GO (IGI/ND/NOT dropped).
- Status: acquired, env built, adapted, dev-scored (see Results).
- Adapter: scripts/models/gcatsl/run.sh (SLB_BENCH, SLB_SPLIT, SLB_VARIANTS) -> scripts/models/feng_suite/run_model.sh.
- Notes: The released train_gcatsl.py only builds its global random-walk matrix on the first call and skips training (`if build_premat==1: continue`); run_model.sh therefore calls it twice. RWR made sparse with early exit (same fixed point). CPU: 187 s/epoch (x200); SLB-1.3 was run on the GPU image (15 s/epoch) with a 40-min cap: stopped at epoch 82 of 200, best-validation-F1 score matrix used. Not scored on SLB-1.2 (CPU too slow; GPU slot used for SLB-1.3).

## Results (rows a variant does not score get the model's median score before `slpbench eval`, i.e. what `--allow-missing` does; for human-only variants the other species therefore sit at 0.5)
### SLB1.3 dev
- gcatsl: **SLB 0.4887**; H. sapiens 0.4661, S. cerevisiae 0.5000, S. pombe 0.5000, B. subtilis n/a, C. elegans n/a, D. melanogaster n/a, M. musculus n/a. Human ancestry: AFR 0.4862, EAS 0.4939, EUR 0.4181, unknown 0.6621. Rows scored by the model: 19,149 (the rest get the model's median score).
### SLB1.2 dev
- not scored on this version.
