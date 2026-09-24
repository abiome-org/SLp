# SL2MF (logistic matrix factorisation with GO/PPI regularisation)

battery: sl2mf (human, faithful inputs), sl2mf__allspecies (bundle inputs, every benchmark species); species: human + all bundle species; needs: GO BP + GO CC Wang similarity and PPI-topology (cosine) similarity as graph-Laplacian regularisers (kNN 45)

- Paper: Liu et al., IEEE/ACM TCBB, doi:10.1109/TCBB.2019.2909908.
- Original repo: https://github.com/stephenliu0423/SL2MF @ 101a97938eaefb17d195e2227892579816acf3ee (no license stated; data on Google Drive). Run here through the Feng et al. 2024 unified implementation (`SL2MF`), see notes/models/feng_suite.md for the shared pipeline, patches and leakage audit.
- Released weights: none. Original training data: SynLethDB (SL_Human_FinalCheck), GO BP/CC semantic similarity, PPI topology similarity (SynLethDB labels include SLB source screens, so no released model could be scored anyway).
- Inputs on SLB: GO BP + GO CC Wang similarity and PPI-topology (cosine) similarity as graph-Laplacian regularisers (kNN 45); importance weight 50 on training SL pairs.
- Leakage status: **clean**. Trained only on SLB human train pairs (fit families; valid families for model selection); dev labels never read. PPI / KG sources and GI removal: PPI = BioGRID 5.0.261 physical rows only (no STRING, no genetic rows); GO similarity from NCBI gene2go with IGI and NOT annotations removed; KG (KG models only) = Feng's SynLethKG copy with the SL_GsG / SR_GsrG / NONSL_GnsG relations verified absent; its gene-gene relations are Hetionet INTERACTS_GiG (physical interactome), COVARIES_GcG (evolutionary rate covariation) and REGULATES_GrG (LINCS knock-down signatures), no STRING; residual risk: SynLethKG's gene-GO edges are not evidence-filtered (possible IGI). allspecies variants: bundle ppi.parquet (BioGRID physical + STRING non-experimental, non-textmining channels) and bundle GO (IGI/ND/NOT dropped).
- Status: acquired, env built, adapted, dev-scored (see Results).
- Adapter: scripts/models/sl2mf/run.sh (SLB_BENCH, SLB_SPLIT, SLB_VARIANTS) -> scripts/models/feng_suite/run_model.sh.
- Notes: Transductive in form (a latent vector per gene), but held-out genes are placed by the GO/PPI Laplacian terms, so they get informative scores.

## Results (rows a variant does not score get the model's median score before `slpbench eval`, i.e. what `--allow-missing` does; for human-only variants the other species therefore sit at 0.5)
### SLB1.3 dev
- sl2mf: **SLB 0.5298**; H. sapiens 0.5894, S. cerevisiae 0.5000, S. pombe 0.5000, B. subtilis n/a, C. elegans n/a, D. melanogaster n/a, M. musculus n/a. Human ancestry: AFR 0.6361, EAS 0.5761, EUR 0.5559, unknown 0.4809. Rows scored by the model: 19,149 (the rest get the model's median score).
- sl2mf__allspecies: **SLB 0.4715**; H. sapiens 0.4546, S. cerevisiae 0.4936, S. pombe 0.4663, B. subtilis n/a, C. elegans n/a, D. melanogaster n/a, M. musculus n/a. Human ancestry: AFR 0.4056, EAS 0.4537, EUR 0.5047, unknown 0.3652. Rows scored by the model: 196,523 (the rest get the model's median score).
### SLB1.2 dev
- sl2mf: **SLB 0.5248**; H. sapiens 0.5991, S. cerevisiae 0.5000, S. pombe 0.5000, S. pneumoniae 0.5000. Human ancestry: AFR 0.6141, EAS 0.6131, EUR 0.5701, unknown 0.5971. Rows scored by the model: 15,048 (the rest get the model's median score).
- Runtime (SLB-1.2, CPU container, 8-16 threads): 25 min.
