# KG4SL (knowledge-graph neural network)

battery: kg4sl (human, faithful inputs); species: human; needs: SynLethKG without SL/SR/non-SL relations (1-hop, 64 sampled neighbours, dim 256)

- Paper: Wang et al., Bioinformatics 37:i418 (2021), doi:10.1093/bioinformatics/btab271.
- Original repo: https://github.com/JieZheng-ShanghaiTech/KG4SL @ be3626a6970479affb95fdf6b1d84a6244e52637 (MIT). Run here through the Feng et al. 2024 unified implementation (`KG4SL`), see notes/models/feng_suite.md for the shared pipeline, patches and leakage audit.
- Released weights: none. Original training data: SynLethDB 2.0 (~36k pairs) + SynLethKG (its kg2id.txt already excludes SL relations) (SynLethDB labels include SLB source screens, so no released model could be scored anyway).
- Inputs on SLB: SynLethKG without SL/SR/non-SL relations (1-hop, 64 sampled neighbours, dim 256); pairs scored for all 78.5M gene pairs at the end.
- Leakage status: **clean**. Trained only on SLB human train pairs (fit families; valid families for model selection); dev labels never read. PPI / KG sources and GI removal: PPI = BioGRID 5.0.261 physical rows only (no STRING, no genetic rows); GO similarity from NCBI gene2go with IGI and NOT annotations removed; KG (KG models only) = Feng's SynLethKG copy with the SL_GsG / SR_GsrG / NONSL_GnsG relations verified absent; its gene-gene relations are Hetionet INTERACTS_GiG (physical interactome), COVARIES_GcG (evolutionary rate covariation) and REGULATES_GrG (LINCS knock-down signatures), no STRING; residual risk: SynLethKG's gene-GO edges are not evidence-filtered (possible IGI). allspecies variants: bundle ppi.parquet (BioGRID physical + STRING non-experimental, non-textmining channels) and bundle GO (IGI/ND/NOT dropped).
- Status: acquired, env built, adapted, dev-scored (see Results).
- Adapter: scripts/models/kg4sl/run.sh (SLB_BENCH, SLB_SPLIT, SLB_VARIANTS) -> scripts/models/feng_suite/run_model.sh.
- Notes: SynLethKG is a human KG: human-only (coverage fact). On SLB-1.2 it was trained with the Feng default for this pipeline (all measured negatives, 1:46): the loss fell to the base-rate entropy and every pair got the same score (0.02146), i.e. SLB 0.500. The SLB-1.3 run uses KG4SL's original 1:1 protocol (fit negatives subsampled to the number of positives, SLB_BALANCE=1, scripts/models/kg4sl/run.sh default).

## Results (rows a variant does not score get the model's median score before `slpbench eval`, i.e. what `--allow-missing` does; for human-only variants the other species therefore sit at 0.5)
### SLB1.3 dev
- kg4sl: **SLB 0.4683**; H. sapiens 0.4050, S. cerevisiae 0.5000, S. pombe 0.5000, B. subtilis n/a, C. elegans n/a, D. melanogaster n/a, M. musculus n/a. Human ancestry: AFR 0.2616, EAS 0.4450, EUR 0.5085, unknown 0.3670. Rows scored by the model: 19,149 (the rest get the model's median score).
### SLB1.2 dev
- kg4sl: **SLB 0.5000**; H. sapiens 0.5000, S. cerevisiae 0.5000, S. pombe 0.5000, S. pneumoniae 0.5000. Human ancestry: AFR 0.5000, EAS 0.5000, EUR 0.5000, unknown 0.5000. Rows scored by the model: 15,048 (the rest get the model's median score).
- Runtime (SLB-1.2, CPU container, 8-16 threads): 125 min.
