# DDGCN (dual-dropout GCN)

battery: ddgcn (human, faithful inputs), ddgcn__allspecies (bundle inputs, every benchmark species); species: human + all bundle species; needs: identity node features + the training SL graph only (no side information)

- Paper: Cai et al., Bioinformatics (2020), doi:10.1093/bioinformatics/btaa211.
- Original repo: https://github.com/CXX1113/Dual-DropoutGCN @ 8bde3877614a3805bcc0498b9875d8c5bc8401d9 (no license stated). Run here through the Feng et al. 2024 unified implementation (`DDGCN`), see notes/models/feng_suite.md for the shared pipeline, patches and leakage audit.
- Released weights: none. Original training data: SynLethDB v1 (SL_Human_Approved.txt, 6,375 genes) (SynLethDB labels include SLB source screens, so no released model could be scored anyway).
- Inputs on SLB: identity node features + the training SL graph only (no side information).
- Leakage status: **clean**. Trained only on SLB human train pairs (fit families; valid families for model selection); dev labels never read. PPI / KG sources and GI removal: PPI = BioGRID 5.0.261 physical rows only (no STRING, no genetic rows); GO similarity from NCBI gene2go with IGI and NOT annotations removed; KG (KG models only) = Feng's SynLethKG copy with the SL_GsG / SR_GsrG / NONSL_GnsG relations verified absent; its gene-gene relations are Hetionet INTERACTS_GiG (physical interactome), COVARIES_GcG (evolutionary rate covariation) and REGULATES_GrG (LINCS knock-down signatures), no STRING; residual risk: SynLethKG's gene-GO edges are not evidence-filtered (possible IGI). allspecies variants: bundle ppi.parquet (BioGRID physical + STRING non-experimental, non-textmining channels) and bundle GO (IGI/ND/NOT dropped).
- Status: acquired, env built, adapted, dev-scored (see Results).
- Adapter: scripts/models/ddgcn/run.sh (SLB_BENCH, SLB_SPLIT, SLB_VARIANTS) -> scripts/models/feng_suite/run_model.sh.
- Notes: Structurally cannot represent a gene without training SL edges: held-out genes keep their random initial identity embedding, so dev scores are nearly constant (SLB-1.3: 0.70703-0.70711, 449 distinct values) and their ranking is a fixed random gene order. The SLB-1.3 human 0.600 is carried by the AFR group (0.72 on 26 positives from 3 lines) and should be read as chance. SLB-1.3 run stopped at epoch ~840 of max 2000 (CPU time budget); the best-validation-F1 checkpoint matrix saved by the code was used.

## Results (rows a variant does not score get the model's median score before `slpbench eval`, i.e. what `--allow-missing` does; for human-only variants the other species therefore sit at 0.5)
### SLB1.3 dev
- ddgcn: **SLB 0.5334**; H. sapiens 0.6003, S. cerevisiae 0.5000, S. pombe 0.5000, B. subtilis n/a, C. elegans n/a, D. melanogaster n/a, M. musculus n/a. Human ancestry: AFR 0.7195, EAS 0.5320, EUR 0.5493, unknown 0.5862. Rows scored by the model: 19,149 (the rest get the model's median score).
### SLB1.2 dev
- not scored on this version.
