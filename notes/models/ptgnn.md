# PT-GNN (pre-trained GNN for biomedical link prediction)

battery: ptgnn (human, faithful inputs); species: human; needs: protein-sequence 3-mer word encodings (CNN) + GAT over the training SL graph

- Paper: Long et al., Bioinformatics (2022), doi:10.1093/bioinformatics/btac100.
- Original repo: https://github.com/longyahui/PT-GNN @ 4767a8be6f3f82318dc20060591f16520ac2741d (no license; no SL fine-tuning script, pre-training data absent). Run here through the Feng et al. 2024 unified implementation (`PTGNN`), see notes/models/feng_suite.md for the shared pipeline, patches and leakage audit.
- Released weights: none. Original training data: SynLethDB v1 + PPI/GO pre-training graphs (SynLethDB labels include SLB source screens, so no released model could be scored anyway).
- Inputs on SLB: protein-sequence 3-mer word encodings (CNN) + GAT over the training SL graph; Feng's fine-tuning does not restore the pre-trained checkpoint (the restore is commented out), so it is trained from scratch here too.
- Leakage status: **clean**. Trained only on SLB human train pairs (fit families; valid families for model selection); dev labels never read. PPI / KG sources and GI removal: PPI = BioGRID 5.0.261 physical rows only (no STRING, no genetic rows); GO similarity from NCBI gene2go with IGI and NOT annotations removed; KG (KG models only) = Feng's SynLethKG copy with the SL_GsG / SR_GsrG / NONSL_GnsG relations verified absent; its gene-gene relations are Hetionet INTERACTS_GiG (physical interactome), COVARIES_GcG (evolutionary rate covariation) and REGULATES_GrG (LINCS knock-down signatures), no STRING; residual risk: SynLethKG's gene-GO edges are not evidence-filtered (possible IGI). allspecies variants: bundle ppi.parquet (BioGRID physical + STRING non-experimental, non-textmining channels) and bundle GO (IGI/ND/NOT dropped).
- Status: acquired, env built, adapted, dev-scored (see Results).
- Adapter: scripts/models/ptgnn/run.sh (SLB_BENCH, SLB_SPLIT, SLB_VARIANTS) -> scripts/models/feng_suite/run_model.sh.
- Notes: Needs UniProt protein sequences (human encodings only prepared).

## Results (rows a variant does not score get the model's median score before `slpbench eval`, i.e. what `--allow-missing` does; for human-only variants the other species therefore sit at 0.5)
### SLB1.3 dev
- ptgnn: **SLB 0.5006**; H. sapiens 0.5019, S. cerevisiae 0.5000, S. pombe 0.5000, B. subtilis n/a, C. elegans n/a, D. melanogaster n/a, M. musculus n/a. Human ancestry: AFR 0.5012, EAS 0.5069, EUR 0.4976, unknown 0.4348. Rows scored by the model: 19,149 (the rest get the model's median score).
### SLB1.2 dev
- not scored on this version.
