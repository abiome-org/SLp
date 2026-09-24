# PiLSL (pairwise interaction learning GNN on enclosing subgraphs)

battery: pilsl (human, faithful inputs); species: human; needs: 3-hop enclosing subgraphs over SynLethKG (no SL relations) + relation 0 = training SL pairs (original PiLSL design

- Paper: Liu et al., Bioinformatics (2022, ECCB suppl.), doi:10.1093/bioinformatics/btac476.
- Original repo: https://github.com/JieZheng-ShanghaiTech/PiLSL @ 12d23be6ba3d0e75eeb33e0bdd0dc6428deff149 (MIT). Run here through the Feng et al. 2024 unified implementation (`PiLSL`), see notes/models/feng_suite.md for the shared pipeline, patches and leakage audit.
- Released weights: none. Original training data: SynLethDB (32,561 pairs, 9,516 genes) + SynLethKG + 600-d omics features (SynLethDB labels include SLB source screens, so no released model could be scored anyway).
- Inputs on SLB: 3-hop enclosing subgraphs over SynLethKG (no SL relations) + relation 0 = training SL pairs (original PiLSL design; Feng's release used all 48M gene pairs as relation 0); learnable entity embeddings; node features random (as Feng: add_feat_emb=False).
- Leakage status: **clean**. Trained only on SLB human train pairs (fit families; valid families for model selection); dev labels never read. PPI / KG sources and GI removal: PPI = BioGRID 5.0.261 physical rows only (no STRING, no genetic rows); GO similarity from NCBI gene2go with IGI and NOT annotations removed; KG (KG models only) = Feng's SynLethKG copy with the SL_GsG / SR_GsrG / NONSL_GnsG relations verified absent; its gene-gene relations are Hetionet INTERACTS_GiG (physical interactome), COVARIES_GcG (evolutionary rate covariation) and REGULATES_GrG (LINCS knock-down signatures), no STRING; residual risk: SynLethKG's gene-GO edges are not evidence-filtered (possible IGI). allspecies variants: bundle ppi.parquet (BioGRID physical + STRING non-experimental, non-textmining channels) and bundle GO (IGI/ND/NOT dropped).
- Status: acquired, env built, adapted (not scored: see Comment).
- Adapter: scripts/models/pilsl/run.sh (SLB_BENCH, SLB_SPLIT, SLB_VARIANTS) -> scripts/models/feng_suite/run_model.sh.
- Notes: Human-only (SynLethKG).

## Results (rows a variant does not score get the model's median score before `slpbench eval`, i.e. what `--allow-missing` does; for human-only variants the other species therefore sit at 0.5)
### SLB1.3 dev
- not scored on this version.
### SLB1.2 dev
- not scored on this version.
- Comment: NOT SCORED (time): inputs are built (scripts/models/feng_suite/build_pilsl_inputs.py: 76,797 train/valid/dev/test pairs to extract 3-hop enclosing subgraphs for, fit SL pairs as relation-0 edges, re-mapped entity types) and the code is patched (graph-vs-extraction pair lists, pair->db index, worker cap), but enclosing-subgraph extraction over the 2.2 M-triple KG plus 30 epochs of CPU-only dgl-0.4 R-GCN training was estimated at many hours and was not started within the session budget. Run: `SLB_BENCH=... bash scripts/models/pilsl/run.sh`.
