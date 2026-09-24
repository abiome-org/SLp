# NSF4SL (negative-sample-free contrastive learning)

battery: nsf4sl (human, faithful inputs); species: human; needs: per-gene TransE_l2 (dim 400) SynLethKG embeddings, re-trained here on the SL-free KG (scripts/models/feng_suite/transe.py)

- Paper: Wang et al., Bioinformatics (2022, ECCB suppl.), doi:10.1093/bioinformatics/btac462.
- Original repo: https://github.com/JieZheng-ShanghaiTech/NSF4SL @ de000dec9993286e3228331c51a543668f6c832d (MIT). Run here through the Feng et al. 2024 unified implementation (`NSF4SL`), see notes/models/feng_suite.md for the shared pipeline, patches and leakage audit.
- Released weights: none. Original training data: SynLethDB 2.0 (CV1/CV2/CV3 splits) + SynLethKG TransE embeddings (shipped) (SynLethDB labels include SLB source screens, so no released model could be scored anyway).
- Inputs on SLB: per-gene TransE_l2 (dim 400) SynLethKG embeddings, re-trained here on the SL-free KG (scripts/models/feng_suite/transe.py); BUIR-style online/target encoders trained on positive pairs only.
- Leakage status: **clean**. Trained only on SLB human train pairs (fit families; valid families for model selection); dev labels never read. PPI / KG sources and GI removal: PPI = BioGRID 5.0.261 physical rows only (no STRING, no genetic rows); GO similarity from NCBI gene2go with IGI and NOT annotations removed; KG (KG models only) = Feng's SynLethKG copy with the SL_GsG / SR_GsrG / NONSL_GnsG relations verified absent; its gene-gene relations are Hetionet INTERACTS_GiG (physical interactome), COVARIES_GcG (evolutionary rate covariation) and REGULATES_GrG (LINCS knock-down signatures), no STRING; residual risk: SynLethKG's gene-GO edges are not evidence-filtered (possible IGI). allspecies variants: bundle ppi.parquet (BioGRID physical + STRING non-experimental, non-textmining channels) and bundle GO (IGI/ND/NOT dropped).
- Status: acquired, env built, adapted, dev-scored (see Results).
- Adapter: scripts/models/nsf4sl/run.sh (SLB_BENCH, SLB_SPLIT, SLB_VARIANTS) -> scripts/models/feng_suite/run_model.sh.
- Notes: Uses positives only (negative-sample-free); measured negatives are ignored by design. map_genes() patched to index all nodes (the original only works when every node has an SL pair).

## Results (rows a variant does not score get the model's median score before `slpbench eval`, i.e. what `--allow-missing` does; for human-only variants the other species therefore sit at 0.5)
### SLB1.3 dev
- nsf4sl: **SLB 0.5170**; H. sapiens 0.5509, S. cerevisiae 0.5000, S. pombe 0.5000, B. subtilis n/a, C. elegans n/a, D. melanogaster n/a, M. musculus n/a. Human ancestry: AFR 0.6097, EAS 0.5082, EUR 0.5348, unknown 0.4567. Rows scored by the model: 19,149 (the rest get the model's median score).
### SLB1.2 dev
- not scored on this version.
