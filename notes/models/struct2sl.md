# Struct2SL (AlphaFold2 structure + sequence + PPI embeddings, MLP)

battery: struct2sl (human, SLB-retrained, leak-free PPI block), struct2sl__released (authors' weights + STRING-derived features, LEAKY); species: human; needs: per-gene AlphaFold2-contact-map node2vec, SeqVec, STRING-physical node2vec embeddings (17,180 human genes)

- Paper: Computational and Structural Biotechnology Journal (2025), doi:10.1016/j.csbj.2025.04.012.
- Repo: https://github.com/hyr-hit/Struct2SL @ ac6a825e1d8bd297ece53dcdf31f9c00cfb5456f (no license stated; figshare data Apache-2.0).
- Released weights: figshare 33439123 bestmodel.pt (trained on SynLethDB 2.0 filtered pairs: leaky).
- Original training data: SynLethDB 2.0 filtered (23,749 positives, computational-only removed) + Human_nonSL + random negatives.
- Leakage status: **clean** for the SLB-trained variants (fit on SLB human train only; no SL/GI edges or GI-derived inputs; dev labels never read), except where noted below. PPI / KG sources and how GI evidence was removed: SLB-trained variant: BioGRID physical from the bundle (node2vec rebuilt); released variant: STRING v12 physical links incl. textmining (leaky).
- What we changed:
  - Structure (AlphaFold2 contact-map node2vec) and sequence (SeqVec) features: the authors' released per-gene files, unchanged. PPI feature: the released one is node2vec on STRING v12 *physical links*, whose combined score includes the experiments and textmining channels (lead's STRING warning); for the SLB-trained variant it is REBUILT as node2vec (repo defaults: 128-d, walk 80, 10 walks, window 10, p=q=1; PyG Node2Vec, scripts/models/struct2sl/ppi_node2vec.py) on BioGRID physical edges of the shared bundle (data/interim/bundle/human/ppi.parquet). Pairs with a gene outside the feature tables are left out (median-filled).
  - slbtrain: MLP trained on SLB fit pairs with measured negatives, early stopping on SLB valid; released: authors' checkpoint, z-score statistics re-estimated from their training pairs (not released).
  - Released code applies BCEWithLogits to an already-sigmoided output (double sigmoid); kept as is.
- Adapter: scripts/models/struct2sl/run.sh; env SLB_BENCH, SLB_SPLIT.

## Results (rows a model does not score get its median score before `slpbench eval`, as `--allow-missing` does)
### SLB1.3 dev
- struct2sl: **SLB 0.5236**; H. sapiens 0.5707, S. cerevisiae 0.5000, S. pombe 0.5000, B. subtilis n/a, C. elegans n/a, D. melanogaster n/a, M. musculus n/a. Human ancestry: AFR 0.6363, EAS 0.5571, EUR 0.5188, unknown 0.4323. Rows scored by the model: 16,825 (the rest get the model's median score).
- struct2sl__released: **SLB 0.4909**; H. sapiens 0.4726, S. cerevisiae 0.5000, S. pombe 0.5000, B. subtilis n/a, C. elegans n/a, D. melanogaster n/a, M. musculus n/a. Human ancestry: AFR 0.4762, EAS 0.4141, EUR 0.5275, unknown 0.3799. Rows scored by the model: 16,883 (the rest get the model's median score).
### SLB1.2 dev
- struct2sl: **SLB 0.4760**; H. sapiens 0.4041, S. cerevisiae 0.5000, S. pombe 0.5000, S. pneumoniae 0.5000. Human ancestry: AFR 0.3537, EAS 0.3895, EUR 0.4690, unknown 0.6150. Rows scored by the model: 13,022 (the rest get the model's median score).
- struct2sl__released: **SLB 0.4867**; H. sapiens 0.4466, S. cerevisiae 0.5000, S. pombe 0.5000, S. pneumoniae 0.5000. Human ancestry: AFR 0.3484, EAS 0.4151, EUR 0.5763, unknown 0.5562. Rows scored by the model: 13,080 (the rest get the model's median score).
- Status: acquired, env built, adapted, dev-scored.
