# KDDSL (knowledge-driven interpretable SL network, A549)

battery: none (not adapted); species: -; needs: 978-d A549 L1000 perturbation signatures per gene + GO hierarchy

- Paper: J Adv Res (2023), doi:10.1016/j.jare.2023.07.xxx (S2090123223003740). Repo:
  https://github.com/Wingswang728/KDDSL @ 7d4228d91571377863172851ca3422b8154de390 (no license). torch 1.7.1. No weights.
- Original data: Data/samples-A549.csv (567 pos / 1,080 neg, A549 only; source not stated); featurematrix-A549.csv =
  978 L1000 landmark genes x 126 perturbed genes. DCell-style GO-structured network (not a KG/graph-embedding model).
- Status: **acquired; not adapted.** Its only gene feature is an A549 L1000 knock-down signature, available for 126
  genes; SLB has ~6k human genes in 50 lines. The training script also references non-existent file names and an
  empty Data/gene2id. (The GO-visible-network family is covered by models-mechanistic's DCell/ontotype runs.)
