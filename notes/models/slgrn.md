# SLGRN (Graph-based Recurrent Network for context-specific SL)

battery: none (not adaptable); species: -; needs: per-gene L1000 A549 knockdown signatures (978-d), a
gene-gene similarity matrix and KEGG/Reactome/GO embeddings for a fixed gene set

- Paper: Science China Life Sciences 2025, doi:10.1007/s11427-023-2618-y.
- Repo: https://github.com/jyygit/SLGRN @ eebb62f84e5430b6220a5615d5962264dd66496d. License: none stated.
- Weights: data/model/12.08_13.15._5-0.best.pt (+ bestparam.pt), trained on A549 fold 0 of the shipped labels.
- Original data: A549 only, 122 genes / 1,141 positive pairs (data/label/A549.csv; source not stated), negatives
  sampled from unlabelled pairs; features fea1 (978-d, matches L1000 landmark genes), a 122x122 similarity matrix,
  640-d pathway/GO embeddings. PyTorch 1.5.1, DGL GraphConv.
- Status: **acquired; not adapted.** The model is transductive over a fixed 122-gene matrix and its main node
  feature is an A549 L1000 knockdown signature that exists only for those genes; SLB has 50 contexts and ~6k human
  genes. Re-training would need L1000 (LINCS) knockdown signatures for SLB genes in each context, which do not exist
  for most genes/lines. The released weights only cover 122 A549 genes (few SLB dev pairs) and were trained on SL
  labels of unknown provenance (leaky).
- Leakage status: released weights leaky (trained on external SL labels). Not scored.
