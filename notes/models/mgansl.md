# MGANSL (2026)

**Sources:** [Li et al., *BMC Bioinformatics* 27, 27 (2026)](https://doi.org/10.1186/s12859-025-06345-4),
[official code](https://github.com/lijinxinchina/MGANSL), acquired at
`external/models/mgansl` (ignored by git; upstream commit
`caa5c642e378f46ce0a3ee526c13a585b10ad2ae`). The repository has no
released weights or feature-generation program.

The model fuses three pairwise network representations: HPRD protein
interactions, Gene Ontology similarity, and Gaussian similarity calculated
from **known synthetic-lethal interactions**. Its released datasets use
SynLethDB and SynLethDB 2.0 positives and randomly sampled unobserved pairs
as negatives. The repo ships 128-dimensional per-gene PPI, GO and Gaussian
features for 6,375 SynLethDB v1 genes only. Its actual supervised network is
an MLP/GAN on those precomputed vectors, rather than a graph neural network.
The SL-network input is label-derived, and its precomputed Gaussian features
cannot be used directly on SLB held-out families. A clean adaptation requires
constructing the Gaussian network from SLB train labels only, excluding
external interaction labels and regenerating all three network views for
SLB's experimentally measured pairs. The official repository does not supply
that pipeline, and its feature vocabulary does not cover all SLB genes.
No MGANSL score is placed on the ranked leaderboard.
