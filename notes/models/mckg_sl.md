# MCKG-SL (2026)

[Publication](https://doi.org/10.1016/j.artmed.2026.103519),
[official code](https://github.com/Qian0711/MCKG-SL) at
`external/models/mckg_sl` (ignored by git). The repository releases C1/C2/C3
folds, `kg2id.txt`, `entity.txt`, 600-dimensional omics features keyed by
integer entity IDs, and code for local pair subgraph extraction with RGCN/GAT.

The graph and omics files do not include a gene-symbol/Entrez-to-integer
entity map. `entity.txt` lists only entity types, and the 9,517 omics vectors
are keyed only by KG integers. The same test gene pair cannot be mapped to
the model's graph, so the released inputs do not define an SLB prediction.
The code also hard-codes 54,012 nodes and a CUDA device. The paper's C1
random-pair results are not a family-held-out evaluation. A fresh graph and
omics construction would change the released model's inputs and requires
data provenance beyond the repository. No score is placed on the leaderboard.
