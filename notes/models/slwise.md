# SLWise (graph-based cell-specific SL prediction)

battery: none (not adapted); species: -; needs: per-cell-line graphs (mutual exclusivity, low-expression /
low-gene-effect DepMap, paralogs, L1000 fold change), node features all ones

- Paper: "Using graph-based model to identify cell specific synthetic lethal effects", bioRxiv
  10.1101/2023.07.23.550246.
- Repo: https://github.com/promethiume/SLwise @ e61e1f6f17dc68c9255bad8c2cd6bd0d98e68544. License: none stated.
  Weights: none shipped (code loads A549cv1_0.pkl etc. from a hard-coded /home/intern path).
- Original data: A549, A375, HT29 SL labels (figshare 10.6084/m9.figshare.24268510.v1, not in the repo);
  torch 1.7.1 + PyG 1.6.3 GraphSAGE + transformer cross-attention.
- Status: **acquired; not adapted.** Several absolute paths are hard-coded; per-cell-line graph views include L1000
  fold-change graphs that exist only for L1000-profiled genes in 3 lines. Lower priority than the Feng suite; a
  re-implementation on SLB would need these per-context graphs for 50 lines.
- Leakage status: n/a (not run).
