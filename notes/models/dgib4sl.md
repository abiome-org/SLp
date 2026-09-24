# DGIB4SL (Diverse Graph Information Bottleneck for SL)

battery: none (no code); species: -; needs: SynLethKG (motif-based high-order KG GNN)

- Paper: "Interpretable high-order knowledge graph neural network for predicting synthetic lethality in human
  cancers", Briefings in Bioinformatics 26(2):bbaf142 (2025), doi:10.1093/bib/bbaf142; arXiv 2503.06052.
- Repo: https://github.com/CXX1113/DGIB4SL @ ff6f39c733bd9eb7482bf054468c14ff854517b8 (cloned to external/models/DGIB4SL).
  License: none stated. Weights: none.
- Status: **acquired only; cannot run.** The repository has a single "Initial commit" containing an empty README.md
  (0 bytes); no code, no data, no weights (checked 2026-09-23, only branch `main`). No fork or mirror with code was
  found on GitHub, Papers with Code or in the arXiv source. Blocker: code not released.
- Leakage status: n/a (not run). The paper trains on SynLethDB 2.0 with SynLethKG; a re-implementation would need
  SynLethKG without SL/SR/non-SL relations (available: data/raw/feng2024_slbench, see notes/models/feng_suite.md).
