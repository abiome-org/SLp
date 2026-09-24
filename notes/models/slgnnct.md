# SLGNNCT (SLGNN per cancer type)

battery: none (covered by slgnn); species: -; needs: SynLethKG w/o SL + SL graph

- Repo: https://github.com/Lydia0228/SLGNNCT @ 60cf78f4b09c4760802f1192a414eef82d5af951, MIT. No weights.
- Original data: SynLethDB split by cancer type (BRCA, CESC, COAD, KIRC, LAML, LUAD, OV, SKCM, total), per-type
  SynLethKG (final kg2id.txt has 23 relation types, SL/SR/non-SL removed). torch 1.10, dgl 0.7.2.
- Status: **acquired; not run separately.** It is the SLGNN architecture (factor-aware KG GNN) trained per cancer
  type; SLB contexts are cell lines, and pooling them is exactly the `slgnn` run (notes/models/slgnn.md). The
  cancer type is hard-coded in src/dataloader.py and src/utils.py.
