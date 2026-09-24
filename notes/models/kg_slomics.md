# KG-SLomics (KG + cancer-type multi-omics GNN for SL)

battery: none (blocked); species: -; needs: the authors' KG (107k nodes, 28 relation types) with its entity and
relation dictionaries, RotatE embeddings of it, DepMap 24Q2 multi-omics per cell line

- Paper: Lee S, Nam H. IEEE/ACM TCBB (2025), record 11239444.
- Repo: https://github.com/GIST-CSBL/KG-SLomics @ 7dc508144f5fed1c4b8cdfac733de530ee0a361a. License:
  CC-BY-NC-SA-4.0. torch 2.6, PyG 2.6 (RGAT + RotatE, LinkNeighborLoader). No weights.
- Original data: SLKB SL labels for 10 cell lines (A375, A549, HeLa, MDAMB468, Jurkat, K562, IPC298, MELJUSO, MEWO,
  PC3); only the id-mapped folds (data/CV1, CV2) are shipped. SLKB contains the SLB source screens, so any released
  model trained on it would be leaky.
- Shipped KG: data/myKG_PyG.zip -> myKG_PyG.pt, a PyG Data object with only `edge_index` (7,966,432 edges),
  `edge_type` (28 types) and `num_nodes` (107,038). **No node names.** The code also needs KG/entity_dict.pkl,
  KG/relation_dict.pkl, KG/KG_genes.txt and KG/myKG_RotatE_batch2048_lr1e-3_margin12.pt, none of which are released.
  A catalogue check found no dedicated SL relation, but type 12 is ~4x enriched for SL pairs (probably PPI).
- Status: **acquired; blocked.** Without the entity dictionary the KG cannot be mapped to gene symbols, so SLB genes
  cannot be placed in the graph, and the relation types cannot be audited for GI/SL-derived edges. Rebuilding the KG
  from its (undocumented) sources would be a new model, not KG-SLomics.
- Leakage status: n/a (not run).
