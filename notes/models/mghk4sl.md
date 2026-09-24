# MGHK4SL (multimodal GAN over hypergraph + knowledge-graph representations for SL)

battery: none (not runnable as released); species: -; needs: SL-adjacency hypergraph + KR4SL KG + CODER embeddings

- Paper: Springer LNCS chapter 10.1007/978-981-95-0030-7_21 (2025). Repo: https://github.com/wyl20181914/MGHK4SL @
  ed24dad9df27e04ddde958dd40c49672d6c552ed (no license). torch 2.4.1, dgl 2.0. No weights.
- Original data: SynLethDB 2.0 (Human_SL.csv, 35,942 rows), KR4SL's KG (GO + pathways; no SL triples).
- Status: **acquired; not runnable.** Missing ./data/SL_adj.npy, ./data/all_entities_pretrain_emb.npy and ./KgData/;
  node count 9,758 hard-coded; transductive by design (hypergraph built from the SL adjacency, random 70/30 edge split,
  random negatives). Held-out SLB genes would have no hyperedges.
