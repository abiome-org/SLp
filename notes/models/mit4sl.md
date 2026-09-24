# MiT4SL (context-aware multi-modal transfer for SL across cell lines)

battery: none yet; species: -; needs: PrimeKG (HGT sampler), contextualized per-cell-line PPI (reference PPI
filtered by RNA-seq TPM >= 400), ESM protein embeddings

- Paper: bioRxiv 10.1101/2025.04.20.649694. Repo: https://github.com/JieZheng-ShanghaiTech/MiT4SL @
  68cfb5c0c2e76a98824ae8511799685c78ef4a93 (MIT per README badge). torch 1.12 + PyG (HGTConv/HGTLoader).
  No weights (result/ holds metrics only).
- Data: Google Drive data.zip (v2026.03.31, 1.03 GB, sha256 d06124f2...) fetched to data/raw/mit4sl and extracted
  (5.9 GB): PrimeKG HeteroData (129,375 nodes, 50 edge types), 1,280-d ESM embeddings for 18,927 proteins,
  RNA-seq for 1,431 lines (Cell Model Passports), reference PPI (839,522 edges) and contextualized PPI subgraphs for
  6 lines only (A375, A549, Jurkat, MeWo, 22RV1, PK-1); SL labels "SLbench" (own compilation) for those lines.
- Status: **acquired + data fetched; not adapted.** Scoring SLB requires contextualized PPI subgraphs for all 50 SLB
  lines (their construction notebook, contextualized_PPI_construction.ipynb, over the shipped RNA-seq table), a
  mapping of SLB genes to PrimeKG node indices, and re-writing the SLbench split files; PrimeKG's protein-protein
  edges would also need auditing for genetic-interaction sources. Not done within this session (time); the released
  labels overlap SLB sources, so no released-model scoring is possible either (no weights).
