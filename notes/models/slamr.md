# SLAMR (LLM-augmented multimodal cold-start SL recommendation)

battery: none (blocked); species: -; needs: per-gene, per-cell-line LLM text summaries (GPT-5.1 / Gemini) embedded
with text-embedding-3-small, SynLethKG graph, ESM-2 embeddings

- Paper: doi:10.1145/3807503.3819499 (2026). Repo: https://github.com/Rrrrachellll/SLAMR @
  90eb8a2236ca2ce222e00e3d6edb9b5f58643713 (license: none stated). torch 2.3, dgl 2.2. No weights.
- Data: Google Drive folder 1dHwiye-hvZP7Wj5JlQK56XhqulkjUuUW fetched to data/raw/slamr (sldb_complete.csv,
  gene_esm2emb.pkl, KG_triple_isomorphic.npy; sha256 in data/raw/MANIFEST.tsv). Shipped splits and LLM embeddings
  (data_slb_filtered/): Horlbeck 2018 K562 / Jurkat (PMID 30033366) and Shen 2017 A549 / HeLa / 293T (PMID 28319113),
  i.e. screens that overlap SLB sources.
- KG: homogeneous SynLethKG (54,012 nodes, 4.24 M edges, no relation types). The code removes validation/test SL
  positives from it (`KG_triple.remove(pair)`, one orientation only), which implies SL edges are present in the KG
  and makes a leak likely; relation types are not stored, so SL edges cannot be filtered out.
- Status: **acquired; not run.** The model needs an LLM-written, cell-line-specific description embedding for every
  gene (a missing gene raises KeyError). The released embeddings cover 458 Horlbeck genes (K562/Jurkat) and 74 Shen
  genes: 13.7 % of SLB human dev rows (K562/Jurkat only) and 0.3 % for A549/HeLa. Generating new GPT-5.1 / Gemini
  summaries for ~6k genes x 50 SLB lines is outside this project's scope (external paid APIs, and the LLMs'
  training data may contain the SL literature, i.e. contamination), and the KG would have to be rebuilt without SL
  edges. Blocker: missing per-gene LLM inputs + SL-contaminated homogeneous KG.
