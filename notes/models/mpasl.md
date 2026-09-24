# MPASL (Multi-perspective learning knowledge graph attention network for SL)

battery: none (structurally cannot score held-out genes); species: -; needs: SynLethKG + per-gene SL history

- Paper: Frontiers in Pharmacology 15:1398231 (2024), doi:10.3389/fphar.2024.1398231.
- Repo: https://github.com/cyttong/MPASL @ 437db90b7bae988cde55b1c5436770b1307c36c2. License: none stated.
  TensorFlow 1.15 / Python 3.6.
- Weights: TF checkpoint src/model/MPASL/misc/SynlethDB/emb/_sw_para_SynlethDB_..._dim_16_parameter.*, trained on
  the SynlethDB split with the leaky KG below. **Leaky.**
- Original data: SynLethDB (36,405 pos + 36,405 sampled neg, data/SynlethDB/sl_file.*), SynLethKG kg_final.txt.
  The shipped kg_triplet.csv is truncated at 1,048,576 rows (Excel limit). kg_final.txt **contains SL_GsG (rel 1,
  32,507 edges), SR (rel 14) and non-SL (rel 24) relations**; 15,921 SL_GsG edges coincide with labelled positive
  pairs, i.e. test labels are in the graph.
- Status: **acquired; not adapted (structural).** MPASL is a RippleNet/KGNN-LS-style recommender: each "user" gene
  is represented by the ripple set grown from its *training SL partners* (genem_history_dict), and
  data_loader_genem_set.load_rating drops every train/eval/test pair whose first gene has no training SL partner.
  In SLB every dev/test pair has both genes in held-out families, so no SLB dev pair has a gene with SL history: the
  model returns no prediction for any of them. Swapping in the SL-free KG (data/raw/feng2024_slbench) would remove
  the leak but not this limitation.
- Leakage status: released weights and shipped KG leaky; not scored.
