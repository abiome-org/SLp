# DSESL (deep stacking ensemble for SL)

battery: none (not adapted); species: -; needs: SynLethKG TransE, KG meta-path counts per pair, GO matrix, pathway
tree + text embeddings, and an SL-graph sub-model over training SL/non-SL edges

- Repo: https://github.com/TOJSSE-iData/DSESL @ 4760a733cbdb2473ae535cc8fd59370b09875360 (no license). IEEE (record
  10830943). torch + PyG 2.4; DGL-KE TransE (precomputed). No weights; stage-1 caches on Google Drive.
- Original data: SynLethDB 2.0 via SynLethKG (C1/C2/C3 x 5 folds; 24,230 pos / 180,319 neg). Its KG (kg2id.txt,
  24 relations) already excludes SL_GsG / SR / NONSL.
- Model: two-stage stacking of graph sub-models (KGE + type-wise KG aggregation, KG meta-path features, SL-graph
  HeteroData GNN, pathway-tree model) and a meta-learner.
- Status: **acquired; not adapted.** Gene universe fixed by gene-id-mapping.csv (6,472 genes); meta-path features
  cached only for pairs in sample-space.txt (regeneration via cache-all-meta-paths.sh over the whole KG); the SL-graph
  sub-model is transductive. A faithful SLB port needs the full two-stage pipeline regenerated for ~12.5k genes; not
  done in this session (time). Posted on the board as possibly feature-like; it is graph-based, so it stays here.
