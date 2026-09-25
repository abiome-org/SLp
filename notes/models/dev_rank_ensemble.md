# Dev Rank Ensemble

An internal benchmark ensemble selected on SLB-1.3 dev. It is separate from the SLp
world-model line. It combines four models fitted on SLB train only:
`go_ppi_gbm` (two parts), `ontotype` (one), `synleaf__allspecies` (one), and
`dekegel2021__allspecies` (one). Each source score is converted to its percentile rank within
species on the prediction split, then averaged. The `loss` finalist adds one part
`depmap_ols__loss` in human only. Both recipes were frozen before test evaluation.

All sources use permitted single-gene, sequence, GO and physical/filtered PPI data. The SynLeaF
all-species knowledge graph is built from the filtered bundle; its SL/GI edges are removed.
`depmap_ols__loss` uses DepMap single-gene screens and omics, no pairwise SL labels.

Run the four base adapters (and `scripts/models/deltadep/run.sh` for `loss`) with the same
`SLB_BENCH` and `SLB_SPLIT`, then run `scripts/models/dev_rank_ensemble/run.sh`. Predictions and a
sidecar containing their source hashes are written to `results/models/<benchmark>/`.
