# MSGT-SL (multi-omics sampling-based graph transformer)

battery: msgt_sl (human); species: human; needs: SL graph + cell-line omics node features

- Paper: arXiv 2310.11082 (2023).
- Repo: https://github.com/MSGT-SL/MSGT-SL @ a6ca9c33379190c107c0d1314cddd5edee9ca816 (no license, no README).
- Released weights: none.
- Original training data: Horlbeck 2018 K562 / Jurkat GI maps (SLB sources).
- Leakage status: **clean** for the SLB-trained variants (fit on SLB human train only; no SL/GI edges or GI-derived inputs; dev labels never read), except where noted below. PPI / KG sources and how GI evidence was removed: only the training SL graph is used by the released transformer model (see below); no PPI/KG.
- What we changed:
  - Released code is work-in-progress: main.py is MVGCN-iSL's GCN_pool; the transformer model (GCN_transformer_pool) is in main_version02.py, which keeps only the first input graph (the training SL graph) and computes random-walk neighbour samples but then overwrites them with an empty list, so the transformer attends over the nodes of each 50-pair batch only. We reproduce exactly that released behaviour.
  - Contexts pooled as for MVGCN-iSL (per-context DepMap node features); max 100 epochs (released default 1), patience 5 (SLB-1.3 run early-stopped at epoch 8, 3.5 min on GPU), AdamW 1e-4; modified_transformer made device-safe (the released one concatenates a CPU tensor).
- Adapter: scripts/models/msgt_sl/run.sh; env SLB_BENCH, SLB_SPLIT.

## Results (rows a model does not score get its median score before `slpbench eval`, as `--allow-missing` does)
### SLB1.3 dev
- msgt_sl: **SLB 0.5166**; H. sapiens 0.5499, S. cerevisiae 0.5000, S. pombe 0.5000, B. subtilis n/a, C. elegans n/a, D. melanogaster n/a, M. musculus n/a. Human ancestry: AFR 0.5857, EAS 0.5237, EUR 0.5404, unknown 0.4607. Rows scored by the model: 19,149 (the rest get the model's median score).
### SLB1.2 dev
- not scored on this version.
- Status: acquired, env built, adapted, dev-scored.
