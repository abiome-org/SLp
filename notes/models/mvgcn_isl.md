# MVGCN-iSL (multi-view GCN for cancer cell-specific SL)

battery: mvgcn_isl (human, DepMap cell-line omics), mvgcn_isl__allspecies (bundle views + fitness/ESM-2 features); species: human (faithful); all benchmark species with a bundle (allspecies); needs: gene graphs (PPI, co-expression, co-essentiality, training SL graph) + per-gene node features

- Paper: Fan et al., Frontiers in Genetics (2023), doi:10.3389/fgene.2022.1103092.
- Repo: https://github.com/kunjiefan/MVGCNiSL @ 85b7b880be9a86ae0b721bef0b26cc4363026882 (MIT).
- Released weights: none.
- Original training data: Horlbeck 2018 CRISPRi GI maps (K562, Jurkat; GI <= -3 = SL) shipped as *_GI_scores.csv; these screens are SLB sources, so the original training data overlap SLB.
- Leakage status: **clean** for the SLB-trained variants (fit on SLB human train only; no SL/GI edges or GI-derived inputs; dev labels never read), except where noted below. PPI / KG sources and how GI evidence was removed: human: BioGRID physical (rows with Experimental System Type = physical of the repo's BIOGRID-9606.csv), CCLE co-expression and DepMap co-essentiality graphs (single-gene data); the BioGRID genetic-interaction view (PPI-genetic) is dropped; no STRING. allspecies: bundle ppi.parquet = BioGRID physical + STRING neighborhood/fusion/cooccurence/coexpression/database channels only (no experimental, no textmining).
- What we changed:
  - Contexts pooled with shared weights: every epoch, each context's pairs are decoded from embeddings computed with that context's node features (DepMap 24Q4 exp / damaging-mut / CN / Chronos of that line; pan-line mean for hTERT-RPE1 and C092); the original trains one model per cell line.
  - Views: PPI-physical, co-exp, co-ess (all shipped with the repo) + SLB fit SL graph; the default `PPI-genetic` view (BioGRID genetic interactions) is dropped as leaky; no per-line expression filter of graph nodes (pooled contexts).
  - Split: SLB fit families train, held-aside train families validate (balanced, as the original); early stopping on validation loss (patience 150, max 500 epochs, AdamW 1e-4, batch 512).
  - Scores averaged over both gene orders (the decoder is asymmetric).
  - Wall-clock cap MV_MAXTIME (default 1800 s) because the GPU is shared: on SLB-1.3 training stopped at epoch 63 (best validation loss at epoch 5; patience would have stopped it at epoch 155), best-validation weights used. allspecies: per-species cap 240 s (MV_MAXTIME_SPECIES); views = SL, bundle PPI, bundle STRING co-expression (>= 400); node features = bundle fitness + 16 PCs of ESM-2 650M.
- Adapter: scripts/models/mvgcn_isl/run.sh (run.py, run_allspecies.py); env SLB_BENCH, SLB_SPLIT.

## Results (rows a model does not score get its median score before `slpbench eval`, as `--allow-missing` does)
### SLB1.3 dev
- mvgcn_isl: **SLB 0.5299**; H. sapiens 0.5896, S. cerevisiae 0.5000, S. pombe 0.5000, B. subtilis n/a, C. elegans n/a, D. melanogaster n/a, M. musculus n/a. Human ancestry: AFR 0.6407, EAS 0.5837, EUR 0.5443, unknown 0.4434. Rows scored by the model: 19,149 (the rest get the model's median score).
- mvgcn_isl__allspecies: **SLB 0.5596**; H. sapiens 0.5442, S. cerevisiae 0.5450, S. pombe 0.5897, B. subtilis n/a, C. elegans n/a, D. melanogaster n/a, M. musculus n/a. Human ancestry: AFR 0.5591, EAS 0.5652, EUR 0.5084, unknown 0.5518. Rows scored by the model: 196,523 (the rest get the model's median score).
### SLB1.2 dev
- not scored on this version.
- Status: acquired, env built, adapted, dev-scored.
