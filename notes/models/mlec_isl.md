# MLEC-iSL (SL connectivity + graph transformer, cell-specific)

battery: mlec_isl (human); species: human; needs: Opticon pathway + BioGRID physical graphs, DepMap population PCA features, cell-line omics

- Paper: Fan et al., Briefings in Bioinformatics (2024), doi:10.1093/bib/bbae425.
- Repo: https://github.com/kunjiefan/MLEC-iSL @ 3cce41d8c31d6ee7ef88720c85ae6e54143aa32f (MIT); data Mendeley 10.17632/7shf34snd3 (Opticon_networks.csv fetched, sha256 in MANIFEST).
- Released weights: none.
- Original training data: Horlbeck 2018 K562 / Jurkat (SLB sources).
- Leakage status: **clean** for the SLB-trained variants (fit on SLB human train only; no SL/GI edges or GI-derived inputs; dev labels never read), except where noted below. PPI / KG sources and how GI evidence was removed: BioGRID physical (repo file, physical rows) + Opticon pathway network (Mendeley data); PPI-genetic view dropped; no STRING; population features from DepMap expression / Chronos.
- What we changed:
  - Connectivity targets (number of SL partners per gene) are computed from SLB fit rows only, per context (the audited repo computes them before the gene split, leaking held-out edges into training targets).
  - Contexts pooled (one optimiser step per context per epoch with that line's omics); PPI-genetic view dropped; node universe restricted to SLB genes + graph genes with mean DepMap log2(TPM+1) > 1 so full attention fits a 24 GB GPU; max 300 epochs / patience 50 (original 1000 / 200) and a 1800 s wall-clock cap (MLEC_MAXTIME; SLB-1.3 run stopped at epoch 32, validation MSE still improving slowly).
  - Pair model: logistic regression on (connectivity_a, connectivity_b), trained on true fit connectivity (balanced 1:1, as the original) and applied to predicted connectivity of dev genes; symmetrised.
- Adapter: scripts/models/mlec_isl/run.sh; env SLB_BENCH, SLB_SPLIT.

## Results (rows a model does not score get its median score before `slpbench eval`, as `--allow-missing` does)
### SLB1.3 dev
- mlec_isl: **SLB 0.5289**; H. sapiens 0.5866, S. cerevisiae 0.5000, S. pombe 0.5000, B. subtilis n/a, C. elegans n/a, D. melanogaster n/a, M. musculus n/a. Human ancestry: AFR 0.5965, EAS 0.6162, EUR 0.5471, unknown 0.5079. Rows scored by the model: 19,149 (the rest get the model's median score).
### SLB1.2 dev
- not scored on this version.
- Status: acquired, env built, adapted, dev-scored.
