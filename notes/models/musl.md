# musl: MuSL multimodal SL predictor (Fang et al., IEEE JBHI 2026)

battery: musl__human, musl__allspecies; species: musl__human = human only (TCGA expression + human ESM2); musl__allspecies = every SLB train species with bundle esm2 + ppi (slb1.2: human scer spom spne; slb1.3: bsub cele dmel human mmus scer spom); needs: data/raw/musl/{protein_embeddings.pt,tcga_all.h5ad} (Zenodo 17098066), BioGRID 5.0.261 human (data/raw/biogrid), data/interim/bundle/<sp>/{esm2,ppi}.parquet, GPU for musl__human (CPU works but slow)

- Paper: Fang et al., "MuSL: Multimodal deep learning for generalizable prediction of synthetic lethality from sequence, transcriptomic, and network data", IEEE JBHI 2026, doi:10.1109/JBHI.2026.3698476.
- Repo: https://github.com/JieZheng-ShanghaiTech/MuSL, commit f8021cfc618fafae8c330b694d0fa7c46db5f1a5 (2026-01-20), cloned to external/models/musl. License: MIT (LICENSE file, "Copyright (c) 2022 JieZheng").
- Data: Zenodo 10.5281/zenodo.17098066 (CC-BY-4.0). Fetched protein_embeddings.pt (ESM2 per-gene, 5120-d i.e. ESM2-15B size, 19,794 genes; sha256 84b57628...) and tcga_all.h5ad (TCGA bulk RNA-seq, 10,090 samples x 27,192 genes; sha256 13a16e96...); md5s match Zenodo. URLs + sha256 in data/raw/musl/SOURCES.tsv. Not fetched: a549/k562 single-cell h5ad (cell-line variant) and GenePT embeddings.
- Weights: none released. Bundled processed_data/ has SynLethDB-derived fold data (human_sl_7684.csv) and all_emb_scgpt.pkl (scGPT embeddings of the 7,684 SL genes; an alternative node init, emb_type=scgpt; not used - the README full model uses ESM2 "protein").
- Architecture (full model, README command): CNN over three 32x32 joint histograms of the two genes' TCGA expression (raw z, log, z-log) + MLP over 35 hand-crafted pair expression statistics (correlations, MI, KL/JS, conditional entropy, distances) + 2-layer GraphSAGE over a PPI graph with ESM2 node features projected to 1024-d; cross-modal attention, adaptive gate fusion, 4 heads, supervised contrastive loss.
- Original training data: SynLethDB human SL pairs (human_sl_7684.csv, 7,684 genes) + random negatives, 5-fold CV1/2/3 => original model would be leaky; we did not run the original folds.

## Leakage
- musl__human: trained only on SLB train human rows. Inputs: TCGA expression (single-gene), ESM2 (sequence), BioGRID 5.0.261 human **physical** interactions only. The bundled PPI file (processed_data/merged_ppi_gene_names.csv, 845,844 edges, provenance undocumented) was NOT used: 710 of its edges are BioGRID genetic-only interactions and 260k are in neither BioGRID physical nor genetic. Node set = all 19.8k ESM2 genes (the original node list, the 7,684 SynLethDB SL genes, is itself label-derived). No SynLethDB labels used => **not leaky**.
- musl__allspecies: SLB train rows of each species; bundle PPI (BioGRID physical + STRING neighborhood/fusion/cooccurrence/coexpression/database >= 400; no STRING experimental/textmining, no genetic rows) + bundle ESM-2 650M embeddings => **not leaky**.

## Changes vs original
- Labels: one row per SLB (context, pair) train row, context-agnostic model; single model instead of 5-fold CV. Early stopping (original criterion: best AUPR of the final head) on a gene-held-out 10% of SLB train (random 10% of train genes; rows touching them), never on dev.
- Model / feature / optimiser code imported unchanged from src/module.py and src/utils.py (customAdamW lr 1e-4, wd 1e-3, batch 128, seed 432, ReduceLROnPlateau). Engineering changes that do not change the computation: stat features computed in chunks (torch.quantile 16M-element limit) with 8 MI worker processes (1 BLAS thread each); expression histograms rendered once and cached (int16 counts, exact); GraphSAGE mean aggregation run on a sparse CSR adjacency (verified identical to the edge-list path on coalesced edges; the edge-list path needed 7.3 GB message tensors).
- Stat-feature normalisation parameters fitted on the fit part of train and applied to val/dev (original: fitted on train fold, applied to test fold).
- Inference: each eval pair scored in both gene orders (the fused head is order-dependent) and averaged. Genes absent from tcga_all.h5ad get an all-zero expression row (as the original get_cell_expression does).
- The original contrastive loss uses an unnormalised multi-positive target matrix, so the total loss sits around 60; kept as is.
- slb1.3 musl__human: patience 4 instead of 10 to bound GPU time (slb1.2 run: best epoch 2 of 12 with patience 10).
- musl__allspecies: GNN branch only (the repo's `--use_gnn` ablation, loss of that branch only); batch 1024, patience 5, <= 30 epochs, training rows capped at 300k per species (label-stratified subsample) so it runs on CPU.

## Status
acquired (repo + Zenodo data) / env built (uv, python 3.10, torch 2.1.2+cu121, torch-geometric 2.6.1, scanpy 1.11.1, anndata 0.11.4, numpy 1.26.4, pandas 2.0.3, scikit-learn 1.5.2; external/models/musl/.venv) / ran original: no (its folds are SynLethDB labels) / adapted / dev-scored: see below

## Dev results
SLB = headline score; species columns = the evaluator's per-species SLB score (fitness-balanced, context x screen stratified; headline line of the .txt). Species a variant does not score are constant-filled (0.5). Reports: results/models/musl__<variant>_dev.txt (slb1.2), results/models/slb1.3/musl__<variant>_dev.txt (slb1.3); native coverage in the matching .coverage.json.

| bench | variant | SLB | H. sapiens | S. cerevisiae | S. pombe | S. pneumoniae | human AFR / EAS / EUR (stratum SLB) |
|---|---|---|---|---|---|---|---|
| slb1.2 | musl__human | 0.5319 | 0.6275 | - | - | - | 0.727 / 0.671 / 0.485 |
| slb1.2 | musl__allspecies | 0.5966 | 0.5974 | 0.5836 | 0.6213 | 0.5843 | 0.637 / 0.621 / 0.535 |
| slb1.3 | musl__human | 0.5215 | 0.5646 | - | - | (unlabelled) | 0.556 / 0.583 / 0.555 |
| slb1.3 | musl__allspecies | 0.5802 | 0.6000 | 0.5884 | 0.5522 | (unlabelled) | 0.570 / 0.668 / 0.563 |

slb1.3 auxiliary species (musl__allspecies; all < 20 dev positives, n/a in the score): bsub stratum 0.425 (4 pos), cele 0.469 (12), dmel 0.491 (10), mmus no positives.
Paralog stratum (slb1.3): musl__human 0.532, musl__allspecies 0.601.

Native coverage (rows scored by the model; rest filled by slb.write):
- musl__human: slb1.2 human 15,002/15,048; slb1.3 human 19,097/19,149 (missing = genes without an ESM2 embedding). Other species: none (constant fill).
- musl__allspecies: slb1.2 human 15,048/15,048, scer 91,939/91,939, spom 24,171/24,171, spne 1,361/1,564; slb1.3 human 19,149/19,149, scer 134,621/134,621, spom 41,484/41,484, bsub 874/934, cele 56/59, dmel 234/234, mmus 42/42 (unscored rows of a covered species get that species' median).

Training (val = gene-held-out 10% of train): musl__human slb1.2 best val AUPR 0.181 / AUROC 0.81 at epoch 2 of 12 (early stop, patience 10); slb1.3 best val AUPR 0.139 / AUROC 0.81 at epoch 2 of 6 (patience 4). Validation AUPR peaks after 2 epochs in both runs: the full model overfits SLB train quickly.

## Runtime / GPU
- musl__human: feature pre-computation on CPU (8 threads): 35 stat features for 90k train pairs ~40 min (mutual information dominates), histogram rendering ~2 min; cached per gene pair in external/models/musl/_slb/{stat,img}_cache.pt and reused across benchmark versions. Training on the RTX 3090: ~100 s/epoch (230 s/epoch when the GPU was shared); slb1.2 ~35 min GPU, slb1.3 ~11 min GPU.
- musl__allspecies: CPU only (8 threads), ~1.2-1.3 h per benchmark for all species (human ~2.5 min/epoch, scer ~2 min/epoch).

## Run
`SLB_BENCH=data/bench/slb1.3 SLB_SPLIT=dev scripts/models/musl/run.sh` -> results/models[/slb1.3]/musl__{human,allspecies}_<split>.parquet (+ .txt/.json/.coverage.json on dev). Checkpoints cached per benchmark in external/models/musl/_slb/ (MUSL_RETRAIN=1 to refit). Set MUSL_ALLSPECIES=0 to skip the bundle variant; MUSL_PATIENCE / MUSL_EPOCHS override.

## Problems / notes
- The README's data list names a `tcga_all.h5ad` while utils.get_expression() reads `all_tcga.h5ad`; we load the Zenodo file directly.
- utils.get_mutual_info hard-codes 48 worker processes; capped at 8 (machine-wide thread limit).
- The node list, PPI and fold data bundled in processed_data/ are all derived from SynLethDB SL genes; none are used.
- The scGPT embedding file listed in our brief is only an alternative node initialisation (emb_type=scgpt, 7,684 SL genes); the README full model uses ESM2 and so do we.
- Not run: the cell-line (A549/K562 single-cell) variant - those labels are SynLethDB/cell-line SL pairs, and SLB has its own context structure.
