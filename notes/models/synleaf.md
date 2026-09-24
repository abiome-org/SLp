# synleaf: SynLeaF dual-stage multimodal SL predictor (Xing et al., arXiv 2603.22369, 2026)

battery: synleaf__human, synleaf__allspecies; species: synleaf__human = human only (TCGA pan-cancer omics + SynLethKG); synleaf__allspecies = every SLB train species with bundle go + ppi (slb1.2: human scer spom spne; slb1.3: bsub cele dmel human mmus scer spom); needs: data/raw/synleaf/data_raw.tar.gz (authors' packaged raw data: SynLethKG 2.0, UniProt 2024-10-26, cBioPortal PanCan PCAWG 2020 cna/exp/mut), data/interim/bundle/<sp>/{go,ppi}.parquet + _go/edges.parquet, GPU for the RGCN stages

- Paper: Xing Z, Zhou S, Wang R, ... Wang Y, Li J. "SynLeaF: A Dual-Stage Multimodal Fusion Framework for Synthetic Lethality Prediction Across Pan- and Single-Cancer Contexts". arXiv 2603.22369 (2026). Web server https://synleaf.bioinformatics-lilab.cn.
- Repo: https://github.com/Jmpax404/SynLeaF, commit 2570240944d6db209fcbe73fcada3b16ce56eadb, cloned to external/models/synleaf. **No LICENSE file.**
- Weights: none released. We retrained.
- Data: authors' packaged `data_raw.tar.gz` (Google Drive id 1IFvkEcOWdfmlkg60hB5hZr3VXT5EQJrB, 5,007,320,294 bytes, sha256 e42fdb69d1b6d9aa8d7579f2793b0290df79a1792551009b2b9b16599a4205a0; data/raw/synleaf/SOURCES.tsv). Only SLKG2/raw_kg.tsv (+ sldb_complete.csv), uniprot fasta and TCGA/pan/{cna,exp,mut}.txt are extracted/used. The SynLethDB server itself returns 403. The SL label files in the archive (Human_SL/nonSL/SR, ELISL pairs) are never read.
- Architecture: omics encoder = 3x3 matrix of VAEs (self-VAE + cross-VAE Product-of-Experts) over per-gene vectors across TCGA samples (pan-cancer setting: cna, exp, mut; no methylation, as in the paper's pan-cancer setting); KG encoder = 2-layer RGCN over the 2-hop subgraph of SynLethKG; stage 1 trains each unimodal teacher; stage 2 (`umt`) trains fresh omics + KG encoders with a joint MLP head plus feature-level MSE distillation towards the frozen teachers.
- Original training data: SynLethDB 2.0 human SL / non-SL / SR pairs (pan-cancer) and ELISL cancer-specific pairs (per cancer). Both would be leaky; not used.

## Leakage
- KG source: SynLethKG 2.0 (SynLethDB 2.0 `sldb_complete.csv`, converted by the authors to raw_kg.tsv; 2,218,039 edges, 27 relation types).
- GI evidence removal, step 1 (original code): the SL-label relations SL_GsG (50,873 edges), SR_GsrG (8,107) and NONSL_GnsG (2,899) are dropped (prep.py asserts no relation matching SL/GsG/GnsG/GsrG remains).
- Step 2 (lead's PPI rule, 2026-09-24): SynLethKG's gene-gene INTERACTS_GiG relation (147,132 edges, undocumented provenance; our audit against BioGRID 5.0.261 human: 93.5k are BioGRID physical, 121 are BioGRID genetic-only, 53k in neither) is dropped and replaced by the shared bundle PPI (data/interim/bundle/human/ppi.parquet) between KG genes: BioGRID physical rows only (982,204 pairs) + STRING v12 neighborhood / fusion / cooccurence / coexpression / database channels >= 400 (221,158 + 74,980 + 1,742 + 1,595 pairs), one relation per channel; STRING experimental, textmining and combined_score are not used. prep.py substitute_ppi(). 21 model genes whose only KG edges were INTERACTS_GiG get a self-loop so they stay KG entities. KG after both steps: 3,280,319 edges over model-gene-relevant nodes, 57 relation types (incl. reversed).
- Remaining gene-gene relations: COVARIES_GcG (61,688; evolutionary rate covariation; 8 overlap BioGRID genetic-only) and REGULATES_GrG (264,044; LINCS knockdown/overexpression signatures; 181 overlap BioGRID genetic-only) - single-gene perturbation / evolution evidence, not GI assays; kept. Other relations are GO / anatomy / pathway / disease / compound edges.
- Labels: SLB train rows only. => synleaf__human **not leaky**. (A first slb1.2 KG teacher trained on the unsubstituted KG was discarded before any dev scoring; no leaky result was written.)
- synleaf__allspecies: KG built from the shared bundle only (GO direct annotations without IGI + GO is_a/part_of hierarchy, BioGRID physical, STRING neighborhood/fusion/cooccurence/coexpression/database >= 400) => **not leaky**.

## Changes vs original
- Pan-cancer setting only (SLB contexts are cell lines of many cancer types; the model is context-agnostic). One training row per SLB (context, pair) train row. SLB human symbols are current HGNC: KG gene names, TCGA Hugo symbols and UniProt GN= are all mapped to current HGNC symbols (HGNC prev/alias), and KG gene entities with the same (mapped) name are merged (the original unifies duplicate names the same way). Gene set = SL-free SynLethKG genes ∩ UniProt ∩ TCGA = 18,878 genes.
- Split: the repo's 5-fold CV replaced by one fit / validation split of SLB train: validation = rows touching a random 10% of train genes (gene-held-out, like dev); test_sl.npy = copy of val (train.py requires one). Dev labels never reach train.py; dev is scored by scripts/models/synleaf/predict.py from the saved checkpoint.
- Epochs / batch (repo default: 200 epochs, patience 999 = none, batch 384), bounded to keep shared-GPU jobs < ~60 min; selection criterion (best val AUC) unchanged:
  - omics teacher: batch 384, <= 50 epochs, patience 8 (CPU). slb1.2: stopped at 10, best epoch 2 (val AUC 0.738); slb1.3: 11 / best 3 (0.763).
  - KG teacher: batch 2048, 6 epochs, patience 2 (GPU). Both runs were still (slowly) improving at epoch 6: slb1.2 val AUC 0.798, slb1.3 0.750.
  - distillation (umt): batch 2048, <= 8 epochs, patience 3 (GPU). slb1.2: 8 epochs, best 5 (0.808); slb1.3: 7, best 4 (0.748).
- Inference: both gene orders scored and averaged (the classifier concatenates gene1|gene2).
- Single process via `accelerate` default config (the paper used 2 GPUs + SyncBatchNorm; SyncBatchNorm falls back to plain BN).
- allspecies: task `only_kg` (the paper's KG-teacher stage, same code and hyper-parameters) on the bundle KG; batch 1024, <= 8 epochs, patience 3; training rows capped at 300k per species (label-stratified subsample; scer and spom are subsampled).

## Status
acquired (repo + authors' packaged data) / env built (uv, python 3.11, torch 2.4.0+cu124, torch_geometric 2.8.0, accelerate 0.34.2, biopython 1.88, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.0.2; external/models/synleaf/.venv) / ran original: training pipeline run end-to-end on SLB inputs only (not on the SynLethDB labels, which are leaky; no released weights) / adapted / dev-scored: slb1.2 and slb1.3, both variants

## Dev results
SLB = headline score; species columns = per-species SLB score from the headline line of the report (fitness-balanced, context x screen stratified). Species a variant does not score are constant-filled (0.5). Reports: results/models/synleaf__<variant>_dev.txt (slb1.2), results/models/slb1.3/synleaf__<variant>_dev.txt; native coverage in the .coverage.json files.

| bench | variant | SLB | H. sapiens | S. cerevisiae | S. pombe | S. pneumoniae | human AFR / EAS / EUR (stratum SLB) | paralog stratum |
|---|---|---|---|---|---|---|---|---|
| slb1.2 | synleaf__human | 0.5248 | 0.5993 | - | - | - | 0.553 / 0.665 / 0.579 | 0.603 |
| slb1.2 | synleaf__allspecies | 0.5421 | 0.5328 | 0.5972 | 0.6184 | 0.4200 | 0.575 / 0.509 / 0.515 | 0.519 |
| slb1.3 | synleaf__human | 0.5413 | 0.6238 | - | - | (unlabelled) | 0.607 / 0.670 / 0.594 | 0.623 |
| slb1.3 | synleaf__allspecies | 0.5999 | 0.6136 | 0.5801 | 0.6061 | (unlabelled) | 0.651 / 0.593 / 0.597 | 0.629 |

slb1.3 auxiliary species (synleaf__allspecies; all < 20 dev positives, n/a in the score): bsub stratum 0.540 (4 pos), cele 0.476 (12), dmel 0.548 (10), mmus no positives.

Native coverage:
- synleaf__human: slb1.2 human 14,963/15,048; slb1.3 human 19,058/19,149 (missing = genes outside KG ∩ UniProt ∩ TCGA). Other species none.
- synleaf__allspecies: slb1.2 human 15,048/15,048, scer 80,184/91,939, spom 23,400/24,171, spne 1,267/1,564; slb1.3 human 19,149/19,149, scer 121,522/134,621, spom 40,533/41,484, bsub 874/934, cele 56/59, dmel 234/234, mmus 42/42 (missing = genes with no GO/PPI edge in the bundle; filled with the species median by slb.write).

## Runtime / GPU
- prep.py ~1 min per benchmark (CPU). Omics teacher 25-32 min CPU (8 threads).
- GPU (RTX 3090, shared): KG teacher ~2.3 min/epoch at batch 2048 on the 3.3M-edge KG (~13-14 min); distillation ~3.2 min/epoch (~23 min). ~40 min GPU per benchmark for synleaf__human. allspecies human KG branch ~1.3 min/epoch on GPU (~11 min per benchmark); other species ran on CPU (scer/spom ~10-12 min/epoch, ~1.5 h each).

## Run
`SLB_BENCH=data/bench/slb1.3 SLB_SPLIT=dev scripts/models/synleaf/run.sh` -> results/models[/slb1.3]/synleaf__{human,allspecies}_<split>.parquet (+ .txt/.json/.coverage.json on dev). Stage checkpoints are cached per benchmark in external/models/synleaf/result/{slb,bundle}_<bench>*; delete to retrain. SYNLEAF_ALLSPECIES=0 skips the bundle variant; SYNLEAF_KG_EPOCHS / SYNLEAF_UMT_EPOCHS override the stage caps. For this report the GPU stages were run via external/models/synleaf/_slb/gpu_jobA.sh (same commands, split into claim-sized pieces) and the CPU allspecies species via _slb/cpu_allspecies.sh.

## Problems / notes
- The SynLethDB 2.0 download server returns HTTP 403; the authors' Google-Drive package was used instead (it contains the same sldb_complete.csv/raw_kg.tsv).
- No LICENSE in the repo: redistribution of code/derived artefacts unclear.
- KG teacher epochs were capped at 6 while validation AUC was still rising slightly; more GPU time could improve synleaf__human.
- slb1.2 synleaf__allspecies is weak on human (0.533) and spne (0.420, 73 positives) while slb1.3 human is 0.614: per-species KG-only models are noisy with few epochs; treat single numbers with care.
- The paper's cancer-specific setting (ELISL cancer types, methylation) was not run: SLB contexts are cell lines and the ELISL/SynLethDB labels are leaky.
