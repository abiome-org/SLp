# Cilantro-SL: Geneformer in-silico-KO viability embeddings + pair classifier + conformal calibration (retrained on SLB)

battery: cilantro_sl; species: human only (coverage fact: the model needs cell-line bulk RNA-seq for Geneformer
and DepMap CRISPR gene effect for viability pretraining; non-human species are written as constant 0.0 by
slb.write, i.e. AUROC 0.5); needs: DepMap 24Q4 expression + CRISPR gene effect, contexts.parquet depmap_id,
Geneformer gf-12L-30M-i2048, Cilantro's Gene2vec (128-d), SLB train labels, GPU (~15 min per benchmark version).

## Source
- Paper: Hu (Kailey H.) et al., "Cilantro-SL", bioRxiv 2026, doi 10.64898/2026.02.25.708096.
- Code: https://github.com/kaileyhh/Cilantro-SL commit c674b55 (no licence file; GitHub licence: none) at
  external/models/cilantro_sl; modified Geneformer https://github.com/kaileyhh/geneformer commit 0e2bf2a
  (Apache-2.0, like upstream) at external/models/cilantro_sl/geneformer_fork (installed -e, used for
  reference; see below).
- Geneformer weights: gf-12L-30M-i2048 (the model named in notebook 2) + its gc30M dictionaries, from HF
  ctheodoris/Geneformer revision 01d3ea8993c2 (the last revision that still ships this model; current main only
  has V1-10M/V2 models). pytorch_model.bin sha256 812f8d85e5ecf9d64c268f052f6ece2c1906bc4f1aecf70d5144b2598386b615;
  all files + URLs in external/models/cilantro_sl/gf_weights/SOURCES.tsv. Licence Apache-2.0. Geneformer is
  pretrained (masked LM) on Genecorpus-30M single-cell transcriptomes: no SL labels.
- Gene2vec: `gene2vec_embs.pt` (24,447 Ensembl genes x 128) from the authors' Google-Drive data folder
  (data/raw/cilantro_sl/SOURCES.tsv, sha256 ea6105e4...). Gene2vec is trained on GEO co-expression: no SL labels.
- Original training data: Human_SL.csv / Human_nonSL.csv (SynLethDB-style SL/non-SL lists) + expression-
  correlation negative sampling (notebook 3). NOT used here. No trained weights are released.

## Leakage
PPI/KG: none (no STRING, BioGRID or knowledge graph is used anywhere in this port).
Non-leaky: the pair classifier is fitted on SLB `train.parquet` labels only; viability pretraining uses DepMap
single-gene CRISPR gene effect (permitted); Geneformer/Gene2vec pretraining uses no SL labels. The per-context
tokenisation gene set uses the gene IDs of train/dev/test inputs (no labels).

## Environment
uv venv external/models/cilantro_sl/.venv: python 3.11, torch 2.5.1+cu121, transformers 4.46.3 (as pinned by the
fork), datasets, scanpy, anndata, geneformer fork (-e, --no-deps). Original: conda, python 3.9, torch 1.13.1+cu117.

## What we run (scripts/models/cilantro_sl/)
1. `isp_embed.py` = notebooks 1_tokenizer + 2_perturber + isp/viability_perturber.py, re-implemented as a single
   batched loop with HF BertForMaskedLM (the original drives InSilicoPerturber + EmbExtractor through temp
   datasets on disk, re-embedding all 1,479 DepMap lines once per gene). Same semantics: DepMap log2(TPM+1)
   used as counts; genes = top-500 HVGs (scanpy, all DepMap lines) + "SL genes"; Geneformer V1 rank-value
   encoding (value / gene median, non-zero genes, descending, first 2,048 tokens, no special tokens);
   perturb_type "delete"; cell embedding = mean over tokens of hidden_states[11] (emb_layer -1 of 12 layers);
   delta = emb(original) - emb(deleted); viability target = DepMap CRISPR gene effect of (line, gene).
2. `train_slb.py`: original nn modules (nn_helpers/net_layers.FiLMNet, SLNet), original hyper-parameters.
   FiLMNet(512-d delta, FiLM on 128-d Gene2vec) -> gene effect, MSE, Adam 1e-3, batch 512, 100 epochs;
   viability embedding = l3 output (32-d). Pair = [via(c, a) | via(c, b)] -> SLNet, class-weighted CE, Adam 1e-4,
   batch 128, 100 epochs, 5-fold CV over SLB-train pairs; the 5 fold nets are ensembled; Mondrian conformal
   (single class, as in notebook 6) calibrated on each net's held-out fold -> p_SL stored alongside.
   score = mean fold P(SL), averaged over (a,b) and (b,a) order.

### Changes vs the original
- Cells restricted to the human SLB contexts (their DepMap line); hTERT-RPE1 (no depmap_id) -> mean expression /
  gene effect of the six DepMap RPE1-ss* clones; C092, CHL-1, KP-1N have no DepMap expression -> no deltas.
- "SL genes" = the SLB genes occurring in that context (per-context set; original: one global set of SynLethDB
  genes). This spends the 2,048-token window on genes the benchmark asks about.
- Dictionary: the fork's __init__ points to the gc95M token/median dictionaries while notebook 2 uses the
  gf-12L-30M-i2048 model, whose vocabulary is gc30M (25,426 tokens). Feeding gc95M token ids to the 30M model
  would silently map genes to the wrong embeddings, so we use the model's own gc30M dictionaries.
- Pair features use the example's own context (SLB labels are context-specific) instead of exploding a
  cell-agnostic label over every DepMap line where both genes have deltas.
- fp16 inference for Geneformer; drivers re-written (via_film / pair_classifier hard-code /work/magroup paths
  and use pandas-1.3 / np.float idioms).

## Status ladder
acquired: yes | env built: yes | ran original: no (the original notebooks need the authors' SL label files and
/work/magroup paths; we ran their modules through our driver) | adapted: yes | dev-scored: yes (SLB-1.2 and SLB-1.3).

## Results (dev; slb.write / slb.evaluate; non-human species constant-filled = 0.5)
| benchmark | SLB score | H. sapiens | human flat AUROC (species=human stratum) | human per ancestry (SLB) |
|---|---|---|---|---|
| SLB-1.3 | 0.5173 | 0.5519 | 0.5327 | AFR 0.556, EAS 0.540, EUR 0.560, unknown 0.307 |
| SLB-1.2 | 0.5158 | 0.5634 | 0.5469 | AFR 0.585, EAS 0.580, EUR 0.526, unknown 0.703 |
scer / spom (and spne, bsub, cele, dmel, mmus) = 0.500 by construction (not scored).
Diagnostics: Geneformer deltas for 54,543 (SLB-1.3) / 54,407 (SLB-1.2) (context, gene) combos, 97% with a Gene2vec
vector; viability FiLM fit to DepMap gene effect r = 0.956 (in-sample, as in the original which trains on all
rows); SL classifier fold-calibration AUROC 0.81-0.85 on random train-pair folds (random pair CV is optimistic:
genes are shared between folds; the gene-family-held-out dev score above is the relevant number).
Files: results/models/slb1.3/cilantro_sl_dev.{parquet,txt,json,coverage.json}, results/models/cilantro_sl_dev.*.

## Coverage (native, from <name>_dev.coverage.json)
A (context, gene) delta exists only if the gene is expressed and ranks inside that line's 2,048-token window;
C092, CHL-1 and KP-1N have no DepMap expression. Human dev rows scored natively: SLB-1.3 12,016/19,149
(63%); SLB-1.2 9,419/15,048 (63%); the rest are median-filled by eval. Train pairs with features: 103,587
(2,471 pos) on SLB-1.3, 99,896 (2,447 pos) on SLB-1.2. All non-human species: 0 (human-only model).

## Runtime / GPU
- Geneformer ISP (stage 0): ~5 GPU-min per benchmark version on the 3090 (fp16, ~180 perturbed forward passes/s;
  ~52k perturbations over 47 lines). The original loop (one EmbExtractor pass over all 1,479 DepMap lines per
  gene via on-disk datasets) would take many GPU-hours.
- Stages 1-2 (FiLM 100 epochs + 5 x SLNet 100 epochs): ~22 min per version on the GPU (tiny MLPs; could run on CPU).
- Total GPU: ~55 min for both benchmark versions.

## Problems / caveats
- Original notebooks cannot run as-is (hard-coded /work/magroup paths, authors' SL label files, pandas 1.3,
  np.float); run via our driver with the original modules.
- Dictionary mismatch in the fork (gc95M dicts with a gc30M model) - we use the matching gc30M dicts.
- Coverage is capped by Geneformer's 2,048-token input: lowly expressed genes get no embedding.
- Fold-level class weights and conformal p-values are kept (p_sl in external/models/cilantro_sl/_slb/<bench>/dev_scores.parquet);
  the submitted score is P(SL), since the single-class conformal p-value is a monotone transform of it per fold.
