# ESM4SL: ESM-2 protein embeddings + cross-attention for cell-line-specific SL (retrained on SLB)

battery: esm4sl__human (ESM4SL attention model + cell-line branch), esm4sl__allspecies (the paper's ESM-2 + MLP
baseline, one model per species); species: human (esm4sl__human); every benchmark species that has proteins in
data/interim/orthology_extra/fasta (esm4sl__allspecies: SLB-1.2 human/scer/spom/spne, SLB-1.3 also bsub/cele/dmel/mmus);
needs: protein sequences (UniProt canonical for human, orthology_extra proteomes otherwise), ESM-2 650M weights,
DepMap 24Q4 omics for the human cell-line branch, SLB train labels.

## Source
- Paper: Dai / Yang et al., "ESM4SL", IEEE EMBC 2025, doi 10.1109/EMBC58623.2025.11254319.
- Code: https://github.com/JieZheng-ShanghaiTech/ESM4SL, commit 348b2bf9a03f (2026-05-15), MIT licence.
  Cloned at external/models/esm4sl (gitignored). Training harness = bundled coach-pl (PyTorch Lightning).
- Weights: no trained ESM4SL weights are distributed. The repo ships data/SLKB (SLKB SL labels,
  `SLKB_original_scores.csv`) and mapping files (`clname2embed.npy` = SynergyX 4079-gene x 6-channel cell tensors
  for 170 lines). Anything trained on data/SLKB would be leaky for SLB (SLKB screens are SLB sources), so we
  retrain on SLB train only and never touch data/SLKB.
- Protein LM: ESM-2 `esm2_t33_650M_UR50D` (fair-esm 2.0.0), https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
  sha256 ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c (+ contact-regression file;
  external/models/esm4sl/_weights/SOURCES.tsv). ESM-2 is pretrained on UniRef50 sequences only: no SL labels.

## Leakage
PPI/KG: none (no STRING, BioGRID or knowledge graph is used anywhere in this port).
Non-leaky: fitted only on SLB `train.parquet` labels (+ permitted inputs: sequences, pretrained ESM-2, single-gene
DepMap omics / CRISPR gene effect for the cell branch). Checkpoint selection / early stopping use a gene-held-out
slice of SLB train (5% of that species' train genes, seed 0; all train rows touching them), never dev.

## Environment
uv venv external/models/esm4sl/.venv: python 3.10, torch 2.5.1+cu121 (CUDA 12.1 wheels, driver CUDA 13.2),
torchvision 0.20.1, pytorch-lightning 2.4.0, fair-esm 2.0.0, rich 13.7.1 (coach-pl imports
`rich.logging.FormatTimeCallable`, removed in newer rich), omegaconf, fvcore. Original used conda + CUDA 11.8.

## What we run (scripts/models/esm4sl/)
- `seqs.py`: one protein per gene. Human: UniProt reviewed canonical isoform via HGNC `uniprot_ids`
  (data/raw/uniprot_human/UP000005640_reviewed_canonical.fasta, sha256 dffdff38...; the original also mapped
  Entrez -> UniProt Entry -> sequence), fallback Ensembl longest isoform; other species:
  data/interim/orthology_extra/fasta/<sp>.fa (longest isoform, canonical SLB IDs). SLB-1.2 bench genes with a
  sequence: human 7,353/7,354, scer 5,795/5,806, spom 2,387/2,403, spne 499/530.
- `embed.py` (adapted from data_preprocess/esm2_gen.py): layer-33 representations. (a) per-gene mean over the
  first 1,022 residues (BOS/EOS excluded) -> data/interim/esm2_650m/<sp>.parquet, which is also the shared
  bundle's esm2.parquet; (b) human per-residue tensors (fp16, <= 2,000 residues = the original collate's max_len;
  proteins > 1,022 aa embedded in consecutive 1,022-residue windows, the original ran full length which is
  infeasible for titin-size proteins). GPU fp16; a CPU fp32 top-up path (ESM_DEVICE=cpu) was used for the
  SLB-1.3 new-species genes.
- `cellfeat.py`: the original cell branch consumes SynergyX's 4079 x 6 tensor (exp, mut, cn, eff, dep, met) that
  is not distributed (only 170 lines in clname2embed.npy, most SLB lines missing). Rebuilt for every human SLB
  context from DepMap 24Q4: 4,079 genes with the highest CRISPR-gene-effect variance; channels = expression z,
  damaging mutation 0/1, CN z, gene-effect z, expression/10, raw gene effect (dep/met unavailable). hTERT-RPE1
  has no depmap_id in contexts.parquet -> mean of the six DepMap RPE1-ss* clones (ACH-002462..67) as proxy.
  C092 (no DepMap model) -> zero tensor; CHL-1 has no expression/effect and KP-1N no expression (partial zeros).
- `prep.py`, `slb_data.py`, `train_slb.py`, `collect.py`: SLB -> the original CSV format (columns 0/1/2, int
  gene ids), new datasets registered into esm4sl's DATASET_REGISTRY, then the ORIGINAL coach_pl
  `train.py` main with the original configs `esm4sl/configuration/{attn,mlp}/new.yaml`.

### Changes vs the original (all in adapter code; no file in the repo was edited)
- esm4sl__human pools all human contexts into one model; the cell tensor is looked up per row (original: one model
  per cell line, cell tensor constant per run). Original train/val/test splits came from SLKB per cell line.
- Collate pads to the longest protein in the batch (<= 2,000) instead of always 2,000 (compute). NB: the original
  attention mask only masks positions where BOTH query and key are padding, so real residues attend to padding;
  batch-max padding changes how much padding they see (less).
- Embeddings kept in RAM instead of one torch.load per sample; SLDataset's hard-coded
  `/home/qingyuyang/.../clname2embed.npy` load bypassed.
- esm4sl__human is trained with `train_attn.py`, a plain PyTorch loop around the ORIGINAL AttnModule/AttnWrap and
  config (same loss, Adam lr/weight decay, MultiStepLR, grad clip 1.0, balanced WeightedRandomSampler, batch 16,
  best-val-avgmtr checkpoint selection evaluated every 3,000 steps + epoch end). Under the coach_pl Lightning
  trainer the attention run logged train/loss = nan from the start at ~1.8 it/s, while the same module and batches
  in a plain loop train finitely at ~6-7 it/s; not resolved within the shared-GPU slot.
- BUG FIX (original code): with the cell branch, `gene_cell_CA(gene, gene_mask, cell, None)` masks whole rows of
  padded query positions with -inf -> softmax NaN, which then propagates to every sample containing a padded
  protein. Verified: the unmodified AttnWrap(use_cell=True) returns NaN for any padded input. We drop query-only
  masks (monkey-patch in train_slb.py); padded positions are still excluded by the pooling masks.
- esm4sl__human: bf16 autocast, proteins truncated to the first 1,000 residues (ESM4SL_MAXLEN; original 2,000),
  1 epoch (ESM4SL_ATTN_EPOCHS; original up to 100 with early stopping patience 20) = ~141k balanced samples,
  ~26 GPU-min per run. These are budget choices on the shared 3090, not tuned on dev.
- esm4sl__allspecies (MLP): trained on CPU (the original ClsModule's `.cuda()` calls are made no-ops;
  identical model). Batch 16 (N <= 50k train rows), 64 (<= 500k), 256 (larger) with the original linear lr
  scaling (lr = 5e-5 * batch/16); epochs = clamp(1.5M / N, 5, 100) (original 100).
- Test CSVs carry dummy alternating labels: the harness computes AUROC on the test set at the end and crashes on
  one class. Eval-split labels are never read; the harness' own "test" metrics are meaningless by design.

## Status ladder
acquired: yes | env built: yes | ran original: harness smoke-tested (1-epoch ESM-2+MLP on SLB human through the
original train.py; the SLKB experiments themselves were not re-run since their labels are not SLB-clean) |
adapted: yes | dev-scored: yes (SLB-1.2 and SLB-1.3, both variants).

## Results (dev; slb.write / slb.evaluate)
Headline = SLB score (context x screen stratified, fitness-balanced AUROC); species columns = per-species SLB
score; "flat" = the species=human stratum row.
| variant | bench | SLB | H. sapiens | S. cerevisiae | S. pombe | other | human per ancestry (SLB) |
|---|---|---|---|---|---|---|---|
| esm4sl__human | SLB-1.3 | 0.5204 | 0.5611 (flat 0.523) | 0.5 (not scored) | 0.5 (not scored) | - | AFR .595 EAS .594 EUR .494 unk .598 |
| esm4sl__human | SLB-1.2 | 0.5284 | 0.6137 (flat 0.575) | 0.5 | 0.5 | spne 0.5 | AFR .634 EAS .640 EUR .567 unk .543 |
| esm4sl__allspecies | SLB-1.3 | 0.5426 | 0.5527 (flat 0.540) | 0.5475 | 0.5276 | aux: bsub .741 (4 pos), cele .403 (12 pos), dmel .228 (10 pos), mmus n/a (0 pos) - all <20 pos, not in headline | AFR .512 EAS .619 EUR .528 unk .524 |
| esm4sl__allspecies | SLB-1.2 | 0.5704 | 0.5828 (flat 0.515) | 0.5799 | 0.5952 | spne 0.5235 | AFR .635 EAS .633 EUR .480 unk .658 |
Files: results/models/slb1.3/esm4sl__{human,allspecies}_dev.{parquet,txt,json,coverage.json} and
results/models/esm4sl__*_dev.* (SLB-1.2). Validation (gene-held-out train slice) avgmtr of the attention model:
0.432 (1.3) / 0.442 (1.2), val AUROC ~0.74.

## Coverage (native, from <name>_dev.coverage.json)
- esm4sl__human: all human dev rows (SLB-1.3 19,149/19,149; SLB-1.2 15,048/15,048); other species constant 0.0.
- esm4sl__allspecies SLB-1.3: human 19,149/19,149, scer 134,621/134,621, spom 41,484/41,484, bsub 874/934,
  cele 56/59, dmel 234/234, mmus 42/42 (missing = genes without a protein sequence in orthology_extra).
  SLB-1.2: human, scer, spom complete; spne 1,361/1,564.

## Runtime / GPU
- ESM-2 650M embedding, SLB-1.2 species (human incl. per-residue for 7,353 genes, scer, spom, spne) + bsub,
  calb: ~70 GPU-min (fp16, 3090; ~16 full-length proteins/s). SLB-1.3 top-up (47 new human genes, cele 703,
  dmel 93, mmus 126 bench genes): ~37 CPU-min (fp32, 8 threads). Cached thereafter.
- esm4sl__human attention model, 1 epoch: ~26-28 GPU-min per benchmark version.
- esm4sl__allspecies (CPU, 8 threads): SLB-1.2 ~4 h (human 40 min, scer 82, spne 49, spom 68); SLB-1.3 ~4.2 h
  (bsub 81 min, human 28, scer 69, spom 47, cele/dmel/mmus < 3 each). Batch 16 on small species is the slow part.
- Total GPU: ~125 min (embedding 70 + attention 2 x ~27).

## Notes on the numbers
- Pooling all human contexts, the attention model is not better than the paper's own ESM-2+MLP baseline on
  SLB-1.3 (human .561 vs .553), and both are near 0.5 on the fitness-balanced, gene-family-held-out split: ESM-2
  embeddings mostly carry gene identity / essentiality, which the fitness balancing neutralises. The within-gene
  (wg) human scores are <= 0.53 for both variants.
- 1 epoch / 1,000-residue truncation was a GPU-budget choice; longer training was not tried.

## Problems
- Original deps: coach-pl needs torchvision and rich<13.8 (not listed in environment.sh).
- The released code hard-codes author paths (clname2embed.npy, ESM roots) - handled in the adapter.
- coach_pl Lightning run of the attention model produced NaN loss (see Changes); the attention variant is therefore
  trained with our plain loop around the original module. The MLP variant uses the original Lightning harness.
- Original cell-branch NaN bug (query-only attention mask) - fixed by monkey-patch, see Changes.
- Shared ESM-2 files (data/interim/esm2_650m/<sp>.parquet): full proteomes for human, scer, spom, spne, bsub, calb; cele/dmel/mmus contain only SLB bench genes (CPU top-up); ecol/mtub/saur not embedded yet (run embed.py with ESM_SPECIES=... on the GPU, ~40 min).
