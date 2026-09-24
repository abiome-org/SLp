# Feng et al. 2024 SL_benchmark code base on SLB (shared notes for sl2mf, grsmf, cmfw, ddgcn, gcatsl, slmgae,
# kg4sl, slgnn, nsf4sl, ptgnn, pilsl, mge4sl)

- Paper: Feng Y, Long Y, Wang H, Ouyang Y, Li Q, Wu M, Zheng J. "Benchmarking machine learning methods for
  synthetic lethality prediction in cancer". Nature Communications 15:9058 (2024), doi:10.1038/s41467-024-52900-7.
- Code: https://github.com/JieZheng-ShanghaiTech/SL_benchmark @ 04274e801b820a81f8dd92eb362d154086b80d9a
  (external/models/SL_benchmark; our changes live on local branch `slb`, full diff in
  scripts/models/feng_suite/slb.patch). License: MIT (LICENSE file, (c) 2023 JieZheng).
- Data: Zenodo 10.5281/zenodo.13691648 `data_small.tar.gz` (5 parts, md5 verified; sha256 in data/raw/MANIFEST.tsv),
  extracted to data/raw/feng2024_slbench/extracted (26 GB). `data_large` (adds the PiLSL subgraph database) fetched
  for PiLSL only.
- One unified re-implementation of 12 methods (the 11 of the paper + MGE4SL) sharing split handling, metrics and
  score-matrix output. We run these implementations (not each paper's original repo) because they share one input
  pipeline, were written/checked by the Zheng lab (authors of KG4SL, PiLSL, NSF4SL, MGE4SL) and are what the paper's
  ranking (SLMGAE best overall) refers to. Original repos are cloned for provenance (external/models/<Name>).

## How SLB is fed to the code (scripts/models/feng_suite/)
- `build_inputs.py` (human): node universe = the 9,845 SynLethDB genes of the original benchmark (ids 0..9844,
  symbols updated through HGNC ids) + every other SLB human gene of any split (2,688 more, ids 9845..12532; labels
  of dev/test are never read). All model inputs are rebuilt for these 12,533 genes:
  - KG: Feng's `fin_kg_wo_sl_9845.csv` = SynLethKG (54,012 entities, 24 relation types, 2.23 M triples). We
    verified that the three label relations of SynLethKG (SL_GsG 73,230, SR_GsrG 7,693, NONSL_GnsG 2,899 edges) are
    absent. Entity ids are permuted so the 2,687 appended SLB genes that are SynLethKG gene entities become gene
    nodes (1 appended gene has no KG entity). Residual risk: SynLethKG's GO-annotation edges (from Hetionet) may
    include IGI-evidence annotations; not separable in the released KG.
  - PPI: BioGRID 5.0.261 human physical interactions (668,214 undirected edges). Feng's own PPI matrix was not
    reused because its name and code show PPI edges coinciding with SynLethDB SL pairs were deleted (edited with SL
    labels).
  - GO similarity (BP, CC, MF): Wang measure + best-match average (the GOSemSim::mgeneSim default Feng used),
    re-implemented in Python/numba on NCBI gene2go (2026-09-23) with NOT-qualified and **IGI** (inferred from genetic
    interaction) annotations dropped, rounded to 3 decimals. Pearson vs Feng's R matrices on 200k random pairs of the
    original genes: BP 0.58, CC 0.65, MF 0.68 (7 years of annotation drift + IGI removal; term-level values match
    GOSemSim's published example within GO-version differences).
  - SL2MF PPI "topology similarity": Feng's recipe is not released; cosine similarity of PPI adjacency rows
    reproduces their matrix at Spearman 0.998, so we use cosine on the new PPI.
  - PT-GNN protein-sequence 3-mer word encodings: Feng's recipe (PTGNN_pre.ipynb) re-applied to UniProt reviewed
    canonical sequences for appended genes; original 9,845 encodings kept.
  - NSF4SL TransE embeddings: re-trained (dim 400, TransE_l2, DGL-KE-style loss) on the SL-free KG above, because the
    shipped file's training graph cannot be verified.
- Split (`slb_pairs.py` + `make_pkl.py`): Feng's `indep_test` pickle format, 1 fold. Human SLB train rows, contexts
  pooled to unique unordered pairs: label 1 if SL in >= 1 context, 0 if measured and never SL (all negatives are
  measured; no random negatives). Train families are hashed 80/20 into `fit` (58,342 pairs, 1,252 SL) and `valid`
  (4,494 pairs, 153 SL; pairs straddling fit/valid dropped) so that validation mimics the two-held-out-gene setting.
  Model selection / early stopping uses `valid` only. The Feng "test" slot is filled with the valid pairs as a
  placeholder: dev labels are never given to the code. Positive:negative ratio ~1:46 (closest Feng setting: -pn 50).
- Run: `-m <MODEL> -ns Random -ds CV1 -pn 50 --indep_test --cell_line slb --save_mat`, SLB_KFOLD=1. The score
  matrix of the epoch with the best validation F1 (Feng's `classify` checkpoint) is saved; `extract_scores.py` reads
  every human dev example off it (mean of M[a,b] and M[b,a]); all 50 contexts of a pair get the same score.
- Environment: Docker `slb-feng:cpu` (scripts/models/feng_suite/Dockerfile: python 3.7, TF 1.15.5, torch 1.5.0 CPU,
  PyG 1.5, dgl 0.4.3) reproduces the paper's stack (their CUDA 10 wheels cannot drive an RTX 3090).
  `slb-feng:gpu` (Dockerfile.gpu: NVIDIA TF1 23.03 container + torch 1.13.1/cu117) for GPU runs.

## Code changes (all in slb.patch)
- node count read from SLB_NUM_NODE instead of the literal 9845; fold count from SLB_KFOLD; CUDA device falls back
  to CPU; `--cell_line slb` loads CV1_50_slb.pkl.
- bug fixes needed to run at all: ChecktoSave clamps the fold index (SLMGAE passes 1-based folds and crashed in
  indep_test mode); F1 uses nanmax (max() returned NaN when precision+recall = 0 at any threshold, which silently
  disabled checkpoint saving); load_kg uses n_entity = max id + 1 and iterates entity ids (original enumerated dict
  keys, whose order is not the id order) and n_relation = max relation id + 1 (original len(set)=24 while ids run
  4..27: out-of-range lookups are silently zero on TF-GPU, a crash on CPU); SLMGAE array2coo vectorised (identical
  output); CMF-W's torch SVD and TF device pinning fall back to CPU; NSF4SL's training DataLoader drops a final batch of
  size 1 (BatchNorm crash on SLB-1.3); GCATSL's random-walk-with-restart uses a sparse transition matrix and stops at
  convergence (identical fixed point; the dense 1,000-step loop took ~11 h for 12.5k genes); `--cell_line slbbal` +
  SLB_BALANCE=1 gives the 1:1 negative-subsampled split (KG4SL's original protocol).
- allspecies variant (`run_model.sh <MODEL> <name> allspecies`, `build_inputs_bundle.py`): per species, node
  universe = that species' SLB genes; PPI = bundle BioGRID physical or STRING (non-experimental, non-textmining
  channels) score >= 400; GO similarity = same Wang/BMA on bundle GO (IGI/ND/NOT dropped). Only for models that need
  no KG (SL2MF, GRSMF, CMFW, DDGCN, GCATSL, SLMGAE). KG models are human-only (SynLethKG is human): coverage fact.

## Structural expectation
Every one of these models except NSF4SL/PTGNN/KG4SL/SLGNN/PiLSL gets its dev-gene representation only from
side-information (GO/PPI/KG); DDGCN has none (identity features + SL graph), so it cannot distinguish dev genes at
all (its dev scores collapse to a handful of values).

## Versions / runs
- SLB-1.2 and SLB-1.3 dev (SLB_BENCH selects; work dirs external/models/_slb_work/<bench>/feng[_<species>]; results in
  results/models/ for 1.2 and results/models/slb1.3/ for 1.3). SLB-1.3 human universe: 12,537 genes (2,692 appended);
  fit 1,281 SL / 61,373 non-SL pairs, valid 155 / 4,120.
- Compute: CPU container (8 threads each, at most 2 concurrent). GCATSL and SLGNN are only feasible with the GPU image
  (slb-feng:gpu, NVIDIA TF1 23.03 + torch 1.13): 187 s per GCATSL epoch on CPU (x200), ~4 h of CPU pair scoring for SLGNN.
- STRING audit (lead's warning): no Feng-suite input uses STRING. PPI = BioGRID physical; KG gene-gene relations are
  Hetionet GiG / GcG / GrG (no STRING); allspecies uses the channel-filtered bundle PPI.
