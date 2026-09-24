# elisl: ELISL, early-late integrated SL prediction (Tepeli, Seale, Goncalves, Bioinformatics 2024)

battery: elisl__slbtrain; species: human only (cell-line/TCGA/GTEx features exist only for human; other species get the shared constant fill); needs: SLB train/dev(+contexts), DepMap 24Q4 (CRISPRGeneDependency, OmicsSomaticMutations, expression, Model), TCGA PanCanAtlas Xena (EB++, GISTIC2, MC3, TCGA-CDR), GTEx v8 TPM + sample attributes, STRING v11.0 (links.full, info), BioGRID physical (shared bundle data/interim/bundle/human/ppi.parquet), UniProt reviewed human FASTA, SeqVec weights; docker image slb/elisl-seqvec (GPU optional), uv venv external/models/elisl/.venv

- Paper: Tepeli YI, Seale C, Goncalves JP. ELISL: early-late integrated synthetic lethality prediction in cancer. Bioinformatics 40(1):btad764 (2024). doi:10.1093/bioinformatics/btad764
- Repo: https://github.com/joanagoncalveslab/ELISL, commit eaf5f8c502a6db580722d73ceef8b41f4e4fe1b7 (external/models/elisl). License: GPL-3.0.
- Weights: none released. The authors' SurfDrive/figshare share has precomputed feature sets only for their own label pairs (not usable for SLB pairs).
- Original training data: SL labels per cancer type (BRCA, CESC, COAD, KIRC, LAML, LUAD, OV, SKCM) compiled from ISLE, DiscoverSL, EXP2SL and related combinatorial-screen collections. We do not use these.

## Leakage
Retrained only on SLB train labels (human rows of `train.parquet`) => **not leaky**. Single-gene sources used:
UniProt sequences -> SeqVec (pretrained on UniRef50, no SL labels); STRING v11.0 *database* channel + BioGRID
physical interactions (node2vec); DepMap 24Q4 CRISPR dependency probabilities, somatic mutations, expression; TCGA
PanCanAtlas expression, copy number, mutations, survival; GTEx v8 expression. **Change for the leakage contract:**
ELISL's PPI graph used STRING `experiments > 0 OR database > 0`; the STRING experiments channel imports
interaction-database records that include genetic-interaction evidence, so it is replaced by BioGRID physical-only
edges (1.00 M) plus STRING v11 database (0.35 M undirected) = 1.30 M edges, 20,876 genes.

## What was rebuilt / changed (all code in scripts/models/elisl/)
Feature families (same definitions as ELISL's src/feature_generation + src/embedding code, vectorised):
- `seq_1024`: |SeqVec(g1) - SeqVec(g2)|. **Faithful:** original `seqvec==0.4.1` + allennlp 0.9 in docker
  (slb/elisl-seqvec; allennlp's dep set no longer resolves on py3.7, so it is installed --no-deps with pinned deps,
  see external/models/elisl/slb_docker/{Dockerfile.seqvec,seqvec_pip_freeze.txt}); `--protein` = mean of the summed
  ELMo layers. One UniProt reviewed protein per gene via HGNC uniprot_ids. 6,292 genes cached (slb1.2 + slb1.3).
- `ppi_ec`: |node2vec(g1) - node2vec(g2)|, 64 d, walk length 30, 200 walks/node, p = q = 1, Word2Vec window 10,
  sg=1, gensim 3.8.3. With p = q = 1 the walks are uniform, so they are generated directly in numpy (the node2vec
  package's transition-probability precomputation is O(E x deg)); batch_words left at gensim default (ELISL: 4,
  throughput only). Graph change as above.
- `crispr_dependency_mut` / `crispr_dependency_expr`: mean CRISPR dependency probability of g1 in lines of the
  cancer type with / without g2 altered (and vice versa); altered = non-silent mutation / |expression z| >= 1.96.
  ELISL used DepMap 18Q3 `gene_dependency.csv` + CCLE 2019 cBioPortal; we use DepMap 24Q4 CRISPRGeneDependency,
  OmicsSomaticMutations (protein-altering / splice consequences) and log TPM z-scored per gene across all lines.
- `tissue` (13 features): TCGA mean expression-z of g1 in g2-mutated vs other tumours (x2), GTEx co-expression
  (r, p), TCGA tumour co-expression (r, p), TCGA normal co-expression (r, p), GISTIC Spearman (r, p), Cox p-value of
  g1&g2 co-alteration (mutation | |z| >= 1.96 | |GISTIC| = 2) stratified by sex, race, age quartile. ELISL used
  per-study cBioPortal/Firehose files; we use the PanCanAtlas Xena versions of the same TCGA samples (EB++ values
  back-transformed to linear scale for Pearson). The Cox model is a vectorised re-implementation (Efron ties,
  strata, Wald p) because lifelines took 3-4 s per fit (~115k fits); it matches lifelines 0.25.7 to 4 decimals
  on test data (e.g. 0.81337 vs 0.81337, 0.99925 vs 0.99932).
- Context -> cancer type (common.py): DepMap OncotreeLineage/OncotreeCode of the context's cell line -> TCGA code
  (BRCA, CESC, COAD, READ, LUAD, LUSC, PAAD, SKCM, HNSC, STAD, OV, LAML; T-ALL Jurkat and CML K-562 -> LAML as
  ELISL's 'Leukemia' group); free-text disease keyword fallback for contexts without a DepMap id (C092 -> SKCM);
  hTERT-RPE1 (non-cancer, 37% of human train rows) -> PANCAN: all tumours (z-scores within type, Cox strata + type),
  all GTEx samples, all DepMap lines. Cell-line groups = DepMap lines mapped to the same TCGA code. GTEx tissue
  per type = ELISL's map + analogous choices for PAAD (Pancreas), HNSC (Minor Salivary Gland), STAD (Stomach),
  LUSC (Lung), READ (Colon - Transverse).
- Model: the original `ELRRF` class (src/models/ELRRF.py) with ELISL's final configuration: 5 feature sets + their
  concatenation -> 6 LightGBM random forests (400 trees, 165 leaves, colsample 0.8, subsample 0.632), StandardScaler
  + VarianceThreshold, NaN -> 0, `undersample_train` with 5 repeats (`fit_predict_2set`), late integration weighted
  by training AUPRC. Changes: (1) one pooled model over all human contexts instead of one model per cancer type;
  undersampling is done within each context (ELISL's balance_cancer_by_index with cancer = context); (2) final
  score = mean over the 5 repeats' ensembled probabilities; (3) the released ELRRF.py has `balance_by_index` /
  `balance_cancer_by_index` inside a triple-quoted block (undefined at runtime), so elrrf.py exec's that exact
  source text into the module; (4) `init_model` in LGBMClassifier.fit needs lightgbm >= 3 (requirements.txt says
  2.3; the authors' conda env pins 3.1.1, which we use).

## Status
acquired / env built (uv venv py3.8 + docker slb/elisl-seqvec) / ran original code (ELRRF class, seqvec CLI) / adapted / dev-scored on slb1.2 and slb1.3

## Dev results
Native coverage (from .coverage.json): human 15,048/15,048 (slb1.2) and 19,149/19,149 (slb1.3); all other species
0 rows scored (constant fill = AUROC 0.5). Feature coverage: SeqVec 100% of pairs, PPI 99.8%.

| bench | SLB | H. sapiens | S. cerevisiae | S. pombe | human AFR | EAS | EUR | paralog pairs |
|---|---|---|---|---|---|---|---|---|
| slb1.2 dev | 0.5361 | 0.6443 | 0.5000 (fill) | 0.5000 (fill) | 0.6961 | 0.6444 | 0.5925 | 0.6205 |
| slb1.3 dev | 0.5369 | 0.6106 | 0.5000 (fill) | 0.5000 (fill) | 0.6148 | 0.6371 | 0.5799 | - |
slb1.3 auxiliary species: all n/a (<20 positives) and not scored. Human flat fitness-balanced stratum: 0.6061 (1.2), 0.5886 (1.3).
Training AUPRCs of the 6 forests (balanced, in-sample, used as ensemble weights): seq 0.97, ppi 0.94, dep-mut 0.77/0.78,
dep-expr 0.78, tissue 0.84/0.86, all 0.96: the embedding forests memorise the balanced train set, so ensemble weights
are almost uniform.

## Runtime
One-off: SeqVec 18 min on the RTX 3090 for 6,046 genes (CPU ~6 s/protein), node2vec ~25 min (8 threads), omics
caches ~15 min; per benchmark: cell-line features ~2 min, tissue features ~20 min for slb1.2 train (PANCAN dominant),
ELRRF training + scoring ~3.5 min (8 threads). Re-runs are incremental (per-gene / per-(pair, cancer) caches).

## Run
`SLB_BENCH=data/bench/slb1.3 SLB_SPLIT=dev scripts/models/elisl/run.sh` (ELISL_CPUS default 8; ELISL_GPU=1 for SeqVec
on GPU after claiming it) -> results/models/[<bench>/]elisl__slbtrain_<split>.parquet (+ .coverage.json, .txt/.json on dev).

## Problems / caveats
- Per-cancer training (ELISL's protocol) is not possible for most SLB contexts (few positives); pooled model instead.
- hTERT-RPE1 rows (largest human context) use pan-cancer tissue/cell-line features, which are not context-specific.
- Old stack: py3.8 + pyarrow 4 sometimes segfaults at interpreter exit after writing outputs; features.py ends with os._exit(0).
