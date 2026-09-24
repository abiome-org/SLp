# dekegel2021: paralog-SL random forest (De Kegel et al. 2021, Cell Systems)

battery: dekegel2021__pretrained, dekegel2021__slbtrain (human), dekegel2021__allspecies (every species with bundle data); species: human (pretrained, slbtrain), all benchmark species (allspecies); needs: Ens111 paralog feature table (Ryan lab, Zenodo 14973633), docker image slb/dekegel2021 (python 3.7 / sklearn 0.23.1) for the released pickle

PPI/KG: __pretrained/__slbtrain use the Ryan-lab Ens111 feature table whose PPI features (interact, n_total_ppi, fet_ppi_overlap, shared_ppi_mean_essentiality) come from BioGRID filtered to `Experimental System Type == physical` (1_data_processing/paralog_features/02_process_ppi.ipynb) - no genetic interactions, no STRING. __allspecies uses the bundle ppi.parquet (STRING neighborhood/fusion/cooccurence/coexpression/database only - experimental, textmining and combined_score dropped - plus BioGRID physical). Not affected by the STRING warning.

- Paper: De Kegel B, Quinn N, Thompson NA, Adams DJ, Ryan CJ. Comprehensive prediction of robust synthetic lethality between paralog pairs in cancer cell lines. Cell Systems 12:1144 (2021). doi:10.1016/j.cels.2021.08.006
- Repo: https://github.com/DeKegel/paralog_SL_prediction (commit 414e7e447158ba58044d108ead386cb4a99c4d3c, 2022-05-29), cloned to external/models/dekegel_paralog_sl. No LICENSE file in the repo.
- Weights: `local_data/results/RF_model.pickle` (RandomForestClassifier, 600 trees, max_depth 3, max_features 0.5, min_samples_leaf 8, 22 features; sklearn 0.23.1).
- Original training data: 3,810 paralog pairs labelled from DepMap 20Q2 (CERES re-run without multi-targeting guides) + CCLE: A2 is a dependency in lines with homozygous loss of A1 ("robust SL") -- i.e. single-gene CRISPR screens in natural loss backgrounds. No combinatorial-screen labels.
- Features: 22 pair features (sequence identity, family size, WGD, yeast/pombe orthologs and essentiality, conservation, age, protein complexes, PPI overlap, GTEx co-expression, ...). We use the Ryan lab's Ensembl 111 rebuild of the same features: data/raw/ryan_paralog_features/ens111_human_allFeatures.csv (Zenodo 14973633, sha256 in SOURCES.tsv; 104,303 human paralog pairs), matched to SLB pairs through Ensembl IDs -> current HGNC symbols.

## Leakage
- `__pretrained`: labels are DepMap-derived paralog dependencies (single-gene screens + genotype), permitted single-gene data under the contract => **not leaky**, with the caveat that the label is a genotype-conditioned single-gene dependency (closest thing to SL that single-gene data give). Some features (complex/PPI essentiality) also use DepMap.
- `__slbtrain`: same 22 features and hyper-parameters, refit on SLB-1.2 train rows only (54,713 human train rows with paralog features, 2,107 SL) => **not leaky**.

## Changes vs original
- Ens111 feature rebuild instead of the paper's Ens93 table; pickle applied unchanged (sklearn 0.23.1 in Docker).
- Human pairs that are not Ensembl paralogs get score 0 (the model is defined only on paralog pairs; 0 = its prior). Other species: not scored (median-filled by eval). Native coverage: 10,035/15,048 human dev rows (95% of same-family rows, 2.5% of different-family rows).
- Retrain: one training row per (context, pair) SLB row; the model stays context-agnostic.

## Status
acquired / env built (Docker py3.7) / ran original (released pickle) / adapted / dev-scored (both variants)

## Dev results (SLB-1.2 dev, `--allow-missing`: 117,674 non-human rows median-filled)
| variant | SLB | H. sapiens | AFR | EAS | EUR | human flat | paralog stratum |
|---|---|---|---|---|---|---|---|
| __pretrained | 0.5412 | 0.6650 | 0.7727 | 0.6783 | 0.5440 | 0.5615 | 0.6279 |
| __slbtrain | 0.5456 | 0.6823 | 0.7788 | 0.7257 | 0.5423 | 0.5650 | 0.6371 |
Reference on the same split: paralog_identity baseline human 0.599, codependency 0.652, lgbm 0.596. Human species score is noisy (AFR has 22 dev positives; `random` scores 0.556 on human).
SLB-train RF importances: fet_ppi_overlap 0.38, conservation_score 0.13, mean_complex_essentiality 0.13, min_sequence_identity 0.09.

Runtime: ~25 s (features + both variants), CPU only.

## Run
`SLB_BENCH=data/bench/slb1.2 SLB_SPLIT=dev scripts/models/dekegel2021/run.sh` -> results/models/dekegel2021__{pretrained,slbtrain}_<split>.parquet (+ .txt/.json on dev). Caches per benchmark version in external/models/dekegel_paralog_sl/_slb/<bench>/.

## allspecies variant (scripts/models/dekegel2021/allspecies.py)
Same RF (600 trees, depth 3, max_features 0.5, min_leaf 8), one per species, fitted on that species' SLB train rows;
features rebuilt from the shared bundle for any species: paralog identity / is_paralog / family size / closest
(Ensembl 116 via slpbench.homology, else DIAMOND/Ensembl paralog edges from slpbench.families_extra), ESM-2 cosine,
PPI interact / n_total_ppi / Fisher shared-neighbour overlap / Jaccard / mean fitness of shared neighbours (bundle
PPI: STRING without experimental + text-mining, BioGRID physical), GO CC colocalisation and GO BP Jaccard (IGI
dropped). GTEx / CORUM / DepMap features are human-only and not used here. Not leaky (bundle inputs + SLB train).
SLB-1.2 dev: SLB 0.6419 (95% CI 0.545-0.702), human 0.645, scer 0.583, spom 0.620, spne 0.720; paired vs
codependency +0.104 [+0.020, +0.155]. Runtime ~12 min (8 threads), dominated by the Fisher tests on scer train pairs.

## Problems / notes
- The released-feature variants cover only Ensembl paralog pairs; the allspecies variant is the one to use across species.

## SLB-1.3 dev results (results/models/slb1.3/; SLB_BENCH=data/bench/slb1.3)
Headline = mean of human/scer/spom; auxiliary columns are the per-species strata (bsub/cele/dmel have < 20 dev positives, not in the headline). Species a model does not score get a constant (AUROC 0.5); `native` lists the species actually scored (from <name>_dev.coverage.json).

| model | SLB | human | scer | spom | bsub | cele | dmel | paralog | native |
|---|---|---|---|---|---|---|---|---|---|
| dekegel2021__allspecies | 0.5865 | 0.6325 | 0.5636 | 0.5634 | 0.6035 | 0.6416 | 0.5188 | 0.6616 | bsub,cele,dmel,human,mmus,scer,spom |
| dekegel2021__pretrained | 0.5524 | 0.6572 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.6257 | human |
| dekegel2021__slbtrain | 0.5572 | 0.6717 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.6248 | human |

Reference on SLB-1.3 dev: lgbm 0.5315 (human 0.558), paralog_identity 0.5392 (human 0.608), codependency 0.5149, random 0.4854.
Paired bootstrap (200) vs paralog_identity on SLB-1.3 dev: allspecies +0.048 [+0.013, +0.102] (human +0.026, scer +0.056, spom +0.061); slbtrain +0.018 [-0.008, +0.060]. (results/models/slb1.3/_models-features_compare_vs_paralog_identity_dev.txt)
(re-run 2026-09-24 with bundle v1.2 incl. ESM-2 for cele/dmel/mmus: 0.5865; earlier run 0.5868. Compare numbers above refer to the earlier file.)
