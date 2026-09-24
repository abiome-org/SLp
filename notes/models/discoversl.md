# discoversl: DiscoverSL (Das et al. 2019, Bioinformatics 35:701)

battery: discoversl__released, discoversl__slbtrain; species: human; needs: TCGA PanCanAtlas (Xena), DiscoverSL_2.0 package (sysdata.rda: RF + KEGG/REACTOME gene sets), docker slb/r-stats

PPI/KG: no PPI; pathway feature = KEGG + REACTOME gene-set co-membership from the package (curated pathways, no GI evidence).

- Paper: Das S, Deng X, Camphausen K, Shankavaram U. DiscoverSL: an R package for multi-omic data driven prediction of synthetic lethality in cancers. Bioinformatics 35:701 (2019). doi:10.1093/bioinformatics/bty673
- Repo: https://github.com/shaoli86/DiscoverSL (commit 32bf8c2, README only); package = release asset V1.0 `DiscoverSL_2.0.tar.gz` (data/raw/discoversl, sha256 in SOURCES.tsv; 349 MB unpacked in external/models/discoversl/pkg). License GPL-2.0.
- Weights: `model` in R/sysdata.rda = randomForest regression, 700 trees, mtry 1, 4 features; trained on 1,265 SL / 422 non-SL literature/screen pairs => **leaky**.
- Features (per ordered pair gene1 primary, gene2 partner): DiffExp p (gene2 expression, gene1-mutant vs WT), Pearson co-expression p, Mutex (Fisher-combined 1 - P[co-occurrence] of mutation, amplification, deep deletion), 1 - pathway co-membership hypergeometric p (package KEGG + REACTOME sets).
- The package fetched TCGA through the retired cBioPortal CGDS API (cgdsr); re-implemented in Python (scripts/models/discoversl/features.py) on TCGA PanCanAtlas (EB++ expression, MC3 mutations, GISTIC2). DEVIATION: DiffExp uses a Welch t-test on log2 expression instead of edgeR exactTest on counts; pan-cancer instead of per cancer type. Pair score = max over the two orientations.

## Leakage
- __released: released RF (literature SL labels) applied unchanged => **leaky**.
- __slbtrain: same RF form (regression, 700 trees, mtry 1) refit on SLB train (273,964 oriented rows, 5,930 SL) => **not leaky**.

## Status
acquired / env built (Docker R 4.3 + randomForest; python features) / released model ran / adapted / dev-scored

## Dev results (human rows lacking a mutated-vs-WT contrast are missing: 435 human rows median-filled + 117,674 non-human)
| variant | SLB | H. sapiens | AFR | EAS | EUR | human flat | paralog |
|---|---|---|---|---|---|---|---|
| __released | 0.4999 | 0.4997 | 0.531 | 0.485 | 0.483 | 0.485 | 0.486 |
| __slbtrain | 0.5202 | 0.5807 | 0.588 | 0.585 | 0.569 | 0.564 | 0.593 |
Runtime: features ~1 min (dev) / ~5 min (train); RF ~2 min.

## SLB-1.3 dev results (results/models/slb1.3/; SLB_BENCH=data/bench/slb1.3)
Headline = mean of human/scer/spom; auxiliary columns are the per-species strata (bsub/cele/dmel have < 20 dev positives, not in the headline). Species a model does not score get a constant (AUROC 0.5); `native` lists the species actually scored (from <name>_dev.coverage.json).

| model | SLB | human | scer | spom | bsub | cele | dmel | paralog | native |
|---|---|---|---|---|---|---|---|---|---|
| discoversl__released | 0.4894 | 0.4682 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.4819 | human |
| discoversl__slbtrain | 0.5293 | 0.5880 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5735 | human |

Reference on SLB-1.3 dev: lgbm 0.5315 (human 0.558), paralog_identity 0.5392 (human 0.608), codependency 0.5149, random 0.4854.
