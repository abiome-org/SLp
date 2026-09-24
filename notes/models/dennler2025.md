# dennler2025: paralog sequence/structure-similarity classifiers (Ryan lab 2025)

battery: dennler2025__released, dennler2025__slbtrain, dennler2025__slbtrain_all58; species: human; needs: Ens111 paralog feature table + released predictions (Zenodo 14973633), xgboost (uv venv external/models/ryan_paralog_seq_similarity/.venv)

PPI/KG: the 36 features are sequence/structure/PLM only (no PPI). __slbtrain_all58 adds the De Kegel features, whose PPI part is BioGRID physical only (no STRING). Not affected by the STRING warning.

- Paper: Dennler O, ..., Ryan CJ. Evaluating sequence and structural similarity metrics for predicting shared paralog functions (in preparation / 2025). Code: https://github.com/cancergenetics/paralog_seq_similarity (commit a3e31298fd9fdcb6b3c5531e7b05442b28f566f2), data https://zenodo.org/records/14973633 (DOI 10.5281/zenodo.14975580 for the code). License: none stated in the repo.
- Model: StandardScaler + XGBClassifier (600 trees, lr 0.1, colsample 0.5) on 36 features (min sequence identity, Foldseek structure similarity, MMseqs2 similarity-search context, ESM-2 and ProtT5 embedding distances).
- Released predictions: `ens111_human_allPredictions.csv`, column `SL | All 36 features XGB` (trained on the DepMap-derived paralog SL labels `ens111_human_SL.csv`, same label definition as De Kegel 2021).

## Leakage
- `__released`: DepMap-derived labels only => **not leaky** (same caveat as dekegel2021).
- `__slbtrain` (36 features) and `__slbtrain_all58` (36 + the 22 De Kegel features): refit on SLB train only => **not leaky**.
- The yeast predictions in the same release use yeast negative-GI labels (Costanzo) and would be leaky; not used.

## Status
acquired / env built / released predictions used / adapted (retrain) / dev-scored

## Dev results (117,674 non-human rows median-filled; human pairs without paralog features scored 0; native human coverage 10,035/15,048)
| variant | SLB | H. sapiens | AFR | EAS | EUR | human flat | paralog stratum |
|---|---|---|---|---|---|---|---|
| __released | 0.5306 | 0.6225 | 0.6860 | 0.6373 | 0.5443 | 0.5547 | 0.6103 |
| __slbtrain | 0.5267 | 0.6067 | 0.6640 | 0.6155 | 0.5405 | 0.5488 | 0.6006 |
| __slbtrain_all58 | 0.5459 | 0.6836 | 0.8027 | 0.6962 | 0.5518 | 0.5705 | 0.6500 |
Runtime ~25 s CPU.

## Run
`scripts/models/dennler2025/run.sh` (SLB_BENCH, SLB_SPLIT).

## SLB-1.3 dev results (results/models/slb1.3/; SLB_BENCH=data/bench/slb1.3)
Headline = mean of human/scer/spom; auxiliary columns are the per-species strata (bsub/cele/dmel have < 20 dev positives, not in the headline). Species a model does not score get a constant (AUROC 0.5); `native` lists the species actually scored (from <name>_dev.coverage.json).

| model | SLB | human | scer | spom | bsub | cele | dmel | paralog | native |
|---|---|---|---|---|---|---|---|---|---|
| dennler2025__released | 0.5539 | 0.6616 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.6270 | human |
| dennler2025__slbtrain_all58 | 0.5534 | 0.6601 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.6168 | human |
| dennler2025__slbtrain | 0.5233 | 0.5699 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5575 | human |

Reference on SLB-1.3 dev: lgbm 0.5315 (human 0.558), paralog_identity 0.5392 (human 0.608), codependency 0.5149, random 0.4854.
