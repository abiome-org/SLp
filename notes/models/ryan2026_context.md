# ryan2026_context: context-specific paralog SL random forest (Ryan lab, bioRxiv 2026)

battery: ryan2026_context__released_ctx, ryan2026_context__slbtrain_ctx, ryan2026_context__slbtrain_full; species: human; needs: DepMap 24Q4 (expression, Chronos, CN, damaging mutations), Ens111 paralog table, De Kegel 2021 released scores (dekegel2021 run.sh), bundle human GO + PPI

PPI/KG: the paper used STRING combined scores (includes experimental + textmining => GI-contaminated); __released_ctx inherits that (already leaky for labels). Our __slbtrain_* variants compute weighted_PPI_* from the bundle ppi.parquet (STRING experimental/textmining/combined dropped; BioGRID physical only) and the De Kegel prediction_score (BioGRID physical). GO from the bundle with IGI evidence removed.

- Paper: "Systematic prioritisation of context-specific paralog pair vulnerabilities in cancer" (Kebabci N, ..., Ryan CJ), bioRxiv 2026, doi:10.64898/2026.01.19.700065.
- Repo: https://github.com/cancergenetics/context_specific_paralog_SL (commit f966551, MIT), external/models/ryan_context_paralog_sl. Weights: data/output/models/{contextualised_model,full_model}.pickle.gz (RF 600 trees, depth 20, max_features 0.2, min_leaf 4) and GIMAP-label versions.
- Original training labels: GEMINI-scored combinatorial screens (Ito 2021, Parrish 2021, Klingbeil 2024, Harle 2025, CHyMErA) => released models are **leaky** (Parrish and Harle are SLB-1.2 label sources).
- Features per (cell line, pair): see header of scripts/models/ryan2026_context/run.py. DEVIATIONS: DepMap 24Q4 instead of 22Q4; PPI weights from the leakage-safe bundle (STRING minus experimental/text-mining + BioGRID physical) instead of STRING combined; GO from the bundle (IGI dropped); Protein_Altering approximated by the damaging flag; Ens111 paralog table instead of De Kegel Table S8.
- __slbtrain_*: refit on SLB train rows with DepMap line data (94,919 rows, 2,938 SL) => **not leaky** (the prediction_score feature is the De Kegel RF trained on DepMap-derived labels).
- Coverage: 12,606/15,048 human dev rows (hTERT-RPE1 and C092 are not DepMap lines; 2,442 rows median-filled) + 117,674 non-human.
- Status: acquired / env (project uv env; pickles load with sklearn >= 1.x warnings) / released model ran / adapted / dev-scored.

| variant | SLB | H. sapiens | AFR | EAS | EUR | human flat | paralog |
|---|---|---|---|---|---|---|---|
| __released_ctx (leaky) | 0.5214 | 0.5857 | 0.618 | 0.569 | 0.570 | 0.566 | 0.598 |
| __slbtrain_ctx | 0.5313 | 0.6250 | 0.695 | 0.622 | 0.559 | 0.565 | 0.643 |
| __slbtrain_full | 0.5384 | 0.6536 | 0.712 | 0.671 | 0.578 | 0.586 | 0.654 |
Top features (slbtrain_ctx): prediction_score 0.18, min_sequence_identity 0.10, min_ranked_A1A2 0.08, rMinExp 0.08, weighted_PPI_essentiality 0.08.
Runtime: feature build ~10 min (train + dev, cached per bench in external/models/_statsl_cache/ryan2026ctx_*), RF ~2 min.

## SLB-1.3 dev results (results/models/slb1.3/; SLB_BENCH=data/bench/slb1.3)
Headline = mean of human/scer/spom; auxiliary columns are the per-species strata (bsub/cele/dmel have < 20 dev positives, not in the headline). Species a model does not score get a constant (AUROC 0.5); `native` lists the species actually scored (from <name>_dev.coverage.json).

| model | SLB | human | scer | spom | bsub | cele | dmel | paralog | native |
|---|---|---|---|---|---|---|---|---|---|
| ryan2026_context__released_ctx | 0.5317 | 0.5951 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5904 | human |
| ryan2026_context__slbtrain_ctx | 0.5337 | 0.6012 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.6075 | human |
| ryan2026_context__slbtrain_full | 0.5439 | 0.6317 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.6178 | human |

Reference on SLB-1.3 dev: lgbm 0.5315 (human 0.558), paralog_identity 0.5392 (human 0.608), codependency 0.5149, random 0.4854.
