# Results

Public SL pair-prediction benchmarks are the evaluation target. This file records what has been reconstructed and what has been run locally.

## Protocol

Scores are not comparable across papers unless dataset version, gene-overlap rule, class prevalence, negative definition, task, metric, and feature access are the same.

Split names used here:

- pair holdout: test pair unseen; both genes may appear in other training pairs
- one-new-gene: exactly one test gene absent from supervised SL training
- two-new-gene: both test genes absent from supervised SL training

## Reconstructed published numbers (not local training)

Feng et al. 2024 complete-label table, NSM random, PNR 1:1, from `summary_all_matrics.csv`:

| model | pair-holdout AUROC/AUPR/F1 | two-new-gene AUROC/AUPR/F1 | two-new-gene NDCG@10 |
| --- | --- | --- | --- |
| SLMGAE | 0.934 / 0.950 / 0.883 | 0.790 / 0.796 / 0.738 | 0.039 |
| GCATSL | 0.943 / 0.951 / 0.883 | 0.678 / 0.660 / 0.692 | 0.002 |
| KG4SL | 0.941 / 0.950 / 0.878 | 0.562 / 0.575 / 0.667 | 0.000 |
| NSF4SL | 0.932 / 0.945 / 0.869 | 0.683 / 0.695 / 0.685 | 0.004 |
| PiLSL | 0.927 / 0.934 / 0.863 | 0.626 / 0.630 / 0.670 | (not reported) |
| PTGNN | 0.925 / 0.939 / 0.869 | 0.529 / 0.517 / 0.670 | 0.010 |

SynLeaF pan-cancer CV3, author table, 1:1 balance, 33746 positives: AUC 0.7407, AUPR 0.7611. Different gene filter and negative construction than Feng.

Cilantro-sl gene-holdout AUPR 0.7148 (KG4SL 0.7325 in that figure). Split equivalence to two-new-gene is unverified.

## Local runs

`slp-bench` on released pair tables, 5-fold, 1:1 random unknowns, seed 123, fold-local SL degree:

| export | genes | positives | pair-holdout AUROC/AUPR | one-new-gene | two-new-gene |
| --- | --- | --- | --- | --- | --- |
| Feng `human_sl_9845.csv` | 9845 | 35913 | 0.928 / 0.942 | 0.818 / 0.868 | 0.500 / 0.488 |
| SynLethDB 2.0 Human_SL.csv | 10218 | 36741 | 0.931 / 0.945 | 0.824 / 0.873 | 0.500 / 0.483 |
| SynLethDB 3.0 Human.SL.detailed.tsv | 10038 | 37536 | 0.928 / 0.943 | 0.827 / 0.874 | 0.500 / 0.484 |

On two-new-gene every test pair has training degree 0. Pair-holdout degree already reaches the 0.93 AUROC range that published graph models report on random pair splits.

Feng et al. released code, splits, the metric tables, and ranked pairs in `predicted_by_model.csv`. They did not release fold-wise SL predictor checkpoints. Retraining those 12 methods is not required for this project.

Released weights being collected under `data/models/weights/`: Geneformer V1-10M / V2-104M / V2-104M cancer-continual, GEARS-Norman, Arc State SE-600M, ST-SE-Tahoe zeroshot `best.ckpt`, ESM-2 650M. Cilantro-sl's Geneformer fork is `data/models/geneformer-cilantro`. SynLeaF and the Feng graph methods ship training code only.

## SL-Predict v0

The executable outline is in `docs/model-card.md`. The protein-aware transition model has six 384-wide transformer layers and 11,919,712 trainable parameters. Its transition accepts single, simultaneous and sequential gene actions and emits a stochastic next-state latent, decoded molecular relations and uncertainty. Pretraining uses no SL labels; reinforcement learning rewards held-out molecular-relation recovery.

The fixed feature model produced these means on the released Feng random-negative 1:1 folds:

| model | pair-holdout AUROC/AUPR | one-new-gene AUROC/AUPR | two-new-gene AUROC/AUPR |
| --- | --- | --- | --- |
| fixed non-SL feature model | 0.7589 / 0.7680 | 0.7354 / 0.7447 | 0.7055 / 0.7088 |
| relation world model + training-fold calibration | 0.8110 / 0.8228 | 0.7681 / 0.7756 | 0.7165 / 0.6994 |
| relation world model, label-free score | 0.6464 / 0.6294 | 0.6460 / 0.6298 | 0.6424 / 0.6264 |
| relation + quantitative viability + calibration | 0.8417 / 0.8593 | 0.7988 / 0.8137 | 0.7456 / 0.7382 |
| quantitative viability, label-free score | 0.5756 / 0.5763 | 0.5749 / 0.5763 | 0.5752 / 0.5718 |
| protein + strict quantitative viability + calibration | — | — | 0.7653 / 0.7616 |
| protein + strict quantitative viability, label-free | — | — | 0.5704 / 0.5759 |
| protein + native interaction magnitude + calibration | — | — | 0.7615 / 0.7523 |
| protein + native interaction magnitude, label-free | — | — | 0.5763 / 0.5762 |
| compact world state + deterministic tabular ensemble | — | — | 0.7716 / 0.7648 |
| 256-landmark world state + deterministic tabular ensemble | — | — | 0.7713 / 0.7688 |
| equal average of compact and 256-landmark states | — | — | 0.7855 / 0.7819 |
| cancer-continued 256-landmark state + deterministic ensemble | — | — | 0.7670 / 0.7656 |
| spectral biological state + deterministic ensemble | — | — | 0.7741 / 0.7724 |
| equal compact + 256-landmark + spectral prediction average | — | — | 0.8027 / 0.8000 |
| safe spectral state + deterministic ensemble | — | — | 0.8074 / 0.8137 |
| equal compact + 256-landmark + safe spectral average | — | — | 0.8116 / 0.8108 |
| 59.6M native interaction model + calibration | — | — | 0.7125 / 0.7050 |

The full relation run used 12 pretraining epochs and three self-critical policy-gradient epochs on a Modal L4. Its checkpoint and fold metrics are in `results/sl_predict/relation_full/latest/`. These numbers are below the released SLMGAE Feng CV3 result and are not SOTA.

The quantitative viability run additionally fitted the transition and outcome decoder to normalized SLKB guide depletion, then applied a second policy-gradient phase. Its checkpoint and metrics are in `results/sl_predict/outcome_full/latest/`. CV3 standard deviation is 0.0443 AUROC and 0.0452 AUPR; one fold reaches 0.8101/0.8010 while another is 0.6861/0.6824. The calibrated mean improves, but the weak label-free score shows that emergence has not yet been established.

Adding State/ESM protein features and excluding every pair found anywhere in the released Feng matrix raises CV3 to 0.7653/0.7616 (SD 0.0326/0.0407); the best fold is 0.8150/0.8258. The checkpoint and metrics are in `results/sl_predict/protein_strict_run/`. This remains below SLMGAE's 0.7901/0.7964 mean and the absolute-viability label-free score remains weak.

Replacing absolute viability with SLKB's native continuous interaction magnitude gives 0.7615/0.7523 after calibration (SD 0.0427/0.0530) and 0.5763/0.5762 without labels. Restoring the fixed modality summaries discarded by latent compression changes calibration only to 0.7628/0.7540. A deterministic readout—ExtraTrees, LightGBM and three explicitly seeded neural fits—reaches 0.7716/0.7648. Increasing each biological relation view from 32 to 256 landmarks gives 0.7713/0.7688. Their equal prediction average reaches 0.7855/0.7819, the strongest confirmatory local result. The 59.6M model improves held-out relation and phenotype losses to 0.01206 and 0.24785, respectively, but falls to 0.7125/0.7050 on CV3. Thus proxy-task scaling does not select better cold-start geometry.

Replacing generic Geneformer tokens with cancer-continued Geneformer tokens slightly improves held-out relation loss to 0.01285, but gives only 0.7670/0.7656 after deterministic calibration and 0.5537/0.5805 label-free. An equal three-representation average gives 0.7852/0.7827. Choosing a 0.2 cancer-state weight after inspecting the outer folds gives 0.7861/0.7833; this is recorded as exploratory hill climbing, not a confirmatory estimate. All three remain below SLMGAE's released 0.7901/0.7964 result.

Replacing random landmarks with 32 leading coordinates from each of six explicitly non-SL biological graphs gives 0.7741/0.7724 with a deterministic readout and 0.6083/0.5957 from the label-free transition score. A stricter retraining zeros the 400-dimensional frozen knowledge-graph block because the training corpus of that exact embedding is unresolved. Despite also removing all SLAMR validation and test pairs from quantitative pretraining, this safe checkpoint reaches 0.8074 AUROC, 0.8137 average precision, 0.8135 trapezoidal PR-AUC and 0.7449 maximum F1. The equal, fixed average of the compact, 256-landmark and safe spectral readouts reaches 0.8116/0.8108/0.8107/0.7523. The standalone safe model and the ensemble exceed the released SLMGAE summary of 0.7901/0.7964/0.7378 on the same balanced random CV3 files. These are development SOTA estimates, not confirmatory estimates: the spectral representation was introduced after outer-fold inspection, and two ensemble components retain the unresolved frozen KG feature. The released SLMGAE number is also developmental because its standard runner evaluates the nominal test fold every epoch and retains maxima selected by test F1 and test NDCG@10.

Marginalizing predicted outcomes over 28 observed study-cell contexts does not explain the remaining emergence gap. Mean depletion reaches 0.6072/0.5913 and top-three depletion reaches 0.6062/0.5916, both below the direct unknown-context score. Non-additive residual scores are weaker.

For the 12.78M 256-landmark model, both reinforcement phases were rejected by fixed label-free validation: relation loss worsened from 0.01296 to at best 0.01580, and phenotype loss worsened from 0.25683 to at best 0.25972. Restoring the pre-reinforcement parameters prevents objective drift but gives only 0.7458/0.7421 with the native head and 0.5527/0.5757 label-free. Validation guarding is therefore necessary for fidelity, but does not by itself produce emergent SL discrimination.

Four secondary heads were rejected. Fold-safe transfer from 51,013 disjoint SLKB binary pairs reduced CV3 to about 0.731/0.715, indicating label-source shift. A six-view biological graph head with fold-local SL messages was near random (about 0.518 AUROC). A cosine metric head was 0.569/0.573. A full-state symmetric neural head overfits at 0.656/0.652. These families were not tuned further on the outer folds.

An additive gene-effect residual fitted from the sparse double-knockout table was rejected: calibration fell to 0.7531/0.7459 and the label-free score was 0.5011/0.5164. The retained checkpoint and metrics are in `results/sl_predict/interaction_full/`. The replacement target is the magnitude of SLKB's native continuous interaction score, normalized within study and cell context without binary SL calls.

The local SLKB dump was independently recounted: 3,578,017 guide-pair count rows and 280,483 original pair results across 11 studies and 22 cell lines, including 16,059 labelled SL rows. T0 and endpoint counts were normalized by sample library size. Before quantitative viability training, every exact pair appearing in any of the 72 local Feng split files was removed; 81,120 mapped trajectories remain. Binary SL calls are not present in the training pack. A compact 135.1 MiB evaluation artifact retains all 24 released CV3 protocols and their 49,973,386 train/test pair occurrences.

An apparent label-free result on SLAMR scenario 3 was disqualified after an overlap audit: approximately 38–44% of its fold-specific test pairs occurred in the quantitative SLKB pretraining set. A new exact-pair-isolated pack removes the union of every mapped SLAMR scenario-3 validation and test pair from A549, Jurkat and K562, retaining 57,788 of 81,120 trajectories. This is exact-pair isolation, not intervention isolation; other pairs containing the held-out genes may remain. External scores from the safe no-knowledge-graph model are reported only after this exclusion.

The safe model produces the following label-free scenario-3 ranking means. Unknown-context scoring is the primary result; the matched-context column uses a trained cell-line context token but still no SLAMR labels during scoring or model fitting.

| cell line | folds with positives | unknown-context MRR / Recall@20 | matched-context MRR / Recall@20 |
| --- | ---: | ---: | ---: |
| K562 | 5 | 0.2340 / 0.4360 | 0.2313 / 0.4409 |
| Jurkat | 5 | 0.2285 / 0.5619 | 0.2546 / 0.6143 |
| A549 | 4 | 0.2715 / 1.0000 | 0.3158 / 1.0000 |

A549 is not a stable headline because one fold has no positive query and the remaining test sets are small. For context only, the local literature transcription reports SLAMR K562 scenario-3 MRR 0.09 and Recall@20 0.25; that author comparison has not yet been verified from a primary accessible table. The K562 and Jurkat scores above are the strongest evidence here that useful SL ranking emerges from the transition model, but exact-pair isolation is weaker than removing all interventions involving held-out genes.

That interpretation does not survive the stronger test. Removing every quantitative trajectory containing any of the 492 genes in a SLAMR scenario-3 validation or test set retains 16,093 trajectories, 4,526 unique pairs and 23 contexts. The K562 and Jurkat screen contexts contribute no training rows. Results with the unchanged label-free strength score are:

| intervention-isolated training rule | Feng CV3 AUROC / AP | K562 MRR / Recall@20 | Jurkat MRR / Recall@20 |
| --- | ---: | ---: | ---: |
| pair-held-out outcome selection | 0.6270 / 0.6196 | 0.0879 / 0.2386 | 0.0562 / 0.2247 |
| inner two-new-gene outcome selection | 0.6347 / 0.6456 | 0.1099 / 0.2172 | 0.0540 / 0.2086 |
| frozen relational backbone | 0.5239 / 0.5110 | 0.0722 / 0.1971 | 0.0810 / 0.2715 |

Thus the high exact-pair-isolated SLAMR result depended on quantitative interventions involving the benchmark genes and is not evidence of fully intervention-cold emergence. Inner two-new-gene selection improves generic CV3 but does not solve cell-specific distribution transfer.

Context-conditioned GPT-5.1 gene-description embeddings score 0.1236/0.2644 on K562 and 0.1285/0.3213 on Jurkat without SL labels. A fixed 50/50 rank fusion with the world-model score does not improve consistently. Adding a 32-dimensional fixed projection of their cell-level mean to the transition token also fails: 0.0960/0.2111 on K562 and 0.0567/0.2060 on Jurkat. The next model requires actual basal cell expression or dependency state rather than a learned ID or averaged semantic proxy.

That measured-state experiment uses the immutable [DepMap Public 24Q2 release](https://doi.org/10.25452/figshare.plus.25880521.v1). The complete expression and CRISPR gene-effect files were streamed, reduced to 19 exactly resolved cell models and verified against the release MD5 values `f00c434db50d811544ac4d047003979f` and `d10d73692672380b3792231549e97d5a`. Parental 293T and RPE1 were left unresolved rather than mapped to engineered derivatives. K562 and Jurkat do not participate in normalization because they have no retained quantitative outcome trajectory. A 130-dimensional state contains independent expression and single-gene dependency projections; its global projection excludes all 492 held-out scenario-3 genes.

Measured state does not improve direct intervention-cold transfer:

| state input | Feng label-free CV3 AUROC / AP | K562 unknown / state MRR | K562 unknown / state Recall@20 | Jurkat unknown / state MRR | Jurkat unknown / state Recall@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 130-dimensional cell state | 0.6375 / 0.6489 | 0.1092 / 0.1090 | 0.1959 / 0.2004 | 0.0504 / 0.0504 | 0.2129 / 0.2058 |
| cell state plus aligned expression/dependency for both genes | 0.6324 / 0.6451 | 0.1109 / 0.0980 | 0.2353 / 0.1818 | 0.0551 / 0.0556 | 0.2396 / 0.1788 |

The aligned model has 11,972,704 parameters. It slightly improves inner two-new-gene outcome loss from 0.48658 to 0.48626, but this proxy improvement does not transfer to the state-conditioned external score.

An additional 11,968,864-parameter model was pretrained to predict single-gene CRISPR dependency from basal expression across 1,063 other DepMap cell models. The auxiliary pack contains 10,203,206 measured cell-gene effects. A549, K562, Jurkat and all 492 scenario-3 held-out genes were excluded from auxiliary fitting; the relational transformer was frozen so only the expression-state projection and state/outcome path learned. Validation Huber loss reached 0.38589. Subsequent intervention-isolated double-knockout training and reinforcement learning produced 0.6327/0.6441 label-free Feng CV3 AUROC/AP.

This all-cell dynamics objective improves some external recall but is not retained as the primary model:

| score | K562 MRR / Recall@20 | Jurkat MRR / Recall@20 |
| --- | ---: | ---: |
| unknown-context transition | 0.0993 / 0.2781 | 0.0721 / 0.1832 |
| expression-conditioned transition | 0.1087 / 0.2690 | 0.0653 / 0.2316 |
| label-free text prior | 0.1236 / 0.2644 | 0.1285 / 0.3213 |

The auxiliary mapping therefore contains useful K562 recall signal, but it does not improve MRR or consistently transfer across cells. Predicting marginal single-gene dependency from expression is insufficient to align the learned transition with context-specific double-knockout ordering.

The local host also contained measured Perturb-seq artifacts that were absent from the original repository inventory. Adamson 2016, Dixit 2016, Norman 2019, Replogle 2022 K562 and Replogle 2022 RPE1 were mapped to the Feng gene universe. Every row containing any of the 492 scenario-3 validation/test genes was excluded from fitting. A deterministic gene-cold partition retains 5,941 training and 1,388 validation pseudobulks; the Norman subset contributes 164 double-perturbation training rows and 16 validation rows for which both genes are unseen in training. Across studies, 207 measured expression genes support a 32-dimensional standardized state target explaining 54.48% of training variance.

The compact 11.92M transformer was initialized from the safe relational checkpoint. Its gene encoder and relation transformer were frozen while the transition path and a 4,128-parameter expression decoder learned single- and double-perturbation state. Training balances studies, upweights double perturbations fivefold, drops the study token for half of supervised examples and selects on held-gene state loss plus held-double loss. Three self-critical stochastic-rollout epochs were allowed, with validation rollback. A subsequent SLKB fit updates only the outcome head, preserving the learned transition. No SL labels enter the state objective or select its score.

| frozen readout | Feng CV3 AUROC / AP | K562 MRR / Recall@20 | Jurkat MRR / Recall@20 |
| --- | ---: | ---: | ---: |
| supervised SL head | 0.7735 / 0.7796 | — | — |
| label-free quantitative outcome | 0.5735 / 0.5519 | 0.0955 / 0.2532 | 0.1165 / 0.3451 |
| expression-state non-additivity | 0.4447 / 0.4587 | **0.1300 / 0.3033** | 0.1102 / 0.2078 |
| fixed gene-description text prior | — | 0.1236 / 0.2644 | 0.1285 / 0.3213 |

The expression-state score is the norm of the predicted double-perturbation state minus the two predicted single-perturbation states. It improves both K562 metrics over the prior fully intervention-cold transition and over the label-free text prior, despite being anti-predictive on generic Feng CV3. This is the first retained evidence here of SL ranking emerging from a measured, held-gene cellular-state objective. It is context-specific rather than a universal SOTA result: Jurkat does not improve, and the primary published SLAMR comparator still requires verification from an accessible primary table. Web verification confirms the paper metadata and DOI (`10.1145/3807503.3819499`) but identifies the electronic edition as closed access; the locally transcribed 0.09 MRR / 0.25 Recall@20 comparison therefore remains provisional.

Two post-result readouts remain exploratory. Averaging the four K562 study-context scores changes K562 to 0.1326 MRR / 0.2994 Recall@20. A fixed 50/50 rank fusion of expression state and text reaches 0.1356/0.3009 on K562 and 0.1065/0.3820 on Jurkat, trading MRR against recall. Neither replaces the unconditioned expression-state primary result.

The architecture also permits a double knockout to be simulated simultaneously or in both sequential orders. Without retraining, the expression-space distance between the simultaneous state and the mean sequential state reaches 0.1335 MRR / 0.3167 Recall@20 on K562, improving both primary state metrics. Averaging the four K562 source contexts—Adamson, Dixit, Norman and Replogle K562—raises this to **0.1507 / 0.3435**. Source membership is fixed by perturbation-corpus provenance, and all scenario-3 held genes remain excluded. A separate run selecting on the sum of unknown- and source-context held-gene state losses converges to the same checkpoint and identical fold metrics. On Jurkat the unknown-context score trades MRR for recall (0.0909/0.2990), and on generic Feng CV3 it is near random (0.4873 AUROC / 0.4824 AP). Because this readout family was introduced after inspecting the primary result, it is exploratory rather than confirmatory. Normalizing the same residual by decoder-projected Gaussian variance was rejected at 0.1224/0.2917 on K562.

The sequential score was then frozen before opening the official MuSL repository's CV3 labels. Both released seeds were named in advance and evaluated without SL-label fitting, calibration, sign selection or fold selection. MuSL uses 7,684 genes and balanced random negatives; every test pair contains two genes absent from that fold's training pairs.

| locked MuSL CV3 readout | seed 42 AUROC / AP | seed 432 AUROC / AP | ten-fold mean AUROC / AP |
| --- | ---: | ---: | ---: |
| unknown-context sequential state | 0.5118 / 0.5071 | 0.5147 / 0.5111 | 0.5133 / 0.5091 |
| four-source sequential state | **0.5367 / 0.5306** | **0.5398 / 0.5314** | **0.5382 / 0.5310** |

This is independent evidence that the cellular-state composition score transfers above chance on a complete cold-start split. It is not SOTA: the authors' released `pancancer.png` reports MuSL at 0.7895 AUROC and 0.8018 AUPR and SLMGAE at 0.7630/0.7699 on their CV3. The official repository commit is `f8021cfc618fafae8c330b694d0fa7c46db5f1a5`; the evaluated world model and decoder have SHA-256 hashes `7832B110...8691` and `E2232039...6BAE`. The complete fold results are in `results/sl_predict/musl_cv3_confirmatory.json`.

The already-defined deterministic supervised readout was then fitted separately to each MuSL training fold. It averages ExtraTrees, LightGBM and three seeded shallow neural heads over fixed world-state pair features. No test-fold early stopping, threshold selection or ensemble-weight selection is used.

| MuSL CV3 fold-local readout | AUROC | average precision | F1 at known 1:1 prevalence |
| --- | ---: | ---: | ---: |
| seed 42, five-fold mean | 0.8325 | 0.8386 | 0.7519 |
| seed 432, five-fold mean | 0.8428 | 0.8452 | 0.7629 |
| both seeds, ten-fold mean | **0.8377** | **0.8419** | **0.7574** |
| MuSL author table | 0.7895 | 0.8018 | 0.7363 |

This is a protocol-specific development SOTA: both released seeds exceed the author table in AUROC and average precision, and all ten folds exceed 0.81 AUROC. It is not a second untouched confirmation because the MuSL labels had already been opened for the frozen-score evaluation. The readout family and hyperparameters predate MuSL inspection, however, and the two-seed replication argues against a favorable split. The author F1 is not directly comparable because MuSL searches the threshold on each test fold; the local F1 uses the known balanced prevalence. Full rows and provenance are in `results/sl_predict/native_spectral_safe_intervention_perturbseq/musl_cv3_calibrated.json`.

MuSL's released 512-dimensional scGPT embeddings were also tested as a label-free feature addition. Standardized coordinates were inserted into the previously blank 400-dimensional safe block for 7,678 genes. After 12 relation-pretraining epochs, held loss was 0.01877 versus roughly 0.013 for the retained compact representations. The candidate was rejected before perturbation-state or benchmark training.

The perturbation-state corpus was then expanded with two local combinatorial screens: Wessels/Satija 2023 THP-1 CRISPR-Cas13 and Joung/Zhang 2023 hESC ORF perturbations. Deterministic cell-barcode pseudoreplication adds 204 and 36 training doubles, respectively. The seven-source pack contains 7,665 pseudobulks, of which 6,269 train and 1,396 form the unchanged held-gene validation partition; 404 double perturbations enter training. All 492 SLAMR scenario-3 genes remain excluded. Across 206 common expression genes, the 32-dimensional target explains 53.04% of training variance.

The same 11.92M model was retrained once on this enlarged state corpus and selected only by held-gene state prediction. Its unknown-context sequential score improves the already-opened MuSL CV3 development result from 0.5133/0.5091 to **0.5336 AUROC / 0.5220 AP**. The earlier four-source contextual average falls to 0.4598/0.4651, so source-context averaging is not retained for the seven-source model. This is an exploratory label-free improvement, not an independent confirmation. The checkpoint, decoder and corpus SHA-256 values are `F3998FB9...6876`, `5B221908...C606` and `F65DAF6F...406F`.

A context-conditioned successor adds a 128-dimensional basal DepMap state to the transition. K562 and THP-1 use exact models, RPE1 uses the mean of five available engineered derivatives and hESC remains explicitly unknown. The model has 11,968,864 parameters and uses 50% context dropout so unknown-context inference remains defined. Held-gene expression-state loss reaches 0.3795 and held-double loss 0.5557; reinforcement learning does not improve the fixed validation objective and is rolled back. This candidate is rejected on MuSL CV3: unknown-context sequential scoring gives 0.4592 AUROC / 0.4836 AP, while averaging over 32 fixed DepMap basal-state representatives gives 0.4711/0.4873. Motivated by the later cell-specific Sanger result, top-three and maximum context aggregation were also tested without fitting; they fall further to 0.4537/0.4726 and 0.4353/0.4628. Thus an exact matched state, not marginalization or context search, is required.

## External experimental tests

The four-source sequential score was locked before downloading outcomes from Harle et al. 2025 (`10.6084/m9.figshare.25954027.v4`). The official hit rule is `mean_norm_gi < -0.5`, FDR below 0.01 and neither gene singly depleted. The intervention-cold primary set contains 157 paralog pairs, 3,496 pair-cell-line measurements and 400 hits. The fixed score correlates in the wrong direction with mean negative interaction strength (Spearman rho **-0.2531**, pair-bootstrap 95% CI -0.4020 to -0.1011; two-sided permutation p=0.00150) and gives 0.4819 macro AUROC. No sign reversal was permitted. This locked confirmation failed.

The basal-context successor was evaluated retrospectively on the same screen, now assigning each measurement its exact checksum-derived DepMap expression state. Twenty-four of 27 cell lines have measured states; C092 lacks a DepMap identifier, while CHL-1 and KP-1N lack expression in the pinned release. After the stricter seven-source intervention filter, 153 pairs, 3,040 pair-cell measurements and 342 hits remain. The prespecified pair-averaged endpoint is positive but nonsignificant (Spearman rho 0.0747, 95% CI -0.0932–0.2382; p=0.361). The cell-specific endpoint is materially different: macro AUROC is **0.6081** and macro AP is **0.1947** at mean cell prevalence 0.1131; pooled AUROC/AP are 0.5861/0.1488. All 24 cell-line AUROCs exceed 0.5. On the same 24-cell coverage, the earlier context-free score gives 0.4718/0.1253 macro AUROC/AP. A post-result cell-line bootstrap gives 0.5894–0.6275 for macro AUROC and within-cell permutation gives p=0.0001. This is retained as retrospective evidence that the model captures context-specific hit ordering, not as an independent confirmation or a successful pair-level universal SL score.

The seven-source checkpoint was also tested against the Billmann/Costanzo 2026 HAP1 genetic-interaction map (`10.17632/bpcpfns6vb.1`). This test is explicitly retrospective because a related HAP1 analysis already existed elsewhere on the local host. The fixed context-free sequential score was evaluated without sign selection on 1,149,657 intervention-cold pairs across 138 query screens. For standard negative interactions versus no-call pairs, query-macro AP is **0.01489** at macro prevalence 0.01498, and macro AUROC is **0.48805**. The 95% bootstrap intervals are 0.01328–0.01661 for AP, -0.00042–0.00025 for AP minus prevalence and 0.48090–0.49559 for AUROC; the within-query permutation p-value is 1.0. This also fails to support general emergence.

The context-conditioned successor was then scored with HAP1's exact DepMap expression state. Because HAP1 lacks DepMap CRISPR coverage, its state was reconstructed by streaming the same checksum-pinned expression and dependency matrices and replaying the original deterministic projection; all 1,066 reference states agree within a maximum absolute error of `2.37e-7`, and no HAP1 interaction outcomes enter the projection. On the unchanged intervention-cold task, macro AP is **0.01548** at prevalence 0.01498 and macro AUROC is **0.50028**. The AP-minus-prevalence bootstrap interval is 0.00017–0.00084, but the finite-sample, tie-correct within-query permutation null has mean AP 0.01600 and gives p=0.999. Basal context removes the strong inverse AUROC without producing significant ranking skill; the candidate is rejected. Protocol, scores and result are in `results/sl_predict/hap1_basal_protocol.json` and `results/sl_predict/native_spectral_safe_intervention_basal_perturbseq_v3_p12_d3_t10_r3/`.

Two fold-local auxiliary ablations asked whether HAP1 could teach only the missing state-to-fitness map. They used 21,190 standard negative-GI pairs plus 84,760 fixed no-call pairs. Before every MuSL CV3 fold, every auxiliary pair containing either held-out test gene was removed. A static pair-tree feature reduces the ten-fold MuSL result to 0.83479 AUROC / 0.83939 AP; a neural feature over both sequential transition orders reaches 0.83524/0.83929. Both are below the retained 0.83768/0.84186 readout and are rejected.

An external-only binary transfer was tested separately, without any MuSL training labels. The local SLKB pack contains 51,013 collapsed pairs, including 2,878 positives over 2,354 genes. Four deterministic readouts were selected only on five fixed SLKB both-genes-new folds. The selected boosted full-state readout reaches **0.8116 AUROC / 0.2483 AP** on that external validation, but only **0.5320 AUROC / 0.5453 AP / 0.5220 F1** on the balanced ten-fold MuSL CV3 task. Before each MuSL fold, every SLKB pair containing either test gene was removed; 32,071–37,031 training pairs remained. This branch is rejected as strong evidence of label-source shift. It is supervised cross-dataset transfer, not emergence.

The missing readout was isolated with two frozen auxiliary decoders. A 16,897-parameter single-gene tolerance decoder was fitted to 10,203,206 raw DepMap CRISPR effects while excluding the three target cell lines and all 492 scenario-3 genes. On held cell lines it reaches Huber loss 0.05078, dependency AUROC 0.9342 at raw effect below -0.5 and correlation 0.7167. Despite that fidelity, multiplying the state-residual score by predicted single-gene tolerance reduces label-free MuSL CV3 to 0.3779 AUROC / 0.4220 AP. Adding 26 fixed tolerance summaries to the supervised MuSL readout gives 0.8381/0.8404, versus 0.8377/0.8419 without them; the feature is rejected because average precision falls.

A separate 17,026-parameter decoder predicts continuous double-knockout outcomes from the frozen joint-state latent. It uses 7,014 quantitative SLKB rows with measured basal context and no binary SL calls. Selection uses 388 validation pairs for which both genes are absent from decoder fitting. Signed depletion transfers (validation correlation **0.5240**); absolute interaction magnitude does not (0.0638). The fixed label-free score is therefore negative predicted depletion, where larger values mean stronger predicted loss after the double knockout.

| retrospective label-free test | fixed endpoint | result |
| --- | --- | ---: |
| MuSL CV3, both official seeds and 32 fixed DepMap state representatives | ten-fold mean AUROC / AP | **0.6227 / 0.6245** |
| Sanger, 114 decoder-intervention-cold pairs and exact state for 24 cell lines | pair-mean Spearman rho | **0.4453** (95% CI 0.2771–0.5856; p<0.0001) |
| Sanger, same measurements | cell-line macro AUROC / AP | **0.5987 / 0.1404** |
| HAP1, 927,801 decoder-intervention-cold pairs and exact state | query-macro AP / prevalence / AUROC | **0.02034 / 0.01511 / 0.5771** |

All ten MuSL folds exceed 0.5 AUROC with the signed score. On HAP1, the bootstrap interval for macro AP minus prevalence is 0.00434–0.00619 and the prespecified within-query permutation gives p=0.0005. The external subsets exclude every gene used in either perturbation-state training or continuous-decoder fitting; 28 exact Sanger training-pair overlaps and 83 exact HAP1 overlaps are thereby removed along with all other exposed genes. These results retain the signed interaction decoder as the first broadly transferable label-free readout in this repository. They are not independent confirmation: all three outcome sets had been inspected in earlier readout experiments before this decoder was proposed. They also do not establish label-free SOTA; MuSL's released supervised result remains 0.7895/0.8018. The protocol-specific supervised SL-Predict readout remains separately reported at 0.8377/0.8419.

The frozen world model remains 11,968,864 parameters; only the 17,026-parameter continuous decoder was added. Its checkpoint SHA-256 is `2FBD6595...E877`. Training, external-test protocols and complete results are in `results/sl_predict/interaction_decoder_protocol.json`, `results/sl_predict/sanger2025_interaction_protocol.json`, `results/sl_predict/hap1_interaction_protocol.json` and `results/sl_predict/native_spectral_safe_intervention_basal_perturbseq_v3_p12_d3_t10_r3/`.

A 165,633-parameter structured successor was tested using the same 7,014 continuous training rows and the same both-genes-new validation set. It combines the joint latent with symmetric differences and products of the two single-perturbation and gene representations, and predicts signed depletion alone. The selected first epoch reaches validation Huber loss 0.3113 and correlation 0.3857, both worse than 0.2693 and 0.5240 for the retained compact decoder. It was rejected without benchmark evaluation. This negative control indicates that added pair capacity overfits the small quantitative corpus rather than improving unseen-gene transfer.

Repeated measurements provide a modest denoising improvement. Pair-mean depletion has estimated reliability 0.89 in training and 0.86 on the both-genes-new validation partition. A same-size successor therefore replaces 15% of each depletion target with its gene pair's mean across measured contexts while retaining the original interaction-magnitude auxiliary target. The coefficient was fixed from molecular variance before benchmark evaluation. Selected at epoch 18 by the sum of row and pair-mean validation Huber loss, it improves row Huber loss from 0.2693 to **0.2619** and correlation from 0.5240 to **0.5525**; pair-mean Huber loss remains 0.2221.

| decoder | MuSL CV3 AUROC / AP | Sanger pair rho | Sanger macro AUROC / AP | HAP1 macro AP / AUROC |
| --- | ---: | ---: | ---: | ---: |
| original continuous decoder | 0.6227 / 0.6245 | 0.4453 | **0.5987 / 0.1404** | **0.02034 / 0.5771** |
| 15% repeated-screen shrinkage | **0.6271 / 0.6275** | **0.4549** | 0.5847 / 0.1355 | 0.01962 / 0.5645 |

The shrinkage decoder is retained as the stronger pan-cancer MuSL development readout, not as a universal replacement: it improves both MuSL metrics and Sanger's prespecified pair endpoint but reduces cell-specific Sanger and HAP1 ranking. Its checkpoint SHA-256 is `A441D4DC...7923`; complete molecular, MuSL, Sanger and HAP1 results are in the basal-model result directory. All external results remain retrospective.

Five further molecular-only alternatives were rejected without benchmark evaluation. A pair-balanced context-invariant head gives 0.4369 row and 0.4546 pair-mean correlation; a 2,852-parameter symmetric low-rank interaction gives 0.4806 and 0.5029; a regularized tree over 552 frozen world-state and pair features gives 0.3967 and 0.4711; a same-size rank-aware decoder gives 0.5541 row Pearson correlation but lowers row Spearman correlation from 0.4445 to 0.4109; and a ridge residual over 12 direct biological-relation features lowers row correlation to 0.5258 and pair correlation to 0.4811. Together with the failed 165,633-parameter head, these controls localize the current gain to modest target denoising rather than decoder capacity, ranking loss or a different predictor family.

A label-free cross-modal conditional-vulnerability residual was the one additional candidate advanced beyond molecular validation. For each ordered gene pair it averages the DepMap correlation between basal expression of gene A and CRISPR dependency of gene B with the reverse correlation, using 1,063 training cell models and no double-knockout or binary SL labels. A single ridge coefficient fitted to the continuous depletion residual improves every both-genes-new molecular endpoint: row Huber loss 0.2619 to 0.2572, row Pearson correlation 0.5525 to 0.5719, pair Huber loss 0.2221 to 0.2149 and pair Pearson correlation 0.5077 to 0.5255. It nevertheless lowers the fixed MuSL mean-context score from 0.6271/0.6275 to **0.6257 AUROC / 0.6268 AP**. It is rejected and was not evaluated on Sanger or HAP1. The source transform, fitted residual and complete ten-fold result are recorded in `results/sl_predict/conditional_vulnerability_protocol.json` and the basal-model result directory.

A more literal additive target was rejected before benchmark evaluation. Raw DepMap single-gene CRISPR effects were matched to 9,337 quantitative SLKB rows across 14 cell contexts; A549, Jurkat and K562 were excluded. Within each context, a nuisance model predicts normalized double depletion from the standardized sum of both single-gene effects using only the 5,873 decoder-training rows. The compact decoder then predicts the remaining residual. On 304 both-genes-new rows over 66 pairs, its selected combined row/pair Huber loss is **0.3385**, worse than **0.3211** for predicting zero, and its row and pair Pearson correlations are **-0.0254** and **-0.2726**. It fails every preregistered advancement criterion, so MuSL, Sanger and HAP1 were not evaluated. The protocol, source audit and molecular result are in `results/sl_predict/depmap_additive_residual_protocol.json` and the basal-model result directory.

The complete HAP1 map was next treated as a quantitative state transition source rather than converted to hit labels. Averaging finite raw qGI over measured orientations yields 1,509,338 undirected pairs over 9,365 genes. A frozen-world-state decoder was trained on 1,017,515 pairs and selected on 47,600 pairs for which both genes are absent from fitting. Neither ordinary Huber training nor a fixed magnitude-weighted variant generalizes: the selected variant has validation Huber loss **0.017853** versus **0.017839** for predicting zero, Pearson correlation **0.00640** and Spearman correlation **0.00635**. A preregistered representation-level successor lets 66,625- and 535,153-parameter symmetric networks relearn each gene from the frozen world encoding alone or from all 1,816 provenance-safe coordinates. These are no better: the selected full-feature candidate has Huber loss **0.019388**, Pearson correlation **0.00421** and Spearman correlation **0.00515**. Both experiments fail every advancement criterion, so MuSL was not evaluated. This shows that HAP1 qGI does not extrapolate across this gene holdout from the current ontology, even with roughly one million continuous training pairs; more readout data or capacity is not the missing ingredient. The protocols, pair pack and results are in `results/sl_predict/hap1_quantitative_transfer_protocol.json`, `results/sl_predict/hap1_qgi_feature_protocol.json`, `results/sl_predict/hap1_quantitative.npz` and the basal-model result directory.

Encoding the same matrix as gene state was also rejected before benchmark evaluation. A 400-coordinate high-variance fingerprint and a 128-coordinate PCA representation give held-relation losses 0.01901 and 0.01905, respectively, versus 0.01892 for the safe representation. A zero-initialized adapter is more stable: updating only 153,600 input-projection weights improves held-relation loss monotonically to 0.01788, while downstream dependency and Perturb-seq validation remain approximately unchanged. That improvement does not reach the double-knockout readout. Under the fixed 15% shrinkage protocol, both-genes-new depletion worsens from 0.2619 Huber loss / 0.5525 Pearson correlation to **0.2940 / 0.4416**. Thus generic cross-modal relation geometry is not the missing continuous interaction state; the adapted model is rejected without MuSL, Sanger or HAP1 scoring.

The previously ambiguous knowledge-graph block was reconstructed independently rather than trusted. The local raw SynLethKG export contains 2,316,994 rows. Removing `SL_GsG`, `SR_GsrG` and `NONSL_GnsG` leaves 2,233,172 non-label rows; the executable compact graph contains 2,231,921 of them over 54,012 entities and 24 relations, with 1,251 rows removed during entity remapping. Exact symbol mapping covers 9,782 of 9,845 genes. A new 6.92M-parameter, 128-dimensional TransE model reaches sampled type-matched MRR **0.3954** and Hits@10 **0.6548** on a deterministic 1% triple holdout.

Inserted into the formerly blank safe block, this prior improves the compact transformer's held relation loss from 0.018922 to **0.018742**. It does not pass the next molecular checkpoint: dependency Huber is 0.385920 versus 0.385892 for the retained model, while the combined perturbation-state score improves only marginally from 1.869513 to 1.868976. A separate fixed ridge residual over the safe KG pair geometry is actively harmful to unseen-gene quantitative depletion: row correlation falls from 0.5525 to **0.3840**, pair correlation from 0.5077 to **0.4192**, and row Huber rises from 0.2619 to **0.3262**. Both branches stop before MuSL. The graph prior is therefore admitted as a provenance-safe representation artifact but rejected as a retained cellular-state or interaction improvement.

The expression endpoint was then expanded without changing the 11,968,864-parameter transformer. A rank audit over the same seven sources gave cumulative training variance of 53.26%, 68.13%, 79.15% and 87.44% at 32, 64, 96 and 128 dimensions. Ninety-six dimensions were preregistered as the molecular elbow before the candidate pack or model existed. The final direct 96-component PCA explains 78.82%; randomized PCA's fitted basis changes slightly with requested rank. The pack retains all 7,665 rows, 404 training doubles, 16 both-genes-new validation doubles and the exclusion of all 492 scenario-3 genes. Explicit target metadata replaces the earlier hard-coded 32-coordinate assumption, so every state coordinate participates in supervised and policy-gradient loss.

The richer endpoint passes the first molecular stage. Dependency Huber remains exactly **0.385892**, while the context-aware held Perturb-seq score improves from 1.869513 to **1.863435**. Reinforcement epoch 2 supplies the selected improvement. The fixed 15% repeated-screen decoder then gives a mixed unseen-gene result: depletion Huber improves from 0.261873 to **0.259526**, pair-mean Huber from 0.222085 to **0.201227**, and pair Pearson correlation from 0.507708 to **0.563168**. Row Pearson correlation falls from 0.552512 to **0.545998**, however, a 1.179% relative regression against a preregistered maximum of 1%. The candidate therefore stops before MuSL CV3 and is not retained as the benchmark-facing model. Its molecular checkpoint remains useful evidence that endpoint rank, rather than transformer capacity, can improve held perturbational-state prediction; it does not yet preserve the row-level interaction geometry required for transfer.

A dual-resolution follow-up combines predictions from the retained 32-dimensional and new 96-dimensional worlds. Five weights were evaluated only on the same 388-row, 66-pair continuous partition. The automatic minimum-Huber rule selected the rejected 100% 96-dimensional endpoint. A distinct Pareto candidate was then frozen before outer evaluation: 25% of the retained prediction plus 75% of the 96-dimensional prediction. It improves molecular row Huber to **0.258704** and row Pearson correlation to **0.554854**, while pair Huber reaches **0.204988** and pair Pearson correlation **0.554190**. Thus every preregistered transfer constraint is satisfied.

On its first MuSL exposure, fixed negative blended depletion averaged over the same 32 deterministic DepMap state representatives reaches **0.63197 AUROC / 0.62823 AP / 0.59628 F1** across ten CV3 folds, versus 0.62705/0.62747/0.59300 for the retained 32-dimensional shrinkage decoder. Both locked primary metrics improve, so the blend becomes the pan-cancer label-free development readout. The gain is small, remains well below supervised SOTA and is retrospective with respect to the larger project, but the blend itself used no MuSL label, calibration, sign selection, context selection or fold selection.

The same 25%/75% score was then locked unchanged for exact-state external tests. On HAP1's 922,249-row intervention-cold primary task, it improves the shrinkage decoder from 0.01962 macro AP / 0.56453 macro AUROC to **0.02027 / 0.57758** at 0.01511 prevalence; the query permutation p-value remains 0.0005. On the identical 114-pair Sanger subset, cell-line macro AUROC/AP improve from 0.58471/0.13546 to **0.59526/0.14269**, but the prespecified pair-mean Spearman correlation falls from 0.45494 to **0.42834**. Thus multi-resolution state improves pan-cancer CV3 and cell/query-level ranking while sacrificing some context-averaged pair ordering. It is not a universal replacement for the original decoder.

A single-checkpoint successor was constructed before any further benchmark exposure. Its first 32 targets reproduce the retained endpoint exactly; 64 additional components model only the expression residual left after reconstructing that endpoint, for 78.88% total training variance. Equal block weighting improves the residual-block held score to **1.84930** and the combined score to **1.86161**, with dependency Huber unchanged at **0.385892**. The exact retained block nevertheless regresses from 1.86951 to **1.87391**, failing its preregistered preservation criterion. The experiment therefore stops before continuous-interaction fitting or CV3. Richer residual state can be learned, but updating one shared transition compromises the validated state; the next candidate must isolate residual adaptation from the frozen 32-dimensional pathway.

That isolation succeeds with a much smaller change. The complete retained RL-refined world and 32-dimensional decoder are frozen, and a 25,024-parameter nonlinear endpoint predicts only the 64 residual coordinates. Its held residual score reaches **1.85729**; combining it with the locked 1.86951 legacy score gives **1.86340**, narrowly passing the preregistered 1.86344 ceiling with dependency behavior invariant by construction. Stochastic-latent refinement does not improve the selected supervised epoch. The resulting checkpoint has 11,998,016 parameters, only 29,152 more than the retained model including its frozen legacy decoder.

A zero-initialized 8,642-parameter correction then adds residual joint and non-additive state to the frozen continuous-depletion readout. On both-genes-new quantitative validation, row Huber improves from 0.26187 to **0.25695**, pair Huber from 0.22209 to **0.21752** and pair correlation from 0.50771 to **0.51810**; row correlation remains within its locked bound at **0.54997**. Its first MuSL exposure improves the single-world mean to **0.62926 AUROC / 0.62948 AP**, versus 0.62705/0.62747 for the retained 32-dimensional readout. This does not dominate the dual-world blend, whose AUROC is higher at 0.63197 but AP lower at 0.62823.

Exact-state external transfer is again mixed but informative. On Sanger's identical 114 intervention-cold pairs, the isolated endpoint reaches **0.45002** pair-mean Spearman correlation and **0.59593/0.13997** cell-line macro AUROC/AP: substantially better pair ordering than the dual-world blend and better cell-line ranking than the retained shrinkage readout. On HAP1 it gives **0.01947** macro AP / **0.56255** macro AUROC with permutation p=0.0005, slightly below the retained shrinkage and dual-world readouts. The isolated endpoint is therefore retained as a compact single-checkpoint pan-cancer improvement, not as a universal external replacement or label-free SOTA.

Directly optimizing continuous ordering does not improve that endpoint. A post-MuSL, preregistered 2,337-parameter RankNet correction compares 100,000 depletion measurements per epoch only within their experimental context and keeps the world and accepted continuous head frozen. Its best eligible epoch changes held row Spearman only from 0.450736 to **0.450775** and leaves pair Spearman at **0.447663**. Later epochs trade rank and Pearson correlation for small pair-Huber gains. The branch fails its required 0.01 rank improvement and stops without further MuSL exposure, indicating that another ranking loss cannot extract the missing cold-start signal from the same residual state.

The additive-by-construction local Perturb-seq v2 checkpoint was audited from executable code and weights. Although it uses no SL labels or graph inputs, its own data-support artifact marks it exploratory and prohibits downstream features because strict-cold pair support and an unselected expression panel are inadequate. Its weights are therefore excluded. The derived low-capacity hypothesis was tested safely on the accepted model: a 192-feature ridge correction over the complete predicted 96-dimensional double endpoint and its additive residual. Nested gene folds select alpha 1000, but locked outer row Huber worsens from 0.25695 to **0.26599**, pair Huber from 0.21752 to **0.22819**, row correlation from 0.54997 to **0.51466** and pair correlation from 0.51810 to **0.48686**. The branch stops without benchmark scoring.

The accepted residual continuous-depletion score was also applied unchanged to fully intervention-isolated SLAMR scenario 3 with exact independent DepMap basal states. It does not transfer to the two well-powered cells: Jurkat averages **0.04822 MRR / 0.20310 Recall@20** and K562 **0.08485 / 0.18697**, slightly below their unknown-context variants. A549 reaches 0.32616/1.0 but has only 18 positive queries across four evaluable folds and no positives in the fifth. Thus continuous depletion and expression rollout capture different biology; the latter remains the stronger intervention-isolated K562 mechanism.

Five source-specific molecular endpoints were then fitted over externally fixed LINCS landmark genes instead of the 206-gene intersection shared by all studies. The frozen world feeds five 20,896-parameter decoders, each predicting a 32-dimensional training-only PCA state. The 104,480 trainable parameters improve unknown-context source-macro Huber loss from 0.27269 to **0.27083** and exact-context loss to **0.27076**; held Norman double-perturbation Huber improves from 0.36095 to **0.34251** and **0.33734**, respectively. Both source-macro cosines are positive, so this endpoint is retained as a molecular model.

Neither tested SL transfer is retained. A preregistered 35-feature ridge over sign-invariant response geometry selects alpha 10,000 in nested gene folds but worsens locked outer continuous-depletion row Huber from 0.25695 to **0.25959**, pair Huber from 0.21752 to **0.22100**, row correlation from 0.54997 to **0.54092** and pair correlation from 0.51810 to **0.50602**. It stops before benchmark evaluation. An independent label-free score averages the five decoded discrepancies between simultaneous and order-averaged sequential knockouts over 32 fixed basal-state medoids. Its locked ten-fold MuSL result is only **0.51241 AUROC / 0.52420 AP**, below the prior four-source state score of 0.5382/0.5310, so Sanger and HAP1 remain closed. Wider measured endpoints improve held cellular-state prediction, but their geometry is not by itself the missing general SL readout.

A separate vulnerability-state branch compresses single-gene CRISPR dependency across 1,063 non-target DepMap cell models. The 64-dimensional PCA basis is fitted only on 7,387 permitted genes and explains 30.69% of their profile variance; 1,847 modulo-5 genes select the endpoint, while 478 well-measured scenario-3 genes remain intervention-isolated. A 25,024-parameter decoder over the frozen single-action transition weakly predicts the complete landscape: generic held-gene Huber improves only 1.99%, below the locked 5% threshold, although the isolated set improves 7.27% with cosine 0.299. Allowing a zero-initialized 128,048-parameter action adapter raises generic improvement to 3.76% and isolated improvement to 13.60%, but held relation loss rises 47.18% and the legacy Perturb-seq score rises 1.85%. Both versions are rejected before outcome fitting.

Signal is concentrated in the leading dependency modes. A separately trained 18,832-parameter endpoint over the first 16 variance-ordered coordinates improves generic held-gene Huber from 0.40493 to **0.38160** (5.76%) with cosine **0.2190**. On the untouched intervention-isolated genes it improves Huber from 0.53059 to **0.44487** (16.16%) with cosine **0.4291**. This compact dependency core is retained as a molecular endpoint. It still does not supply double-knockout geometry: a preregistered 626-parameter correction over predicted joint and non-additive dependency state selects its zero-initialized epoch. Every trained epoch worsens the accepted continuous model, so MuSL, Sanger and HAP1 remain closed. Dependency landscapes are therefore a genuine cold-gene cellular state but not an additional synthetic-lethal state under the current composition operator.

Pair composition was then isolated at the transition itself. A symmetric 4,368-parameter residual adapter was trained only on 404 measured double-perturbation pseudobulks representing 102 pairs from Norman, Wessels and Joung; the accepted world, legacy endpoint and residual endpoint remained frozen, and the adapter was inactive for single interventions. Selection used 16 Norman pseudobulks over four pairs for which both genes were absent from fitting, under both unknown and exact source context. The zero-initialized epoch remains best at mean held-double Huber **0.54883**; every trained epoch is worse, reaching 0.55109 at epoch 30. The unchanged biological-relation loss is 0.01904. The adapter is rejected, no continuous-outcome model is fitted, and MuSL, Sanger and HAP1 remain closed. With all locally usable combinatorial perturbation sources already represented, 102 distinct training pairs do not support transferable correction of the frozen world's simultaneous transition.

A four-scalar alternative asks whether native composition paths are already sufficient. It mixes frozen simultaneous prediction with order-averaged sequential rollout and the sum of both single-intervention endpoints, selecting separate legacy and residual weights only on the 404 fitting doubles. The fit assigns zero weight to sequential rollout and additive weights 0.8 and 0.3 to the legacy and residual blocks. Mean held-double Huber improves slightly from **0.54883** to **0.54758** (0.23%) and all four context/block losses remain within bounds, but the gain is far below the locked 2% requirement. This candidate is also rejected without continuous-outcome fitting or SL benchmark evaluation. Additive single-state information is weakly useful, but neither learned latent repair nor constrained native-path mixing closes the pair-composition gap.

Abundant single interventions do not rescue composition through a global action calibration. A bounded 128-parameter diagonal scale was fitted to 5,865 single and 404 double pseudobulks while the world and all admitted endpoints remained frozen. The identity at epoch zero retains the best held expression score of **1.863475** and held-double mean Huber of **0.548830**; every one of 20 trained epochs is worse. Selecting the identity preserves relation loss and generic/intervention-isolated dependency-core Huber exactly. This branch is rejected before outcome fitting and rules out simple action-amplitude mismatch as the source of the pair-state gap.

A bounded low-rank rotation tests the stronger single-to-pair emergence hypothesis without fitting any double perturbation. Its 2,048 parameters are trained on 5,865 singles over 996 genes and selected on 1,380 singles over 232 unseen genes. Epoch 29 gives only a **0.038%** held-single improvement, from 0.761967 to 0.761677. The 16 double rows are then opened for the first time: mean Huber worsens slightly from **0.548830** to **0.548894**. Relation loss rises 0.28%, generic dependency-core Huber 0.17% and intervention-isolated Huber 0.40%, all within preservation bounds. The branch is nevertheless rejected because improved single-action calibration does not produce improved pair state; no outcome is fitted or benchmarked.

The released GEARS-Norman weights are not admissible for this test: executable inspection shows learned lookup tables for 3,127 perturbations and 5,045 gene positions fitted without the intervention-isolated partition. Its explicit perturbation-fusion principle was therefore reconstructed from scratch using only the accepted inductive actions. A 3,888-parameter symmetric endpoint correction over action products, absolute differences, the simultaneous latent and predicted relations selects epoch 30, but improves held-double Huber only from **0.548830** to **0.548762** (0.012%). Both residual-block losses worsen. It is rejected without outcome fitting.

The resulting limitation is empirical and sharply scoped. Across all seven executable sources, only Norman, Wessels and Joung contain doubles: 164, 204 and 36 fitting rows, totaling 404 rows over 102 pairs. All 16 both-genes-new selection rows, representing four pairs, come from Norman; there is no independent-source cold-double validation. Five safe composition families now fail molecular advancement. Additional optimization on the same four held pairs would be test-set hill climbing; the required next evidence is a more diverse intervention-isolated double-perturbation corpus with an independent-source cold partition.

That required public evidence was found in Dixit et al. GSE90063 and admitted without using an SL outcome. Exact guide assignments yield 390 pair-context conditions and 16,255 pair-assigned cells across BMDC at 3 hours, BMDC at 0 hours and high-MOI K562, with matched singles and controls. After deterministic pseudobulk construction, the ten-source pack contains 852 double-intervention fitting rows and 55 both-genes-new selection rows across four sources, rather than 404 fitting rows and 16 Norman-only selection rows. The nested 96-dimensional endpoint retains 201 common genes and explains 76.05% of training expression variance. All 492 scenario-3 genes remain excluded from fitting.

Three preregistered molecular-only composition tests fail despite the improved evidence. A new decoder plus latent pair adapter improves both-genes-new loss from 0.487673 to 0.487484 (0.039%); explicit symmetric latent composition reaches 0.487569 (0.021%); and direct molecular-residual prediction improves 0.478093 to 0.475669 (0.507%). Each preserves the retained state but misses the locked 2% threshold. No binary SL label, continuous double-knockout outcome, rank, sign, calibration or benchmark score selected these models, and no SL benchmark was opened. GSE90063 therefore solves the independent-validation shortage but does not validate the tested compact composition operators. The accepted 11,998,016-parameter world model and its frozen benchmark results remain unchanged.

A second public route is now identifiable but not yet admitted: Genentech GSE337988 reports a DLD1 pilot across multiplicities of infection and a scaled 5,000-gene Perturb-seq screen at low, medium and high multiplicity, with processed H5 artifacts. Its approximately 24.7 GB raw release requires a local guide-assignment and matched-single audit before it can enter intervention-isolated training.

A prespecified signed target based on within-study log recovered-cell abundance was rejected. Its frozen direction gives 0.4045/0.4334 on Feng CV3, 0.0893/0.2276 on K562 and 0.0861/0.2935 on Jurkat. The sign was not reversed after label inspection; recovered cell count is too confounded by guide capture and study design to serve as intervention fitness.

Explicitly supervising expression non-additivity was also rejected. All 180 measured doubles have source-matched singles, including the 16 fully gene-cold validation doubles, but adding their residual loss reduces K562 state ranking to 0.1020/0.2376 and Jurkat to 0.0683/0.2349. The retained objective therefore predicts endpoint state and computes non-additivity only at inference.

SLAMR scenario 3 also supplies fold-local SL training and validation data. A frozen LightGBM ranking probe over world-state pair features is therefore reported separately from emergence. It is trained on each fold's training pairs, stopped on that fold's validation queries and evaluated once on its test queries. The cold-joint features reach 0.1240 MRR / 0.3845 Recall@20 on Jurkat and 0.1086 / 0.2599 on K562. Exposing the independent gene-aligned DepMap measurements changes these to 0.1046 / 0.3365 and 0.1247 / 0.3116, respectively. Fold-local validation selection between the label-free model and the text prior reaches 0.1285 / 0.3331 on Jurkat and 0.1127 / 0.2298 on K562. These supervised readouts show that the frozen state is useful, especially for recall, but they are not evidence that SL performance emerged without benchmark labels.

## Hard molecular generalization gate

A source-only validation change adds deterministic five-fold exact-action-set,
composition-gene, intervention-gene, context, source, condition and compound
source-plus-gene protocols. Each fold records exclusions and evidence counts,
and refuses missing provenance rather than substituting source, context or
condition identifiers. Perturbation-delta outcomes are scored against
cardinality-matched mean and matched-single additive baselines. The aggregate
Perturb-seq builder now preserves complete upstream context and experimental
condition metadata and records when either axis is incomplete. Seven focused
NumPy tests pass; the two PyTorch contract tests remain skipped in the local
environment because PyTorch is absent. No model was trained, no SL benchmark
was opened and no performance claim changes. Existing generated packs must be
regenerated and pass the gate before they can support a new benchmark run.

## Exact label-free CV3 matrix

The compact native-score checkpoint was evaluated over all 9,845 genes with the authors' gene-wise ranking semantics. PR-AUC below is trapezoidal area under the precision-recall curve, whereas the training table above reports average precision. These results do not establish emergent SL ranking.

| protocol | AUROC | PR-AUC | NDCG@10 |
| --- | ---: | ---: | ---: |
| full dependency 1:1 | 0.4494 | 0.5009 | 0.00129 |
| full dependency 1:5 | 0.4937 | 0.1853 | 0.00129 |
| full dependency 1:20 | 0.5374 | 0.0624 | 0.00129 |
| full dependency 1:50 | 0.5551 | 0.0278 | 0.00129 |
| full expression 1:1 | 0.5962 | 0.6340 | 0.00129 |
| full expression 1:5 | 0.5907 | 0.2657 | 0.00129 |
| full expression 1:20 | 0.5838 | 0.0836 | 0.00129 |
| full expression 1:50 | 0.5837 | 0.0350 | 0.00129 |
| full random 1:1 | 0.5763 | 0.5757 | 0.00129 |
| full random 1:5 | 0.5764 | 0.2173 | 0.00129 |
| full random 1:20 | 0.5763 | 0.0669 | 0.00129 |
| full random 1:50 | 0.5761 | 0.0280 | 0.00129 |
| non-computational dependency 1:1 | 0.4619 | 0.5199 | 0.00183 |
| non-computational dependency 1:5 | 0.4953 | 0.1972 | 0.00183 |
| non-computational dependency 1:20 | 0.5377 | 0.0665 | 0.00183 |
| non-computational dependency 1:50 | 0.5556 | 0.0300 | 0.00183 |
| non-computational expression 1:1 | 0.5733 | 0.6255 | 0.00183 |
| non-computational expression 1:5 | 0.5733 | 0.2695 | 0.00183 |
| non-computational expression 1:20 | 0.5727 | 0.0856 | 0.00183 |
| non-computational expression 1:50 | 0.5788 | 0.0365 | 0.00183 |
| non-computational random 1:1 | 0.5676 | 0.5660 | 0.00183 |
| non-computational random 1:5 | 0.5732 | 0.2233 | 0.00183 |
| non-computational random 1:20 | 0.5733 | 0.0703 | 0.00183 |
| non-computational random 1:50 | 0.5724 | 0.0294 | 0.00183 |

## Public multi-study perturbation atlas v1

To address the corpus limitation that blocked composition-cold selection (102
distinct fitting pairs, four both-genes-new selection pairs, all from Norman),
seven public GEO perturbation-expression deposits were acquired with
checksum-pinned manifests and converted to one canonical action/mode/dose
schema. `src/training/merge_perturbation_corpora.py` builds
`data/perturbation-atlas/public-multi-study-v1` from the seven constituent
packs, each with its own audit:

| accession | population | modality | rows | action targets |
| --- | --- | --- | ---: | ---: |
| GSE220974 | K562 | CRISPRa/i singles and pairs | 256 | 22 |
| GSE221321 | THP-1 + LPS, compressed composite | knockout/repression | 53,735 | 599 |
| GSE337988 | DLD1 CRISPRi, six MOIs | repression | 3,056 | 35 |
| GSE213957 | THP-1 Cas13d CaRPool | RNA knockdown | 3,122 | 28 |
| GSE200201 | MOLM13 mSWI/SNF knockout | knockout, up to 8 actions | 761 | 28 |
| GSE208240 | Calu-3 CRISPRi host-factor screen | repression | 370 | 183 |
| GSE278572 | primary CD4 T-cell CRISPRi | repression | 437 | 28 |

Merged over the exact 124-gene expression-feature intersection: 61,737 rows
(27,900 single-action, 20,980 two-action, 9,857 three-plus-action), 853 unique
action targets, eight contexts, and study-balancing weights. Each pack keeps
source, biological context and perturbation condition as separate explicit
fields; unreported axes (donor, activation, infection, duration) are recorded
as unreported rather than inferred. No SL benchmark label was applied to this
corpus, and the merge performs no cross-study normalization, projection or
imputation.

On the preregistered hard molecular gate
(`src/training/validate_generalization.py`, five folds, seed 731) the atlas is
eligible in every fold for pair-cold, composition-gene-cold,
intervention-gene-cold, context-cold, source-cold and condition-cold
protocols, but source-gene-cold is eligible in only 2 of 5 folds because the
four smallest sources (256-761 rows) cannot satisfy the 32-row/16-action-set/
8-gene test minimums once held out with their genes. The gate therefore fails
closed: this corpus supports molecular training and six hard validation
protocols, and does not yet support a source-gene-cold claim.

Engineering note: the GSE278572 matrix (1.03 billion entries) is streamed in
newline-aligned binary blocks because buffered CSV chunking over a text stream
split lines and silently corrupted coordinates; the builder asserts coordinate
ranges and a clean end-of-stream. Constituent packs remain untracked generated
artifacts; the data-preparation scripts are tracked source.
