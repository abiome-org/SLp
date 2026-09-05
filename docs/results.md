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

## Supervised-readout relation-topology ablation

The fold-local MuSL CV3 readout stack (`embed_pairs` + `pair_summary` + `observed_relations`) was refit with the observed-relation block removed, holding the checkpoint, folds, seeds, ensemble and metrics fixed. On the released 59.7M SLp-1 checkpoint the ten-fold mean falls from 0.8154/0.8208 AUROC/AP to 0.7298/0.7298 (paired Δ +0.086 ± 0.008 AUROC, +0.091 ± 0.012 AP; all ten folds drop). The compact 11.92M intervention-isolated checkpoint replicates the direction at smaller magnitude (0.8300→0.7557 under the same rebuilt feature pack). Roughly the entire supervised margin above the 0.73–0.76 world-state-only floor is relation topology rather than learned transition geometry.

This is a diagnostic, not a byte-level reproduction. Local input artifacts had been cleaned after the August publication pass, so the feature pack was reconstructed from pinned public sources: MuSL folds and metadata at commit `f8021cfc618fafae8c330b694d0fa7c46db5f1a5`, Geneformer-V2-104M and SE-600M embeddings from their original releases, with the Geneformer, protein and PTGNN blocks verified against the surviving conditional pack to float16 precision (per-column |corr| = 1.000, identical coverage masks). The spectral views were rebuilt with the current tracked recipe; the original pack's exact 192 spectral columns derive from an older graph construction, so absolute levels are reconstruction-level. Both arms of each pair share identical conditions, which is what the deltas require. The conclusion is that further MuSL optimization of this readout family is benchmark-specific tuning rather than progress toward the label-free objective.

A provenance audit of this ablation also found that the scaled directory's plain `musl_cv3_calibrated.json` is byte-identical to the compact run's file; the plain deterministic readout had not previously been executed on SLp-1. The 0.89339 AUROC cited on the SLp-1 card comes from the later stacked variant with six additional public relation features and inner-fold simplex weighting, which is a further instance of the same relation-topology dependence.

The ablation's fold-level outputs are published at `results/ablations/musl-cv3-relation-topology-v1/` on `potteryrage/SLp`: the four result JSONs (SLp-1 and compact, each with and without the observed-relation block) and an audit with input checksums, reconstruction verification levels and the paired deltas. The compact checkpoint is SHA-256 `7832b1108d30f3d66b5e1d09f3e2771f1280227575a35bad445535070c208691`; the SLp-1 checkpoint is the released `177991d2c0aec316985e3f47949fdcdf381e71ef24d9c0adcf6d1bde1e0d3b78`.

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

Genentech GSE337988 provides a second public route. Six pilot H5/RDS pairs align exactly and contain 93,305 filtered DLD1 cells. Strict guide reconstruction finds 12,617 exact two-guide/two-gene cells and 76,692 one-to-eight-gene intervention cells before universe and isolation exclusions; every dose has matched single-target and NTC-only cells, and 1,380 source-pair conditions have at least four cells. A training-only 64-dimensional endpoint over all 3,022 measured genes leaves 28,803 fitting and 8,105 fully gene-cold validation cells, including 282 cold pair cells. The forced global expression intersection is only 134 genes and fails its registered 150-gene criterion, so it is not used.

Two permutation-invariant set-composition tests are rejected. On raw pair cells, a 9,104-parameter correction changes source-macro Huber from 0.249914780 to 0.249914497, approximately 0.00011%. Exact-condition means reduce the baseline loss to 0.111298963, but the selected correction reaches only 0.111235544, a 0.057% improvement. Both preserve the retained ten-source state and relation prediction, both miss the unchanged 2% threshold, and neither opens an SL outcome.

The scaled GSE337988 differential-expression release was then audited as an alternative source of composition information without downloading the approximately 25 GB raw archive. Four matrices align exactly over 6,451 perturbation conditions and 2,827 response genes; 2,813 conditions over 2,348 unique target genes survive universe mapping and scenario-3 exclusion. The two low-MOI matrices correlate only 0.1293 globally and 0.0739 within condition at the median. Their RMS disagreement is 0.10353, larger than the high-minus-low RMS of 0.07370, for a contrast-to-noise ratio of 0.7119. This fails both registered reliability thresholds, so no action adapter is fitted. The accepted world model and every SL benchmark result remain unchanged.

The public medium-MOI cell objects provide a sounder reconstruction. They align at 399,359 cells by 36,601 features; strict high-confidence assignments retain 58,669 intervention-isolated cells, including 18,481 multi-gene cells. A training-only 64-dimensional endpoint over 2,827 response genes leaves 50,864 fitting cells and 7,805 fully gene-cold validation cells. Among condition means, 9,634 supported multi-gene conditions train a 9,104-parameter correction and 300 cold pairs select it. Mean unknown/exact residual Huber changes only from **0.30708745 to 0.30700873** (0.0256%), far below the locked 2% threshold. Full state and prior molecular behavior are preserved, but the branch is rejected without opening an SL benchmark.

An orthogonal experiment avoids genetic double-perturbation training altogether. The official Combi-Seq GSE174695 release supplies 1,263 K562 profiles covering 380 drug combinations, 36 unique matched single agents, controls and three biological replicates. Of the fixed 2,827 response genes, 2,024 map directly. Replicate correlation is 0.5590 and non-additive contrast-to-noise is 1.4355, passing the preregistered data criterion. A 16,896-parameter symmetric response composer fits 224 neither-held-drug combinations, excludes 138 mixed combinations and improves both-drugs-new Huber on 18 frozen validation combinations from **0.21696 to 0.12643** (41.73%). This establishes transferable molecular composition across drugs.

That molecular success does not become genetic SL ranking through the fixed single-gene bridge. After freezing the model, the positive L2 norm of its exact-K562 predicted correction was opened once on both official MuSL CV3 seeds. Mean performance is **0.48513 AUROC / 0.48834 AP**, below chance and below the retained 0.63197/0.62823 label-free dual-world score. The direction was not reversed and no context, calibration or readout variant was attempted after exposure. Drug-response composition is therefore useful alternative supervision, but correction magnitude is not a valid SL score in this construction.

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

## Public alternatives to double-perturbation training

DepMap 24Q2 provides a public, label-free route to test whether naturally occurring gene loss can act as a pseudo-intervention. Official absolute copy-number and damaging-mutation matrices pass their release-manifest checksums and align 859 cell models to the fixed 9,845-gene dependency universe. Loss was defined before reconstruction as damaging mutation or absolute copy number below 0.5; conditional dependency was the lineage-residualized CRISPR-effect shift between loss and wild-type cells. Across deterministic disjoint model halves, 8,676,825 supported directions from 941 shared loss genes have Pearson **0.0213**, Spearman **0.0208**, and sign agreement **0.5076**. This misses the preregistered 0.15/0.55 source thresholds, so the observational target is rejected before fitting a pair model or viewing any SL labels. No double-perturbation data or SL labels were used.

A second preregistered estimator asks whether this failure is attributable to measured confounding. Five-fold cross-fitted ridge models residualize both loss and dependency against the first 32 fixed basal-expression coordinates and Oncotree lineage. Across 8,788,701 shared directions from 904 loss genes, adjusted half estimates have Pearson **0.0011**, Spearman **0.0197**, and sign agreement **0.5071**. Adjustment therefore does not rescue natural loss as reliable pseudo-intervention supervision; again, no pair model is fitted and no SL result is opened.

The public PRISM primary screen provides actual single-agent interventions rather than observational pseudo-interventions: 4,518 compounds across about 578 cancer lines. Exactly single-gene annotations yield 1,215 conditions and 70 targets with at least two distinct compounds in each deterministic compound half after alignment to 465 local models. Context-specific half profiles reproduce (pooled Pearson **0.2641**, median target Pearson **0.1722**), but their target consensus correlates only **0.0450** with the matching independent CRISPR dependency profile, below the preregistered 0.10 genetic-concordance threshold. PRISM is therefore admitted as pharmacology but rejected as genetic-action viability supervision; no model or SL evaluation follows.

DEMETER2 combined RNAi is a second public single-gene intervention source, spanning Achilles, DRIVE and Marcotte screens. The 94 MB matrix contains 655 models and 17,309 genes; 536 models and 8,925 genes align locally. Across 3,851,305 paired observations for 8,750 genes with at least 100 paired models, within-gene standardized RNAi and CRISPR effects have pooled Pearson **0.0700** and median per-gene Pearson **0.0338**, below the preregistered 0.15/0.10 admission thresholds. Cross-platform RNAi supervision is therefore rejected before outcome-head fitting or SL access.

A direct single-intervention route succeeds molecularly. A 16,897-parameter outcome head is fitted over the frozen accepted world using only DepMap CRISPR effects, excluding the three benchmark target cells and every SLAMR scenario-3 validation or test gene. On all 366,375 measured observations jointly held across 1,847 genes and 202 cells, Huber improves from a zero baseline of 0.09095 to **0.05403** (40.59%), with **0.66554 Pearson**, **0.42366 Spearman** and **0.92020 dependency AUROC**. The head is admitted as a cold-gene/cold-cell single-perturbation model.

Its preregistered sequential readout does not transfer. For each pair, predicted dependency after applying B to the A-transitioned state and A to the B-transitioned state is compared with the corresponding basal single effects, symmetrized and averaged over 32 fixed DepMap medoids. One frozen evaluation on both official MuSL CV3 seeds gives **0.37510 AUROC / 0.42541 AP / 0.39621 F1**, far below the retained 0.63197/0.62823 dual-world score. The sign is not reversed and no context, calibration or mixture is searched. Single-action outcomes generalize, but the second latent transition is not an outcome-calibrated cellular state.

An explicit basal-context transition then tests that diagnosis without double perturbations. The original DepMap expression transform is reconstructed from the checksum-matched 461 MB primary matrix and reproduces all 1,066 stored 128-dimensional states to **2.37e-7** maximum absolute error. Future-minus-control expression from 7,149 single interventions across Adamson, Dixit, Norman and Replogle K562/RPE1 is projected into those exact coordinates; 1,372 intervention-gene-held rows select a 99,200-parameter no-lookup shift model. Source-macro Huber improves 47.78% and cosine is 0.2196, but RPE1 drives the scale-weighted gain. Four sources are worse than zero in absolute Huber, and held Dixit cosine is **-0.0165**. The model fails its all-source criterion and is rejected without SL evaluation.

Source-balanced angular training removes that magnitude artifact and improves held-gene direction substantially: source-macro cosine is **0.37308**, with Adamson 0.52533, Norman 0.39462, Replogle K562 0.26026 and Replogle RPE1 0.58605. Dixit's 16 held interventions reach **0.09914**, narrowly below the preregistered 0.10 per-source minimum. The threshold is applied without rounding; the unit-direction transition is rejected and MuSL remains closed.

The preregistered extension adds only non-held single interventions: 68 Wessels, 20 Joung and 2,294 GSE337988 medium-dose rows. All 519 held GSE conditions are discarded, and the original five-source 1,372-row selection set is unchanged. Its 99,200-parameter source-balanced head clears the molecular rule at **0.38059** macro cosine; Dixit rises to **0.14248**, while the other sources range from 0.25118 to 0.58479. The authorized pair score then asks whether either gene's predicted single-dependency becomes worse after adding the partner's unit context shift, symmetrized over 32 fixed DepMap medoids. Its one MuSL CV3 evaluation reaches **0.59877 AUROC / 0.61390 AP / 0.57023 F1**, below the retained 0.63197/0.62823 dual-world result. The molecular context model is admitted, but this SL readout is rejected without variants, reversal or blending. No double-perturbation data or SL labels fit either head.

The cancer-continued Geneformer checkpoint was then tested through actual in-silico knockout rather than the static token already present in SL-Predict. Exact 4,094-gene K562 and RPE1 rank encodings yield 635 and 759 requested last-layer deletion deltas. A zero-initialized 59,072-parameter residual over the frozen extended context head reproduces the 0.380586 baseline exactly at epoch zero. Its best trained epoch reaches only **0.380733** source-macro cosine, a gain of 0.00015 versus the preregistered 0.01 requirement; later epochs regress. The candidate is rejected before any SL benchmark. No double perturbation or SL label is used.

TCGA PanCancer Atlas MC3 supplies a larger public non-perturbational relation. The checksum-recorded UCSC Xena matrix contains 8,741 primary tumors across 32 cancer types. After residualizing nonsilent mutation profiles for cancer type and log mutation burden, 6,010 genes support 18,057,045 pairs with at least 20 altered tumors per deterministic patient half. Half estimates correlate at **0.16887 Pearson / 0.16834 Spearman**, and their top-one-percent overlap is **2.879-fold** above random, passing all registered source criteria. A graph of 102,438 edges is therefore admitted as a reproducible artifact. Its 32 spectral coordinates do not improve SL-Predict: the unchanged 11,919,712-parameter transformer finishes 12 matched relation-pretraining epochs at **0.0191825** held loss versus **0.018922** for the retained representation and a 0.0187328 advancement boundary. The model branch stops before dependency, Perturb-seq, interaction, reinforcement or SL evaluation. No double-perturbation data or SL labels are used.

The same public relation is useful as an independently validated readout rather than as input to the shared transition. A 33,537-parameter symmetric decoder over frozen accepted gene actions is fitted on one patient half and reaches **0.17604 Pearson / 0.17418 Spearman** on 749,700 relations from the other half where both genes were excluded from fitting. Before opening any SL outcome, its positive prediction was assigned the fixed reliability weight 0.174175 and combined with the retained dual-world depletion score after separate within-fold average-rank normalization. One evaluation on both official MuSL CV3 seeds improves the ten-fold mean from exactly reproduced 0.63197 AUROC / 0.62823 average precision to **0.64170 / 0.63501**. The TCGA relation alone reaches only 0.55698/0.58096 and varies by fold, so it is retained only as corroborating state information. No sign, weight, raw-versus-rank choice, context, subset or score-family variant was tried after exposure. This is the strongest label-free result in this repository, but it remains retrospective development evidence and does not establish SOTA.

A final TCGA decoder test asked whether the earlier held-gene bottleneck came from compressing each intervention to its learned action alone. A fixed 128-dimensional PCA of provenance-safe input state, fitted only on non-held supported genes, was concatenated with frozen actions and passed through a 66,817-parameter symmetric head. On 749,700 pairs for which both genes were held from fitting, the selected epoch reached **0.18963 Pearson / 0.18464 Spearman**, short of the preregistered **0.19604 / 0.19418** thresholds. It stopped before any SL evaluation. Thus input features recover some external relation structure, but do not repair cold-gene decoding enough to replace the accepted action-only TCGA relation model.

The original capacity hypothesis was then tested directly with the unchanged objective and exposure. A 768-wide, 256-state, eight-layer transformer has 59,638,240 pretraining parameters, approximately five times the compact model. After the same 12 relation-pretraining epochs, its selected held-relation Huber is **0.01321235**, 30.17% below the registered compact baseline of 0.018922 and well past the 0.01873278 advancement boundary. No SL label, measured genetic double perturbation or benchmark score enters training or selection. This admits the pretraining stage only; dependency, Perturb-seq and relation preservation remain pending before any SL evaluation.

The unchanged downstream schedule preserves most of that gain: final held relation Huber is **0.01335694**, 29.84% better than the compact final checkpoint, and dependency Huber improves slightly from 0.385892 to **0.385035**. Its combined Perturb-seq score improves only from 1.869513 to **1.865409** (0.22%), however, short of the registered 1.850818 boundary. The complete scaled candidate is therefore rejected and no SL benchmark is opened. Because the first Perturb-seq epoch is selected while training loss continues downward and every held block worsens, one final optimizer correction is registered before execution: multiply only the Perturb-seq and molecular-RL learning rates by the square root of the compact-to-scaled parameter ratio, with all data, epochs, seeds and gates unchanged.

That controlled optimizer correction also fails. It reproduces dependency Huber exactly at **0.385035** and preserves relation Huber at **0.01333291**, but its best Perturb-seq score is **1.869647**, worse than both the original scaled run and the 1.850818 gate. The registered stop rule forbids further learning-rate, schedule, epoch, architecture or checkpoint variants in this branch. The useful conclusion is narrow: fivefold capacity substantially improves held relation modeling, but the existing Perturb-seq objective does not convert that capacity into a jointly advancing cellular world. No SL benchmark is opened.

An executable data audit corrects one statement in both historical scaling protocols. `perturbseq_world_v3.npz` contains 5,865 fitting singles, **404 fitting doubles**, 1,380 validation singles and **16 validation doubles**; the executed path sampled all fitting rows and upweighted doubles fivefold. Thus the 59.64M relation-only checkpoint is genuinely double-free, but both downstream scaling runs are not. Their molecular numbers remain valid as double-exposed experiments, while their `double_perturbation_data_used: false` metadata is false. A new matched experiment therefore filters cardinality to one before sampling, fitting, validation and block scoring for both compact and scaled models.

That corrected experiment uses exactly 5,865 fitting and 1,380 held single-intervention rows for both sizes; every reported double component is identically zero. The clean compact control selects **0.75684063** unknown-plus-exact Huber. The scaled model reaches **0.75632200**, only a 0.0685% improvement and far short of the preregistered 0.74170381 threshold. Dependency improves slightly and held relation loss remains substantially better, but scale again does not repair intervention-cold expression. The claim is rejected without opening an SL benchmark. The next independent test asks whether the stronger frozen 256-dimensional relation actions can decode the already admitted public DepMap co-dependency relation for two unseen genes.

That scaled inductive decoder improves held co-dependency Pearson from 0.14895 to **0.18001**, but Spearman reaches only **0.10294** on 1,638,955 pairs for which both genes are absent from fitting. It misses the unchanged 0.15 rank threshold and stops before SL. The measured relation itself remains admissible, so a separate alternative-data test is registered: directly fuse split-half DepMap co-dependency and TCGA mutual exclusivity using only their independently measured reproducibility, without a learned decoder or genetic double perturbation.

The direct public-relation score is clean but weak. Reliability-weighted DepMap/TCGA rank fusion reaches **0.52347 AUROC / 0.51458 AP** on MuSL CV3; co-dependency alone is 0.50917/0.51823 and raw TCGA is 0.55201/0.57327. The signs, median missing-value rule and 0.74044/0.25956 weights were fixed before the single evaluation. No reversal, reweighting or subset variant follows. The next registered test permits fold-local SL supervision while freezing the double-free, molecular-RL-refined 59.7M world, separating representation quality from label-free emergence.

That controlled supervised test succeeds. With the world frozen, the unchanged deterministic ExtraTrees/LightGBM/three-seed neural ensemble is fitted independently to each official MuSL CV3 training fold. Across both seeds it reaches **0.84845 AUROC / 0.85048 AP / 0.76772 F1**, improving the prior 0.83768/0.84186/0.75743 result on all three metrics. Both test genes remain absent from each fold's supervised training genes, and no test-fold selection or world update occurs. This is a fully cold-start supervised development result, not label-free emergence.

The preregistered six-feature ablation also succeeds. Adding split-half mean, disagreement and support for DepMap co-dependency and TCGA adjusted mutual exclusivity to the same frozen world and fold-local learner reaches **0.86403 AUROC / 0.86461 AP / 0.78347 F1**. This improves the unaugmented scaled result by 0.01558 AUROC, 0.01413 AP and 0.01575 F1. Missing relations are represented as zero value, zero disagreement and zero support; no relation subset, transform, sign, imputation or ensemble-weight variant follows the result. This was the strongest fully cold-start supervised development result before the independently admitted expression-silencing extension below. The public relations contain no genetic double-perturbation measurements, but MuSL training-fold labels supervise the readout, so the result is not emergent label-free performance.

DepMap co-dependency provides a much more reproducible single-intervention relation but does not pass the same inductive test. Splitting 1,063 permitted cell models by identifier hash gives 530 and 533 disjoint halves. Across two million deterministic pairs from 9,081 eligible genes, independently reconstructed CRISPR-effect correlations reach **0.53100 Pearson / 0.48022 Spearman** with **32.695-fold** top-one-percent overlap, so the source is admitted. A 33,537-parameter frozen-action decoder then fits half 0 on 7,270 genes and is selected on 1,638,955 half-1 pairs where both genes are unseen. Epoch 12 reaches only **0.14895 Pearson / 0.06667 Spearman**, missing both locked 0.15 thresholds. The claim is rejected without rounding, extra epochs, capacity changes or SL exposure. This prevents a highly reproducible but poorly decoded relation from being rewarded as a benchmark feature.

A second DepMap relation tests conditional vulnerability directly without genetic double perturbations. For each pair it averages expression(A)-dependency(B) correlation with the reverse direction. Across the same 530/533 hashed cell halves, 9,510 eligible genes and two million deterministic pairs, this relation is strongly reproducible: **0.47715 Pearson / 0.42156 Spearman / 31.015-fold** top-one-percent overlap. The source is admitted before any new SL evaluation. Its independently measured Spearman then fixes one 42.156% rank fusion with the retained label-free cellular-state/TCGA score. The relation alone reaches only **0.48369 AUROC / 0.50673 AP**, and the locked fusion falls from 0.64170/0.63501 to **0.60641 AUROC / 0.60195 AP**. The transfer is rejected without reversing the sign or reducing its weight. This separates reproducible natural conditional vulnerability from the pair phenotype MuSL measures.

A prespecified functional-redundancy conjunction is also rejected before SL exposure. Across two million pairs from 8,993 genes, protein similarity and the four safe GO/PPI spectral views overlap 5.945-fold in their top one percent, but their global Spearman correlation is only **0.08370**. Protein/co-dependency and function/co-dependency top-tail overlaps are only **1.770-fold** and **2.155-fold**, below the locked 3-fold criteria. Three of four source requirements fail, so no geometric conjunction or CV3 score is computed. Direct sequence/function resemblance and shared single-gene viability do not define a sufficiently coherent redundancy target in these representations.

The 59.7M single-only world does not improve the one public relation already known to help label-free MuSL. With the world frozen, the unchanged symmetric TCGA decoder has 66,817 trainable parameters and fits the identical half-0 pairs and modulo-5 gene split as the compact decoder. Its selected epoch reaches only **0.15509 Pearson / 0.15642 Spearman** on 749,700 half-1 pairs where both genes are unseen, below the compact 0.17604/0.17418 baseline and the preregistered 0.18604/0.18418 thresholds. The branch stops without MuSL evaluation. Capacity improves supervised CV3 features but not inductive recovery of this label-free tumor relation.

The authoritative 59,736,544-parameter single-only checkpoint was next exposed once to the official Feng full-data balanced-random CV3 benchmark using its frozen direct outcome-strength endpoint. With no double-perturbation training, SL fitting, calibration, sign reversal, context selection or score fusion, the five-fold mean is **0.46859 AUROC / 0.47745 average precision / 0.47674 trapezoidal PR-AUC**. Only one fold exceeds chance. The direct emergent readout is rejected without a post-result variant; the checkpoint remains authoritative for molecular state prediction and the separately identified supervised MuSL representation.

Pooling the two disjoint TCGA patient halves reduces target noise but does not pass the stricter cross-half molecular rule. The compact decoder reaches half-0 **0.19516 Pearson / 0.19759 Spearman**, half-1 **0.19131 / 0.18943**, and pooled **0.25405 / 0.24996** on the same 749,700 two-unseen-gene pairs. It improves half 1 and the pooled target but misses the registered half-0 minima of 0.19823/0.20049. The scaled decoder misses all six thresholds, reaching 0.19714/0.19591, 0.16523/0.16449 and 0.23873/0.23357. Both branches are rejected without opening MuSL or trying a patient aggregation, head, loss or representation variant.

A nonlinear DepMap alternative treats bottom-quartile lineage-residual expression as natural silencing and contrasts partner dependency with the corresponding top quartile. Across 2,000,000 deterministic relations from 9,510 genes, disjoint cell halves reproduce at **0.23578 Pearson / 0.21142 Spearman / 9.13-fold** top-one-percent overlap, so the source is admitted without SL labels or double perturbations. Frozen compact and scaled world actions do not inductively encode it: matched decoders reach only 0.05385/0.03438 and 0.06386/0.04408 on 1,783,216 pairs where both genes were absent from fitting. Both fail the locked 0.15 thresholds, so label-free MuSL remains closed.

A bounded world-state integration tests whether the missing relation can enter the compact action without disrupting it. A 64-dimensional PCA of half-0 profiles against 256 non-held anchors drives an 8,192-parameter zero-initialized residual capped at five percent of base-action RMS. The selected residual uses 2.61% RMS and preserves held biological-relation, dependency, legacy Perturb-seq and residual Perturb-seq metrics to within 0.05%. Nevertheless, its new decoder reaches only **0.06387 Pearson / 0.04773 Spearman** on the same 1,783,216 two-unseen-gene half-1 pairs. It fails the 0.15 molecular rule, so no continuous interaction head or SL benchmark is opened. Small safe side-channel capacity is not the missing mechanism.

The independently admitted relation does add a small amount of supervised information. Appending its fixed split-half mean, disagreement and support to the existing six public features in the unchanged fold-local scaled-world learner raises the ten-fold MuSL CV3 mean from 0.86403/0.86461/0.78347 to **0.86523 AUROC / 0.86560 AP / 0.78684 F1**. The gains are 0.00120, 0.00099 and 0.00336. This is the repository's new fully cold-start supervised development best; the world and public features remain free of genetic double perturbations, but MuSL training-fold labels supervise the readout, so it is not emergent performance.

Cross-species genetic interaction supplies a genuinely new public alternative. The official Costanzo 2016 archive contains 20.71 million finite yeast SGA epsilon measurements, and Alliance 8.3 stringent reciprocal-best orthology maps 958,232 consensus pairs into the fixed human universe. Among 333,297 pairs measured in both query-array directions, epsilon reaches **0.34348 Pearson** and the most negative one-percent tails overlap **35.914-fold** above random. Full-rank agreement is only **0.16861 Spearman**, however, below the locked 0.20 source threshold. The continuous conserved-supervision claim is rejected before decoder fitting. A tail-only objective is not introduced after observing this discrepancy, and no human SL benchmark is opened.

The public Arc Institute SE-600M and State Transition Tahoe checkpoints provide a larger single-intervention prior without double perturbations. Local reconstruction maps 19,086 DepMap genes into the exact 2,058-dimensional Tahoe input, repeats bitwise in the fixed smoke test, and produces nonzero basal-context-dependent changes. The implementation now follows the released loader exactly by normalizing the supplied log-expression values directly; earlier values from exponentiating those inputs are superseded. Independent validation uses 64 target-annotated drugs and 48 cell lines, split separately by SHA-256 before PRISM outcomes are read. A fixed 32-component ridge decoder reaches **0.37829 Pearson / 0.23662 Spearman** on 717 measurements where both drug and cell are held out, with **0.79890 AUROC** for bottom-quartile sensitivity. Transition magnitude alone correlates with sensitivity at 0.19642 Pearson. Tahoe remains admitted as a public pharmacologic state source.

The first cold-gene bridge does not pass. Direct target attribution covers only 0.34% of MuSL CV3 pairs. Under corrected preprocessing, a preregistered map from spectral-safe target-gene state to a 16-component Tahoe response has held-drug mean cosine **-0.08601** and Spearman **-0.04951**; target-isolated mean cosine is **-0.17371**. The branch stops without MuSL evaluation. This avoids rewarding sparse target annotations: the source is independently validated, but the current gene representation cannot translate it to unseen genetic actions.

A second public-data route uses Replogle K562 and RPE1 single-gene CRISPRi responses only. Zero-shot SE expression erasure is rejected at 0.06943 source-macro cosine. A fixed source-specific SE-token-to-response ridge learns direction on held genes but initially worsens Huber loss because its magnitude is miscalibrated. Fivefold shrinkage fitted only from training-gene out-of-fold predictions repairs this defect; one confirmation on untouched intrinsic-validation genes reaches **0.27476 cosine / 7.89% Huber improvement** in K562 and **0.37812 / 18.92%** in RPE1. The 1 MB calibrated action bridge is admitted as a cold-gene single-perturbation state model. It uses no SL labels or double-perturbation measurements.

The action bridge does not yet supply pair geometry. On two million pairs whose genes are absent from every Replogle fitting set, its fixed response cosine correlates with two independent DepMap co-dependency halves at only **0.00396/0.00329 Pearson/Spearman** and **0.00303/0.00212**. A separate ridge residual over its frozen coordinates improves the accepted dependency-core Huber by 0.22% on generic held genes and 1.28% on intervention-isolated genes, but reduces isolated cosine from 0.42910 to 0.42242. Both preregistered downstream tests are rejected and MuSL remains unopened. The evidence supports unseen-gene single-action prediction, not emergent synthetic-lethality scoring.

Two further benchmark-closed checks reach the same boundary. An 8,709-parameter zero-initialized residual asks whether the confirmed action coordinates improve the accepted inductive TCGA decoder. On 749,700 patient-half relations with both genes unseen during fitting, Pearson rises only from 0.17604 to **0.17870** and Spearman from 0.17418 to **0.17707**, missing the fixed 0.01 gains. Separately, replacing the 256-dimensional projected State protein view with the exact released 5,120-dimensional vectors worsens its agreement with independent functional and co-dependency evidence. Neither branch opens MuSL.

The independently admitted lineage-adjusted expression-silencing relation receives one locked label-free test because it improved the supervised readout. Its positive direct score is anti-aligned with MuSL at **0.48588 AUROC / 0.50325 average precision**. Weighting it only by its 0.21142 split-half Spearman reliability lowers the retained world-state/TCGA score from 0.64170/0.63501 to **0.63442/0.62728**. The branch is rejected without sign reversal, weight search or follow-up variants. Expression silencing remains useful supervised information, not an emergent pan-cancer SL mechanism.

The confirmed SE/Replogle action is not already latent in either transformer. Fixed training-gene-only ridges recover it from the compact and scaled worlds at only **0.18043** and **0.21763** held-gene cosine, failing all registered representation criteria. This stops action distillation and Modal spend. A separate 10,496-parameter residual does improve the admitted five-source context transition from **0.380586 to 0.391687** macro cosine, but Dixit falls by 0.02193, narrowly exceeding the 0.02 preservation limit. The candidate is rejected without choosing another epoch or opening MuSL.

The bundled SLMGAE BC pathway and protein-complex matrices are also excluded as alternative evidence. They are 332-gene benchmark-specific features co-packaged with explicit positive and negative SL edges, and the released trainer appends supervised SL adjacency over those same nodes. Without independent mapping and primary provenance they are not a label-free public prior; none enters SL-Predict.

A new source-preserving integration resolves the molecular near miss without changing its held genes. Five independent ridge corrections use only single-gene SE/Replogle actions; four-fold fitting-gene predictions select a nonnegative gate for each assay before held outcomes are opened. Macro held-gene cosine rises from **0.380586 to 0.391133**, while Dixit and Norman are exactly preserved by zero gates and Replogle K562/RPE1 improve to 0.28990/0.59366. This transition is admitted without measured genetic doubles or SL labels.

Its preregistered label-free readout does not improve synthetic-lethality ranking. The unchanged symmetric dependency-change equation, using nearest K562/RPE1 basal context over 32 fixed medoids, reaches **0.59707 AUROC / 0.61097 average precision / 0.56790 F1** on both official MuSL CV3 seeds. This is below the prior context readout and retained 0.64170/0.63501 result. No context, sign, weight or fusion variant follows; better single-action state still does not identify the missing pair mechanism.

Three public pair-state routes were audited without changing their preregistered rules. A raw Horlbeck reconstruction is reproducible within Jurkat and K562 but yields only 88,410 shared pairs and 385 mapped genes, below its coverage criteria. The authors' processed matrices contain 100,576 K562 and 75,078 Jurkat pairs with exact symmetry, but cross-cell Spearman is **0.18470**, below 0.30. SPIDR supplies 149,787 complete public RPE-1 scores, but only 346 of its 548 DNA-repair genes map to the fixed world-model universe; 179 exact MuSL pairs were identified without reading labels. All three suitability claims are rejected, no thresholds are relaxed and no decoder is fitted.

Two single-perturbation alternatives test whether stable co-dependency can be learned without genetic doubles. DEMETER2 RNAi is internally reproducible at **0.34191 Pearson / 0.31883 Spearman**, yet agrees with CRISPR co-dependency at only 0.03061/0.02303 and covers 4,577 eligible genes. The public Project Score matrix covers 17,995 genes across 325 Sanger screens; after excluding 252 matched models, 9,040 genes remain against 812 independent DepMap models. Sanger split halves agree strongly at **0.74084 / 0.68789**, but cross-center agreement is only **0.05374 / 0.04110**. Both relations are rejected before SL evaluation. Stable within-study technical structure is not rewarded as biological transfer.

The public-data objective is therefore satisfied at the access layer: checksum-recorded Horlbeck, SPIDR and Project Score artifacts are local and reproducible. None clears the independent suitability boundary for world-model or benchmark use. The later nested supervised readout raises the headline supervised result to **0.89339/0.89143** AUROC/AP; the label-free result remains **0.64170/0.63501**.

Official Ensembl 116 supplies an outcome-independent human paralog relation without double perturbations or screen membership. A release-specific, complete-row-validated BioMart acquisition maps 31,231 unique pairs over 6,973 fixed-universe genes; 27,455 pairs are evaluable in both independent DepMap halves. Sequence identity has reproducible positive association with co-dependency at **0.14397/0.15519 Spearman**, but paralog membership reaches only **0.55759/0.55818 conservative AUROC** against two deterministic degree-aware control sets, below the locked 0.60 criterion. The public source is admitted and checksum-recorded, while its functional-redundancy claim is rejected before model fitting or SL evaluation. No identity, family-age or pair subset is selected after seeing the result.

A second benchmark-closed test asks whether public natural loss supplies the missing conditional intervention specifically for these paralogs. Across all Ensembl directions, 6,780 have supported estimates in both independent DepMap halves. They reproduce at only **0.08696 Pearson / 0.04810 Spearman / 0.51770 sign agreement**. Pearson exceeds the stronger gene-marginal-matched non-paralog control by 0.06076, but Spearman exceeds it by only 0.01736; four of five locked criteria fail. The relation is rejected without an adjusted estimator, identity subset, model fit or SL evaluation.

Retaining each single-perturbation assay's full measured transcriptome, rather than only the 206 genes shared by all sources or 175–493 LINCS landmarks, produces a stronger molecular endpoint. Five fixed source-specific 32-dimensional projections cover approximately 5,000 measured genes each. With the 59.7M single-only world frozen, a 372,640-parameter endpoint reaches **0.20475 unknown-context and 0.19779 exact-context macro cosine** on 1,372 untouched-gene rows, improving source-macro Huber by 2.49% and 2.74%. Every source is positive and both large Replogle panels exceed 0.15. No double perturbations or SL labels enter fitting or selection.

The induced exact-context response geometry also generalizes independently. Across all 24,531 pairs among 222 genes absent from every endpoint fitting set, positive five-source response cosine correlates with disjoint DepMap co-dependency halves at **0.27808 Pearson / 0.26306 Spearman** and **0.24617 / 0.22959**. This is admitted as molecular evidence. Its single preregistered addition to the retained supervised MuSL readout does not help: the ten-fold mean falls to **0.86427 AUROC / 0.86484 average precision / 0.78424 F1**. No source, context, sign, transform, feature-set or ensemble variant follows. Full-transcriptome single-perturbation state is useful world-model supervision, but its symmetric cosine is redundant with the public relations already in the supervised readout.

A final benchmark-closed integration fits a zero-initialized 33,280-parameter transition residual while freezing the transformer and endpoint. Epoch 0 remains optimal at 0.359303 combined held Huber; every trained checkpoint is worse. The residual is rejected without capacity or optimizer variants. The admitted full-transcriptome endpoint is retained as a sidecar rather than folded into the world state.

A directed cross-response test extracts mutual partner-expression induction from Replogle K562 and RPE1 singles. Observed fitting-gene relations reproduce across the two cell types at **0.32219 Pearson / 0.29199 Spearman**. On 231 pairs among 22 genes absent from endpoint fitting, the frozen world predicts the relation at 0.40953/0.46164 in K562 and 0.35878/0.30975 in RPE1; equal-source consensus reaches **0.42458/0.39864**. This narrow molecular relation is admitted, with 1,819 response-supported genes in its derived score pack.

Its one locked label-free MuSL fusion is mixed and therefore rejected. The independently weighted score raises average precision from 0.63501 to **0.63638** but lowers AUROC from 0.64170 to **0.64049**; the direct relation alone is near random and covers 5.02–9.44% of fold pairs. Both metrics were required to improve. No sign, reliability weight, support rule, supervised feature or fusion variant follows.

One bounded full-transcriptome REINFORCE continuation trained 691,712 transition parameters while freezing the encoder, transformer, endpoint and outcome heads. Epoch 1 improved exact-context Huber by only **0.16%** (0.17943 to 0.17914) and slightly worsened unknown-context Huber (0.17988 to 0.17993), despite cosine gains from 0.19779 to 0.20886 and 0.20475 to 0.21064. Relation, dependency and prior single-state losses remained within 0.15% of baseline. The candidate therefore fails the registered one-percent endpoint rule, is rejected without MuSL evaluation or variants, and does not replace the original world checkpoint.

The public GSE337988 medium-dose matrix supplies a larger independent single-intervention endpoint. Repeated promoters and constructs are averaged by target gene, leaving 1,903 fitting and 445 completely held genes on the source's fixed 64-dimensional state. An 82,752-parameter head over the frozen 59.7M world reaches **0.20664 unknown-context and 0.20647 exact-DLD1 cosine**, with 3.039% and 3.032% Huber improvement over zero. This endpoint is admitted without changing the world or using double perturbations or SL outcomes.

Its preregistered pair interpretation fails independently. Across all 95,266 pairs among 437 eligible held genes, measured GSE response cosine has only 0.00727/0.00532 Pearson/Spearman against one DepMap co-dependency half and 0.00061/0.00032 against the other. World-predicted response cosine reaches only 0.02309/0.02576 and 0.01770/0.02444. The pair branch is rejected without sign, dose, subset, similarity or benchmark variants. GSE337988 improves cold-gene cellular-state coverage but does not encode general co-dependency geometry.

The admitted five-source full-transcriptome profile is also tested as a molecular bridge to the TCGA component that currently improves label-free MuSL. Two 75,137-parameter heads receive identical frozen actions and 17.31 million eligible half-0 fitting pairs; only the candidate receives the frozen response profile. On 7,875 half-1 pairs among 126 genes excluded from both endpoint and relation fitting, the capacity-matched baseline reaches **0.15763 Pearson / 0.14393 Spearman**, while the candidate reaches **0.15708 / 0.13911**. Candidate Huber is slightly lower, 0.39384 versus 0.39509, but both correlations regress. The branch is rejected without MuSL evaluation or representation variants. Co-dependency-like response geometry does not supply the missing tumor mutual-exclusivity relation.

A deterministic pretraining attempt then exposes the 691,712 transition parameters to six equally weighted frozen single-intervention endpoints: the five full-transcriptome assays plus gene-averaged GSE337988. Epoch 6 improves five-source unknown/exact Huber by only **0.294% / 0.261%**, the six-source macro by **0.224% / 0.200%**, and GSE337988 by less than 0.01%. All macro cosines improve, and relation, dependency and prior single-state losses remain within 0.09% of baseline. The candidate therefore preserves state but misses the registered one-percent macro and half-percent per-family error reductions. It is rejected without the planned reinforcement continuation, MuSL evaluation, or training variants.

A transition-specific capacity test addresses the frozen-core bottleneck directly. One zero-initialized 768-wide transformer layer adds 7,087,872 trainable action-dynamics parameters to the 59.7M world while remaining exactly identity at epoch 0 and never entering static gene encoding. Epoch 7 improves the original five-source full-transcriptome Huber by **0.614% unknown / 0.516% exact** and prior single-state loss by 0.282%, but six-source macro Huber improves only **0.417% / 0.356%**, GSE337988 Huber regresses 0.189%/0.134%, Norman cosine falls by more than 0.04, and relation Huber worsens 1.518%. The 66.824M candidate fails the locked multi-source and preservation rules, so reinforcement learning and CV3 remain closed.

The permitted supervised branch produces a large fully cold-start gain without changing the world or its public features. Within each outer MuSL training fold, a deterministic second two-new-gene split selects nonnegative weights for the established ExtraTrees, LightGBM and neural components by inner log loss. Outer train/test gene overlap is independently verified as zero in all ten folds. The single registered run reaches **0.89339 AUROC / 0.89143 average precision / 0.81130 F1**, improving the previous supervised best by 0.02816/0.02584/0.02446. Inner selection assigns 90–100% LightGBM weight in nine folds and 70% in one, showing that equal weighting diluted the strongest cold-generalizing learner. This is the new protocol-specific MuSL CV3 SOTA, but it is benchmark-supervised rather than emergent.

A final decoder-control asks whether the TCGA relation bottleneck is merely the neural readout. One fixed LightGBM Huber regressor receives product and absolute-difference features from frozen 128-dimensional compact-world actions and fits 1.5 million half-0 relations among non-held genes. On all 749,700 half-1 pairs of two held genes it reaches only **0.16832 Pearson / 0.16811 Spearman**, below both the admitted neural decoder's 0.17604/0.17418 and the registered 0.18604/0.18418 thresholds. The branch is rejected before MuSL and no tree, target or feature variant follows. The remaining label-free deficit is therefore not explained by generic readout capacity.

Two newly located public single-perturbation resources were evaluated without double perturbations or SL labels. The KOLF2.1J CRISPRi Perturbation Cell Atlas is admitted: its compact reconstruction contains matched up/down signatures for 11,680 genes, maps 7,014 targets and 9,691 response genes to the fixed universe, and leaves 769 assay-supported targets untouched. A frozen-world 82,752-parameter endpoint reaches **0.09927 cosine** and improves Huber by **1.788%** on those held targets. Both values miss the registered 0.15 and 2% requirements, so the endpoint is rejected and no world continuation or SL evaluation follows.

LINCS L1000-XPR was tested in two separately registered forms. The public top-250 signed-set export reconstructs 26,497 target-cell profiles across all 19 cancer contexts but has median disjoint-context cosine **0.04287**, below 0.05, and fails its pair-count and response-coverage criteria. The independently downloaded 7,016,682,296-byte quantitative coefficient matrix contains 140,945 profiles. Stable symbol, Ensembl and NCBI identity resolution yields 71,374 single-knockout signatures, 2,892 fixed-universe targets and 32,164 target-cell states. Its response is strongly reproducible across contexts—median cosine **0.19460**, with **94.45%** positive—but only 745 directly measured landmarks map to the fixed universe, below the preregistered minimum of 800. Both L1000 sources are therefore rejected before endpoint fitting. No admission threshold, response subset, inferred-gene substitution or benchmark result was used to rescue them.

An action-information test then combines the admitted SE/Replogle coordinates with the broad five-source full-transcriptome endpoint. The audit removes every confirmation gene used either for endpoint fitting or to fit the SE bridge, leaving 43 strict genes; 23 select epochs and 20 remain unopened for confirmation. Capacity-matched 568,160-parameter heads receive identical frozen world transitions plus either zero auxiliary coordinates or the standardized SE action. On 84 confirmation profiles across all five assays, the candidate improves unknown-context Huber by only **1.282%**, worsens exact-context Huber by **0.130%**, and lowers source-macro cosine by **0.03304 / 0.04738**. Source regressions exceed 0.12. The candidate fails its registered molecular rule, so pair testing, world integration, reinforcement learning and CV3 remain closed.

A second benchmark-closed test learns one 602,291-parameter gene state jointly from independent halves of DepMap co-dependency, TCGA adjusted mutual exclusivity and expression-silencing relations. The shared universe contains 5,614 genes; 1,142 genes and 651,511 half-1 pairs per source are absent from fitting. Later training would improve TCGA to **0.21085/0.20778 Pearson/Spearman** and co-dependency to **0.11445/0.09320**, but expression silencing remains below 0.04 and its Huber loss worsens. The registered common selector therefore chooses epoch 1, where the three relations reach only 0.03848/0.03560, 0.18339/0.17905 and 0.00481/0.00521. The all-source claim is rejected without selecting separate favorable epochs, removing the difficult source or opening CV3.

Tahoe-x1 supplies the requested public 50–100M-scale alternative: its Apache-2.0 Tx1-70M checkpoint has 70,996,993 parameters, maps 9,842 of the 9,845 fixed genes, and runs locally with 586 MB peak GPU allocation for a fixed 32-cell panel. Its proposed gene-state transfer does not pass molecular admission. The ordered static representation reaches **0.02348/0.02253**, **0.14062/0.13928** and **0.01125/0.01090** Pearson/Spearman on held-gene co-dependency, TCGA and expression-silencing relations. Contextual hidden states improve these to **0.06175/0.05817**, **0.17639/0.17298** and **0.01861/0.01832**, but still miss every all-source threshold. The lower aggregate Huber is not treated as success; the branch is rejected without double-perturbation data, SL labels or benchmark exposure.

The public Tahoe-100M drug metadata offers a separate action-level alternative. Exact inhibitor/antagonist filtering and frozen action matching yield 197 drugs over 198 fixed-universe targets, compared with 36 supported drugs in the earlier projection. A checksum-registered model fits 100 drugs across 159 targets and evaluates 97 held drugs over the same 32 contexts. Held mean cosine/Spearman reach **0.10825/0.10583**; on 31 drugs whose complete target sets never occur in fitting, they fall to **0.04207/0.03500**. The public source is admitted, but the cold-target action claim is rejected. No gene atlas, reinforcement stage or SL score is emitted.

## 2026-09-03 — SLp-1.1 OpenModelFactory reset

**Hypothesis.** A species-aware query-decoder trained on separately versioned
human and yeast molecular trajectories will predict perturbation-specific
effects for intervention genes absent from every fitting and reward trajectory
better than both a training-set mean and fixed ridge baseline. The fixed first advancement rule is
zero leakage and benchmark-label records, at least 2% validation Gaussian-NLL
improvement, at least 0.10 effect Pearson correlation, and non-negative NLL
improvement in every represented species.

**SLp-1 audit.** The proof-of-concept established several boundaries worth
retaining: a data-free frozen model package, explicit benchmark-opening guards,
quantitative rather than binary molecular targets, intervention-gene exclusion,
and a record that preserves rejected results. It also exposed structural limits
that are not carried forward. `model/v1/world.py` assumes one fixed 1,816-wide
feature layout and six hard-coded relation slices. `modules/training/world.py`
is a 181,395-byte experiment accumulation containing more than 70 fitting and
evaluation paths. `src/training/run_modal.py` binds dozens of local artifacts,
model choices and benchmark evaluators into one 59,636-byte deployment script.
The architecture emits one small fixed latent state rather than answering
versioned sparse molecular queries. The historical yeast path first collapses
Costanzo SGA through strict one-to-one human orthology; its reciprocal mapped
measurements reached 0.34348 Pearson but only 0.16861 Spearman, failed its
registered source criterion, and was never fitted. These facts reject using
SLp-1 as the scaffold for a larger corpus.

**Factory protocol.** The project was mapped to OpenModelFactory 1.0 source
commit `ef26eea2cb694596f7680a4bce400371738cbb4b`. New versioned resources define
the project, Linux-local binding, admission policy, molecular evaluation, and
pretraining graph. A new corpus schema requires stable CURIE identifiers,
explicit NCBI taxonomy IDs, roles, modalities, intervention-gene inventories,
record counts and SHA-256 shard identities. Pretraining, molecular validation
and molecular reward are three independent OMF snapshots. The first stage
verifies every shard and rejects benchmark-bearing data or any validation
intervention gene found in pretraining or reward.

**Fresh model contract.** `modules/slp-1-1-world` contains a new PyTorch model,
not a modification of `model/v1`. It encodes variable-size context and action
sets without positional information and decodes sparse readout queries. Every
entity is supplied through versioned molecular features; there is no learned
gene-ID embedding or fixed gene universe. Species is an explicit continuous
feature block. Yeast remains species-native and may align to human through
separate sequence or orthology evidence, never by relabeling a yeast phenotype
as a human outcome. Maximum-likelihood pretraining and a molecular-only
self-critical continuation stream bounded shards; every reinforcement epoch is
discarded unless held-gene NLL improves without worsening the worst species.

**Validation.** The three focused corpus-audit tests pass, including positive
two-species isolation, exact held-gene leakage rejection, and digest-drift
failure. The two architecture tests pass, including context/action permutation
invariance and absence of gene-named parameters. A one-epoch, two-species CPU
trainer smoke test produces finite held-molecular evidence. Eight initial OMF resources
validate against the exact 1.0 schema, the pretraining workload passes semantic
graph projection, and both module contracts and dependency digests validate.
The fixture audit reports zero leakage, zero benchmark records and NCBI taxa
4932 and 9606. No biological dataset, GPU training, checkpoint selection, or SL
benchmark was opened.

**Decision and limits.** The repository advances to the SLp-1.1 factory and
architecture-contract phase; it does not advance a biological model. The
Costanzo rights document remains quarantined with `trainingAllowed: false`
until explicit training and redistribution terms are verified. The world
module currently relies on an operator-provided PyTorch/NumPy runtime and is
not release-eligible until its dependency lock is hash-pinned. OMF 1.0 supports
local execution only on Linux x86-64 and passes JSON protocol state, but not a
materialized large checkpoint artifact, to its independently captured
inference adapter. Portable SLp release promotion is therefore closed until an
upstream artifact-to-adapter contract is implemented and tested; absolute run
paths and weights serialized into metadata are explicitly rejected.
Only the Linux-local executor is bound. Historical Modal launch code is not an
OMF executor integration, so large remote training is also closed until a
provider implements and passes the `omf.executor/v1` transport, cancellation,
recovery and scale checks.

**Factory runtime verification.** The reset was bootstrapped on Ubuntu 24.04
under Python 3.12 from the pinned OpenModelFactory source revision. The
bootstrap plan contained only untracked `.omf/` runtime state. `omf doctor`
passed all eight checks, including Git, policy, signing identity, database and
artifact-store integrity. Bounded agent context reported zero datasets, runs,
deployments or blockers, and capability discovery exposed only the built-in
local POSIX executor; no remote executor was inferred. The active signed OMF
goal `slp-1-1-frontier` has revision
`sha256:12fd72391c72993761245698f64da2db9ff15c3495055822ff41d89f50efa338`
and fixes the molecular advancement rule plus the no-benchmark-selection,
species-provenance, pinning and release constraints. All eight initial OMF
manifests validate against the pinned 1.0 schemas. No workload or biological
compute was allocated.
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

## 2026-09-03 — SLp-1.1 pre-compute contract hardening

**Hypothesis and fixed rule.** Before any biological training, an identical
molecular query must produce an identical marginal distribution regardless of
the other queries in its panel, and a passing corpus audit must become invalid
if any admitted manifest, trajectory-gene inventory or shard byte changes. The
first biological candidate remains closed until it achieves both at least 0.02
nats per observed-target and 2% Gaussian-NLL improvement over the strongest
fixed mean/ridge baseline, at least 0.10 training-centroid-adjusted molecular
Pearson, and non-negative source- and species-level NLL deltas. The absolute
delta was added before data exposure because a relative density-NLL percentage
alone depends on the additive likelihood constant and registered value space.

**Architecture correction.** The initial Transformer decoder allowed query
self-attention. In a direct adversarial probe, the same query changed by
0.24558 in predicted mean and 0.49919 in log scale when unrelated queries were
added. It was replaced with stacked pre-norm cross-attention and feed-forward
blocks with no query-to-query communication. The corrected CPU probe is
bitwise identical under panel subset, reordering and chunking (maximum mean and
log-scale deltas 0.0). Masked query features and `readout_type=-1` sentinels are
now inert.

**Admission correction.** The corpus audit now attests SHA-256 identities for
each exact manifest, trajectory-gene file and shard, plus an aggregate content
digest. The trainer recomputes those identities before constructing a model.
Numerical shards require fixed-width stable record, source, perturbation and
action identities; exact action-CURIE inventory agreement; declared taxon to
species-vector agreement; compatible shapes and dtypes; finite tensors; and at
least one observed molecular target per represented species. Reusing a valid
audit after substituting another corpus is rejected before training. Gaussian
NLL now includes its normalization constant and emits an absolute NLL delta.
Molecular reinforcement is schema-disabled until a matched deterministic
continuation and rollback-safe source/species preservation gate exist.

**Species-native yeast boundary.** A new self-contained preparation module
accepts only an immutable checksum-pinned raw snapshot with verified
`trainingAllowed: true` rights, taxon 4932, and SGD CURIE actions. It rejects
missing, false or drifted rights; mutable release aliases; raw digest or count
drift; symbol identities; taxon relabeling; duplicate/unsorted records;
non-finite values; and unbounded token shapes. Output is deterministic,
bounded, trainer-validated NPZ shards plus a corpus manifest, exact intervention
inventory, provenance report and tar artifact. Costanzo remains quarantined and
was not prepared or admitted.

**Molecular evaluator and OMF boundary.** A separate self-contained evaluator
checks checksum-bound molecular references and predictions, nested held-gene
overlap, source/species identity and benchmark-like fields. It reports ordinary
errors plus training-perturbed-centroid Pearson/cosine and common-panel centroid
accuracy by species and source. This is explicitly Systema-inspired for sparse
multi-modal profiles, not an exact reproduction of the dense single-cell
benchmark. Source inspection of OMF commit
`ef26eea2cb694596f7680a4bce400371738cbb4b` and a direct resolver probe show
that a same-workload generated artifact is captured but cannot be pinned for a
dependent stage: resolution fails with `reference input was not pinned at
admission`. The supported design is two runs, with the second workload pinning
the first run's literal prediction-artifact digest. No sibling-stage path or
metadata-embedded payload workaround was added.

**Validation and decision.** The merged repository passes 40 unit tests,
including adversarial query, audit-substitution, non-finite tensor,
action-inventory, species-vector, rights, species-native preparation,
perturbed-mean and nested-intervention checks. All changed Python modules
compile, three fixture manifests validate against the corpus schema, and the
world, audit, yeast-preparation, molecular-evaluation, workload and evaluation
manifests validate against pinned OMF 1.0 on Ubuntu/Python 3.12. The
seven-study public perturbation-atlas history from the concurrent main branch
was retained, but it is not silently admitted as an SLp-1.1 snapshot. No
biological model, checkpoint, molecular result or SL benchmark was produced.

## 2026-09-03 — first admitted OMF audit-to-world execution

**Hypothesis and fixed smoke rule.** The new factory should execute an
audit-to-training DAG from three immutable, rights-bearing molecular snapshots
without a repository-relative module import, unpinned input, hidden benchmark
record or intervention-gene overlap. This engineering smoke passes only if all
three snapshot payloads verify, the audit reports zero leakage and zero
benchmark records, both synthetic taxa 4932 and 9606 are represented, the run
terminates successfully, and the checkpoint file hash equals the digest
reported by the world module. It cannot advance a biological model regardless
of its toy metrics.

**Exact admission.** OMF 1.0 source commit
`ef26eea2cb694596f7680a4bce400371738cbb4b` admitted Git commit
`e52a0af4164a6c9404527077f92cf6eb60090c78`. The workload resource revision is
`sha256:a944b6ef699216a416ee9391d7551380f8aa1719a87f3e940c1f302aba7ea6c7`;
the audit and world module artifacts are
`sha256:d9a63134c6082691d89079b555e7fb1e73e52deb0dfb682cf0239b3e14dd56f1`
and
`sha256:f059e9ec9bc7e3b958040595e3ae12dcf0acb055032e5d786e7630a46e38ddba`.
The CC0 synthetic snapshot revisions are pretrain
`sha256:a17db98246ee3f9185d3809347e3354eaa5e0c09be101fe5c0c523dbe6319927`,
validation
`sha256:0ac5b846cffde156d949033f8d2586eb6a8b5d0b0e971824f55990a2ee9a082e`
and reward
`sha256:43fb342a3e655196f9e79888ff2882a82dbb7a39f4bf0a47157dd87d74af8152`.
Every snapshot contains eight synthetic records split evenly across the two
taxa, with globally disjoint intervention genes.

**Execution evidence.** Executor preflight returned ready with no missing
capabilities. Run `01a06b0c-9477-7982-9e22-f9e2313a7ccd` reached `Succeeded`
at result revision
`sha256:f80ce282fb11164a7cb76774ca4d6a564703a579fb2bc0b1927cdf412afe720a`.
The audit attested all six corpus-file digests, 24 records, two species, zero
leakage and zero benchmark-label records. The one-epoch CPU training stage used
Python 3.12.3, NumPy 2.3.3 and PyTorch 2.8.0+cpu as recorded in OMF's runtime
inventory, created 1,450 trainable parameters, and emitted a 27,098-byte
checkpoint. Its file SHA-256 is
`3374a0397c18f6c5d9337d4593ec225261ad89ae466e0f1e5c1f7409b6e464d4`,
exactly matching `checkpointSha256`; OMF wrapped it in immutable checkpoint
artifact
`sha256:3ed1d807c73640ed8e401e66285fd4d42bdaa3daf82e7111038c0fe57f0e327c`.
Stage-level lineage links both module artifacts and all three dataset revisions
to both audit and training activities, and links each dataset to its imported
source digest.

**Decision and failures retained.** The aggregate toy NLL improved by 0.12899
nats (7.88%) and the raw toy effect Pearson was 0.28138, but the worst-species
NLL delta was -0.43494 nats (-53.44%). The scientific advancement rule therefore
fails even on this deliberately tiny fixture; no checkpoint is selected and no
benchmark is opened. No immutable `EvaluationResult` was created because this
workload tests execution compatibility rather than the frozen biological
evaluation. The run also exposed three operational defects that are now fixed:
newline-only dependency locks incorrectly triggered isolation, a per-user
`RLIMIT_NPROC` of 16 could not spawn under shared WSL, and two validation
fixtures violated their own required input contracts. OMF's module-test runner
also reuses `module-0` for every manifest named `module.yaml`; tests were
isolated by clearing only that ephemeral run directory until upstream supplies
unique test identities. All four isolated OMF module fixtures and all 45 local
tests pass. The empty dependency lock and disposable CPU environment remain
ineligible for release; a hash-pinned wheelhouse is still required.

**First biological acquisition decision.** Primary-source and repository
rights review identifies the CC-BY-4.0 yeast deletion single-cell atlas
(`E-MTAB-14004`, versioned Zenodo DOI `10.5281/zenodo.14062629`) as the first
transcriptomic source: 1,061,865 profiled cells, 710,952 assigned/QC cells and
more than 3,500 deletion genotypes in control and 0.4 M NaCl at 15 minutes. The
CC-BY-4.0 genome-wide knockout proteome (Mendeley Data version 2,
`10.17632/w8jtmnszd9.2`) is second: 4,699 strains with an average of 2,520
proteins per strain. No bytes were downloaded or admitted. The current v1
contract cannot make NaCl, time, medium, temperature, chemical identities,
metabolite readouts or morphology traits model-visible with stable namespaces
and units. Encoding them as fake SGD identifiers, source strings or anonymous
numeric covariates is rejected. A typed context/action/readout v1.1 contract is
the next prerequisite; only then will one global held-intervention roster be
frozen across transcriptome and proteome snapshots.

## 2026-09-03 — typed sparse world and global held-intervention contracts

**Hypothesis, fixed rule, modalities and snapshots.** Before allocating
biological training compute, a dictionary-size-independent model should accept
typed, sparse molecular observations without learning stable IDs; the same
outcome-blind yeast roster should assign every intervention covered by both
protected sources to exactly one immutable role; and uncertainty diagnostics
must not silently modify the molecular gate. This contract milestone passes
only if dictionary permutation and extension leave parameters and identical
queries unchanged, sparse missingness and typed likelihoods fail closed on
ambiguous inputs, the roster is deterministic from the exact common SGD-CURIE
set and mapping revision, and both modules validate with pinned OMF 1.0. It
cannot advance a biological candidate. The accessible modalities were the
published yeast knockout proteome payload and metadata, plus the yeast
single-cell-atlas README; no biological OMF `DatasetSnapshot`, RData payload,
checkpoint, molecular reward set, final holdout, or benchmark snapshot was
opened or created.

**Typed sparse candidate.** `slp.corpus/v1.1` separates checksum-pinned entity,
query and panel dictionaries from record-local CSR targets. Context, action,
record and observation covariates declare `world`, `likelihood`, or `audit`
access; missing numerical values require an explicit false mask and zero
storage. Readouts declare units, implicit-zero semantics, and Gaussian or
negative-binomial likelihoods. Model batches contain features, missingness
masks and ontology-type indices but no stable entity/query IDs or dictionary
indices. The independent cross-attention decoder is bitwise query-invariant,
and the parameter count does not depend on dictionary cardinality. Exact
active nonzero-taxon action entities must equal `trajectoryGenes`; neutral
chemical actions are excluded from that gene inventory. Source weights produce
deterministic quotas followed by source→perturbation→replicate→record cycling.
The OMF module accepts only a copied SHA-256-pinned `DatasetSnapshot` at the
materialization path implied by its resource. It deliberately returns
`trainingImplemented: false`, emits no checkpoint, and is not a model result.
Negative-binomial library-size offsets and the biological optimizer remain
blocked contracts.

**Outcome-blind global yeast roster.** A separate module accepts two or more
immutable identity-only inventories and rejects quantitative fields. Every
inventory must use taxon 4932, canonical `SGD:S#########` CURIEs, and the exact
same immutable SGD mapping ID and digest. The candidate set is the intersection
of QC-passing identities across every protected source. Assignment is the
first 64 bits of SHA-256 over the domain-separated string
`slp-1.1-yeast-global-held-v1\x00<SGD-CURIE>` modulo 100: buckets 0–9 are
molecular final, 10–29 molecular validation, and 30–99 pretraining. There is no
outcome access, optimization or reroll. Stable-identity mapping remains a hard
source-admission prerequisite; no biological roster was generated in this
milestone.

**Source probe.** Four allowlisted files from Mendeley Data
`10.17632/w8jtmnszd9.2` were downloaded to an untracked temporary directory and
matched their upstream checksums: `yeast5k_noimpute_wide.csv` SHA-256
`69a9df05b6db011f595a4e0b3ce25c1cc247f22cbdd066c79e6da9a706aa1df9`,
metadata `48864282c82d516ae929dc87aff7fae9e05e9b922e316c001f3d29dce0ff878b`,
knockout-detection metadata
`ca7c8f2ac33272df3763807add7b8982b8a8b52d4276bd929a61ecf19e0ae405`,
and documentation
`4078289dc86dd6b526d9b0c963e6df61d53acdfdf6260abdeae307588623f828`.
The matrix has 1,850 protein rows and 5,476 sample columns whose order and set
match the metadata exactly, 255,715 literal `NA` cells, no empty cells, and one
quoted comma-bearing sample header. The metadata has 4,699 knockout, 389 QC and
388 HIS3 rows over 57 plates; the 4,699 knockout rows contain 4,549 distinct raw
ORF strings, including 247 extended systematic identifiers requiring an
explicit pinned map. The values are positive batch-corrected MaxLFQ relative
intensities, not the later log2 differential-analysis table. Any target
transform must be frozen from fitting records and WT controls only; published
all-knockout centering cannot include held interventions. The Zenodo atlas
README SHA-256 is
`268533b10c59d3f4ca941ff31ac8b9c108b61f55f00d85792d44b3a90b3b9da8`;
it confirms separate control and NaCl objects in `seus_split.RData`. The 5.9 GB
RData structure, cell identities, raw-count layer and missingness semantics
remain unprobed, so the atlas stays contract-blocked.

**Molecular comparison protocol.** The frozen comparison contract now names
intervention-gene-cold, context-cold-with-declared-basal-access, and double-cold
tasks separately. It requires context-only, TxPert mean/additive and
feature-bilinear ridge baselines on molecular fitting folds. BDS remains
inadmissible at or below 0.5 but is contract-blocked until replicate halves and
anchors exist; differential-expression metrics are blocked until their method,
FDR, effect threshold and feature universe are frozen. Energy and Wasserstein
metrics are prohibited for the current marginal-output model. The evaluator
reports inclusive central-Normal 50% and 90% empirical coverage and mean full
interval width overall and by species, source and species×source. These are
diagnostics only; the molecular decision function and thresholds are unchanged.

**Validation and decision.** On Windows, all 73 repository tests pass with one
expected symlink-creation skip. The 36 focused sparse, roster, evaluator and
source-manifest tests also pass on Ubuntu/Python 3.12, including the symlink
checks. Changed Python files compile and the JSON schema validates. OMF module
validation produced sparse package
`sha256:c2476b34284651c3f723f806836e8e09ae75961609393a6b4ca2b42ee4478620`
with artifact manifest
`sha256:bde1c96757e7fb16b8f08e6777ad403c01681d986e7d443dd596a347ce8ba41f`,
and roster package
`sha256:aa046b9041bbf51ffe29e8801e833467b4125c05c7279dcbb46a61759532c102`
with artifact manifest
`sha256:e52f95a625cbd228e4302a1e985d9a362b58b3e21d66c174ad2faa5a08ec8672`.
OMF's fixture runner still lacks `omf.sdk` in the empty dependency-lock
environment and reuses a fixed module-test identity; no interpreter-path or
metadata-payload workaround was added. This milestone admits the contracts,
not a model. No biological metric, selected checkpoint, SL benchmark result,
novelty result, or SOTA claim exists.

## 2026-09-04 — sparse optimizer, molecular baselines, and exact SGD identity map

**Hypothesis, rule, modalities, and fixed inputs.** Before admitting biological
outcomes, the typed sparse candidate should support a deterministic
pretraining-only optimizer, the molecular protocol should have executable
leakage-safe simple baselines, and the exact SGD source objects should produce a
canonical stable-identity relation map without guessing symbols, case, retired
redirects, or one-to-many targets. This engineering milestone passes only when
two identical sparse fixture runs have identical parameter, prediction and
report hashes; every validation intervention is absent from pretraining
trajectories; baseline fitting obeys its frozen task and source/species access
rules; the six SGD objects match their exact version, byte count and SHA-256;
and Windows, Linux and OMF validation pass. It cannot advance a biological
candidate. Accessible modalities were synthetic sparse Gaussian molecular
records, SGD current-feature identities, typed external-accession relations,
retired/merged records, the already verified Mendeley proteome matrix and
metadata, and the atlas README only. The exact biological source contracts
were `slp-sgd-map:2026-08-28-object-set-v1`, Mendeley
`10.17632/w8jtmnszd9.2`, and Zenodo `10.5281/zenodo.14062629`; no quantitative
biological OMF snapshot, RData payload, reward set, final holdout, benchmark
snapshot, or biological checkpoint was opened.

**Sparse optimizer boundary.** The new pure-library AdamW loop reads only
`pretraining` records for updates and only `molecular-validation` records for a
fixed before/after diagnostic. Each scheduled record contributes its own mean
observed-target NLL and then receives equal batch weight, preventing dense
panels from dominating sparse records. Epoch schedules are independently
domain-separated while retaining exact source quotas; benchmark-like fields,
role drift, held-gene overlap and contract drift fail before optimization. On
the deliberately tiny fixed fixture, 18 epochs of eight records each allocated
72 records to each of two sources. Overall per-target validation NLL changed
from 1.5602428317070007 to 0.5222099050879478 over three observed targets;
source A changed from 2.0367143154144287 to 0.7435191869735718 and source B
from 0.6072998642921448 to 0.07959134131669998. Two independent runs produced
parameter hash
`9880a5eaa35ddda710e7565f874901715ff87e968c2329dbbef623b39847f044`,
prediction hash
`d19f2ac329f1e31b829e6e4c2ced89326cb8de57da8cf974eed16995f28467a7`,
and report hash
`2ee5c359f7fd1b2a0f4e12457af87235387ec95053945545cfa0121c8ebb8aee`
exactly. These are initialization-to-final engineering diagnostics, not a
training-mean/ridge comparison, checkpoint-selection result, biological metric
or advancement decision. The OMF sparse entry point remains validation-only
and emits no checkpoint.

**Frozen molecular point baselines.** A separate self-contained module now
implements context-only and TxPert mean/additive predictions over immutable
centroid-profile snapshots. It requires an aggregation-protocol digest and an
exact fitting-manifest link. Context-cold reference interventions must occur in
another fitting context in the same NCBI-taxon/source stratum; gene-cold and
double-cold interventions remain absent from all fitting outcomes. Exact,
single-intervention and global TxPert effects never cross that stratum, and
missing inputs remain null. The outputs deliberately omit a probabilistic
scale, so the molecular evaluator blocks them rather than manufacturing
uncertainty. Feature-bilinear ridge remains blocked until released query and
action features exist.

**SGD acquisition and mapping.** Six exact SGD objects totaling 18,804,370
bytes were retrieved only to untracked temporary storage and matched locally
computed SHA-256 values recorded in
`sources/sgd-stable-id-mapping-2026-08-28.yaml`. The rights declaration is
CC-BY-4.0 with training and redistribution separately allowed and attribution
required. The mapper verifies the exact file set, documentation markers,
line/record bounds, copy-materialized DatasetSnapshot shape, symlinks, bytes
and hashes before output. A disposable real-payload smoke emitted 6,613 current
ORFs, 228,320 typed relation keys, 19,276 one-to-many typed keys and 105
retired/irregular quarantine rows, including five irregular physical rows. Its
canonical mapping digest was
`6fd789df6099b78a8842baa8f1d20ab0a3fe77f27ce512ee783444eb2627ef2a`.
The smoke output is not retained or admitted; the digest must be reproduced
from the admitted raw DatasetSnapshot before it may enter the global held-gene
roster. The Mendeley protein keys remain UniProt accessions connected to all
exact typed current-ORF relations, including five one-to-many keys; none is
arbitrarily collapsed to a gene.

**Superseding source-probe correction.** The previous section's counts of
4,549 distinct raw knockout strings, 150 duplicates and 247 extended rows were
produced by a case-insensitive PowerShell enumeration and are superseded. Exact
case-sensitive counting finds 4,550 unique strings and 149 duplicate rows:
4,451 exact uppercase simple rows, 246 exact uppercase suffixed rows and two
noncanonical mixed-case values (`YAL043C-a` and `YML009c`). Case folding merges
`YML009c` with a separate `YML009C` row. Therefore the admission policy keeps
only exact current one-to-one systematic mappings and quarantines every other
intervention row; it never uppercases an unmatched value.

**Validation and decision.** Pinned OMF 1.0 again passed all eight health
checks; the bounded context still exposes only the Linux-local executor. The
101-test Windows suite passed with two expected symlink-permission skips, and
the 28 focused tests passed on Ubuntu/Python 3.12 with symlink rejection active.
The SGD workload schema validates. OMF module packages/artifact manifests were:
sparse world
`sha256:3f446ef45aa7af51416492450caaee7768bd95ceebbdf9e7622bb9b57fec558c` /
`sha256:18e66120add2c0bf47146e68d411e4e776d4e2873932cfb15e823a6d7320980c`,
molecular baselines
`sha256:275b2e4429359c6e73c6a60079c534ab2d0ae69f587c6c6db397aed699ff4b4e` /
`sha256:317a58382b6e6a5b09b8f9b33a7769755ab64faa46807beb748e4107ffc09e90`,
and SGD map
`sha256:ef19d6d6584cbc9ffd0b472989d30641bce518be52cf24920643b0689064ba50` /
`sha256:c8970cfc56c7c0270997a42057b398b8f385259707f56ac068c5120d9083cef2`.
No candidate advances: there is still no admitted biological corpus, OMF sparse
training/checkpoint path, frozen probabilistic baseline scale, fixed ridge
implementation, molecular-gate result, SL benchmark result, novelty result, or
SOTA evidence.

## 2026-09-04 — admitted SGD identity normalization

**Admission and fixed acceptance rule.** The exact six-file SGD object set was
admitted only as an identity-normalizer input under the reviewed CC-BY-4.0
rights declaration. DatasetSnapshot
`omf://abiome/slp/datasetsnapshot/slp-1-1-sgd-map-raw-2026-08-28@sha256:061906684f67100bd855cfbd3e0ed4df2b1b3f0339ebbc34679ba8e535a214cf`
has manifest digest
`sha256:70af3a16a69895580d96243e3da1667b9a9a2fe3d6fe72a86542241730922da0`;
`omf data verify` returned valid. The run could pass only if its canonical
mapping digest exactly reproduced the preregistered disposable-smoke digest,
all pinned input and output counts/hashes matched, and current, ambiguous and
retired relations remained distinct. It could not authorize model fitting or
advance a model.

**Failures retained.** Run `01a06b56-3e29-716f-ab9b-e70eb102d289` failed
before module execution because an absolute invocation of the OMF CLI without
activating its environment let relative `python3` resolve to unsupported
CPython 3.13.12 without `omf.sdk`. OMF's documented contract is to activate its
supported Python 3.11/3.12 environment; no `PYTHONPATH`, absolute module
interpreter, or dependency-lock workaround was added. With the pinned Python
3.12 environment activated, run `01a06b57-de6b-706b-bc77-6bcbc3b69ce3`
reached the module and exposed an actual entry-point defect: OMF 1.0
`ProtocolRequest` has no `outputs_dir` attribute. Commit `4c79246` corrected
artifact placement to the directory containing the documented
`OMF_RESULT_FILE` and added a regression guard. Both failed runs remain
immutable failed evidence.

**Successful execution and lineage.** The clean, pushed Git revision was
`4c7924663010409a7aaaec644afd33fa2e865c9a`. Module validation produced package
digest
`sha256:aca55394cf2853bdb0daa60da347f23968d5b3eda697c85203e361f67db4bc88`
and source artifact
`sha256:1f9cae8a1c44adcba888ef30bbcd7f3dd41308276b357f9e312d8e93b52aca65`.
The binding digest was
`sha256:5dc181c5643d2a98ec43b1c79791764e4f5ba5ad4edbd5730d7f455ced4d6e2f`,
the workload digest was
`sha256:56f4bdb3d5a76e3fff72c24bec33fb76c2bf0cc9abb73ee4a06f99cc8f7f00ff`,
and the admitted environment digest was
`sha256:2857f0de3e6a95a520da1d0b3ba81b0974bac0a1f8c88a6d048320edff836169`.
It records CPython 3.12.3, `open-model-factory==1.0.0`, the exact interpreter
digest, and network isolation. Its empty lock has `realization: null` and the
SDK is an editable installation from the pinned temporary OMF checkout, so the
runtime is attested but is not a portable dependency closure or release
environment. Run `01a06b59-cd6a-748c-9347-79b515d0a622` succeeded with RunResult
`omf://abiome/slp/runresult/result-01a06b59-cd6a-748c-9347-79b515d0a622@sha256:0283aa638d96dc06ab48f0cb775402f4b7018639fb03ea419f240c658c56fc6b`.
Upstream lineage contains the exact DatasetSnapshot and module source; four
generated edges lead to the current-ORF, typed-relation, quarantine and mapping
manifest artifacts.

**Outputs and decision.** The admitted run reproduced canonical identity digest
`6fd789df6099b78a8842baa8f1d20ab0a3fe77f27ce512ee783444eb2627ef2a`
exactly. The mapping manifest content SHA-256 is
`570557ab1201913a18de9790f8adc5ee2e3cb56c6bb0e8d588fe43660c0214e1`.
OMF artifact digests are current ORFs
`sha256:e67f0e8773feae108ecdb687139885e01ca972ff4aec95cd1358b33db1ea1192`,
typed external relations
`sha256:75e0fef99bbae3bb4e4dc3e2f24cfd0ab62919c0e6e3e321e8d82f3bd557f4da`,
retired/irregular quarantine
`sha256:07ea82f877224496c24effc2aa2a2b684c01b85017e616a70f003a5363f6925f`,
and mapping manifest
`sha256:c74ea81ce604357b998e5f09130dff85bf8a7a26504b9b2426f8038608c52d9c`.
Counts remain 6,613 current ORFs, 228,320 typed keys, 19,276 one-to-many keys,
100 retired/merged rows and five irregular rows. OMF Knowledge revision
`sha256:2d1e731b274d373904b45d951cb4a31cf84606224435847f37390ba69d6204c9`
records this as identity provenance with no biological-model claim. The exact
mapping digest may now bind source-specific identity inventories, but the
mapping artifacts are not biological pretraining, validation, reward, final or
benchmark snapshots. No model or benchmark gate was opened.

## 2026-09-04 — yeast source admissions and outcome-blind identities

**Hypothesis, rule, modalities, and fixed inputs.** Before any biological
optimization, independently published yeast proteome and single-cell sources
should yield one stable-ID intervention population without reading molecular
outcomes or resolving by symbols, case normalization, retired redirects, or
one-to-many guesses. This milestone passes only when immutable rights-bearing
raw snapshots verify, both identity adapters reproduce their preregistered
counts and mapping digest, phenotype/quantitative mutations cannot alter the
identity outputs, and the emitted inventories remain separate from fitting,
reward, truth, and benchmark data. Accessible modalities were proteome sample
metadata plus protein accessions, the atlas genotype/cell-count summary, and
the admitted SGD relation artifacts. No biological outcome was admitted as a
training, reward, validation, final, or benchmark corpus.

**Raw DatasetSnapshots.** The four-file Mendeley proteome release was copied
under its reviewed CC-BY-4.0 declaration as
`omf://abiome/slp/datasetsnapshot/slp-1-1-proteome-raw-v2@sha256:5392d4df7e962c9f59798b83fbdf8e71cd568b30c78a498702d50fabc059397e`
with manifest digest
`sha256:7f25f6e11d3deb73624d1c59f7aead59aef77641be6a54b8d0ee838e305f2213`;
`omf data verify` returned valid. The exact one-file Zenodo
`ptb_summary.Rdata` snapshot was copied under its reviewed CC-BY-4.0
declaration as
`omf://abiome/slp/datasetsnapshot/slp-1-1-atlas-genotype-summary-raw-v1@sha256:c7cad889b43f293fe5b59e3fd2486f5dabf0b0b362964968eb4858f6917268ef`
with manifest digest
`sha256:97df177ff586d3409d6348926562c5ba4c4943ab4789b1823d059bf6c708fa31`;
it also verified. The atlas summary physically contains phenotype columns, so
its authorization is narrower than its bytes: the identity adapter may access
only exact genotype assignments and non-null integer cell counts above five,
and the snapshot is prohibited from fitting, reward, or molecular evaluation.

**Identity contracts.** The proteome adapter admits 4,623 knockout rows over
4,476 exact current SGD interventions and quarantines 76 rows; it retains all
1,850 UniProt readouts and all five exact one-to-many SGD relations. Its binary
matrix scan decodes only the header and first identity field, and tests show
that changing quantitative bytes cannot change identity artifacts. The atlas
adapter takes the exact non-WT intersection of control and NaCl assignments:
3,151 candidates produce 2,941 current SGD interventions, while 19 retired or
merged and 191 unmatched assignments remain quarantined. Its `rdata==1.1.0`
runtime converts the small frames, but the adapter never indexes, inspects,
uses, or emits the seven phenotype fields; mutation tests hold all emitted
identity bytes fixed. The dependency lock is hash-pinned but is not an offline
wheelhouse or portable release closure.

**Quantitative atlas constraint and decision.** The separately downloaded,
untracked `seus_split.RData` matched 5,907,877,873 bytes, upstream MD5
`65bb56efd8120f32f65c044de5f040aa`, and local SHA-256
`da99869c11d1a6c034454568098aa50bc3313cd4508dbd506d43241b0fb4695d`.
A streaming gzip integrity check passed and measured 21,596,869,016
uncompressed serialization bytes before R/Seurat object expansion. That
already exceeds the 15 GiB local memory envelope, so local full parsing is
forbidden; quantitative conversion requires an admitted remote executor or a
source-native streaming path. Commits `f88f11f`, `88a2e63`, `8af455c`, and
`4aa8315` record the reviewed adapters, admissions, pinned workloads, and
source facts. The inventory OMF runs and separate admission of their emitted
bytes remain pending a clean worktree. This milestone establishes identity
and compute boundaries only; it supplies no molecular metric, candidate
advancement, SL benchmark result, novelty evidence, or SOTA claim.

## 2026-09-04 — target-separated OMF training and evaluation boundary

**Hypothesis, fixed rule, modalities, and snapshots.** Before biological
optimization, a sparse world-model run should be able to fit and serialize a
checkpoint without any protected molecular target being present in its process,
and every later score should be an exact evaluator-only join against a frozen
target-free query. This engineering milestone passed only if checkpoint,
prediction, and report bytes were deterministic; changing protected
validation/final corpus fingerprints could not change checkpoint bytes; the
query covered the complete validation-roster intervention domain; every
validation/final intervention remained absent from pretraining and reward
trajectories; all active OMF artifacts were file-valued; and the complete test
and OMF validation surfaces passed. Accessible modalities remained the tiny
synthetic sparse contract fixtures plus outcome-blind identities from the exact
SGD, Mendeley proteome, and Zenodo genotype-summary snapshots already recorded
above. No quantitative biological corpus, molecular reward, protected truth,
checkpoint-selection metric, or SL benchmark was opened.

**Training and truth boundary.** `slp-1-1-world-sparse` now accepts exactly one
pretraining `DatasetSnapshot`, one target-free molecular-query snapshot, one
prior-run corpus-audit file artifact, and the outcome-blind held-roster
snapshot. It performs fixed-epoch AdamW updates only from pretraining targets
and emits a bounded canonical checkpoint, training report, and predictions.
Predictions repeat exact query identities and typed distribution parameters but
contain no target, observed mask, benchmark field, or target-derived inclusion.
The independent evaluator v2 alone receives fitting-only centering,
evaluator-only held truth, query, admitted audit, roster, exact prediction, and
checkpoint artifacts. It exact-joins profile and readout panels, scores
Gaussian and negative-binomial NLL only where truth is non-null, removes the
fitting perturbation centroid, and reports source/species strata. Its current
correlation thresholds are diagnostic only; they do not replace the frozen
mean/ridge NLL advancement rule in `MODEL_CARD.md`.

**Audit and OMF artifact compatibility.** Corpus-audit v1.1 independently
reconstructs the exact QC-passing intersection from at least two protected
source-inventory snapshots, recomputes held assignments, checks exact sparse
NPZ members and record-level active interventions, and isolates both validation
and final genes from pretraining and reward. Inspection of pinned OMF 1.0 found
that directory artifact import supplies `logical_kind` twice, which raises a
Python `TypeError`. The repository does not patch OMF or pass data through
metadata. All active producers now emit regular files: multi-file identity and
baseline outputs are separate immutable artifacts, the audit is one JSON file,
and predictions are one deterministic uncompressed tar containing exactly
`evaluation.json` then `profiles-000.jsonl` with canonical zero-time metadata.
The evaluator verifies member order, type, ownership, permissions, sizes,
hashes, counts, and target-free structure before streaming records.

**Validation and decision.** The Windows suite passed 140 tests with three
expected symlink-permission skips; changed Python compiled and `git diff
--check` reported no errors. On the supported Linux OMF environment, all active
workload and evaluation schemas validated. Module package/source-manifest pairs
were: proteome inventory
`sha256:166fed5cc0423a3913c677d0d2b5e5d5b4d1aa3466e294ce2f9471d8a029af7a` /
`sha256:72ef0f3ffbb0f91524af02bb8bc804d2e03d83cc427238eb5da4ec12ae8b7bc5`;
atlas identity inventory
`sha256:32d867cb47c16a0dbeebd011c5b86595eeca7be6cd920fa00983869bd52162bf` /
`sha256:95cc95a3d76bcffe2089548d9fadbe94a909605293c62468ccda3f3ad7f670ea`;
corpus audit
`sha256:ed5f1d93535e46b4864f83e381fdf22ee5f6ba6222a3dbf29e8159f67a39a00b` /
`sha256:4881ac861adbe6cb2471d103900bad67de64a7167851716fda1636391b4a7dcb`;
sparse world
`sha256:f0cbae8d5cf35a79f9bf049b7b7dcb58670d85345d5be49a2492ea2e9e715372` /
`sha256:b19cee10183e3de4e43361475afe2e25e1e4bb341f756a4cd3846de12ab3ac60`;
molecular evaluator
`sha256:54edac6b6a6d0fc0942c7a5cbb69911d1e32bf7a2aa9aa5224dc84e2835f5a3c` /
`sha256:6e6b98eba15362ebd6f8b090812c8b9c227f4e4f27371cce347c7d77e8cf44bf`;
and molecular baselines
`sha256:735fa3b539f3bf6efd47b59e1438b5836fb10b4132c3ee5c27d99635fa50c41b` /
`sha256:efee5272fba05cace33919e1f4cc6ede29855e24f5daa65d38097d92b30aeb1c`.
The dense target-bearing prototype workloads and obsolete trainer tests are
quarantined, with an invariant test preventing re-entry into the active graph.
This advances the executable factory boundary only. No biological candidate,
molecular gate, SL benchmark result, release, novelty result, or SOTA claim
exists; identity runs and separate derived-snapshot admission remain next.

## 2026-09-04 — executed identity boundary and frozen yeast held roster

**Hypothesis, fixed rule, modalities, and snapshots.** The preregistered
engineering hypothesis was that the two outcome-blind adapters would reproduce
their frozen identity populations through OMF's supported file-artifact path,
and that their exact intersection would deterministically reproduce one global
held-gene roster. Advancement required successful terminal state and lineage,
the exact preregistered counts and content digests, and a separately admitted
rights-bearing roster snapshot; any mismatch stopped the sequence. Accessible
modalities were proteome sample metadata and protein accessions, atlas genotype
assignments and cell-count identity summaries, and the exact SGD identity map.
The quantitative proteome matrix, atlas phenotype fields, atlas transcriptomic
matrix, molecular targets, reward data, and SL benchmarks were inaccessible to
this decision. Before execution, OMF doctor reported all eight checks ready,
the only admitted executor was local Linux, and every run used a clean Git
worktree.

**Failed probes and corrections.** Proteome run
`01a06cfb-8140-7929-a242-4fa22430b762` failed because the adapter rejected the
actual OMF file materialization shape `inputs/<name>/payload/payload`; commit
`dc15a90` corrected both identity adapters and added rejection of the obsolete
shape. Run `01a06cfe-0553-75cb-ad02-1463ac3634d0` then stopped on the frozen
retired-row count. An independent exact-case reconstruction found 35, not 36,
retired rows: `YKR099C-A` occurs in a 12-column SGD row where 13 columns are
mandatory, so the mapper had correctly retained it as a malformed source row
without inferring a systematic identity. Commit `1615a51` reconciled the
partition to 35 retired and 41 unmatched rows while preserving 76 quarantined
rows. Neither failed run was used as advancement evidence.

**Outcome-blind identity executions.** Proteome run
`01a06d02-0fa0-7b70-b202-299269125458` succeeded from clean commit `1615a51`;
its immutable result is
`omf://abiome/slp/runresult/result-01a06d02-0fa0-7b70-b202-299269125458@sha256:48f7833ef4e2122450a45978ffa88311464351ef3908ec7e11cc8ac99b26e0b7`.
It admitted 4,623 knockout rows over 4,476 exact current SGD interventions,
with 76 quarantined rows, and retained 1,850 protein accessions including five
exact one-to-many relations. The inventory and intervention-record content
digests are respectively
`dd683a2585a15377282e669f61dce38c44ea9d3d9d55be71b24842048c05f3e5`
and
`15e011d9f3bbea2e034f47dd06b260f834475ffee8adb452046dbb2701ead497`;
their file-artifact manifests are
`sha256:88773fa08823a7eb1de21ce269a5e5a9668b02bf6ff3d6f9ef3e80b9fb409cd3`
and
`sha256:cc9e9a6479d8e789b59f9544b47da4c67540679786723e86b2ca2cb94442663d`.
The module package digest was
`sha256:25ea27935195662a32be1ce5b1c18ac10b9df5c151f504d76b97e6ea3e9c8e6b`
and its admitted source-artifact manifest was
`sha256:e43684e71c1703155b75f5d114c7f177668684c7be25cf2af3fa86a0e0ed2a5e`;
the workload semantic digest was
`sha256:e80c5e34beb1f419431e6a89bce1e21e75edd8abf6fbd1c1c8638ac95f4dd2a1`
and its WorkloadSpec revision was
`sha256:1fb38a29caafd0d01fd018078c081c277e16be85602a2f4fb69a74bf94492893`.
The admitted environment digest was
`sha256:2857f0de3e6a95a520da1d0b3ba81b0974bac0a1f8c88a6d048320edff836169`.

Atlas run `01a06d02-b27f-7d1b-a6bd-ce1658c586cb` also succeeded; its immutable
result is
`omf://abiome/slp/runresult/result-01a06d02-b27f-7d1b-a6bd-ce1658c586cb@sha256:607fcc0a63a8fb7eefa6adad29ef574d2bb9397d1e26544e0efd7140b4c37069`.
The exact non-WT intersection of control and NaCl assignments contained 3,151
candidates: 2,941 exact current SGD interventions, 19 retired or merged
assignments, and 191 unmatched assignments. The inventory and record content
digests are
`a722677a61996f89a3a402d096d0bceedaf34d1439c3ae4ce72c491729b07774`
and
`5d71c846aa8740f4eb7284ccbcc7cbf857f209c80064d1bbeb823fde9dada66a`;
their file-artifact manifests are
`sha256:eedc782ac4088b5d106119349879258bd3fca109baabf8ba269e505874451f6f`
and
`sha256:de3e56b05692bcce07f174f2e8108d29eed585cf1bd80ee5a60f61d0b3887eb4`.
The module package and admitted source-artifact digests were
`sha256:c66b224a20e50d16345a752328ed4343b2ad5595c3df5cf0ed0fc9e8e5af4bef`
and
`sha256:43ce78f7e4b3da83e5124f5855c6a5e2ef001ad968169c87d63a75aa66ae01f7`;
the workload semantic digest was
`sha256:af9f476aa5f4d9300ea22624e8663204754993aceafe307e0cedbe715174dbdc`
and its WorkloadSpec revision was
`sha256:2d43786b17af4a0089ad02c1843c572d348a6ef56f85154929be8c8b2734d328`.
The admitted environment digest was
`sha256:2dc5606a6dca6e2f3e22f00586ebc49c9a1d2eba5cefd0239de8a7847a3a43dc`.
The adapter converted only the bounded genotype-summary frames and accessed
only genotype assignments and cell counts; it never indexed or emitted a
phenotype field.

**Derived identity snapshots.** Commit `779c6e5` pinned source-specific rights
to the two successful RunResults and their selected file artifacts. The exact
restored bytes were independently hashed, admitted, and verified as separate
OMF datasets:

- proteome identity inventory:
  `omf://abiome/slp/datasetsnapshot/slp-1-1-proteome-intervention-inventory-v1@sha256:bd688dffdf4d96c01d4147580b1a8705c2149acadbc843a719537817a74505d9`,
  directory manifest
  `sha256:a1f5222f3dca31d2ca68ca46a271d39cdca3425a903b5dceb7373481450ada36`;
- atlas identity inventory:
  `omf://abiome/slp/datasetsnapshot/slp-1-1-atlas-intervention-inventory-v1@sha256:3d48478089105b77431f9a7459df3d84bfc41aefe2e2906e6f057b1a6399ae41`,
  directory manifest
  `sha256:9fcf5373923c83e93d5a1d6a7dedce6cfd57bd7aadfad4debb7665436c13bd2a`.

Both `omf data verify` calls returned valid. These are identity inventories,
not pretraining, reward, validation, final, or benchmark corpora.

**Global held-roster execution and admission.** The exact protected-source
intersection contained 2,700 stable SGD CURIEs. The frozen domain-separated
hash rule assigned 1,903 to pretraining eligibility, 529 to molecular
validation, and 268 to molecular final holdout. The headerless, sorted roster
is 248,524 bytes with content SHA-256
`c27eb11a20f593235131f28fc29d8fbd69735f8a0aea88736104850bb875117a`;
coverage is 262,302 bytes with SHA-256
`c746218cbe5a8312e4d00f771d2155ab902d33795381b8c14ada1f9a876e1cbf`.
Run `01a06d0b-bd40-7663-a7a3-b1dfb1c1ebbd` succeeded from clean commit
`71c0c1ba1678567fa5de2879f099c741aa9dba48`; its immutable result is
`omf://abiome/slp/runresult/result-01a06d0b-bd40-7663-a7a3-b1dfb1c1ebbd@sha256:f6ce5a5383e144c1db788419c8d257ec4ce864042b98b03f11f9fc622466e22b`.
It collapsed 147 identical duplicate intervention records, excluded 2,017
identities absent from at least one protected source, and observed zero source
QC failures. The roster and coverage file-artifact manifests are
`sha256:857db5a2ced03b7dc5a88de96406e1f706fdc317e56bf7f5a3fbf951884bf5e2`
and
`sha256:2b54117856cb239cf4614f919872cca10e2d43a7e5cf88415f88240f0f628895`.
Module package and source-artifact digests were
`sha256:5a6e36170b6cc619b36c42afccd818135132d01daed642e97ed412300b77be0a`
and
`sha256:98084356b839dcd3026c7725c1be0517dcf1d949627acf88e98f103126f852dd`;
workload semantic and manifest digests were
`sha256:12776faff683c028514f0d5288c0bab6d9a2c346082b35f116de9825f323c4fb`
and
`sha256:cb3062f3645d4c98ac1a21547337e830ff87b7caf9342445e157e5c208d3e05b`.
The binding and admitted-environment digests remained
`sha256:5dc181c5643d2a98ec43b1c79791764e4f5ba5ad4edbd5730d7f455ced4d6e2f`
and
`sha256:2857f0de3e6a95a520da1d0b3ba81b0974bac0a1f8c88a6d048320edff836169`.

Commit `9f523bf` narrowed derived rights to the roster and coverage files. Their
independently restored bytes reproduced both content hashes and the exact
2,700-row role partition before admission as
`omf://abiome/slp/datasetsnapshot/slp-1-1-held-roster-v1@sha256:1b9a4800370a5398bf83e0a636007f466bf6ca5a6232e2ebb8fc64c5beb63450`,
with directory manifest
`sha256:f8aac504a2d56fdc9e13cc9b1c9fa87a08ebc7ff2d7036c0b6b135c26d187425`.
`omf data verify` returned valid. The advancement rule passed for the identity
and split boundary only. No quantitative biological corpus has been admitted,
no world model has been trained on these sources, no molecular metric has been
measured, and the SL benchmark remains closed; there is therefore no model,
novelty, frontier, or SOTA performance claim. All three execution environments
are attested but remain nonportable release closures: the empty locks have no
realization, while the atlas lock does not provide an offline wheelhouse.

## 2026-09-04 — leakage-separated proteome pretraining observations

**Hypothesis, fixed rule, modalities, and snapshots.** The falsifiable
engineering hypothesis was that the admitted Mendeley proteome release could
be transformed into a deterministic, source-normalized, fitting-only
pretraining observation corpus while never numerically decoding a molecular
validation, molecular final, quarantine, or analytical-QC knockout column.
Advancement required two successful executions from the same clean commit,
identical bytes for every emitted file, exact protected-role exclusions, and
separate rights-bearing admission of the pretraining observations and basal
control. Any byte, count, identity, provenance, or access-boundary mismatch
would stop advancement. Accessible modalities were the pinned raw proteome
matrix and metadata, the admitted intervention inventory, typed UniProtKB-to-
SGD protein relations, the frozen held-gene roster, and the separately decoded
HIS3 controls. Atlas quantitative values, validation and final target values,
molecular reward, SL labels, and all benchmark data were inaccessible to this
decision. The exact input snapshots were:

- raw proteome:
  `omf://abiome/slp/datasetsnapshot/slp-1-1-proteome-raw-v2@sha256:5392d4df7e962c9f59798b83fbdf8e71cd568b30c78a498702d50fabc059397e`;
- intervention inventory:
  `omf://abiome/slp/datasetsnapshot/slp-1-1-proteome-intervention-inventory-v1@sha256:bd688dffdf4d96c01d4147580b1a8705c2149acadbc843a719537817a74505d9`;
- typed protein relations:
  `omf://abiome/slp/datasetsnapshot/slp-1-1-proteome-protein-relations-v1@sha256:acad3427907644f8ab8af38ed36066a6e1148ef92557b727351b0a4fba2b446c`;
- held roster:
  `omf://abiome/slp/datasetsnapshot/slp-1-1-held-roster-v1@sha256:1b9a4800370a5398bf83e0a636007f466bf6ca5a6232e2ebb8fc64c5beb63450`.

The protein-relation snapshot above contains 1,850 accessions and preserves
all five exact one-to-many relations. Its content-manifest and record digests
are `8d559638f48ee4516f7e6fce9e0248e9a1762d58803fe2ed761eff8734f45f86`
and `c72996b4ddc6870a3ab722060eef2fa2747fa9dd121d3e70514dd196c5283b8d`;
its admitted directory-manifest digest is
`sha256:c159573f4f7a2e41b18930d724dea9fb297452a659bdf6050e4718efc1a6c58a`.

**Reward and access contracts.** Commit `d7ede41` advanced the corpus-audit
schema to `slp.corpus-audit/v1.2`. Active training now requires the exact three
roles `pretrain`, `molecularValidation`, and `molecularFinal` plus
`rewardEnabled: false`; it rejects a reward input, reward identity, a missing
flag, a true flag, or an older audit schema. The world trainer and molecular
evaluator accept only this v1.2 contract. This is an enforced absence of
molecular reward, not a zero-valued reward workaround. The final admitted
corpus-audit module package and source-artifact manifest are
`sha256:3a6b431fec4f7c8cb7efe2c21b2ddac4d0ee115d099a41eed003de5a4d3ebe02`
and
`sha256:ff4366c67de4914e24e4f1430c9a6e2716565b4c7b9dedc64008df6a1eeb50c4`.

The self-contained proteome preparation module reconstructs each raw matrix
column from exact metadata coordinates and current SGD identity rather than
assuming row order. It parses the shared readout header, then converts only
the 3,811 pretraining columns and 388 HIS3 control columns to numbers. It does
not convert or validate the 537 molecular-validation, 275 molecular-final, 76
quarantine, or 389 analytical-QC columns. Positive finite abundances are
represented as log2 values without a pseudocount; missing values remain absent
from record-local CSR targets. Raw filenames and raw ORF strings are omitted
from the output shards. The HIS3-derived basal profile is a separate artifact,
is not subtracted from targets, and uses no knockout or QC outcome.

**Failed execution and bounded correction.** Run
`01a06d40-6bd3-7751-b5de-2c2f9e25da93` stopped at source line 6,137 because
the adapter's current-ORF syntax admitted nuclear systematic names but rejected
the legitimate mitochondrial current SGD identifier `Q0010`. An exhaustive
audit found exactly 28 current mitochondrial systematic names, all matching
`Q[0-9]{4}`. Commit `ff15ded` added that exact alternative and adversarially
rejected lowercase, short, long, nonnumeric, and wrong-prefix variants. It did
not loosen any expected count, role, or source revision. The failed run was not
used as advancement evidence.

**Successful executions.** Runs
`01a06d42-8493-7cd5-8c57-779dc8512436` and
`01a06d43-4260-707f-8fa5-4beee63c7856` both succeeded from clean commit
`ff15ded`. Their immutable results are respectively
`omf://abiome/slp/runresult/result-01a06d42-8493-7cd5-8c57-779dc8512436@sha256:6ec04e07cd917b66e16274a04f565844fa7acc1b538fb460c408f9276b5f694c`
and
`omf://abiome/slp/runresult/result-01a06d43-4260-707f-8fa5-4beee63c7856@sha256:d81d5c4f8a790542f3b2c16ed6d0954b27907784aa946e157cee76406c3b22a6`.
Both emitted 3,811 records over 3,679 stable intervention genes with 6,865,493
observed log2 targets and 184,857 missing values. The separate basal artifact
contains 388 controls and 701,619 observed values, with 1,843 of 1,850
readouts supported. The ordered trajectory-list, trajectory-set, and basal-
locator-set digests are
`a37fbd5ba56ba4f38cf4ec0655d7dd9734e4727e77f68064739c24d025d3b7e1`,
`f6083da5b795d5653e630d41758e52855ab1e931d9a6311a1b7ae7350b59b838`,
and `aa012c10e56552108051049fa86c5782461cfd2114ccc6ce4171f947471a6d27`.

Independent `sha256sum` and byte comparisons reproduced identical files from
the two run directories: observation archive
`1f533d7dfb5bd76489b5b4576268e5d5b58fc6200416362876b5a2301c611f0b`,
basal archive
`9be4596a59f3730e7b16995ba562e6561b8f424f46f99aecaeaee78ffe536a71`,
and preparation audit
`9a651c55700644d2522065e4fae48a56188862c3e65aeb92ddc649c09be6463d`.
The first run's OMF artifact-manifest digests were
`sha256:da147a203b93a89e0807e624a43b51cd4037d6e80169fd1601f40a5b3a4250ab`,
`sha256:e3e011fcb5543c714a1b7d2032e6493f4026da54dc865ef4de848b6bde53380a`,
and
`sha256:7fb263f10f6e4b24fa4cdff9861ea22501f94d2ed72ad784059a3371f18892db`;
the second run's corresponding manifests were
`sha256:102c6b48e0e48f1fcdde2fe56790d6f3a0edde23e092b307110e1c75d8a8949d`,
`sha256:0bf29c0b15dd8909496fa2ff3f704c9a89b32f24f4767410ec23a2ce77e33a51`,
and
`sha256:62ce3d3e4cbacc2746eab1068f944d7d7fde993bcfc23ea45bad2ca529829366`.
Manifest digests differ because OMF binds run context; the underlying file
bytes are identical.

The final module package and source-artifact manifest are
`sha256:8803da89c98e103d9fb0dce203cebf9219097a42fb2fd6e5a11c87e81c10384b`
and
`sha256:1c0d9d12989017b6853ee9064026e90ce91da827708a6868111de7a4b480b3f3`.
The workload semantic and WorkloadSpec manifest digests are
`sha256:fd4546ec6bb493312c8015d4682d29af74ff36cdd822d8da30fbc7caa861e0a2`
and
`sha256:f49fc7c9ebf55cfbbb6a25362e79b0e5cf612511446f0c656878ae6aabcb57bd`;
the binding and realized environment digests are
`sha256:5dc181c5643d2a98ec43b1c79791764e4f5ba5ad4edbd5730d7f455ced4d6e2f`
and
`sha256:c2060448ac2beedca12bfe46d125cf4eeedeefb0b22ee47c6933c9031e38e1ab`.
The module imported Python 3.12.3 and pinned NumPy 2.2.6, but realization still
used the package index with no offline wheelhouse and the attestation exposes
interpreter-site-package layering. It is therefore reproducible evidence for
this host, not a portable or release-eligible environment closure.

The fixed dual-run rule passed for pretraining observation preparation only.
The artifacts are source-normalized observations, not yet an admitted
`slp.corpus/v1.1`; they contain no static gene features or model-facing query
tensors. Separate protected molecular-validation and molecular-final snapshots
still need to be prepared and access-controlled, then all three roles must pass
the corpus audit before training. No world-model checkpoint or molecular
metric exists from this work, and the external SL benchmark remains closed.
There is no performance, novelty, frontier, release, or SOTA claim.

**Rights-bearing admission.** Commit `bd011ea` pinned each derived rights
scope to one named tar file, all four input snapshots, both SGD input artifacts,
the primary RunResult and artifact manifest, the reproduction RunResult and
artifact manifest, and the shared content hash. The exact staged files were
rehash-checked before admission. OMF admitted and verified them separately as:

- fitting-only proteome observations:
  `omf://abiome/slp/datasetsnapshot/slp-1-1-proteome-observation-pretrain-v1@sha256:631f66e32a218e167af9edb60115a04514d0bcf675a13bcb244c465ffab2f751`,
  directory manifest
  `sha256:0bc00463f8641fc91d6fcb82266b6f41d4c55cc78275b737eaad257dd2053130`;
- HIS3 basal control:
  `omf://abiome/slp/datasetsnapshot/slp-1-1-proteome-basal-control-v1@sha256:5abaa79409a9e342785ce610083908ffe5054353ddc1c728ab91ddaa704e112a`,
  directory manifest
  `sha256:930b156dc5f97f27b6283439931ae1fef943d5900c2b18eabd81eb31b21bd4dc`.

Both `omf data verify` calls returned valid. The immutable OMF knowledge record
`slp-1-1-proteome-pretraining-boundary-v1` revision
`sha256:8fca68ef568673559920447ee65b337130e14e8dbdec2f6d29dc9822f0808305`
links both runs and both snapshots while explicitly limiting the assertion to
quantitative data-boundary evidence. This satisfies the admission portion of
the fixed rule; the larger corpus-construction step remains open. Independent
OMF graph verification passed for all six run-output artifact graphs and traced
each immutable RunResult or artifact reference through the producing stage to
the exact input snapshots and SGD artifacts. OMF 1.0 does not automatically
create a first-class DatasetSnapshot-to-producing-RunResult lineage edge when
admitting a restored file; the snapshot graph records only the staging-source
digest. The exact producer lineage is retained in the rights declaration and
knowledge record, but this platform limitation remains a release-lineage
blocker rather than being papered over.

## 2026-09-04 — protected proteome molecular-validation observations

**Hypothesis, fixed rule, modalities, and snapshots.** The falsifiable
engineering hypothesis was that the frozen molecular-validation partition of
the same proteome release could be normalized reproducibly without converting,
semantically validating, or emitting any pretraining, molecular-final,
quarantine, analytical-QC, or HIS3-control quantitative column. Advancement
required two successful executions from clean commit `da810b4`, byte-identical
observation and audit files, exactly 537 records over 529 validation genes,
967,019 observed and 26,431 missing values, all frozen identity/order digests,
and one-file rights-bearing admission. Any role crossing, extra file, count,
hash, lineage, or byte mismatch would reject the preparation. Accessible
modalities were the raw MaxLFQ proteome matrix and metadata, the outcome-blind
intervention inventory, typed UniProtKB-to-SGD relations, the held-gene roster,
and the SGD identity artifacts. No model checkpoint, prediction, reward,
synthetic-lethality label, or benchmark datum was accessible to the decision.
The exact input snapshots were:

- raw proteome:
  `omf://abiome/slp/datasetsnapshot/slp-1-1-proteome-raw-v2@sha256:5392d4df7e962c9f59798b83fbdf8e71cd568b30c78a498702d50fabc059397e`;
- intervention inventory:
  `omf://abiome/slp/datasetsnapshot/slp-1-1-proteome-intervention-inventory-v1@sha256:bd688dffdf4d96c01d4147580b1a8705c2149acadbc843a719537817a74505d9`;
- typed protein relations:
  `omf://abiome/slp/datasetsnapshot/slp-1-1-proteome-protein-relations-v1@sha256:acad3427907644f8ab8af38ed36066a6e1148ef92557b727351b0a4fba2b446c`;
- held roster:
  `omf://abiome/slp/datasetsnapshot/slp-1-1-held-roster-v1@sha256:1b9a4800370a5398bf83e0a636007f466bf6ca5a6232e2ebb8fc64c5beb63450`.

`omf doctor` was ready, both protected workload documents passed `omf schema
validate`, the local Linux binding preflight passed, and all four dataset
snapshots reverified before compute. The molecular-final document received
schema validation only; its workload was not executed.

**Executed validation preparation.** Runs
`01a06d60-6e5c-7552-b9ce-01cf6d046c31` and
`01a06d60-e6d5-7326-a0ed-2c2dda05a18b` both succeeded. Their immutable results
are respectively
`omf://abiome/slp/runresult/result-01a06d60-6e5c-7552-b9ce-01cf6d046c31@sha256:3efaf6bf8db78894f478badc4b46b54432b54008556aefb9cbe3844760424dde`
and
`omf://abiome/slp/runresult/result-01a06d60-e6d5-7326-a0ed-2c2dda05a18b@sha256:2eff9f50be60e4dd80be3dcd2915c08cc1d33ec272777d544563f25ad0c1be96`.
Each selected exactly 537 validation records over 529 stable SGD intervention
genes and emitted 967,019 finite observed log2 values; 26,431 assay-missing
values remained absent sparse entries. It reported all 4,939 unselected rows as
not numerically decoded: 3,811 pretraining, 275 molecular-final, 76 quarantine,
389 analytical-QC, and 388 HIS3-control rows. The ordered trajectory-list,
trajectory-set, metadata-order action-sequence, and raw-locator-sequence
digests were respectively
`32ea7b84790202f0b9a87c95e31434c9aba3d8588a68a090964ce9edd2373558`,
`932a2750d3ae3cff3bbbcc165f985bcefbef6f3fa3cf369512fa6d71b217fa76`,
`4d5bb70c6c2fb22f1a938a7134f15601a9251cf771a3cdc92096e5aa8320cdc4`,
and `f226453f029b8fa789acdc1e5ac264136e07584c57bebba8443264988efac2ed`.

Independent file hashing and byte comparison showed identical 8,939,520-byte
archives with SHA-256
`f8263d4813282799625182e8286a0af42311d5e76d58c84071ae9071e8a4bc69`
and identical preparation audits with SHA-256
`26e304410b66e6c7ec0562297854ef0854f7160bbce20f11f51bb0c6babcf85b`.
The primary observation and audit artifact manifests are
`sha256:c2ae4c787504f39f01e78e88e46c3009e32542002adb00b802b6c98e2611d87c`
and
`sha256:0dce17c1fe51f2ce1d265173303ff7b339f3f6a63e02ae74e006bdbdf1563c5d`;
the reproduction manifests are
`sha256:36f6edcdf585319553093e172b05412bfe5b80377bdd6964fd411f5a4ce08d52`
and
`sha256:38dc5d89a2cac5670e4325337edbbc5d454bd27cae196d55167c5d6b8fa52cba`.
Their differing manifest digests bind distinct run contexts; their underlying
files are byte-identical. Upstream graph inspection for all four artifacts
reached the producing stage, exact four dataset revisions, protected module
source, both SGD artifacts, and the SGD normalization lineage.

The protected module package and source-artifact manifest are
`sha256:df9ffa44415c60035de045028fa81a17d4bd3e40d2b4e2e68f05e8854d534894`
and
`sha256:6363bdd341b990e77500da06a112945872fc8a526ab797f9cdb59e4919ba894b`.
The validation workload semantic and WorkloadSpec manifest digests are
`sha256:6fbd1bfdbd1b69f50c3d20c619cad61df2273e7e6d58287fce3a0738a6814070`
and
`sha256:b86c393c0c3e1268adc38a1113739b8f2e23be9cdcb7c1c986657e4b0004408c`.
The binding, realized environment, and dependency-lock digests are
`sha256:5dc181c5643d2a98ec43b1c79791764e4f5ba5ad4edbd5730d7f455ced4d6e2f`,
`sha256:c2060448ac2beedca12bfe46d125cf4eeedeefb0b22ee47c6933c9031e38e1ab`,
and
`sha256:2b8837ecc5287dd25a4100a7ed4e30c60ae8aaf6c1273c4ffd3fa3ebb39372d9`.
The environment used CPython 3.12.3 and NumPy 2.2.6, but still used the package
index without an offline wheelhouse and exposed interpreter-site-package
layering. It is host-reproducible evidence, not a portable release closure.

**Rights-bearing admission and access limitation.** Commit `0eafd14` bound one
named `observation-corpus.tar` to both RunResults, both observation artifact
manifests, the common content hash, the four input snapshots, and both SGD
artifacts. The rights manifest distinguishes legal `trainingAllowed: true`,
which OMF requires for admission, from operational authorization: fitting,
reward, final-holdout use, and current-factory evaluation are false. It also
states that OMF 1.0 does not enforce those purpose restrictions. After an exact
staging rehash, OMF admitted and verified:

`omf://abiome/slp/datasetsnapshot/slp-1-1-proteome-observation-molecular-validation-v1@sha256:6bdefb2ff86c56a5d86fadcf9ffd8c6e3d759183fde6a706d18086a6e2f2341a`,

with directory-manifest digest
`sha256:e25b93ac4ab0ce5f018881f8515f6b1005fd1e026356b18d6ea46b96d44ebe10`.
The immutable knowledge record
`slp-1-1-proteome-molecular-validation-boundary-v1` revision
`sha256:d8408f110fe58cc4e73c8f0a2d4844560b6b688def6689e52d3063215c35914b`
records the same narrow evidence and limitations.

This execution occurred in the full-source custodian factory, not a physically
isolated validation service. The wide CSV parser necessarily materialized each
row's cells as strings before converting only selected validation columns; the
valid claim is therefore that other roles were not numerically converted,
semantically validated, or emitted, not that molecular-final truth was
physically inaccessible. OMF 1.0 actor strings, policies, and the local
executor do not supply that confidentiality boundary. The admitted archive
must never be copied into the clean training factory; a distinct OS/service
identity and store are required before evaluation. The final workload remains
unexecuted until candidate and decision-rule lock under independent control.

The fixed rule passed for protected validation-source preparation and admission
only. The artifact is `slp.source-observation-archive/v1`, not an evaluator-ready
`slp.molecular-evaluation/v2` truth set or composed `slp.corpus/v1.1`. The
governed `tests/` suite passed 170 tests with 3 skipped. No model was trained,
no checkpoint or molecular metric was produced, and the SL benchmark remained
closed. This evidence supports no performance, novelty, frontier, release, or
SOTA claim.

## 2026-09-04 — signed clean-training audit contract v1.3

**Hypothesis, fixed rule, modalities, and intended snapshots.** The falsifiable
engineering hypothesis was that a training factory can prove complete held-
intervention exclusion without receiving either molecular-validation or
molecular-final quantitative truth. The proposed proof uses the exact composed
optimizer corpus, the outcome-blind held roster, every outcome-blind protected-
source inventory, and a recipient-bound custodian signature. Advancement was
fixed before implementation: an ephemeral-key positive fixture had to produce
byte-identical audit files, while any signature, canonicalization, key,
recipient, challenge, protocol, DatasetSnapshot identity, inner-content, held-
roster, inventory, reward, benchmark, or active-held-action drift had to fail
closed. The production path had to remain non-runnable until an independently
controlled key ceremony and a physically separate clean training factory
exist.

The accessible biological modalities were only the already admitted fitting-
only proteome observations and outcome-blind intervention identities. Their
exact intended safe-source snapshots were:

- pretraining observations:
  `omf://abiome/slp/datasetsnapshot/slp-1-1-proteome-observation-pretrain-v1@sha256:631f66e32a218e167af9edb60115a04514d0bcf675a13bcb244c465ffab2f751`,
  directory manifest
  `sha256:0bc00463f8641fc91d6fcb82266b6f41d4c55cc78275b737eaad257dd2053130`;
- held roster:
  `omf://abiome/slp/datasetsnapshot/slp-1-1-held-roster-v1@sha256:1b9a4800370a5398bf83e0a636007f466bf6ca5a6232e2ebb8fc64c5beb63450`,
  directory manifest
  `sha256:f8aac504a2d56fdc9e13cc9b1c9fa87a08ebc7ff2d7036c0b6b135c26d187425`;
- atlas inventory:
  `omf://abiome/slp/datasetsnapshot/slp-1-1-atlas-intervention-inventory-v1@sha256:3d48478089105b77431f9a7459df3d84bfc41aefe2e2906e6f057b1a6399ae41`,
  directory manifest
  `sha256:9fcf5373923c83e93d5a1d6a7dedce6cfd57bd7aadfad4debb7665436c13bd2a`;
- proteome inventory:
  `omf://abiome/slp/datasetsnapshot/slp-1-1-proteome-intervention-inventory-v1@sha256:bd688dffdf4d96c01d4147580b1a8705c2149acadbc843a719537817a74505d9`,
  directory manifest
  `sha256:a1f5222f3dca31d2ca68ca46a271d39cdca3425a903b5dceb7373481450ada36`.

No static sequence/protein/annotation/phylogeny feature snapshot, composed
optimizer `slp.corpus/v1.1` snapshot, signed authorization snapshot, validation
service, or final service exists. Consequently these sources were not composed
or passed to the new module, and no biological workload was run.

**Implemented boundary.** Commits `48dda03` and `261a8a5` added the separate,
self-contained `slp-1-1-training-corpus-audit-v1-3` module while leaving the
historical v1.2 module byte-identical. Its input schema admits exactly one
pretrain corpus, one held roster, one custodian authorization, and two to 64
`protectedInventory*` snapshots. The interface cannot name validation truth,
final truth, reward, raw-source, checkpoint, prediction, or benchmark inputs.
The report schema is `slp.corpus-audit/v1.3`; it contains only the pretrain
identity, outcome-blind roster/inventory identities and protected-set hashes,
and signature identity.

The authorization is canonical JSON plus a detached Ed25519 signature over the
domain-separated, length-framed exact statement bytes. It binds an immutable
UUID, issuer key, recipient namespace, clean-factory signing identity, 256-bit
challenge, training-safe protocol flags, the exact DatasetSnapshot resource and
outer manifest for every safe input, and independently recomputable inner
corpus, roster, coverage, and inventory-manifest digests. The verifier derives
the key ID from the raw public key and requires both a compiled key ID and the
compiled SHA-256 of the canonical public-key text file. Runtime input and
configuration expose no trust-root override. The source-pinned production key
and digest constants are intentionally absent, so every production call fails
before reading the large corpus. Positive tests generate ephemeral keys only;
no private key or signing helper was committed.

After signature verification, the module scans every sparse record action and
the complete trajectory-gene list, recomputes the protected-source intersection
and deterministic held assignments, and rejects any validation/final active
action. Held genes remain permitted as static-only entity rows. The signature
authenticates the exact final corpus bytes but does not independently establish
their source-to-corpus derivation. A biological run therefore also requires a
provenance-complete composition report and verified OMF lineage from the exact
observation, static-feature, roster, and basal snapshots. Stateless Ed25519
verification also cannot prove one-time consumption for the same recipient and
inputs; the coordinator needs a consumed-authorization-ID ledger.

**Verification and limitations.** OMF was ready and exposed signing identity
`sha256:115da768a6712a5ab58a128c9f6809fbb0ce7df2c69981ca35e21c44daf166bc`.
From clean commit `261a8a51339874de7ecb3295feffc0fc8f90fe67`, module validation passed with
package digest
`sha256:3b1c1690814aed30efbb59fa156ab679a12a96da7323d44b23dce14967f2ed58`
and source-artifact manifest
`sha256:23ddd0e5574cc321a47b5f94449d8ece40c94b3a07646a9d85e166ac2cac73c4`.
The intended validation-only fixture worker later wrote exit code zero and the
exact expected v1.3 output, including `auditPassed: 0` and
`custodianSignatureVerified: false`; it did not pretend to verify biological
inputs. However, the `omf module test` command is not accepted as clean
compatibility evidence. OMF 1.0 derives every fixture directory from the
manifest filename stem, so every repository `module.yaml` reuses
`.omf/runs/module-tests/module-0`. It does not clear prior completion/result
files before submission and can return a previous module's result while the
intended worker is still running. This invocation first surfaced a stale held-
roster failure despite the correct v1.3 request subsequently completing. No
repository or `.omf` workaround was added; collision-free upstream module-test
identity remains a platform blocker. The completed worker used dependency-lock digest
`sha256:9eca4b24f57234e5479dc6b3b8c0e46039be34014d2141410a3a6bef60e7b57e`,
environment digest
`sha256:4904ffa109a662919b3ae18bdd5e8d2cac38c2ebf9c0f5e7bea18f883e0b4afd`,
and argv digest
`sha256:7e3a0a25bb475aa43dec53f554af2c2202e165ba2259e7caf367938d01105ae3`.
The lock pins `cryptography==46.0.7`, `cffi==2.1.1`, and `pycparser==3.0` by
exact wheel hashes, but realization still used the package index, no retained
offline wheelhouse, and an inherited interpreter site-package layer. It is not
a release-eligible runtime closure.

The governed `tests/` suite passed 183 tests with 3 skipped. The v1.3-focused
suite contributed 13 passing adversarial tests. The initial compatibility
fixture used empty placeholder inputs and was correctly rejected by the strict
nested contract; `261a8a5` replaced them with syntactically valid immutable
dummy objects before the recorded pass. No DatasetSnapshot, Run, RunResult,
checkpoint, metric, evaluation, or knowledge claim was created from the v1.3
module. This milestone is executable boundary engineering, not training or
scientific performance evidence, and supports no novelty, frontier, release,
or SOTA claim.

## 2026-09-04 — exact SGD sequence source and relation-closed static universe

**Hypothesis, fixed rule, modalities, and snapshots.** The falsifiable data-
engineering hypothesis was that the current outcome-blind yeast identity
sources could yield one deterministic, species-native, relation-closed entity
universe suitable for later static features without consuming held assignments
or quantitative outcomes. Advancement required two clean OMF executions to
produce identical payload bytes while preserving every typed one-to-many
relation; exact source, outer-manifest, inner-manifest, record-set, mapping,
count, identity-set, composite-key-set, and relation-edge drift had to fail
closed. The resulting artifact also had to validate independently rather than
trusting hashes declared by its own manifest.

The accessible modalities were stable SGD and UniProtKB identities, NCBI
taxonomy, typed protein-to-current-ORF relations, and static SGD protein
sequence. No quantitative observation, held roster, reward, checkpoint,
prediction, or benchmark input was accessible to the universe module. Its two
inputs were exactly:

- `omf://abiome/slp/datasetsnapshot/slp-1-1-proteome-intervention-inventory-v1@sha256:bd688dffdf4d96c01d4147580b1a8705c2149acadbc843a719537817a74505d9`,
  outer manifest
  `sha256:a1f5222f3dca31d2ca68ca46a271d39cdca3425a903b5dceb7373481450ada36`;
- `omf://abiome/slp/datasetsnapshot/slp-1-1-proteome-protein-relations-v1@sha256:acad3427907644f8ab8af38ed36066a6e1148ef92557b727351b0a4fba2b446c`,
  outer manifest
  `sha256:c159573f4f7a2e41b18930d724dea9fb297452a659bdf6050e4718efc1a6c58a`.

Separately, the release-labelled SGD object
`orf_trans_all_R64-5-1_20240529.fasta.gz` was pinned by S3 VersionId
`GRiDuJlE44rFsMHE63VZUFxVcA4GBun6`, compressed SHA-256
`17e8b47e1ae23178c6000fbc4ab548f102d1b250ef9dff5d811feb3f03dd2c5b`,
and decompressed SHA-256
`e01f9e1ef7e5a01ff7cd0ee7a843e6d1c1da8c3777fdfac3a5293711d4c56518`.
All 6,722 headers carry Genome Release 64-5-1; all 6,613 current ORFs are
covered and 109 non-current records are explicitly accounted for. Its exact
three-file source snapshot is
`omf://abiome/slp/datasetsnapshot/slp-1-1-sgd-protein-sequences-r64-5-1@sha256:3b76017f5ac74d8d96efb1db52d14af91c9fb15995062110558ce4651cf3ba0c`,
outer manifest
`sha256:8f88480196b5cd8f3c15d65dbdbc09f83305c371fb476c70a38825dad2be4283`.
It is admitted for static feature construction only; it is not a quantitative
training corpus or a sequence representation.

**Implemented and reviewed contract.** Commits `e591dc5` and `c657223` added
the self-contained `slp-1-1-static-entity-universe-v1` module, strict schemas,
workload, and adversarial tests. Independent review found that the initial
archive validator trusted self-declared inner hashes, omitted composite taxon
from its semantic entity hash, allowed noncanonical JSONL, hashed records
before enforcing byte bounds, incompletely checked ancestor symlinks, and
misstated the absence of roles despite emitting action/readout usages. All
were corrected before biological execution. The final validator reconstructs
canonical USTAR bytes; parses canonical entity and relation rows; recomputes
counts, exact relation closure and all semantic hashes; rejects extra fields
and label-like keys; and binds the authoritative `(ncbiTaxon, entityId)` set.

The relation-closed result contains 7,037 entities: 5,187 SGD genes and 1,850
UniProtKB proteins. Of the genes, 4,476 are action eligible; 1,855 are relation
targets, 1,144 of those overlap the action set, and 711 are relation-support
only. All 1,855 typed edges and all five two-target protein records are
preserved without selecting a first target. The current model-facing action
plus readout interface remains 6,326 keys. The composite entity-key digest is
`82b8e2885939577fe6946e3b974a10cb947834118f2070e1bcbe4c2f2e6a5fd9`;
the ID-only compatibility digest is
`e7231d3bb859ca4818364c76d9aa9fee54d6b1d9a64050c2d3ab8af81a9b3eb9`;
the relation-edge digest is
`8a75c42d5a0f24a86be16ecea2616d6d13d25d90de18a80d3dd22cd188afc6d1`.

**OMF execution and reproducibility.** OMF doctor was ready with signing
identity
`sha256:115da768a6712a5ab58a128c9f6809fbb0ce7df2c69981ca35e21c44daf166bc`.
The module and workload schemas validated. An initial attempt using full
`omf://` DatasetSnapshot URIs failed at input-contract admission in run
`01a06db9-a4dc-7b84-8261-4e99f45097d8`: OMF 1.0 recognizes DatasetSnapshot
workload inputs only as `dataset/<name>`. The workload now uses that supported
grammar; OMF pins the immutable revision at admission and the module separately
requires the compiled full resource URI and outer digest. A later run
`01a06dbc-f81f-7cd6-ac7f-e92256023415` failed at `import omf` because the host
Conda interpreter, not the dedicated OMF environment, was selected. No module
source or biological input was read in the first failure and no source was
read past module import in the second; neither emitted an artifact. Activating
the same dedicated OMF runtime used by prior factory runs resolved the executor
selection without changing data or numerical behavior.

Clean runs `01a06dbd-82ad-7755-951c-7ac08a13f5e8` and
`01a06dbd-b980-7909-a2e4-fa336a598ecc` succeeded from commit `c657223`. Their
immutable RunResults are respectively
`omf://abiome/slp/runresult/result-01a06dbd-82ad-7755-951c-7ac08a13f5e8@sha256:b6afd306c6db582d7f1ee64ac76952d995bbcee1541c51933b40677532bc83eb`
and
`omf://abiome/slp/runresult/result-01a06dbd-b980-7909-a2e4-fa336a598ecc@sha256:a8230ea2171b696734dcaf4388485032758c356cb1e1740ee1b2e62c7d70a22d`.
Both emitted the same 1,525,760-byte archive with SHA-256
`d947bf618b854dd33a7157ac0f0380c544e9a4377bddb00806c9ca07f689a544`
and the same 4,880-byte audit with SHA-256
`339412ea008cf383db2258d0788d71c2cf357183b331d49f4168aa7f113f1a0f`.
Their OMF artifact manifests differ because they bind distinct run contexts:
archive manifests `sha256:5df1dc1b535f3ec5c58d9a0fb94a3c0144a4ca23ec575e730eee2c83f93df2d8`
and `sha256:0c8e3ea3a13415b9593151195c270ff601c2d1336d4084474a87b5ad7c67b644`,
and audit manifests `sha256:d6967eec1b6669515fe7713fb7e3dca75bac3df42f6f44ffce723b7cd90a4f49`
and `sha256:92e855df2626a3bf20275c5a09bb3efd656d3b0d26c57705e22a1db312e67560`.
The workload, workload-resource, binding, environment, and empty dependency-
lock digests were respectively
`sha256:f89909b2563c30008e11092ff14b9c039e114840448f12df21ac33dd78d3b33c`,
`sha256:c4610c2fb0b9e139e782234736c531b65ab59ca00a9b2ffb9342bfcce6a0b127`,
`sha256:5dc181c5643d2a98ec43b1c79791764e4f5ba5ad4edbd5730d7f455ced4d6e2f`,
`sha256:2857f0de3e6a95a520da1d0b3ba81b0974bac0a1f8c88a6d048320edff836169`,
and `sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
The runtime was CPython 3.12.3; the empty lock is host-reproducible evidence,
not an offline portable runtime closure.

**Admission, evidence, and next boundary.** Commit `3a52b83` bound the exact
two output files, both RunResults, all four run-specific artifact manifests,
and both parent DatasetSnapshots to the CC-BY-4.0 derived rights declaration.
OMF admitted and verified
`omf://abiome/slp/datasetsnapshot/slp-1-1-static-entity-universe-v1@sha256:de3efddf5a9e4f66496a1edda14b04de774e972bc7b9efd30964644de2a56cac`,
directory manifest
`sha256:a65f94081c0b60a8b486ed968b58fc4d021ba3ea7f5f11425d3a1635cbb10684`.
Knowledge revision
`sha256:ec2b7bda50256739d4ba7c6dda1b63fe0291dacb361128a1579e35f69b13803c`
records the same narrow identity-boundary claim.

The next fixed step is an outcome-blind 21-dimensional sequence-statistics
feature block—protein length divided by 4096 plus amino-acid fractions—joined
by composite key across this universe and the admitted SGD sequence source.
It is intentionally a weak deterministic baseline for later protein-language-
model, domain, and phylogeny blocks, not the proposed frontier representation.
The historical corpus v1.1 and sparse consumer use globally unique bare IDs;
they remain frozen and cannot honestly consume multi-species identities. A
new corpus v1.2, audit v1.4, signed handoff, and world consumer must enforce
composite joins before biological training. The external synthetic-lethality
benchmark remained closed. No model was trained, no checkpoint or molecular
metric was produced, and this evidence supports no transfer, performance,
novelty, frontier, release, or SOTA claim.

## 2026-09-04 — deterministic sequence-statistics feature baseline

**Hypothesis, fixed rule, modalities, and snapshots.** The falsifiable static-
feature hypothesis was that the exact admitted SGD R64.5.1 protein sequences
and relation-closed yeast entity universe could produce a deterministic,
complete numerical baseline without consuming quantitative outcomes, held
assignments, or benchmark data. Advancement required two clean executions to
emit byte-identical payloads with exactly 7,037 composite-keyed rows, 21 present
values per row, zero missing or ambiguous entities, explicit accounting for
all 109 non-current sequence records, and exact peptide consensus for all five
proteins with two current-ORF relations. Any source, relation, identifier,
feature-definition, member, row-order, dtype, shape, mask, provenance, or
excluded-record drift had to fail closed.

The feature definition was fixed before execution as peptide length divided by
4096 followed by amino-acid fractions in `ACDEFGHIKLMNPQRSTVWY` order. Values
are IEEE-754 little-endian float32 in C row-major order. There is no fitting,
clipping, learned or numerical identifier feature, post hoc normalization, or
trainable parameter. The only biological modality is static *Saccharomyces
cerevisiae* sequence for NCBI taxonomy 4932 and S288C strain taxonomy 559292.
The exact DatasetSnapshot inputs were:

- sequence source
  `omf://abiome/slp/datasetsnapshot/slp-1-1-sgd-protein-sequences-r64-5-1@sha256:3b76017f5ac74d8d96efb1db52d14af91c9fb15995062110558ce4651cf3ba0c`,
  outer manifest
  `sha256:8f88480196b5cd8f3c15d65dbdbc09f83305c371fb476c70a38825dad2be4283`,
  tree
  `sha256:823a18ed8039ee44ee44b860551fea749b9012c941e6b9cd5163938da19b168a`;
- static entity universe
  `omf://abiome/slp/datasetsnapshot/slp-1-1-static-entity-universe-v1@sha256:de3efddf5a9e4f66496a1edda14b04de774e972bc7b9efd30964644de2a56cac`,
  outer manifest
  `sha256:a65f94081c0b60a8b486ed968b58fc4d021ba3ea7f5f11425d3a1635cbb10684`,
  tree
  `sha256:7ec354f427cfd8a2fcc3de1004c7e4ac77402a78b5cc0a0b5ef89ba24656fd3f`.

The two additional immutable inputs were the 6,613-row current-ORF artifact
`artifact:sha256:e67f0e8773feae108ecdb687139885e01ca972ff4aec95cd1358b33db1ea1192`
(2,135,394-byte payload,
`df7b717cad88dc3672f72f8148f6a9132d12abe6ba020b220b091a8da8f7004d`)
and mapping-manifest artifact
`artifact:sha256:c74ea81ce604357b998e5f09130dff85bf8a7a26504b9b2426f8038608c52d9c`
(3,818-byte payload,
`570557ab1201913a18de9790f8adc5ee2e3cb56c6bb0e8d588fe43660c0214e1`).
The mapping identity remained
`slp-sgd-map:2026-08-28-object-set-v1`, digest
`6fd789df6099b78a8842baa8f1d20ab0a3fe77f27ce512ee783444eb2627ef2a`.
No quantitative outcome, held roster, partition assignment, reward, model,
checkpoint, prediction, or benchmark input was available to the module.

**Implemented contract and review.** Commit
`6887f28b25ae47d6a46e8faf3e7c82cc4f584c65` added the self-contained
`slp-1-1-sequence-statistics-feature-block-v1` module, strict nested artifact
schemas, workload, entry-point tests, and adversarial contract tests. It parses
the structural SGD header fields rather than using description prose as a
feature or class signal. Every gene row uses its own current-ORF peptide. Every
protein row uses its typed current-ORF relation; the five one-to-many relations
are accepted only when all related peptides are identical, never by choosing a
first target. The validator independently reconstructs the canonical USTAR
archive, semantic provenance, excluded-record set, `.npy` headers and payloads,
row ordering, exact `(4932, entityId)` key set, all-present mask, feature values,
and declared counts. The resulting block contains 5,187 gene rows and 1,850
protein rows, with 147,777 present values, zero missing entities, zero ambiguous
entities, 1,426 current ORFs outside the relation-closed universe, and 109
excluded non-current sequences. Its entity-key digest is
`82b8e2885939577fe6946e3b974a10cb947834118f2070e1bcbe4c2f2e6a5fd9`.

**OMF execution and reproducibility.** OMF module validation produced package
digest
`sha256:b2793ed5250222bc095365bc7e642a1a84d31f45abc590ebb122bedda6d59bc9`
and source-artifact manifest
`sha256:d04e02041286f9085eb7e2916910e9ab1497363fa8242982df0e7e3ee2e29a89`.
The admitted workload and workload-resource manifest digests were respectively
`sha256:a248766c727f6a380f9208eb4923bc6545f583ffb3495cd1f5ac66ffebdcb0e9`
and
`sha256:9cedcb7754a924b96083475d01ae4f6e8395c6e80d57bd75bd58d82d4225fa5e`.
An initial submission as actor `local-user` was denied by the existing factory
policy and created no run. The policy-compliant executions
`01a06df0-2427-7737-9321-1615583dedd8` and
`01a06df0-6255-7e42-bf6d-e87425e8a19c` then succeeded. Their immutable
RunResults are respectively
`omf://abiome/slp/runresult/result-01a06df0-2427-7737-9321-1615583dedd8@sha256:72a9b2509069c05ed8aae82734fc31402f02229a64d8b39cd2f7afd06496a53b`
and
`omf://abiome/slp/runresult/result-01a06df0-6255-7e42-bf6d-e87425e8a19c@sha256:bf66d15da02370b2bfb0b1989cfd473876b2c55f482a7c9ed272c96886084eb2`.

Both runs emitted an identical 4,392,960-byte archive with SHA-256
`1b0aaec738b10ad3baa082d907d0c962c35c9b159b89fffca893fa1ecf5a7bed`
and identical 7,851-byte audit with SHA-256
`5d3a9fba29e9c31979fbda5a07951f244b66b35cf6c45de53c27fd231586a5e7`.
The archive members were exactly:

- `entities.jsonl`: 727,401 bytes,
  `e487f428c6eb1eb58de0d3e8ca74f016841713c3014cc55370048eb3e8304572`;
- `excluded-non-current.jsonl`: 42,781 bytes,
  `aeb6be983ff828c517aaed9def31d9401a111b312e344e8349678eae75e7972f`;
- `manifest.json`: 7,139 bytes,
  `5cfd73bb2c55ca0bfc381b6d9e883fcc7d4ab793ee9dfc94ef5f3f02cab5ff65`;
- `present.npy`: 147,905 bytes,
  `c6a282c63fc45d94a4dd932c4b064d345b02939e4e40c84dd4b70d871dc96716`;
- `sequence-provenance.jsonl`: 2,869,698 bytes,
  `5955ae6f8503b87370bf5116fdae8699ced9c4e3a0a378fd3843baaa7c2965fe`;
- `values.npy`: 591,236 bytes,
  `3f1bd1c02d56fb6b9ab100d95fb567a931fed39f6d8b6352a6389a8cae301f05`.

The semantic feature-definition digest was
`2e2152471d3cb487775159d91ff5073c8b19997b62a5a09f94be637efeb75620`.
Run-specific OMF artifact manifests correctly differ because they bind
different lineage: archive manifests
`sha256:8d7eba8e435ad4ff5a020bec68d452969b923b0722cd13227ed058f05236d878`
and
`sha256:d6f76d8c09cd68c9e461a249e64f25db14a9d97d9f52588b2f20a4b9bca0e30c`,
and audit manifests
`sha256:559129602ac30a4903a0823665941e583e8dbd79bc61402d3318a565c607ac52`
and
`sha256:1001f228cffa5da235ca079af845722b5f450aef2fe32d898b518867ecd54809`.
Both used CPython 3.12.3, environment digest
`sha256:2857f0de3e6a95a520da1d0b3ba81b0974bac0a1f8c88a6d048320edff836169`,
and the empty dependency-lock digest
`sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
The final governed repository suite passed 226 tests with 9 skipped, and all
six strict schema tests passed in the pinned Linux OMF environment.

**Admission, evidence boundary, and next step.** Commit `69dd844` bound both
clean RunResults, their four lineage-specific output manifests, the exact
payload digests, and both parent snapshots to the derived CC-BY-4.0 rights
declaration. OMF admitted and verified
`omf://abiome/slp/datasetsnapshot/slp-1-1-sequence-statistics-feature-block-v1@sha256:e9733974c551bca3af93c4cb488972f5167da5e7e3cf48ef5803348cd20d91e5`,
directory manifest
`sha256:6b4b32c794d7787b9b9076d78726ea0ad7706d64fd82b5f918f0c6da20da0d2a`,
and tree digest
`sha256:3f4549114a181c162596d60ef1b94d222ec494282d23ece8da7e19142135cb8d`.
Immutable knowledge revision
`sha256:4e106b60fb660e97cf291c03d439041e91c75379e49fb62750160490dccee53d`
records the same narrow static-feature claim.

This evidence establishes deterministic construction and static identity and
sequence provenance only. The 21 hand-designed statistics are an intentionally
weak baseline, not a learned protein representation, and static coverage of
held genes does not authorize any held quantitative outcome for fitting or
reward. The block has only yeast sequence: it has no human feature coverage,
domain structure, protein-language-model representation, annotation, phylogeny,
context, or experimental measurement. It has not yet been composed into corpus
v1.2 or consumed by a world-model workload. No model was trained, no checkpoint
or molecular metric was produced, the external synthetic-lethality benchmark
remained closed, and this milestone supports no transfer, performance, novelty,
frontier, release, or SOTA claim. The next boundary is a new composite-keyed
corpus contract and consumer; the historical bare-ID implementations remain
frozen.

## 2026-09-04 — composite-keyed proteome corpus v1.2

**Hypothesis, fixed rule, modalities, and snapshots.** The falsifiable
composition hypothesis was that the exact admitted pretraining observations,
static sequence-statistics block, and outcome-blind held-intervention roster
could be joined into a model-facing corpus using `(ncbiTaxon, entityId)` at
every identity boundary without changing a quantitative target byte or static
feature byte and without admitting any protected intervention. Advancement was
fixed before execution: two clean governed runs had to emit byte-identical
canonical archives and audits; protected-intervention overlap had to equal
zero; benchmark and reward flags had to remain false; all 3,811 records,
6,865,493 targets, 7,037 feature rows, and 3,679 trajectory interventions had
to survive exact independent validation. Any mismatch stopped advancement.

Accessible modalities were quantitative *Saccharomyces cerevisiae* proteome
responses and the deterministic 21-dimensional protein sequence-statistics
baseline. The module had no molecular-validation or molecular-final outcomes,
reward records, synthetic-lethality labels, checkpoint, prediction, or external
benchmark input. Its exact DatasetSnapshot inputs were:

- fitting observations
  `omf://abiome/slp/datasetsnapshot/slp-1-1-proteome-observation-pretrain-v1@sha256:631f66e32a218e167af9edb60115a04514d0bcf675a13bcb244c465ffab2f751`,
  outer manifest
  `sha256:0bc00463f8641fc91d6fcb82266b6f41d4c55cc78275b737eaad257dd2053130`,
  tree
  `sha256:fc1f812308af999c601bee9b53ce21035bdd6fd9952cead11451c72b612a833f`;
- static features
  `omf://abiome/slp/datasetsnapshot/slp-1-1-sequence-statistics-feature-block-v1@sha256:e9733974c551bca3af93c4cb488972f5167da5e7e3cf48ef5803348cd20d91e5`,
  outer manifest
  `sha256:6b4b32c794d7787b9b9076d78726ea0ad7706d64fd82b5f918f0c6da20da0d2a`,
  tree
  `sha256:3f4549114a181c162596d60ef1b94d222ec494282d23ece8da7e19142135cb8d`;
- outcome-blind held roster
  `omf://abiome/slp/datasetsnapshot/slp-1-1-held-roster-v1@sha256:1b9a4800370a5398bf83e0a636007f466bf6ca5a6232e2ebb8fc64c5beb63450`,
  outer manifest
  `sha256:f8aac504a2d56fdc9e13cc9b1c9fa87a08ebc7ff2d7036c0b6b135c26d187425`,
  tree
  `sha256:ba62f5855f46e693f2a27f4ed06efeec046ccd99c9993c145f9983807dfed0b1`.

**Implemented contracts and adversarial review.** Commit `2c2f25b` added the
self-contained `slp-1-1-proteome-corpus-compose-v1` module, strict
`slp.corpus/v1.2` schema, workload template, and focused tests. It requires
exactly those three snapshots; constructs canonical action identifiers from
composite keys; rejects opaque query identities; preserves target and feature
bytes; and emits deterministic uncompressed NPZ members inside canonical
USTAR. Commit `84d5552` added the independent
`slp-1-1-training-corpus-audit-v1-4` module and hardened member paths to
POSIX-only relative names without backslashes, whitespace, empty segments,
drive prefixes, or `.`/`..` traversal. The audit verifies canonical entity,
query, panel, feature, trajectory and shard structure; shapes and dtypes;
species and covariate fields; target-panel membership; compound actions;
exact lineage and feature-pack hashes; and a recipient-bound signed custodian
authorization. Historical numerical modules were not modified.

The final composition module package and source-artifact manifest are
`sha256:6e46a366f9ce9768f63eff52d975052769708d63cacaa941fd4bc9eb38b88798`
and
`sha256:a1e43a368c10323440fcc768c1fa18a9257a9bec1714168505011c2de147834d`.
The v1.4 audit package and source-artifact manifest are
`sha256:870e4cd9c26cd76dec36d8717eb5e856d3296d259ef8de8b171ffd6eed45e849`
and
`sha256:405d3bc5a8949ff6a27b2014062fbbf11275239be0d9546fc1d045e8c90dce2e`;
its 303-byte dependency lock has SHA-256
`9eca4b24f57234e5479dc6b3b8c0e46039be34014d2141410a3a6bef60e7b57e`.
Module and schema validation passed. The production trust key is intentionally
unprovisioned, so the audit fails closed outside its ephemeral-key test fixture
and no signed v1.4 workload or training handoff exists.

The focused producer and workload suite passed 10 tests; v1.4 audit tests
passed 25 with one skipped; combined v1.3/v1.4 audit tests passed 38 with one
skipped; and Linux schema-plus-audit validation passed 36 tests. After the
derived-rights test was added, the full governed `tests/` suite passed 263
tests with 20 skipped and 17 existing PyTorch nested-tensor warnings. Ruff and
`git diff --check` were clean apart from Git's configured LF-to-CRLF notices.

**OMF execution and byte-level result.** The composition WorkloadSpec revision
is
`sha256:80b21b446363454f60a07499126cd16387d34fac50c6ee9085651bdf655f5bc4`,
with spec digest
`sha256:ad7396e20487b96f2bd50aaf276eb4941b032671b0b2dfe0bdd5d179388876df`.
The admitted run-status workload and binding digests are
`sha256:141f1c0aee727feee82c540fd3c2b4af03d9dc53aaba46de8421f60916d6d9de`
and
`sha256:5dc181c5643d2a98ec43b1c79791764e4f5ba5ad4edbd5730d7f455ced4d6e2f`.
Preflight was ready with network denied.

Runs `01a06e27-c6b8-7eee-ba08-54c27d8ada57` and
`01a06e28-2f3f-7ccb-b71d-8b7654fc26ca` both succeeded. Their immutable results
are respectively
`omf://abiome/slp/runresult/result-01a06e27-c6b8-7eee-ba08-54c27d8ada57@sha256:8d4362b2d77d855abea6357b6e34abc0b2052fdc71675355548ded16390d9281`
and
`omf://abiome/slp/runresult/result-01a06e28-2f3f-7ccb-b71d-8b7654fc26ca@sha256:58116fb6b3a075ff188d47141d326cd55d895d43cf2f18666b35ae371501348d`.
Both emitted the same 89,149,440-byte `corpus-v1-2.tar`, SHA-256
`0a5322c46e15e8a15d17000e8993c0ad642fcc70bc8fff00cbba8fb2905708bf`,
and the same 5,277-byte `corpus-compose-audit.json`, SHA-256
`898e4069b2bd9575bd7380b57ed6214bf3d75043feb401c0fa50371972623c52`.
Independent review confirmed that each output directory contains only those
two regular files and that the archive has exactly 13 canonical
`composite-corpus/` members.

The two runs' corpus artifact manifests are
`sha256:9c208900fc871b2a60ceddb1a5d72ea5670a327feea4b2bdf39f92132376a0c6`
and
`sha256:fb4fa62ccd06b921947bccc01f4ebb4155d4239df74a644e5105962c5cf7198f`;
their audit manifests are
`sha256:ae9e3fa7ee9207dae17fef63b906f218904fdf5a18d534534801dd96a7bac286`
and
`sha256:bf3f020466c284f2814f6cee7ed2569417ed9b7ba49258ef78e03936114c0d2e`.
Manifest identity differs because OMF binds each run context; payload identity
is exact.

The archive contains 7,038 entities, comprising the 7,037 static biological
entities and one taxon-4932 experimental context; 7,037 feature rows; 1,850
queries; one panel; 3,679 trajectory interventions; 3,811 records; 6,865,493
finite target values; and eight shards. Its corpus-manifest, entity-key-set,
feature-entity-key-set, and feature-pack digests are respectively
`d91cbbc0b98ea05ccbf56201f50143f57c2b71ffe53211a5a5128ce706c60ad7`,
`9ca16d4f44ca97b4940bd389ca8bbdafe0c6fd711d557a98743218a83caeb87d`,
`4a7e15d5aca02862a80acbd182f5a52c86c35e4dbadf8d95297c2ba47a95dce5`,
and
`016753a94bacd6e2b8dd299abc7906fa874c3d5926ff73605a1f9c913a12d66b`.
Feature-present bytes, feature-value bytes, and target-value bytes were
preserved exactly with SHA-256
`08de1975edffb1a14cbea7d27d7fde8abedf8e2cc1899f70838d68a9b5b287af`,
`3f51a98266c855800917ca6c7b87e205d9df7268260c44b4ba0341605840b7a0`,
and
`7eda2fee14728865518c1133e4a7c122a2180ac29b9c5977946518a0c3d46af6`.
Protected-intervention overlap was zero; reward and benchmark presence were
both false. The v1.4 parser independently loaded the production archive and
reproduced its counts and entity/feature-pack digests. That parser check is
format compatibility, not the missing signed authorization.

The realized environment digest was
`sha256:c2060448ac2beedca12bfe46d125cf4eeedeefb0b22ee47c6933c9031e38e1ab`.
It used `/usr/bin/python3.12`; direct import selected NumPy 2.2.6, while the
environment inventory exposed both NumPy 2.2.6 and 2.3.3 distribution metadata.
This layering anomaly does not alter the reproduced payloads but prevents a
portable runtime-closure claim and must be removed before release evidence.

**Rights-bearing admission and evidence boundary.** Commit `469d0ce` pinned
the three parent snapshots, both RunResults, all four run-specific artifact
manifests, both payloads, exact corpus counts and semantic digests, and the
fitting-only CC-BY-4.0 purpose boundary. OMF first admitted revision
`sha256:432f51f3dfb5034d39fb8ccff3ef130320595ce76ffd876d8621820b4ff0c31e`
with its default generic sample schema. Rather than rewrite it, the factory
created and verified the current schema-corrected revision:

`omf://abiome/slp/datasetsnapshot/slp-1-1-proteome-composite-corpus-v1@sha256:e91cad825b8a2e972da293902c630331a92ab664c5d14a95a65ff38090db6c48`,

typed `slp.corpus-bundle/v1.2`, with outer directory manifest
`sha256:9029d5a97a9945fa88000366252c103d3955b4ea0670b7d27239123949f2718e`,
tree digest
`sha256:4d47930f2d508c1c95617a70a1544c3dbe248302c7f38bfe1c2347c7f5b623c5`,
and spec digest
`sha256:8b49d6db112837878bfd75e60ea49ec1a3390a82a5ba570ff1baa852db9cfc6a`.
`omf data verify` returned valid. Knowledge revision
`sha256:dd334e610923a7c6c1f50bb446e5f8e20d98318ed56ad995d220e7c0715fa3cb`
supersedes the initial generic-schema observation
`sha256:3e157594b95d48e15d3513c064dcfad7580beb40ea76a71b7818aa91fb64befb`
and records the narrow reproduced-corpus claim.

OMF lineage traversal from the RunResult did not expose the three upstream
snapshot edges, and `data add` records the derived snapshot's content source
rather than a first-class producing-RunResult edge. Exact parent and run
lineage is therefore bound inside the corpus manifest, composition audit,
rights declaration, and knowledge assertion, but native graph traversal is
incomplete and remains a release-lineage blocker. `omf data verify` establishes
stored-byte integrity, not semantic validity or signed clean-training
authorization. OMF 1.0 purpose restrictions also remain documentary rather
than independently enforced.

The fixed rule passed for deterministic composite corpus construction and
rights-bearing admission. It did not pass the next training gate: no
independent custodian trust key or authorization snapshot exists, no v1.4
signed audit workload has run, and the historical world consumers cannot
consume this contract safely. No model was trained, no checkpoint or molecular
metric was produced, and the external synthetic-lethality benchmark remained
closed. This evidence supports no transfer, performance, novelty, frontier,
release, or SOTA claim. The next admissible implementation step is a new
application-neutral composite-keyed world consumer, followed only after an
independent signed audit handoff.

## 2026-09-04 — packaged clean-training audit v1.5 correction

**Discovered integration failure and fixed rule.** Independent consumer review
found that frozen audit v1.4 could parse a direct corpus or a snapshot whose
sole file was `corpus-v1-2.tar`, while the rights-correct admitted pretraining
DatasetSnapshot contains exactly two files: `corpus-v1-2.tar` and
`corpus-compose-audit.json`. The earlier v1.4 compatibility check had passed a
tar payload directly and therefore did not test the actual OMF handoff shape.
A production v1.4 workload would fail before corpus parsing. Re-admitting a
tar-only snapshot would discard the companion evidence and was rejected as a
workaround; v1.4 remains immutable.

The falsifiable correction hypothesis was that a new audit could consume the
exact admitted two-file bundle while treating the producer's companion as an
assertion to reconstruct, not as proof. Advancement required: exact bundle
loading; independent agreement on archive, manifest, inputs, counts, composite
identity, feature and target bytes; actual canonical USTAR verification; signed
binding of both outer payload hashes; strict boolean and integer types; no
change to v1.4; and green schema, OMF, focused, combined, and full-suite checks.
The only quantitative modality opened by this compatibility check was the
already admitted fitting-only yeast proteome corpus
`omf://abiome/slp/datasetsnapshot/slp-1-1-proteome-composite-corpus-v1@sha256:e91cad825b8a2e972da293902c630331a92ab664c5d14a95a65ff38090db6c48`.
No validation/final truth, reward, benchmark, or production authorization was
accessible.

**Immutable v1.5 implementation.** Commit `ce80904` added
`slp-1-1-training-corpus-audit-v1-5` without modifying v1.4. The new boundary
requires exactly the two admitted top-level regular files and rejects direct,
tar-only, missing, renamed, extra, nested, or symlinked layouts. It parses the
companion with duplicate-key rejection, bounded UTF-8, exact fields, and the
producer's sorted pretty-JSON representation. It then independently binds the
actual tar name, byte count and SHA-256; inner corpus manifest and input
lineage; counts; composite entity and feature key sets; context identity;
feature-pack digest; recomputed feature-value, feature-presence and target-value
bytes; zero protected overlap; and false reward/benchmark declarations.

The audit also verifies that the archive itself—not merely its `formats`
string—has exact sorted regular-file USTAR members, canonical mode/owner/time
headers, contiguous offsets, zero member padding, canonical 10,240-byte record
rounding, and an all-zero trailer. Companion counts reject Boolean-as-integer
confusion, and leakage flags require exact Boolean identity plus an exact
integer zero overlap. The v1.5 corpus identity adds
`bundleArchiveSha256`, `compositionAuditSha256`,
`compositionAuditSchema`, `compositionCompanionValidated: true`, and
`sourcePreservationIndependentlyRecomputed: false`. The signed pretrain claim
now binds both bundle hashes in addition to the inner manifest and content
digests. Source-side preservation strings must match reconstructed composed
bytes, but the audit correctly refuses to claim it independently recomputed
parents that are absent from the clean-training boundary.

Final OMF validation returned valid with package digest
`sha256:df9a16a207e5ad64ff0d5123a33a921eda8e75d13aa272ac6b4be4bebf6dcb0b`
and source-artifact manifest
`sha256:8b574838f6700cea1a271165a930824d1f43ef15c2c0f528d534a5a9a87452ed`.
The 303-byte dependency lock remains
`sha256:9eca4b24f57234e5479dc6b3b8c0e46039be34014d2141410a3a6bef60e7b57e`.
Eight focused tests passed with one optional Windows schema skip; combined
v1.4/v1.5 tests passed 33 with two skips; all nine v1.5 tests passed in the
pinned Linux environment; Ruff and compile checks passed. The full governed
repository suite passed 271 tests with 21 skipped and 17 existing PyTorch
nested-tensor warnings.

An independent read of the exact production bundle through v1.5 reproduced
3,811 records, 6,865,493 target values, entity-key-set SHA-256
`9ca16d4f44ca97b4940bd389ca8bbdafe0c6fd711d557a98743218a83caeb87d`,
feature-pack SHA-256
`016753a94bacd6e2b8dd299abc7906fa874c3d5926ff73605a1f9c913a12d66b`,
archive SHA-256
`0a5322c46e15e8a15d17000e8993c0ad642fcc70bc8fff00cbba8fb2905708bf`,
and companion SHA-256
`898e4069b2bd9575bd7380b57ed6214bf3d75043feb401c0fa50371972623c52`.
This passes the packaging and semantic compatibility rule only. Immutable OMF
lesson revision
`sha256:e5446a64bb97f60fe3d32a1270c00f3b5a3486bd487d7284f1b569982adf6e09`
records why v1.5, rather than frozen v1.4, is required for the packaged handoff
while making no training or performance claim.

The production public trust anchor remains intentionally unprovisioned, so no
signed v1.5 audit workload or authorization artifact exists and no training
consumer may treat this parser result as permission. A future consumer must
independently reverify the original signature and full corpus/roster/inventory
identities before importing Torch or allocating a model. Native OMF producer
lineage, one-time authorization consumption, freshness, rights enforcement,
and physical clean-factory isolation also remain separate gates. No model was
trained, no checkpoint or molecular metric was produced, and the external
synthetic-lethality benchmark remained closed. This correction supports no
transfer, performance, novelty, frontier, release, or SOTA claim.

## 2026-09-04 — principal-scientist audit and selective scientific reset

**Question, rule and access.** At clean Git commit `399ae15`, the audit tested
whether the current admitted data, numerical consumer and evaluation contracts
could support the stated unseen-intervention and unseen-context program.
This was a code-and-contract audit, not a preregistered biological experiment.
The audit criterion was conjunctive: the data must identify the claimed task;
the consumer must preserve the admitted identity and access boundary through
inference; and the evaluator must compare the candidate against the complete
frozen molecular baseline rule. A missing component rejects readiness, without
opening protected truth or changing thresholds. This rule failed: the project
is ready for further engineering, not biological candidate selection or scale.

Accessible evidence was tracked code, source/rights declarations, prior ledger
entries, bounded OMF metadata, and synthetic numerical probes. No biological
outcome arrays were opened by this audit. In particular, no molecular-validation,
final, reward, or SL benchmark truth was loaded. The proposed feasibility study
would use molecular proteomics and static sequence statistics only, with its
exact existing resource revisions recorded in
`evaluations/slp-1-1-scientific-reset-v1.yaml`. Those pins do not grant new
training or evaluation access.

**Factory state.** The existing Linux interpreter metadata referenced
`/tmp/slp11-omf-venv`, which no longer existed. The CLI was absent from PATH.
The exact README OMF source revision
`ef26eea2cb694596f7680a4bce400371738cbb4b` was checked out and installed in a
disposable CPython 3.12 environment. Temporary installation files did not
survive a subsequent invocation; restoration and diagnostics were therefore
performed within one Linux session. No retained offline runtime closure follows
from this repair, and its newly resolved installation dependencies are not
scientific execution provenance.

All three required diagnostics then completed: `omf doctor` returned
`ready: true`, eight passing checks and zero failures at
2026-09-04T22:21:15Z; `omf agent context` returned a bounded activity view
(20 of 21 operations, with the truncation explicit); and
`omf agent capabilities` returned catalog digest
`sha256:c5ac5c4ab25071858a53b32fa562bf95e7aa2e0cf10a3ec3f60712e3fd23ea97`.
The latest two listed runs were the successful composite-corpus runs already
recorded above. The context reported zero deployments. Factory health is not
evidence of baseline performance, clean training isolation or release eligibility.
No OMF training, admission, promotion, token creation or independent signing
ceremony was performed; no runtime-state files were manually edited.

**Model shape and consumer findings.**

1. The sparse model is a conditional scalar-readout predictor with deterministic
   shared context/action memory and separate Gaussian/NB heads. It has no learned
   entity-ID embedding, which should be retained. Independent scalar marginals
   do not learn residual cross-readout dependence or establish a calibrated joint
   cell population. There is no implemented temporal transition objective. A set
   encoder can express non-additivity, but single-intervention training alone
   cannot identify general double-intervention effects. Keep these distinctions
   explicit rather than changing the name of the output into evidence.
2. `workloads/slp-1-1-world-sparse.yaml.tmpl` fixes `dModel: 16`, one encoder and
   one decoder layer, 12 epochs and 32 record draws per epoch, with learning rate
   0.01. `train_sparse_world` explicitly puts the model on CPU and uses one CPU
   thread. Its 384 scheduled draws are not 12 complete passes through 3,811
   biological records. The 256-wide `WorldConfig` default is not the effective
   training template. With the composite corpus ontology dimensions, the smoke
   shape has 5,476 parameters. This is engineering scaffolding, not an RTX 4070
   training implementation or a justified architecture/budget selection.
3. The admitted corpus is composite-keyed v1.2 and uses the packaged v1.5 audit,
   while the frozen sparse consumer remains on the earlier corpus/audit boundary.
   `TargetFreeQueryIndex.materialize` joins entities by bare ID, masks every
   context token, and marks continuous covariates missing. The composer supplies
   an active context token during fitting. Even one-context prediction therefore
   needs matching semantics, before broader context transfer can be considered.
   The materializer also requires an unambiguous readout for **both** supported
   likelihood families even when a query is Gaussian-only. A new consumer must
   resolve requested families only; adding a dummy NB type is not a remedy.
4. Technical sample locators are correctly marked `access: audit` in the
   composer. The old consumer's `_select_covariates(..., "world")` excludes
   them from representation inputs. The suspected locator shortcut was not
   substantiated. Preserve that safeguard and re-test it in the new consumer.
5. Every output query currently runs separately through attention and output
   projection to preserve exact bitwise chunk equality. The mathematical panel
   independence is useful; mandatory bitwise equality across changed kernel
   shapes is an expensive implementation constraint. A new numerical version
   should use bounded batched queries, explicit tolerances and reproducibility
   within its pinned runtime. Frozen implementations were not changed.

**Data findings and limits.** The composition contract and previously admitted
evidence describe 3,811 fitting records, 3,679 intervention genes, one yeast
biological context, one assayed panel of 1,850 proteins, 6,865,493 observed
values and 184,857 missing values. These counts were not independently rederived
from biological arrays during this audit. The number of scalar readouts must
not be presented as the number of independent interventions. The broader
species-aware program currently has neither an admitted human fitting corpus
on this path nor a within-corpus context-transfer test.

The source is the Messner yeast deletion proteome release, with positive
MaxLFQ intensity transformed to log2 and `NA` omitted, not zero-imputed. It
samples measurable proteomes in the viable deletion collection, not all genes,
all molecular abundances or lethal double mutants. Thus observed-target NLL
estimates performance conditional on detection and this population. Report
readout coverage, intervention-level uncertainty and source limitations.
The HIS3 control artifact has 1,843 supported readouts of 1,850; missing basal
support must have a frozen evaluation policy rather than post hoc panel pruning.

`sources/yeast-proteome-v2.yaml` records upstream plate-median normalization
and scaling by the median of all plate medians, preceding our held-gene split.
This is an unresolved preprocessing dependency: our row-level exclusion proves
neither that source normalization was fitted exclusively on allowed controls
and fitting rows nor that held outcomes could not influence retained values.
Its effect has not been quantified here. End-to-end isolation requires a
source-method audit and, where necessary, a rights-admitted reprocessing from
raw fitting/control inputs with frozen transforms. Until then, label any future
result processed-release retrospective prediction. Do not silently describe
the current cleaned rows as a wholly inductive preprocessing pipeline.

Static features are 20 amino-acid fractions plus scaled length. Reversing a
synthetic canonical peptide produces exactly identical features, by construction.
This proves loss of residue-order information, not that all real genes collide
or that the feature arm must fail. Keep it as a weak matched control. A proposed
order-sensitive sequence representation needs separate model/sequence rights,
immutable model and extraction revisions, coverage and contamination review.
No particular pretrained model or novel architecture has been selected by
benchmark scores in this audit.

The yeast single-cell atlas has an outcome-blind genotype inventory, but that
does not imply an admitted quantitative transcriptomic corpus. Its potential
environmental contrast is a future source of context evidence, subject to
counts/exposure, basal-state and replicate contracts. A count likelihood without
a source-appropriate library-size contract is insufficient. Likewise, row-based
replicate IDs are not proof of independent biological replicate halves.

**Evaluation and benchmark decisions.** The molecular evaluator v2 is explicitly
diagnostic-only: it computes typed likelihood and perturbation-centroid metrics
but does not compute the complete frozen baseline-NLL gate. The context/TxPert
module emits point predictions without frozen probabilistic scales; ridge is
not implemented there. The older `slp-1-1-molecular.yaml` refers to trainer-side
outputs and does not encode the complete protected-source baseline comparison.
It must not be repurposed to certify the new target-separated path. Fit baseline
scales and all hyperparameters only inside gene-grouped fitting folds, then
compare exact matched predictions within the independently controlled evaluator.
Report every seed and protected source/species, primary per-intervention macro
metrics, observed-target NLL and intervention-clustered uncertainty. Readout
count does not justify narrow gene-transfer confidence intervals.

Continuous-density NLL is unit-dependent and can be non-positive. The historical
dense code uses `(baseline - model) / max(abs(baseline), 1e-8)`. Preserve that
historical fact; the next independent evaluator must explicitly freeze its
value space, baseline-specific denominator and aggregation rather than silently
inventing a new meaning of the 2% rule. The 0.02-nat and 0.10 adjusted-Pearson
floors remain unchanged. The existing bound-discrimination and DE requirements
remain contract-blocked; neither should be fabricated from unreplicated means.

Primary-literature checks reinforce this interpretation. Systema documents how
shared perturbation shifts inflate ordinary metrics and motivates centering on
perturbation-specific references
([Nature Biotechnology](https://doi.org/10.1038/s41587-025-02777-8)). The 2026
Signal, Bounds, and Baselines preprint argues for signal-sensitive metrics,
empirical bounds and meaningful linear comparators
([bioRxiv v2](https://www.biorxiv.org/content/10.64898/2026.04.20.719650v2)); it is
methodological context, not a reproduced SLp result. Feng et al.'s benchmark
shows sensitivity to split, negative sampling and label provenance
([Nature Communications](https://www.nature.com/articles/s41467-024-52900-7)).
These sources support evaluation design, not a universal current model ranking.

Retain the SLp-1 benchmark history as retrospective evidence. Future primary SL
confirmation should name a context-specific measured interaction/viability
endpoint, the screened population and negatives, and both-gene quantitative
exclusion. Unknown pairs are unlabeled, not automatically tested negatives.
Freeze the label-free application score before opening benchmark truth; a
fold-local supervised readout is a separate result. Compare degree, feature,
linear and reproducible published methods on identical access and splits, with
prevalence-aware AUPRC and ranking at fixed experimental budgets. An untouched
or prospective independent confirmation is required for a SOTA claim. No such
comparison or confirmation occurred here.

**Implemented decision and validation.** Added the design document
`evaluations/slp-1-1-scientific-reset-v1.yaml` and reproducible, biological-data-free
probe `scripts/audit_slp11_model_shape.py`. Updated root `MODEL_CARD.md` and
`README.md` to distinguish actual data/model status from the program objective.
The design is explicitly not an OMF resource or executable workload, and has no
production admission claim. Historical numerical modules, source snapshots,
thresholds and ledger entries were retained.

The retained probe on CPython 3.11.9, Torch 2.11.0+cu128, CPU, one thread and
seed 731 reproduced the feature-order collision and 5,476-parameter smoke
shape. For one forward call with 1,850 synthetic queries and two memory tokens,
the serial decoder took 0.5143 seconds and its algebraically batched layers
0.0018 seconds; maximum absolute difference was 7.153e-7, passing rtol=atol=1e-5.
An earlier ad hoc probe gave 0.5247 versus 0.0027 seconds. These noisy CPU
microbenchmarks establish an implementation opportunity, not GPU throughput,
training speedup, full-model equivalence or biological accuracy.

The focused suite passed **77 tests, one optional schema skip**, in 24.60 seconds,
covering sparse architecture/training/OMF entrypoint, molecular baselines and
evaluation, composition, packaged audit v1.5, factory isolation and active
workload boundaries. Ten existing nested-tensor warnings occurred. The probe
passed; Ruff, Python compilation and design-YAML parsing passed. No biological
training or paid remote compute was allocated. The probe source SHA-256 is
`285d777f5a12a4d928d1452e985f15297b46b993865bd750ed9bf82a88139d84`;
the design SHA-256 is
`f32cbbb1037e413faf6956e7cc4ca5eae444e187f9ff00b1f173dff412ddc3e2`.

**Decision.** Reject readiness to scale or claim a biological world model.
Retain the useful provenance infrastructure and query interface; replace the
smoke consumer and complete the probabilistic baseline evaluator. The next
falsifiable study is a small, same-context yeast held-gene comparison at matched
features and budget. Even success would not establish human transfer, context
transfer, combinations, temporal dynamics or SL. Biological execution remains
blocked by independent custodian authorization, clean physical boundaries,
consumer/evaluator compatibility and the source-normalization question.
This audit does not self-provision an independent signer or waive those gates.
No checkpoint, molecular performance metric, release or SOTA evidence was
produced.

## 2026-09-04 — Biological transition development replaces the audit-only stop

**Authorization and scope.** The investigator explicitly authorized autonomous
data acquisition, local biological training and bounded GPT-5.6 Sol scientific
subtasks. This supersedes the preceding entry's independent-handoff prerequisite
for exploratory work. We used native Windows CUDA explicitly, not an unreported
OMF executor fallback. No release signature, independent confirmation, or OMF
deployment claim was created. Historical model/v1 and earlier evidence remain
unchanged. The original protected molecular snapshots and SL benchmark outcomes
were not opened for these experiments.

**Hypothesis and advancement.** A feature-conditioned intervention-to-molecular
state model should improve development held-gene Gaussian NLL by at least 0.02
nats per observed target over both fitted mean and feature-ridge, with
centroid-adjusted profile Pearson at least 0.10. Checkpoints minimize validation
gene-macro NLL, not the more favorable correlation at another epoch. Human
contexts must each pass; averaging cannot hide regression. These development
comparisons cannot establish a launch or SOTA claim. New feature/decoder arms
are explicitly adaptive development, not an untouched final evaluation.

**Implementation.** `modules/slp-1-1-world-transition-v1/` is a self-contained
native experimental module. It includes composite-keyed loaders, complete mean
and feature-ridge Gaussian controls, grouped fitting-only OOF scale calibration,
static feature alignment, a feature-conditioned action encoder, measured basal
context encoder, query decoder, and optional low-rank shared Gaussian factors.
It has no learned gene-ID vocabulary. The default biological runs use diagonal
uncertainty; joint likelihood and sampling were checked against dense Gaussian
references, not established as biological population models. Tensor-file
inference reloads without a corpus or OMF. Human runs retain source copies;
initial yeast runs retained source hashes but predate automatic source copying.
Neither local tensor reload nor a dependency lock certifies OMF deployment.

**Data and splits.** Yeast uses fitting-only composite SHA-256
`0a5322c46e15e8a15d17000e8993c0ad642fcc70bc8fff00cbba8fb2905708bf`.
The new internal split hashes `slp11-development-v1|731|taxon|stable_id`, using
the first eight digest bytes modulo 100: <70 training, 70–84 validation, and
85–99 reserved test. All repeated intervention records share a group. Yeast
has 2,656 training records/2,562 genes, 553 validation records/541 genes, and
602 reserved records/576 genes. The original protected intervention roster
was excluded upstream from the entire fitting corpus. Upstream processed
proteome normalization remains a retrospective limitation.

Human acquisition uses the authors' Replogle 2022 K562 essential day-6 and RPE1
essential day-7 CRISPRi raw bulk summaries from Figshare+ v1, DOI
[10.25452/figshare.plus.20029387.v1](https://doi.org/10.25452/figshare.plus.20029387.v1).
The CC BY 4.0 release and author aggregation code identify these as per-cell
mean abundance, not integer sums. The first adapter uses the metadata-only
intersection of 7,226 ENSG readouts and `log2(1+10000*x/sum(shared_panel))`.
Its development NPZ SHA-256 is
`82904b7b52ab34d71e94abb2311c93a420321697d53eab12dabae5b247376f75`:
3,281 training and 726 validation records over two contexts. Another 713
records were routed to a separate reserved test file and not scored. Five RPE1
rows with unresolved intervention identity were excluded. The same stable-gene
split applies across both contexts. These raw-bulk targets are now a diagnostic
arm, pending the data correction below.

**Accessible modalities.** Sequence composition (21), ordered dipeptide
statistics (421), ESM2 protein embeddings (320), and direct GO MF/CC SVD features
(256) were tested for yeast. ESM2 is `facebook/esm2_t6_8M_UR50D` revision
`c731040fcd8d73dceaa04b0a8e6329b345b0f5df`; long proteins use overlapping
windows with residue-weighted pooling rather than truncation. Human ESM has
320 protein features plus a presence flag, exact Ensembl gene identity, and
7,401 translated/141 missing rows. Its NPZ SHA-256 is
`9c0ade1b580f46f26938e5eab6e0222b9e543e44bc2c7d5113336c80459bfb52`.
Archived 2022 GO features exclude NOT and perturbation-derived evidence codes;
the human pack covers 7,315/7,542 entities, SHA-256
`208be756b81229b3881af8229e18ba2f5e806f5be85180b6f5560c3f2d07c0ea`.
Feature builders, source versions, mapping details and manifests accompany the
ignored data artifacts. No quantitative interaction graph or SL label feature
was included.

**Yeast results, seed 731.** Each cell below is validation gene-macro NLL /
centroid-adjusted profile Pearson. NLL is in nats per observed log2 value.

| Feature/decoder arm | World | Matched feature ridge | Decision |
|---|---|---|---|
| Composition 21, learned scale | -0.06512 / 0.0053 | -0.06412 / 0.0185 | Fail; initial non-OOF calibration |
| Dipeptide 421, learned scale | -0.06362 / 0.0127 | -0.06199 / 0.0192 | Fail |
| ESM2, learned scale | -0.06421 / 0.0075 | -0.06442 / 0.0333 | Fail |
| ESM2, fixed OOF scale | -0.06400 / 0.0110 | -0.06442 / 0.0333 | Fail |
| GO MF/CC, fixed OOF scale | -0.06603 / 0.0301 | -0.06649 / 0.0462 | Fail |

OOF mean NLL is -0.06358. Ordinary profile correlation near 0.99 conceals
weak perturbation-specific predictions. Later checkpoints sometimes improved
adjusted correlation while worsening NLL; they were not selected. A separately
frozen 128-landmark Nyström RBF GO grid selected bandwidth factor 0.5/alpha1000:
NLL -0.06304, adjusted r 0.03139, also failing against mean. These small runs
reject sequence/GO scaling alone as the next investment on this yeast source.

**Human raw-bulk results, seed 731.** The first measured-context model has 321
action/query features, hidden width 128, state width 64, fixed OOF scales, and
64 basal tokens selected by control-only context differences. The controlled
decoder revision adds 32 query descriptors from SVD of standardized training
responses only; action features are unchanged. These descriptors explain
35.29% of standardized training variance and require an assay-measured query
panel. They do not establish transfer to unmeasured readouts.

| Context | Mean NLL | ESM ridge NLL / adjusted r | Protein-query world | Response-query world |
|---|---|---|---|---|
| K562 | -1.11419 | -1.12356 / 0.1289 | -1.12375 / 0.1048 | -1.13055 / 0.1503 |
| RPE1 | -0.74568 | -0.75497 / 0.1621 | -0.75178 / 0.1787 | -0.75611 / 0.1819 |

The response-query model improves both context point estimates but fails the
0.02-nat rule against ridge. Selected epoch 20, 190,208 parameters, 66.83 seconds
including baselines/evaluation; checkpoint SHA-256
`df2889473e7c7a03caff22a65c1fd9eca0b3a7533e394bea9096d288d5d2a24d`.
Artifacts: `results/slp11-transition/human-esm2-context-seed731-v2/` and
`human-esm2-response32-seed731-v1/`. A separate training-response reduced-rank
ridge grid (ranks 16/32/64, alpha100/1000/10000) selected rank32/alpha10000 but
did not improve on unrestricted feature-ridge: NLL -1.11780 K562, -0.75458 RPE1.
Every OOF fold refitted the response basis. Test artifacts remained unused.

**Actionable data diagnosis.** An independent code/numerical audit found that
the raw-bulk summaries have unequal precision: K562 has 5–1,996 filtered cells
per targeting row and RPE1 2–3,580. Among core non-targeting controls, profile
RMSE versus log(cell count) correlates -0.986 and -0.947 respectively. Equal
training weights and query-only variance miss this measurement noise. The
authors' phenotype analysis also uses per-cell UMI scaling and per-gemgroup
control normalization, which cannot be reconstructed from already collapsed
raw means. The adapter averages all controls despite core-control indicators.
These findings motivate a new control-normalized dataset version and explicit
measurement exposure; cell count must not become a post-intervention predictor
feature. Duplicate ENSG records often represent P1/P2 library constructs with
different efficacy, not biological or single-guide replicates. Their agreement
is not a biological noise ceiling. Global uncertainty rescaling could improve
validation NLL by only about 0.0014/0.0006 nats and cannot rescue the rule.

The audit also identified tiny false mean-baseline adjusted correlations from
subtracting a float32 reference from a float64 fitted mean. The scorer now
retains the exact fitted centroid; the mean residual is undefined by
construction. Earlier saved reports retain those numerical artifacts, which
are not interpreted as signal. Source-target reconstruction error was <4.8e-7.
Audit artifact SHA-256:
`453ebfb6dcd6a209ef690eb02b998412a230215a11fca229a2acaacd2b1ab184`.

**Scientific context and next action.** The need to remove average effects is
consistent with [Systema](https://doi.org/10.1038/s41587-025-02777-8). Recent
[TxPert](https://doi.org/10.1038/s41587-026-03113-4) provides a relevant public-code
comparator, but its strongest proprietary graphs are not available to this
program. Neither paper's headline results are substituted for comparisons on
our splits. Continue with corrected human measurements and exposure-aware
likelihood, then compare the static-feature and response-query arms. Keep
combination and cross-context capabilities as untested until directly trained
and evaluated. No candidate in this entry is launch-ready.

**Compute and verification.** Native Python 3.11.9, Torch 2.11.0+cu128,
NumPy 2.4.4, SciPy 1.17.1, scikit-learn 1.9.0 on RTX 4070; deterministic seed731,
bounded per-run time caps, no purchased remote compute. Protein extraction took
about 27 seconds for yeast and 52 seconds for human. Individual model pilots
finished in seconds to about a minute. The full repository suite passed
337 tests/21 skips before the response-query addition; the subsequent seven
focused architecture/inference/response-query checks passed. Artifacts and
real data remain ignored; code and versioned declarations are reviewable.

## 2026-09-04 — Corrected human data, repeatable baseline gains, and combination data

**Data correction.** The author-normalized Replogle development NPZ is
`data/derived/slp11-human/replogle-k562-rpe1-author-normalized-development-v2.npz`,
SHA-256 `88de5164fca4e2504ac5b459ab4226c161eb586dd04700d5784da4bb53048659`.
It preserves the v1 identities and split arrays exactly: 3,281 training and
726 validation records, 7,226 queries. Targets are author per-gemgroup,
core-control-standardized pseudobulk means, not log2 abundance. All values on
this shared panel are finite. Another 713 records were routed to a separate
test-only artifact and not evaluated. The full sources have SHA-256
`c1ca6456c9c9f1aa2b02c496eb64d1dc3e6a852edbd744d682b8d2c95fd36829`
(K562) and `a3c5bfd0f15d63938bc80c9b8874b9cd761e3a23caf5ffe7966bae4e887ec89d`
(RPE1); upstream MD5 checks also pass. Core-control pseudobulks number 97/113.
Raw core-control expression supplies a separate measured basal context feature,
avoiding the near-zero basal state implied by z-scored targets. Cell counts
enter measurement uncertainty only. No intervention outcome or observed
knockdown efficacy is used as an inference feature.

**Fixed comparison.** The hypothesis remains improvement by 0.02 nats per
observed target against both mean and ridge, with adjusted r >=0.10 in each
context. The feature pack combines ESM2 and archived GO MF/CC (577 dimensions),
SHA-256 `b3de49e18d3c75676985b8790d1ce85de0d87d526bbd7c0c5b555828a1fb11a0`.
The neural model uses hidden128/state64, response-query rank32, dropout0.2,
AdamW lr0.0005/weight-decay0.1, batch64, maximum180 epochs, patience30, and a
30-minute wall-clock cap. It selects the minimum equal-context gene-macro NLL.
The uncertainty model fits per-query/context biological + sampling/n variance
from fitting-only OOF residuals and core controls. Sampling slopes were
identifiable from controls for every query. References remain independent of n.

The comparator grid freezes full ridge alpha100/1000/10000/100000 and
response-basis ranks16/32/64 at those alphas. Every OOF response basis is
refitted without the held intervention genes. Alpha10000 full ridge wins;
equal-context NLL -0.378277 versus -0.343732 for mean and -0.376511 for the
best reduced-rank arm (rank64/alpha10000). The grid took 39.27 seconds on CPU.

**Results.** Entries are development gene-macro NLL / centroid-adjusted Pearson;
NLL units are nats per observed author-control-standardized molecular value.
They must not be compared numerically with the preceding log2-space NLLs.

| Model | K562 | RPE1 | Best epoch |
|---|---|---|---|
| Mean | -0.48459 / undefined | -0.20287 / undefined | — |
| Matched full fusion ridge | -0.52211 / 0.2170 | -0.23444 / 0.2454 | — |
| Fusion world, seed731 | -0.54167 / 0.2307 | -0.24916 / 0.2653 | 20 |
| Fusion world, seed732 | -0.54370 / 0.2363 | -0.24491 / 0.2461 | 93 |
| Fusion world, seed733 | -0.54389 / 0.2305 | -0.24668 / 0.2619 | 45 |
| GO-only world, seed731 | -0.54180 / 0.2438 | -0.24083 / 0.2470 | 40 |
| Fusion ridge-reference correction, seed731 | -0.54735 / 0.2412 | -0.24887 / 0.2688 | 20 |

The GO-only matched ridge is -0.51793/0.2342 K562 and -0.23036/0.2688 RPE1.
The correction model uses grouped OOF ridge forecasts as its training reference
and full-training ridge at inference; it never fits corrections to in-sample
ridge residuals. Its uncertainty inherits the ridge OOF exposure components.
The model's reference is still a quantitative molecular quantity, not an SL
score or label-dependent readout. A saved linear-reference artifact reproduces
the reference for arbitrary subsets of the fitted query panel.

All three fusion seeds improve likelihood in both contexts. No listed model
passes the 0.02-nat rule in both contexts. Seed731 bootstrap resamples
intervention genes, retaining all their records, 1,000 times. NLL benefits over
ridge are 0.01956 [0.00832,0.03210] K562 and 0.01472 [0.00393,0.02682] RPE1.
Adjusted-r benefits are 0.01372 [-0.00689,0.03403] and
0.01994 [-0.00770,0.04715], respectively. These intervals are conditional on
the adaptive development process, not independent confirmation. Absolute world
adjusted-r intervals are [0.2003,0.2584]/[0.2249,0.2998].

**Uncertainty diagnosis.** With inherited mean uncertainty, world residual
second moments for <30/30–99/100+ contributing cells are 1.493/1.012/0.744
(K562) and 1.127/0.946/0.754 (RPE1). High-count summaries are overconservative,
low-count summaries underconservative; heavy tails remain. A separately frozen
calibration experiment therefore fits neural OOF residuals from three global
gene folds, exactly 20 epochs each, using no outer-validation outcome. Every
fold refits references, normalization and response geometry; 726 outer
validation rows remain excluded. It preserves the core-control sampling
component and estimates the neural residual biological component. Its effect
on outer development validation is a separate evaluation, not scale fitting
from these diagnostics.

**Artifacts and practical inference.** Seed731 checkpoint SHA-256 is
`40f69aefea1e895fcbfccd89677c3b8df05ef5bfc5ed4b8b1a2c7c8aedfe39f6`.
Each run directory is under `results/slp11-transition/` and contains immutable
weights/configuration/references, source copies, the pre-fit protocol and
reports. Source changes create new run artifacts. Target-free CPU reload of
the earlier response32 artifact matched its snapshotted runtime exactly for
means, latent state and uncertainty. Tests verify that a measurement-scale
override cannot change molecular means or latent state. Query-subset inference
supports both fitted context means and saved linear references. The scientific
figure is reproducible with `scripts/plot_slp11_development.py`; outputs are
`results/slp11-transition/figures/human-development-v2.{png,pdf}`.
The candidate-audit report SHA-256 is
`8d00bec4cf75b0b9c97198a60601518f827d402b17a53f203c24d33101129cd7`.
Full repository verification reached 361 tests passed/21 skipped before the
subsequent inference and combination-data additions; focused checks for those
additions are retained in their tests.

**Combination data and decision.** Official GEO GSE133344 supplies Norman 2019
K562 CRISPRa: 91,168 eligible cells covering control plus 105 single and 131 double
perturbations. The first imported mean-UMI/CP10k aggregate is diagnostic; a new
per-cell/control-normalized adapter is being prepared. Stable-gene hash routing
assigns 130 intervention records to training, 40 to validation, 66 to test,
with a test or validation constituent excluding the entire double record from
training. Test-only outcomes are not used for modeling. All 105 intervention
genes now have ESM features in a 7,605-entity static pack; existing 7,542 feature
rows are byte-identical, and new GO rows are projected into the reconstructed
original SVD basis without refitting. This enables a genuine molecular
combination test instead of treating multi-token API support as evidence.

Continue with fitting-only uncertainty calibration and the combination transfer
pilot. The positive, repeatable human likelihood gain warrants further model
development. It does not establish SOTA, unseen-context generalization, SL
performance, or readiness for release. Local experiments remain bounded and
no remote compute or original benchmark outcomes were used.

## 2026-09-04 — Ensemble development passes; broader molecular tests remain necessary

**Hypothesis and fixed rule.** Averaging the three already selected human
molecular models reduces forecast error, and fitting observation uncertainty
from their ensemble held-gene OOF residuals improves likelihood without using
outer-validation outcomes for calibration. The rule remains a per-context gain
of at least 0.02 nats/observed target against both mean and full-feature ridge,
with centroid-adjusted profile Pearson at least 0.10. The ridge alpha remains
10,000, selected in the earlier development grid. No SL outcomes enter this work.

**Exact model and data.** Seeds 731/732/733 use selected epochs 20/93/45.
Their three global calibration folds exclude every outer-validation gene and
refit normalization, references and response-query descriptors inside each
fold. Epochs are fixed within OOF calibration. The normalized human development
snapshot is `88de5164fca4e2504ac5b459ab4226c161eb586dd04700d5784da4bb53048659`;
static fusion is `b3de49e18d3c75676985b8790d1ce85de0d87d526bbd7c0c5b555828a1fb11a0`.
The experiment takes 177.53 seconds on the local RTX 4070. Sources remain
Replogle 2022 human K562 essential day 6 and RPE1 essential day 7, measured as
author core-control-standardized pseudobulk means over 7,226 RNA queries.

| Development context | Ensemble NLL | Gain over ridge (gene-bootstrap 95% interval) | Adjusted r | Fixed rule |
| --- | ---: | ---: | ---: | --- |
| K562 | -0.558232 | 0.036119 [0.022602, 0.049188] | 0.247285 | pass |
| RPE1 | -0.257585 | 0.023143 [0.010486, 0.037244] | 0.269301 | pass |

Intervals use 1,000 intervention-gene resamples and remain conditional on
adaptive development. The rule uses point estimates; an interval lower bound
of 0.02 was not stipulated and is not silently added. Independent latent
coordinates are returned on separate member axes, never averaged. Ensemble
variance is calibrated Gaussian measurement uncertainty, not a Bayesian
posterior or a validated single-cell generator. Residual heavy tails and
count-dependent coverage error remain.

Artifact: `results/slp11-transition/human-normalized-fusion-response32-ensemble731-733-v1/`.
Manifest SHA-256 `a972d994f80c124f948b9b4a313d9e76bdd5c1a3477ebc4082c143ae96c50a70`;
exposure SHA-256 `3526b9c30b7f16e26fde17e8f2adece7271dd4d5c0ef3d38f45de40aee6de929`;
report SHA-256 `dc8368481ec2af2b9c4a15fe0a3de2272bc50f5ea970c2af6a8334c1a5c8f610`.
Recomputing seed731 OOF variance differs by at most 4.06e-7 from the earlier
artifact, so numerical agreement, rather than byte-identical calibration, is
claimed. Seven focused ensemble checks pass. A self-contained CPU package at
`results/slp11-transition/packages/human-ensemble-crispri-dev-v1/` produces
BRCA1 molecular forecasts using only its own runtime, weights, references and
static features. Its packager verifies the exact frozen feature checksum
before creating an output. This is a local experimental package, not a public
or certified release.

**Measured-context dependence.** With the selected seed731 model and correct
references held fixed, matched/swapped/masked context tokens give K562 NLL
-0.54167/-0.48658/-0.53185 and RPE1 -0.24916/-0.20888/-0.23299. Across 304 shared
genes, context-response-difference profile r is 0.23619/-0.10800/0.13676.
The model uses measured context beyond its reference; this does not establish
transfer to a previously unobserved context. Audit:
`results/slp11-transition/human-context-dependence-v1/report.json`.

**Combination pilot rejects current transfer hypothesis.** Norman 2019 human
K562 CRISPRa raw cells are normalized per cell (full-library CP10k, log2), then
aggregated by exact construct and standardized by control-cell mean/population
SD. This transformation is implemented by SLp, not supplied as an author
normalized matrix. Development v2 SHA-256 is
`ab81e7ed07d7f111b3dfc964cece28a2db7de0dcf5975f6ff1a3bc2db0be683e`.
There are 160 training and 49 validation constructs, with 7,182 measured queries
from the 7,226 panel. All validation constructs have exactly one held gene;
none tests two held constituents. Additive summed-feature ridge alpha1000
achieves validation RMSE 0.14555, adjusted r 0.3595 and double r 0.5239.
Random/human-initialized neural models achieve RMSE 0.1486/0.1491 and double r
0.4746/0.4710. Transfer fails the fixed rule. Report SHA-256
`a67408fbea62e3e93ac7872e24a41cfc64f85c36a516107e163482223e2d4d1e`;
12.02 seconds local GPU. No Norman test outcomes are scored. A single-CRISPRi
success does not imply combination or activation-mode generalization.

**Decision.** Freeze the successful ensemble, baselines, uncertainty and rule,
then perform its first evaluation on the reserved new human molecular test
snapshot `7bf755248513f41c552e4a4bde2d5958f0f5ea4243eeeb5ec77128642b0697d1`.
Do not refit on development validation before confirmation. Separately expand
training with the official K562 genome-scale day-8 screen. The three-context
development snapshot is
`baac863d7050fbd71ac332a680215af1e400f759ad441534019905bd521fda96`, containing
10,719 training and 2,339 validation records. Selection by fitting observation
availability yields 7,036 fully measured queries without imputation; shared-panel
snapshot SHA-256 is
`006b4bb127a09073a7f409d81a7bccce96bb961879cb5e57dce56b48eb8e664b`.
Its query availability selection has two focused passing tests. New results
must state the smaller panel explicitly and use identical-panel comparators.
The original protected molecular snapshots and SL benchmark labels remain
unopened. SOTA, new-context transfer and launch readiness are not established.

## 2026-09-05 UTC — Frozen human molecular confirmation fails the advancement rule

This entry follows the September 4 local-time development work. The frozen
three-seed ensemble was evaluated exactly once on 713 reserved human molecular
records, after the protocol and synthetic contract checks were written. No
weights, epochs, uncertainty, baseline alpha or threshold changed afterward.
Mean and full-feature ridge were fitted to the original 3,281 training rows
only; the 726 development validation rows were not folded into fitting.

| Confirmation context | Records / genes | Ensemble NLL | Gain over mean | Gain over ridge (gene-bootstrap 95% interval) | Adjusted r | Decision |
| --- | --- | ---: | ---: | --- | ---: | --- |
| K562 essential day 6 | 323 / 309 | -0.559072 | 0.045215 | 0.014646 [0.003575, 0.026303] | 0.228200 | fail |
| RPE1 essential day 7 | 390 / 364 | -0.226893 | 0.031360 | 0.010336 [0.001877, 0.019008] | 0.251521 | fail |

The fixed point-estimate threshold is 0.02 nats over both baselines in each
context; both miss the ridge criterion. The small positive same-source
held-gene gain survives, but the stipulated advancement claim does not.
This is retrospective confirmation within the same sources and selected
experimental populations, not unseen-context or prospective evidence.
No SL benchmark was accessed. This molecular holdout is retired from model
selection, and later candidates need a fresh confirmation source/protocol.

Observation calibration remains imperfect: squared standardized residuals in
<30 / 30–99 / 100+ cell bins are 1.473 / 1.022 / 0.778 in K562 and
1.338 / 0.979 / 0.801 in RPE1. These are diagnostic findings only; no scale is
recalibrated from these outcomes.

Directory: `results/slp11-transition/human-normalized-fusion-response32-ensemble731-733-molecular-confirmation-v1/`.
Protocol SHA-256 `bd6fc2fcb4943d9869a08e347a63b23e17133e28b91d7b87d8beb2b9e9e761a7`;
report `60525de962553b7550b577da071c036de05c1d14b5564d4bd7b86bc4c7cd57f3`;
predictions `f7716025d43d548c3b694970e956e9e2ce46436557e56414e38a5b877865271d`;
baseline exposure `7b36fd62753643f69f6941e6e99de27127f6c6aa33f4a210110fb29e9566c51f`.
Five focused confirmation checks pass. The repository suite reached 396 passed,
21 skipped before those additions. An unrestricted pytest collection entered
an old ignored benchmark-model directory and failed on an inaccessible Windows
log symlink; explicit `python -m pytest tests -q` runs the intended source suite.
No benchmark labels were read by that failed collection.

**Continuing experiment, already initiated before confirmation.** Broader
K562 genome-scale training uses the fixed 7,036-query shared panel. Source
roster coverage is 7,079, but 43 additional queries have sporadic nonfinite
author-normalized training entries, rather than zero source variance. They are
masked in the original adapter and excluded from the complete-panel experiment;
no missing outcome is treated as measured zero. The static extension now has
10,231 entities and 577 features, SHA-256
`a2f3153478c00c191e5a9e218badb3327a180a56948a4c9c6a6926cc506ff02b`.
All prior 7,605 rows are byte-identical. ESM/GO covers 9,825/9,696 of 9,852 GWPS
actions, respectively. ESM extraction adds 2,615 proteins in 17.1 seconds,
retaining full-protein windows; 11 new proteins lack sequence. GO projects onto
the unchanged archived basis and omits 560 new-only terms without refitting.

Freeze a single seed731 pilot with the existing numerical architecture and
hyperparameters, broadened training only, and the same per-context .02/.10 rule
against context-local mean/ridge in all three contexts. Compare the old ensemble
on the identical 7,036-query development panel descriptively; that comparison
changes both corpus and ensemble size and cannot isolate a pure data effect.
Original protected molecular and SL benchmark snapshots remain unopened.

## 2026-09-05 UTC — Genome-scale training and physical-relation features

**Broader-corpus hypothesis.** The predeclared seed731 model uses the same
architecture, 32 response-query descriptors, hidden128/state64, dropout0.2,
AdamW lr0.0005/weight-decay0.1, 180 epochs maximum/patience30. Complete-panel
snapshot SHA-256 `006b4bb127a09073a7f409d81a7bccce96bb961879cb5e57dce56b48eb8e664b`
and ESM/GO pack `a2f3153478c00c191e5a9e218badb3327a180a56948a4c9c6a6926cc506ff02b`
are unchanged. There are 10,719 fitting/2,339 validation records and 7,036
queried RNAs. All results below are adaptive development, with no test access.

| Context | World NLL | Adjusted r | NLL gain vs mean / ridge | .02/.10 rule |
| --- | ---: | ---: | --- | --- |
| K562 essential day 6 | -0.543392 | 0.258046 | 0.062103 / 0.024067 | pass |
| RPE1 essential day 7 | -0.249635 | 0.282514 | 0.049403 / 0.017571 | fail |
| K562 genome-scale day 8 | -0.913336 | 0.102432 | 0.011459 / 0.005396 | fail |

Epoch61 is selected; training takes 370.13 seconds and the full pilot/comparator
386.08 seconds on one RTX 4070. The fixed all-context hypothesis fails. On the
identical 7,036-query panel, the earlier three-seed ensemble has better NLL in
K562/RPE1 (-0.555869/-0.255335), while the new single seed has higher adjusted r
than its 0.248482/0.270051. This comparison does not isolate a data-scale effect:
it also changes ensemble size, optimization and sample composition.

Artifact root: `results/slp11-transition/human-gwps-complete-panel-fusion-response32-seed731-v1/`.
Protocol SHA-256 `3f013a89b388615133a0f3de24dad4c5de3995badd9bcec89d07b950ea57bbe8`;
summary `203f0f015505b67e30edee2ab4eeaeb7428ac4c1e9ebedd58c110b052f1cb37e`;
checkpoint `66f0eb42faaf310f330c3da9734531d99ae2d1ea7f1daeec97acc69701d2b97c`;
reference `244aa4b59f0ae0fad4d04079b286618851ef891633740f0c4cb40a86f1867d43`;
exposure `aeba30010e3e8ae5680526cf00f187ae8ed3f91d622b58f53d5d8aa3f58da7cd`.
Sixteen focused pilot checks pass.

**Separate static physical-relation screen.** Official STRING12.0 physical
relations and unique exact Ensembl_gene aliases supply direct human
experimental-confidence edges >=700/1000. Text mining, transferred evidence,
database channels and combined scores are excluded. Symmetric gene edges are
collapsed by maximum confidence, self edges removed. Within the 10,231-gene
cache this gives 34,155 edges and 3,145 genes with neighbors. The original577
features remain exact; append their weighted known-neighbor mean577, log1p
induced degree and a coverage flag. Feature SHA-256 is
`2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7`.
This static2023 relation graph supports retrospective feature development;
it is not molecular perturbation truth or evidence of causal direction.
Sources/rights are recorded in `sources/string-human-physical-v12.0.yaml` and
`rights/string-human-physical-v12.0.yaml` under the official CC-BY-4.0 terms.

The fixed CPU screen compares ridge alpha10000 with original577 features,
original+degree/coverage, and full1156 features. It reports point metrics only;
no in-sample uncertainty is presented as calibrated likelihood. Advancement
requires >=1% gene-MSE improvement over both controls and no adjusted-r
regression in every context.

| Context | Base / degree / physical MSE | Base / degree / physical adjusted r | Screen rule |
| --- | --- | --- | --- |
| K562 essential | .026671 / .026609 / .025431 | .218316 / .223448 / .257969 | pass |
| RPE1 essential | .059120 / .058935 / .055860 | .246379 / .252739 / .302361 | pass |
| K562 genome-scale | .012457 / .012424 / .012346 | .063444 / .068132 / .078508 | fail |

The screen takes 30.56 seconds CPU and fails its all-context rule. Two focused
aggregation tests verify duplicate handling and identity equivariance. Report
SHA-256 `736968925a96806e1384cf71663e37ffd84fb70c2c4077ed0f240c4dc7a8c4a3` at
`results/slp11-transition/physical-features-ridge-screen-v1/report.json`.
This is not promoted by changing the threshold. A separate fixed neural trial
will test nonlinear use of these relations against the stronger full-feature
ridge and the prior neural world, while retaining all-context .02/.10 and
neural nonregression requirements.

**Architecture and future context evaluation.** A separate control-anchored
module now makes empty intervention return the supplied control baseline
exactly. Its four numerical contract checks pass. A matched biological pilot
is running; the representational fix alone is not a performance claim. The
single-intervention pilot freezes its unidentifiable pair projection. Average
action pooling does not enforce additive molecular combination effects.

Official Nadig HepG2 raw single-cell data have been acquired (5,614,460,941
bytes; SHA-256 `e1ad7c3c5a201c861a207a858aa7e59f5e6ac1955674c415f7de0d1dadadb52e`).
Only metadata and 4,976 non-targeting control rows have been examined, across
56 GEM groups. Perturbed expression rows remain unread. The verified
Replogle code uses linear full-UMI scaling and per-GEM control sample SD,
without log or clipping, for target normalization. A control-only HepG2
implementation is explicitly SLp-computed, not Nadig author DESeq2 logFC/SE.
Artifact SHA-256 `3f72db203e989cb60d9ecd65874a11d2c83af0772a8011bafcb559a65c459951`.
Four focused checks pass. Current saved basal descriptors have differing
normalization denominators/aggregation, so a common control descriptor is
being prepared before any unseen-context scoring. A transferable decoder also
cannot require the new context's unobserved perturbation residual amplitude.

## 2026-09-05 UTC — Physical neural improvement and control-anchoring failure

The physical-neighbor neural trial changes static features to the frozen1156
pack while retaining seed731, hidden128/state64, response-query rank32 and the
same optimizer, splits, panel and stopping rule. Its primary rule is .02/.10
in each context against mean and the stronger full1156-feature ridge, plus
no NLL or adjusted-r regression against the preceding neural model.

| Context | World NLL / adjusted r | NLL gain vs physical ridge | NLL gain / r change vs preceding world |
| --- | --- | ---: | --- |
| K562 essential | -.550340 / .280207 | .012017 | +.006948 / +.022161 |
| RPE1 essential | -.250529 / .304302 | .002006 | +.000894 / +.021788 |
| K562 genome-scale | -.914052 / .107656 | .004746 | +.000716 / +.005224 |

Nonregression passes everywhere; the primary rule fails everywhere. The
strongest matched ridge is -.538323/.257969, -.248522/.302361 and
-.909306/.078508, respectively. Epoch43 is selected in 311.70 seconds GPU,
318.25 seconds for the full pilot. Sixteen focused checks pass. Artifact root
`results/slp11-transition/human-gwps-physical-fusion-response32-seed731-v1/`;
summary SHA-256 `344bb7dab606d496b6f7533e1407eae1f0f894b0ca2cf47e1931d07a00248950`;
checkpoint `159f23afd213ee0535c6f32e0cdd2807e56589c6da42a10ac89e43950b6e6fd6`;
reference `ea12f1c8cb1371fe86fd5e6a1de050c60e6f5b88b9afffce271d02d8cff2f1a9`.
The local self-contained experimental package
`results/slp11-transition/packages/human-gwps-physical-crispri-dev-v1/` has
passed CPU CLI inference from its own directory. No release was uploaded.

**Control-anchored v1.** The separate model satisfies exact control mean/scale
identity and zero latent/molecular intervention delta for an empty intervention
in all3 contexts x7036 queries. Its singleton-unidentifiable pair projection is
zeroed/frozen. However, selected epoch52 gives NLL/r -.53337/.23341 K562,
-.23773/.26792 RPE1 and -.91134/.09700 GWPS, regressing against the original
architecture. Both the primary and no-regression rules fail. The experiment
changes architecture as well as anchoring and does not isolate their causal
contributions. Report SHA-256 `f49ba0fbdb69af09e6e1ce49729febf11856670acae84d94715da96e47783e87`;
checkpoint `2beb8ff4258fedf32fd62c2db9899d378472ed6a2a8eca192499b77f68f198dc`.
Strict source/safetensor reload, query chunk identity and six focused checks
pass. The post-run audit SHA-256
`e42fa26291f5a8b1fa538fa90e4ebda9d47bf2eb5e22b059c2634a360cebefa1`
records two limitations: average action pooling does not enforce additive
molecular single effects, and the decoder amplitude still uses each represented
context's training perturbation residual scale, unavailable in an unseen context.

**Controlled next tests.** A minimal anchoring revision keeps the original
encoder/transition parameterization, uses a single pooled training-only query
amplitude and handles empty actions algebraically. A separate physical-feature
run doubles only latent state64 to128. The geometry diagnostic motivating that
capacity test projects existing ridge forecasts onto the frozen base-world
64-dimensional decoder without fitting to target outcomes. Physical-ridge MSE
rises .54% K562 and1.05% RPE1 but falls1.04% GWPS after projection. This identifies
a possible representational loss in RPE1, not a theoretical oracle ceiling or
proof a larger network will improve. Diagnostic report:
`results/slp11-transition/gwps-decoder-span-audit-v1/report.json`.

**Control descriptor correction.** Source review could not prove that Replogle
raw pseudobulks preserve original full-cell UMI totals: only8248–8749 source
genes remain, and the available unfiltered UMI statistic describes a different
cell set. The first proposed full-library denominator equivalence is therefore
superseded, without overwriting its artifact. Guaranteed common descriptors
now normalize pooled raw control counts by the pooled count over exactly6789
shared stable ENSGs, selected from source IDs and the frozen fitting-complete
panel. This is fixed-panel relative abundance, not full-library CP10k.
Roster SHA-256 `046891d3ceb0766e3fd09441677d6ae078fa7ac7d81ddb1f1c30866007d0d959`;
new7036-target development snapshot
`55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c`;
standalone HepG2 descriptor
`382626401ee38e8d5084ac9f86ffc44bd10408826fb85a94ede8eb908cdf5b27`.
All prior targets, masks, controls and partitions remain exact NPY copies.
Context masks exclude247 unsupported tokens consistently. Nine normalization
and aggregate/single-cell equivalence checks pass. Zero HepG2 perturbed
expression rows have been read. No SL benchmarks or retired confirmation
outcomes enter these new model choices.

## 2026-09-05 UTC — Capacity, transferable controls, and uncertainty experiments

All experiments below use human Replogle CRISPRi development partitions only,
seed731, local RTX4070 or two CPU threads. No SL benchmark or retired molecular
confirmation outcomes inform their choices. Static modalities are sequence,
archived MF/CC annotation and, where specified, direct physical neighbors.

**Latent capacity ablation.** Doubling only the original physical-feature
world's latent state64 to128 retains hidden128, response-query rank32 and the
7036-query complete development panel (`006b4bb127a09073a7f409d81a7bccce96bb961879cb5e57dce56b48eb8e664b`).
The fixed rule remains >=.02 nats versus mean and full physical ridge,
adjusted r>=.10 and no regression versus state64 in every context.

| Context | NLL / adjusted r | NLL gain vs physical ridge | NLL gain / r change vs state64 |
| --- | --- | ---: | --- |
| K562 essential | -.551662 / .290639 | .013340 | +.001323 / +.010432 |
| RPE1 essential | -.257087 / .331079 | .008565 | +.006559 / +.026778 |
| K562 genome-scale | -.914132 / .108128 | .004826 | +.000080 / +.000472 |

The primary rule fails in every context; nonregression passes. Selected epoch43
uses 584832 parameters and 303.22 seconds GPU (309.72 total). The response
geometry hypothesis receives limited development support in RPE1, without
establishing transfer or advancement. Artifact root
`results/slp11-transition/human-gwps-physical-fusion-response32-state128-seed731-v1/`;
summary `12c5dde62527b867fa9b1f80ea12ad8bc2c5f2c829277d4bd1ffb8bf99051b29`;
checkpoint `d85e898e3486741c7aa87e6bc18ecf1c8607746117ec06bac266c567f6b31bbf`.
Sixteen focused checks pass.

**Minimal transferable control model.** The separate self-contained
`modules/slp-1-1-control-transition-v2/` retains the original encoder topology,
anchors empty interventions exactly to supplied controls, and replaces
context-specific decoder amplitudes with one fitting-only pooled query vector.
Observation scale cannot change molecular means or state. The fixed control
snapshot is `55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c`;
the common basal panel has 6789 queries, with 247 unsupported tokens masked.
This permits inference from new-context controls without fitting its
perturbation response statistics. It does not prove such inference is accurate.

The base577/state64 pilot selects epoch61 and gives NLL/r
-.533897/.246223 K562, -.244893/.264745 RPE1 and -.914183/.101173 GWPS.
Ridge margins .01457/.01283/.00624 fail the fixed .02 rule. Report
`665edc70ce283df3187ea0a16485e36f2b3c061fdb5d933065c5b1f85cddf3f9`;
checkpoint `429791272736c59ae77cca72ccd5a6b51f60736c2213493fa6d924f215611d2d`.

A separately fixed synthesis uses physical1156 features and state128, so it is
not an isolated causal ablation. It compares with the stronger physical ridge
and requires nonregression against the minimal base577/state64 pilot.

| Context | NLL / adjusted r | NLL gain vs physical ridge | NLL gain / r change vs minimal base577 |
| --- | --- | ---: | --- |
| K562 essential | -.545383 / .284144 | .007061 | +.011486 / +.037922 |
| RPE1 essential | -.251684 / .315596 | .003162 | +.006791 / +.050851 |
| K562 genome-scale | -.910775 / .104834 | .001470 | -.003408 / +.003661 |

The primary rule fails in all contexts, and GWPS NLL nonregression fails.
Epoch61, 568448 parameters, 430.28 seconds total. Exact empty-control identity
passes across all three contexts and7036 queries. Eleven focused checks pass.
Artifact root
`results/slp11-transition/human-gwps-fixed-context-minimal-control-physical-state128-response32-seed731-v1/`;
protocol `ba5db27cd29a6f3c8fd796d5d20e013fc21b00529c3026e93aaa7d1d01aeaba2`;
summary `8480b1f1b192edb878cb0e25eb9abc57ab9f6b67aa76f85408eab489dfa7a0ca`;
checkpoint `b1e55f2bcc8a29b6b2467a92ebedfdc1cc80ff8c343a6ab36916d638b9c48cf3`.

**Action-dependent uncertainty screen.** Physical-ridge means remain exactly
fixed. A training-OOF-fitted action multiplier changes only biological variance,
with alpha10000 and multiplier bounds [.25,4] fixed beforehand. The rule is
>=.01 nats improvement in every context. NLL gains are +.021626 K562,
-.007983 RPE1 and +.013699 GWPS: the rule fails. Low-count groups worsen in
all contexts. This is not adopted or retuned from validation. CPU34.31seconds;
four focused checks pass. Report at
`results/slp11-transition/human-gwps-physical-ridge-action-uncertainty-v1/report.json`,
SHA-256 `3f719e1d500a6229e73bca1ceb9d6364ff6e2f59b114d0721fb492bcbb54dae6`.

**Next diagnostic frozen before target access.** New-context point prediction
is a separate scientific question, not a replacement advancement gate for the
failed candidates above. HepG2 metadata define2544 exact construct populations
and2390 genes:1665 source-fitting genes and725 genes absent from source fitting.
The physical/state128 minimal-control candidate and five source/control-only
baselines will be frozen before first HepG2 perturbed-expression processing.
The baseline-only protocol hash is
`943baee44f25a9be7a9fe99e87bd89a364483902e429a7acbd2e2a92df6e74ed`.
Controls select RPE1 as nearest source. Strong comparators include same-gene
source response transfer and full physical-feature ridges. No HepG2 perturbed
expression values have been accessed at this entry; no context-transfer
performance or launch claim is made.

**First HepG2 outcome processing.** Final scoring protocol
`77f03b7a142077bbb19c414ef39b31217c8af0090bc95750f1a52e990ab1998d`
pins the world forecast (`c6d6e6569d8d915886f28aaef024e49d82f55f7f6b219e7fcee5713640d6248d`),
all five baselines, exact population/query axes and scoring implementation.
Its primary estimand first averages construct profiles within gene, then gives
each gene equal weight. Correlation independently removes prediction and truth
query-wise gene centroids within each seen/unseen stratum. The diagnostic rule
requires >=2% MSE improvement against every baseline, r>=.10 and correlation
nonregression against every defined nonconstant baseline in both strata.
1000 gene-block bootstrap draws seed731 re-estimate scoring centroids; intervals
are descriptive and cannot change the decision. Six focused scoring checks pass.

After that freeze, the adapter processed140138 targeted cells into2544 molecular
records, preserving per-query contributing-cell counts and masking247 absent
queries. All records support the6789 measured queries. The first outcome
snapshot is `data/derived/slp11-human/nadig-hepg2-frozen-context-diagnostic-v1/molecular.npz`,
SHA-256 `013c13534a5b33c8667a1f0c82d10416efec2b41d13d27f2ca94a29ab32b22e7`.
Scoring is underway. This is jointly cross-study, cell-context and control-cohort
normalization transfer: Replogle core controls and Nadig all-non-targeting
controls are not equivalent populations. It is neither pure cell-context
transfer nor replication of the author's DESeq2 endpoint. No new performance
claim is supported until the frozen report completes.

**Frozen HepG2 point results (bootstrap pending).** The unchanged point rule
fails in both strata. For1665 source-fitting genes, world MSE is .0616399,
at least2.3058% below every baseline, but independently centered profile r
.262490 falls below same-gene source response transfer (.281151). For725 genes
absent from fitting, world MSE .0614861 is1.0362% worse than equal-source
physical ridge (.0608555), despite slightly higher centered r (.171143 versus
.163890). Thus lower magnitude error and better response geometry are not
interchangeable accomplishments. The model beats zero and average-response
baselines in both strata, which is evidence of transferable molecular signal,
not superiority over strong transferred predictors or launch readiness.
Point report `results/slp11-transition/hepg2-context-transfer-frozen-scoring-v1/point-report.json`,
SHA-256 `9646be33f5ff16c4762addf3948c16c163fcf96f48e547b9fbbfb5eb3577558a`.
The descriptive1000-draw bootstrap remains running; no confidence claim is made
from these point estimates alone. HepG2 outcomes are excluded from subsequent
candidate selection.

The separate target-free context-sensitivity audit reports that55.48% of
across-context forecast-difference energy is gene-specific, with44.52% a common
query profile. The model therefore does not respond to controls solely through
an average shift. The two K562 control contexts yield nearly identical deltas
(RMSdifference .001684; flattened r .999898), whereas RPE1 differs substantially.
No explicit time or assay metadata enter this model, so this is not a test of
learned time dynamics. Audit report
`results/slp11-transition/minimal-control-physical-state128-context-sensitivity-v2/report.json`,
SHA-256 `3f9643049c77016df91f12fe7f7e52ffaf804e50d1e33d5eb532bfcc53a62725`.

## 2026-09-05 UTC — Follow-up representation and objective hypotheses

These next experiments use the original three-source development outcomes;
HepG2 remains excluded from candidate selection after its frozen diagnostic.

**Static encoder capacity.** Existing local MIT ESM2-t33-650M weights match
Hugging Face revision `08e4846e537177426273712802403f7ba8261b6c`. A representative
full-protein profile estimates975seconds extraction for10079 available peptides
among10231 exact Ensembl116 genes, with152 explicit missing rows. The workload
has6067594 residues and11773 windows. Conservative runtime including25% margin
is1218seconds; peak allocation2.82GB. Profile report
`results/slp11-sequence/esm2-t33-650m-ensembl116-profile-v1/report.json`, SHA-256
`f16c349fc82372358aaf841c6650a3cb668603c99fbb9128558290c362abff23`.

Full extraction is authorized with float32 final-layer1280 vectors,
1022-residue windows/128 overlap, inverse-overlap full-residue weighting,
float64 accumulation, batch1, deterministic algorithms and TF32 disabled.
The primary feature comparison uses PCA320 fitted on unique source-training
action-gene static vectors only (present peptides, centered/no whitening,
randomized solver, power7, seed731). GO and physical-graph recipes stay fixed.
The fixed source-development ridge screen requires at least1% gene-macro MSE
improvement and no adjusted-r regression in all three contexts. Full1280 is a
separate secondary arm and cannot rescue a failed primary comparison. No
protein-scale performance claim is made before this screen completes.

**Training objective mismatch.** Current training averages rows while checkpoint
selection averages context-specific gene-macro NLL. Fitting rows are1522 K562,
1759 RPE1 and7438 GWPS; consequently GWPS contributes69.39% of rows, while
evaluation gives it33.33% context weight. Within-context unique genes number
1443/1666/6864. A separately controlled trial will weight each fitting row by
`N / (3 * unique_genes_in_context * records_for_gene_in_context)`, with global
mean weight1 and no per-batch renormalization. Architecture, features, random
row order, optimizer and calibration remain fixed. This changes the statistical
objective explicitly; it is not a claim that minority contexts must improve.

**Latent decoder consistency.** A synthetic numerical counterexample shows
that minimal-control-v2 can give identical latent states for an empty action
and a nonempty action with learned zero intervention delta, while predicting
different molecular means. Its decoder uses total latent state and a separate
action-presence gate. The independent numerical revision
`modules/slp-1-1-control-transition-v3/transition_model.py` decodes intervention
delta instead, algebraically linear `D(state)-D(basal_state)`, and removes the
molecular action gate. It adds no parameters. Zero latent effect now gives
exactly the supplied control mean irrespective of action presence. Two focused
checks reproduce the v2 counterexample and verify v3's linear-difference and
empty-action contracts; Ruff passes. Module SHA-256
`75a487046d30d399000bd50dbe7bf2c642fb1acf297a1c78e1d65b5adbb5a832`.
This revision is untrained; existing v2 checkpoints and failed decisions remain
v2 evidence. Objective weighting and decoder consistency must be tested
separately before any synthesis.

**Next source and controls.** Official GSE264667 Jurkat data are acquired:
9366490264bytes, SHA-256
`ffbe15f2c8f7ffcfd7b0ba9e6937d4ebc2d03b0179fa8234648a59bcb82c04a3`.
Metadata define2390 stable actions,2544 populations and12013 controls across55
GEM groups. Only control expression rows have been processed. Their full-UMI,
per-GEM sample-SD normalizer SHA-256 is
`81c46faac7728b737a610c4bf401b4fd49b3ae03dbd11e1093620b5af4b7d169`.
A new common basal panel across the three Replogle contexts plus HepG2 and
Jurkat contains6517 measured genes on the original7036 query axis, with519
masked tokens. Its exact roster hash is
`1a863ba69f514ba9c1f3752cebde707af4a54ecbf1b54000d7e1320207838a79`.
Source targets/masks/splits/control arrays are unchanged NPY payloads in a new
development snapshot `e925b4406bd1a1ad3ebe0dd31de5e6d356d65b54cdbe908b650ee47844c85c81`.
This is preparation for a future candidate; it does not modify the6789-token
HepG2 diagnostic or the planned objective-weighting comparison. Thirteen
focused control aggregation/access checks pass. No Jurkat perturbed-expression
values or outcomes have been accessed.

**Protein encoder screen completed: reject wholesale replacement.** Extraction
finished in1068.7seconds GPU,1090seconds including assembly. All10079 present
gene rows map to verified peptides, with152 explicitly missing. The11773
entity-expanded windows correspond to11752 unique-peptide windows because21
entities share normalized peptides; no residues were omitted. Original GO,
presence and physical graph construction are verified unchanged. Feature
manifest SHA-256 `ecc855c7c273adc6903c46957684da6daa5925f1222277603be49872c3de08f1`;
PCA320 physical pack `3ab70147fb7d1e1ff9a04d5029fd264f862cd048d2d57af0b130f1c3edc7512b`;
full1280 physical pack `7db0b887cc4c9878f53b342f9915f6c849d8bff1c383a4e7c5f6fcfae70bcc4f`.
PCA fits6846 unique source-fitting genes with present peptides and retains
97.1391% of their static embedding variance; this is not response variance.

The fixed alpha10000 ridge screen uses original complete-panel snapshot
`006b4bb127a09073a7f409d81a7bccce96bb961879cb5e57dce56b48eb8e664b`.

| Context | 8M MSE / adjusted r | 650M PCA320 MSE / adjusted r | 650M full1280 MSE / adjusted r |
| --- | --- | --- | --- |
| K562 essential | .0254312 / .257969 | .0252076 / .267823 | .0253234 / .253827 |
| RPE1 essential | .0558602 / .302361 | .0551368 / .308836 | .0552971 / .294192 |
| K562 genome-scale | .0123456 / .078508 | .0123771 / .076286 | .0124087 / .077933 |

The dimension-matched primary improves K562 MSE by0.8794%, below1%, and RPE1
by1.2952%, but regresses GWPS MSE by0.2556% and its correlation. Only RPE1
passes. The full-width secondary also regresses correlations and cannot rescue
the primary failure. Independently query-centered correlations, declared as a
secondary scoring check before this run, show the same direction of changes.
Thus increased protein encoder scale alone does not solve the cross-source
prediction problem under this fixed representation and ridge protocol. It is
not evidence that every possible use of the650M model is inferior. The packs
are retained but not adopted as the next neural default. CPU64.41seconds;
three identity/alignment checks and ten feature-extraction checks pass.
Report `results/slp11-transition/protein-encoder-ridge-screen-v1/report.json`,
SHA-256 `fe0c5c5f7cfa6fd38d33f5a991c2ae030613e7ee8debedddfc287cd228a69da3`;
protocol `8a6c45b5f13d8581e61d3a7a2bdc0a2b8effa92203a179670285d770e061ab2a`.

**Balanced objective completed: reject this candidate.** The matched v2
physical1156/state128 experiment changes only fitting-row weights, on the
original6789-token source snapshot `55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c`.
Context weight totals are3573 each and global mean weight is1; minibatches are
not renormalized. The default unweighted loss retains exact legacy behavior.

| Context | Balanced NLL / adjusted r | NLL gain vs physical ridge | NLL gain / r change vs unweighted v2 |
| --- | --- | ---: | --- |
| K562 essential | -.549030 / .286987 | .010707 | +.003647 / +.002843 |
| RPE1 essential | -.249603 / .318764 | .001081 | -.002081 / +.003167 |
| K562 genome-scale | -.910613 / .097481 | .001307 | -.000163 / -.007352 |

The ridge margin fails everywhere. RPE1 NLL nonregression fails; GWPS fails
both nonregression checks and the .10-r minimum. This single-seed result
rejects the candidate, not all possible balanced optimization procedures.
Epoch47,445.05seconds local compute. No HepG2/Jurkat perturbed outcomes were
accessed. Protocol `b7418348b9204efba681b16997d26c4ae57cfa4a021778dee93f6a83366ef49c`;
summary `be93987d9234f460fbd9d4510b528ce93e70161acbf2ef2581c29db583b8f8bb`;
checkpoint `13341283172db0d774e00f2b2890e156bd4362e747a89ea1de366a9a0a2e623d`.
Artifact root
`results/slp11-transition/human-gwps-fixed-context-minimal-control-physical-state128-balanced-objective-seed731-v1/`.
The explicit weighting API and trainer model-source pinning are versioned and
snapshotted; the broad suite passes477 tests with21 skips and17 legacy warnings.
The v3 decoder experiment runs separately with the original row objective.

### 2026-09-04 — Completed transfer uncertainty and representation redesign

**State-difference decoder completed: reject this candidate.** The isolated
v3 experiment changes the mean decoder from gated total state to latent
intervention delta, retaining physical1156 features, state128, uniform row
loss and the source `55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c`
snapshot. Empty actions and nonempty zero latent effects now give exactly the
control mean. This consistency property does not yield a performance gain.

| Context | v3 NLL | Adjusted profile r | Fixed decision |
| --- | ---: | ---: | --- |
| K562 essential | -.54037 | .27646 | Fail: ridge margin and v2 nonregression |
| RPE1 essential | -.24292 | .30663 | Fail: ridge margin and v2 nonregression |
| K562 genome-scale | -.91336 | .10350 | Fail: mean/ridge margins and v2 r nonregression |

Epoch24;306.83 seconds local CUDA. Reload maximum error2.98e-7,
direct-delta formula error0, decoded-state-difference error1.19e-7.
Report `results/slp11-transition/human-gwps-fixed-context-state-difference-physical-state128-seed731-v1/model/report.json`,
SHA-256 `1e8b3b9ce951a3b4164a8f187577760ccf1721c8c6b4e721754cbd3cb9e4600e`;
checkpoint `ee7769d8a2ca463758ac3f1b602629c49d95c1b62b30fdfbf0e33e766a32cb05`;
outer protocol `10cb7481a561fdefad769fbde52041c8a2be5f713b0f5b0a71f8555020765a40`.

**Frozen HepG2 diagnostic completed: fail both strata.** The unchanged
preaccess protocol and all forecasts precede outcome materialization. The
1,000-draw seed731 gene bootstrap recomputes centroids per draw and leaves the
point-based decision unchanged. Seen-gene world adjusted r is .262490,
95% descriptive interval [.250217,.274086], versus same-gene source-response
.281151 [.270181,.291667]. Unseen-gene world r is .171143
[.150365,.190675]. Intervals for individual correlations are not a paired
test of their difference.

A separately saved, descriptive paired-MSE supplement uses identical gene
draws for world and every baseline. For unseen genes, equal-source ridge
minus world MSE has 95% interval [-.0010694,-.0001755], or fractional world
improvement [-1.7326%,-.3016%]. Only .001 of bootstrap draws favor the world
model. For seen genes, fractional improvement over the closest MSE baseline,
equal-source ridge, has interval [1.6479%,2.9890%]. These intervals do not
change or retroactively tighten the frozen advancement rule. Constant
forecasts have undefined centered correlation in every draw.

CPU runtime2782.56 seconds. Final report
`results/slp11-transition/hepg2-context-transfer-frozen-scoring-v1/report.json`
SHA-256 `2bda36b03a3da2b65c7dc494d959d2d3cd2d602402f18eabb99520c32a454b03`;
paired report `ac549362850e72aee8f1c0943e59bf2013c1cef0f3df474de40a464d4601ba16`;
paired raw draws `c9802677feab8dadf395cd0f6f196bcc0a270eadc5182341726a43cde4c5a504`.
The final PNG/PDF figure is
`results/slp11-transition/figures/hepg2-frozen-transfer-final-v1`.
Six scoring checks pass. HepG2 is now retrospective evidence and is excluded
from subsequent candidate selection. Jurkat perturbed outcomes remain unread.

**Typed intervention and assay metadata completed.** Immutable sidecars in
`data/derived/slp11-action-observation-metadata-v2/` cover the current13,058-row,
7,036-query Replogle development artifact, Norman CRISPRa and yeast deletion
proteomics. The manifest hash is
`7627bef96866f5cef8325311239ee6d9a4eac34da8e839090cb71d19b63e10d0`;
the current Replogle sidecar is
`a5aae27575cdbd6a026010bfff54d20728b56830f8209476233b4a671278a1bd`.
Its three sources have1,853/2,154/9,051 records at days6/7/8. Quantitative
efficacy and dose are missing, not assumed equal to1. Construct identifiers
are provenance-only. Eight contract checks pass.

Replogle responses are already author per-gemgroup control-standardized
population means and receive no second log transform. Norman uses each
cell's full33,694-gene library denominator, CP10K, log2(1+x), construct mean,
then per-query control-cell mean/population-SD standardization. Yeast uses
log2 MaxLFQ protein measurements. These value spaces require separate
observation heads. Mode/time/source confounding remains explicit; sidecars
alone do not establish cross-mode transfer or justify pooling target values.

**Next falsifiable hypothesis, fixed before compute.** Learning to encode and
reconstruct fitting molecular responses will provide a more predictable
latent representation than direct intervention-to-response regression alone.
One new self-contained observed-state module retains the v3 forecast topology
and adds a training-only query-keyed response encoder. Its input is the masked
response relative to control, scaled by the shared fitting amplitude. The
encoder subtracts its zero-response output. The existing decoder reconstructs
the posterior response. Inference never accepts the perturbed target.

Total fitting loss is forecast Gaussian NLL +0.1 times reconstruction Gaussian
NLL +0.1 times squared distance between L2-normalized predicted latent delta
and stop-gradient normalized posterior delta, epsilon1e-6. Components are
reported separately. Early stopping remains forecast gene-macro NLL only.
Use the exact source55def snapshot, physical8M features hash2cbf1220...,64
basal tokens, query-response rank32, hidden/state128, seed731, batch64,
learning rate.0005, dropout.2, weight decay.1,180 epochs, patience30 and a
1,800-second cap. Accessible inputs remain protein/GO/physical features and
source control RNA; auxiliary targets are fitting-gene RNA only. The rule
remains .02 NLL improvement over mean and physical ridge, adjusted r at least
.10 in every source, and NLL/r nonregression against v2 physical128. This is
an auxiliary representation-learning experiment, not evidence of dynamics or
individual-cell generation. No result is available at protocol declaration.

**Degree and coverage controls: world exceeds both simple controls.** A
separate source-development diagnostic tests whether v2 physical128 performs
better than a ridge model using only log1p physical interaction degree, and
another using degree, graph coverage and protein-sequence availability.
Both use fixed alpha1 with fitting-standardized inputs and an intercept;
there is no alpha search. Three gene-grouped fitting folds estimate their
own residual variance, with the same core-control sampling model.

| Context | Degree adjusted r | Degree + coverage adjusted r | v2 world adjusted r |
| --- | ---: | ---: | ---: |
| K562 essential | .113547 | .127551 | .284144 |
| RPE1 essential | .177405 | .172785 | .315596 |
| K562 genome-scale | .064357 | .076946 | .104834 |

World MSE is also lower than both controls in every context, passing the
predeclared shortcut-control question. This does not change the failed
stronger feature-ridge advancement rule. The low-dimensional controls
themselves have nonzero adjusted correlation, so reporting only a constant
baseline would overstate the learned model's contribution. Same source
development outcomes as the neural runs; no new external outcomes.
CPU29.22 seconds, two focused covariate-selection checks pass. Report
`results/slp11-transition/degree-coverage-controls-v1/report.json`,
SHA-256 `a7f27b51b63c4729d6c4d4740b179e37911ea1bc02268de3d7ce631b0383bcf3`;
protocol `d8f4e0db8860b0a86b4ab3f85ab0cc12b00ee07389bb0e68ae08dc576ea54739`;
predictions `1ce1b3cd6e19efae9472ec3ac117dd099214afdb8960140e2ac38a9a0da98f65`.

**Queued model-specific uncertainty experiment.** The present three-source
world means use uncertainty inherited from fitting-only mean-model residuals,
whereas ridge uses its own residuals. A new calibration experiment will fit
v2 physical128 in three global fitting-gene folds, excluding each fold's
genes from every context. All response descriptors, action scaling, means
and shared amplitudes are refitted within each fold. Neural fitting uses61
fixed epochs, the selected full candidate's epoch count; no fold-held or
outer-validation outcome selects an epoch. Frozen full-model means remain
bit-identical. Only the measurement variance changes, using model-specific
OOF residuals and unchanged control sampling components. The existing .02
NLL margins and .10 adjusted-r minimum apply in each source. GPU cap1,800
seconds after the observed-state pilot. This tests uncertainty calibration;
it cannot fix the completed HepG2 mean-prediction failure.

**Response compression diagnostic and matched comparison.** A fitting-only
standardized response SVD retains28.37% of residual variance at rank32 and
37.05% at rank128. Projecting each measured validation response into this
basis gives oracle adjusted correlations .5575/.6895/.3522 in K562 essential,
RPE1 essential and K562 GWPS. The predeclared question required at least.8 in
every context, and fails. This oracle consumes the outcome being reconstructed
and is not a forecast. Lost observed variation includes possible measurement
noise; the result does not establish a biological representation deficit.

Review found an important comparison mismatch: the original diagnostic's
feature-to-latent ridge pooled contexts, while full feature ridge fitted each
context separately. A separate supplement corrects that mismatch with the
same frozen basis, features, alpha10000 and source rows. Projecting full-ridge
forecasts through the response basis is algebraically equivalent to fitting
context-local latent ridge; a synthetic explicit-fit check verifies this.

| Context | Pooled rank128 forecast r | Context-local rank128 MSE / r | Full feature ridge MSE / r |
| --- | ---: | --- | --- |
| K562 essential | .2221 | .025340 / .2712 | .025431 / .2580 |
| RPE1 essential | .2374 | .055837 / .3107 | .055860 / .3024 |
| K562 genome-scale | .08470 | .012210 / .09247 | .012346 / .07851 |

The matched low-rank forecast improves on full ridge, reversing the earlier
unmatched comparison. It still does not satisfy the diagnostic's .8-oracle
premise. Its gains motivate testing response representations, rather than
concluding that static features or low rank alone are inadequate.

Fixed cell-count bins [1,20),[20,100),[100,500),[500,infinity) show increasing
oracle correlations in the essential-gene screens; the highest bins contain
only3 K562 and5 RPE1 genes. GWPS remains .312/.360/.348/.381 across bins.
Genes can span bins through different constructs, so bins are not independent
replicates. The frozen control-sampling contribution averaged over validation
queries is .00893/.01663/.00797 squared outcome units. These variance estimates
are not a measured biological noise ceiling.

Original report `results/slp11-transition/response-compression-diagnostic-v1/report.json`,
SHA-256 `525cae361e2d7888a08ef70b4b5543bf13834b2a0467f55cff8d1c5fd638b51b`;
protocol `7898252301d5488fabd45fea1be5739acc6417e99c5ead4165b301435dfd6b9a`.
Matched/count-bin supplement
`results/slp11-transition/response-compression-count-bin-supplement-v1/report.json`,
SHA-256 `90c0fff0237d333da6d9fc4b84abeba75149a9624ddee6d6660fc74098d93aec`;
protocol `2a1748acca36f75922e635ef6601233f7716d4a8250857837a99707e930a2df6`;
matched predictions `1e97ecae8b485cd3eb7ed1b3a9f689e901c5c298e488a808b8b0f8052bbfda6b`.
CPU20.9+15.2 seconds; five focused checks pass. No protected or external
outcomes enter either diagnostic.

Clarification to the queued uncertainty protocol:61 epochs are inherited from
the full candidate's earlier development-selected checkpoint. Thus development
validation influenced this fixed hyperparameter historically; the claim that
it was excluded from all epoch choice was too broad. No fold-held outcomes
or new validation feedback choose epochs during OOF fitting. The calibration
arrays exclude all outer-validation intervention genes, and the result remains
adaptive development evidence. The numerical plan and fixed rule are unchanged.

Repository verification at this point:511 tests pass,21 skip,17 historical
PyTorch warnings;41.81 seconds. Later adapter additions require their focused
checks. No historical model code or protected benchmark was changed.

**Observed-state auxiliary experiment completed: reject this candidate.**
The unchanged declared objective fits770,432 parameters and selects epoch96
after773.98 seconds of local CUDA work. The forecast topology and its initial
parameters match v3 exactly; the additional response encoder is used only by
the training objective. The following are forecast metrics, not reconstructions
conditioned on the outcomes being evaluated.

| Context | Forecast NLL | Adjusted r | NLL gain over physical ridge |
| --- | ---: | ---: | ---: |
| K562 essential | -.543546 | .284244 | .005223 |
| RPE1 essential | -.252012 | .320829 | .003490 |
| K562 genome-scale | -.911729 | .100239 | .002423 |

Every ridge margin fails. K562 NLL regresses against v2 physical128; GWPS
regresses in adjusted r and misses the mean-model margin. Relative to the
isolated v3 decoder, essential-gene performance improves, while GWPS NLL and
correlation regress. Auxiliary reconstruction has not delivered the required
generalization improvement in this experiment.

At the selected epoch, row-weighted fitting loss components are forecast
NLL-.786314, reconstruction NLL-.817685, normalized latent MSE.005742 and
total-.867509. These training aggregates have different source weights from
the equal-source validation selection score and cannot directly quantify a
train/validation generalization gap. Empty-action identity remains exact;
source reload error is2.38e-7. A trap proves the response encoder is never
called during forecast reload. Fourteen focused checks pass.

Artifact root
`results/slp11-transition/human-gwps-fixed-context-observed-state-auxiliary-physical-state128-seed731-v1/`;
report SHA-256 `266152286bed210bc4bcf78d15de1fe3a10f00a47ce259e671e8888248d66d3f`;
checkpoint `62cd59591fb13160f5f96d0638d7afa8fb2dbdaaaf1ab99dae90ef10cedc7d7d`;
outer protocol `b4d5fecf3306f5788623f5a412826442ffdfad55a69b7a8a77e014ccb798d23a`.
The GPU is released to the separately frozen three-fold v2 uncertainty run.
No HepG2, Jurkat, protected molecular or SL outcomes enter this experiment.


### 2026-09-05 — Calibration result, landscape correction and paired modality data

**Frozen-v2 neural OOF calibration completed: reject advancement.** Three
fitting-gene folds each train for the predeclared 61 epochs, with fold-local
feature and response statistics. Total CUDA time is 302.86 seconds. No new
held-fold or development feedback selects epochs; the historical development
influence on the inherited epoch count is recorded in the preceding amendment.

| Context | Calibrated NLL | Gain over full physical ridge | Mean forecast changed? |
| --- | ---: | ---: | --- |
| K562 essential | -.553117 | .014794 | No |
| RPE1 essential | -.255993 | .007471 | No |
| K562 genome-scale | -.914383 | .005078 | No |

All .02-nat ridge margins fail; GWPS also misses the mean-model margin.
Biological variance totals relative to mean-model calibration are
.831/.851/.897; control sampling components have zero drift. Saved/reloaded
mean forecasts are bit-identical. Calibration cannot repair the already
observed HepG2 mean-prediction failure. Five focused checks and Ruff pass.
Artifacts: `results/slp11-transition/human-gwps-physical-state128-neural-oof-calibration-v1/`;
report SHA-256 `08999fb60dad104d9992185af9af43f938a162347da89b2cd375531958cd36f2`;
protocol `feb8d7f5115471262f5238b767d164c3327c0eaa659f5a4ee960f8e6f39a3874`;
pre-execution clarification `714421df69c5a568e7a290efe0a55f1545f676c6d840efea9434c4ea1eaa0afb`;
calibrated predictions `1015d68d7d4c6fbf161074311094c665f9a576bc6ed73e312b125110ce033500`.

**Independent source landscape audit.** Collapse constructs equally per gene,
remove each prediction matrix's own per-query centroid across validation
genes, remove the truth matrix's corresponding centroid separately, then
compute Pearson across queries within each gene and average genes equally.
No fitting or hyperparameter selection occurs. All source validation outcomes
were already development data; this is an adaptive diagnostic.

| Candidate | K562 essential r | RPE1 essential r | K562 genome-scale r |
| --- | ---: | ---: | ---: |
| Control v2 | .294913 | .306128 | .107899 |
| Linear state-difference v3 | .288597 | .302147 | .108239 |
| Observed-state auxiliary | .290936 | .314163 | .103227 |
| Full physical ridge | .269269 | .315038 | .084352 |

On the same collapsed profiles, common-fitting-centroid RPE1 correlations are
.324444 for v2 and .310897 for ridge. The apparent advantage reverses after
independent centroid removal. This change is not explained solely by
construct aggregation. The previous candidates already failed; those decisions
stand. Future candidates must also avoid regression against full ridge under
the independently centered gene-profile metric in every source.
Report `results/slp11-transition/source-landscape-centering-audit-v1/report.json`,
SHA-256 `32b46978eb47827f74049170eaf083926d59b8bfc6f96693f416b09b70638777`.

**Frangieh paired RNA/protein adapter completed.** Processed scPerturb v1.3
Frangieh2021 counts are CC BY 4.0 and upstream-MD5 verified. RNA SHA-256 is
`cc42ef38bcf703a00e0c77c7945dd53159b12d814c438ae8afeaed9bc71f48d1`;
protein SHA-256 is
`1f85827b5afad11a30d8ac99399772231110a1c3723bed6ecf7981a12cc3dbcc`.
Raw DUOS-controlled material was not accessed. Exact paired barcode order is
verified across 218,331 cells. Development retains 1,399 fitting and 403
validation guide-resolved pseudobulks, 151/43 intervention genes, 84,121 target
cells, 18,063 stable RNA genes and 20 molecular antibody channels. Control
profiles use 39,347 verified all-nontargeting/no-guide cells. All 21,000 test
cells are excluded before matrix-value reads. A further 73,863 cells are
quarantined for mixed targets, guide disagreement, unresolved or truncated
assignments; these are not relabeled as clean single interventions.

RNA values are per-cell ln(1 + 10000 * count / sum of all 23,712 retained
source columns), then averaged per guide group. Direct selected-row sums
match source ncounts exactly; this is not claimed to be an unfiltered full
library denominator. ADTs use the authors' max(0, ln((target UMI + 1) /
(matched-isotype UMI + 1))) transform. Four isotypes remain QC channels.
Stable ADT assay identifiers are not forced onto single gene identities.
Paired means are observations of RNA and protein in the same endpoint cells,
not before/after measurements of those cells. Co-culture reflects surviving
cells, and source-wide filtering limits evidence to retrospective analysis.

Development `data/derived/slp11-frangieh/paired-development-v1/development.npz`,
SHA-256 `4bbb1eec9ede66211f1316b2841bb0037032ef975cd6c92d34aba0adb5fed744`;
protocol `7a45073599c5104f7b7d39550d17a0a0455f4466b81741484d2a8c67e814940f`;
manifest `9f4c47d86473ab704a4313e31529ed908a34c2fa8a6cd0376db92556b9c4bb4f`.
Seven adapter checks and Ruff pass.

**Next controlled neural hypothesis.** The bilinear observation decoder
restricts intervention-response matrices to the latent width. Replace only
this decoder with a 64-hidden-unit nonlinear shared observation function,
using D(basal + intervention_delta, query) - D(basal, query). Action/context/
transition encoders remain v3 topology, hidden/state 128, physical1156 plus
response32 queries, uniform rows, seed731, batch64, lr .0005, decay .1,
dropout .2, 180 epochs/patience30 and 1,800-second GPU cap. Exact data
55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c and
features 2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7.
Advancement retains .02 NLL margins, .10 adjusted r and v2 nonregression in
every source, and adds the independently centered ridge nonregression gate
above. Profile before execution; this tests a nonlinear measurement map,
not identified temporal dynamics. HepG2, Jurkat and benchmark outcomes do
not enter this experiment.


**Nonlinear observation decoder completed: reject.** Revision v4 selects
epoch96 after649.55 seconds of CUDA training. NLL/adjusted r are
-.544380/.28319 (K562 essential), -.240803/.29142 (RPE1 essential), and
-.913206/.10268 (K562 GWPS). All required ridge NLL margins fail; RPE1 is
worse than ridge. Independently centered gene-profile r is
.289794/.291271/.104618, versus ridge .269269/.315038/.084352. The added
RPE1 gate fails. Relative to the isolated v3 decoder, K562 improves, RPE1
regresses, and GWPS is nearly unchanged with slight regression. This run
does not support observation nonlinearity alone as the missing ingredient.
Exact empty-action identity and direct nonlinear decoder consistency pass;
source reload error is5.36e-7. Output root
`results/slp11-transition/human-gwps-nonlinear-decoder-physical-state128-seed731-v1/`;
protocol SHA-256 `cf190316810cff60720ad34b67a5d6398911009b0cbf8b5f64067acd164d8fc0`;
report `afd9a5a1c2f2da7e43549584f41ff57a59f95c63f79fb90231759da790919b99`;
checkpoint `b3f24b2c87b15310ba2155e32f7fc5a604a4123390415733580710593949be29`.

**Frangieh baseline scoring correction.** The first static-vs-target-basal
ridge runner mislabeled ordinary within-profile Pearson as independently
query-centroid-adjusted Pearson: it subtracted row means, but omitted the
required per-query centroid across genes. Review caught the error before
using the result to advance a model. Original predictions and report are
preserved; a separate immutable scoring correction removes the prediction
and truth query centroids before Pearson. No model is refitted.

Corrected static -> static+target-basal r:
Co-culture RNA .01458 -> .01468, ADT .03263 -> .03414;
Control RNA .01583 -> .01594, ADT .07319 -> .07384;
IFNg RNA .02465 -> .02464, ADT .08244 -> .08294.
The mean model now correctly has undefined landscape correlation for all43
genes per stratum. Every MSE gain remains below1%, so the original no-advance
decision stands, with0/6 passing strata. The high earlier .99 correlations
were shared abundance profiles, not intervention-specific prediction.
Original report SHA-256
`95ba68ff7e54af7496091eaba8da083acf4540933d2817423ca0adc139359cfd`;
corrected scoring report
`results/slp11-transition/frangieh-target-basal-ridge-corrected-scoring-v1/report.json`,
SHA-256 `7b34b443b9969ca197e5cdc808bbf62a825388a31217e1baed66466beef5509a`;
correction protocol `ae779fd694d776f07ec0a89fe12515e1348d276aae19ff4cf90f772f3dfb292d`.
Focused checks cover arbitrary shared-query landscape invariance, undefined
constant predictions and rejecting test genes from a development input.

**Paired-state pilot specified before CUDA allocation.** New self-contained
`modules/slp-1-1-paired-state-v1/` encodes molecular controls into a common
state, applies an intervention-feature transition, and decodes RNA and ADT
through separate nonlinear observation heads. Each head computes the
observation difference between changed and basal states. RNA queries and
actions use specieswide static1156 features;20 ADTs use fixed assay-component
one-hot descriptors. This is a fixed measurement panel, not a learned gene
vocabulary. No extrapolation to unseen antibody components is claimed.

Hypothesis: this shared endpoint state predicts unseen intervention genes
better than fitting means and context-local static ridge in each of the
three Frangieh environments and both modalities. Fixed advancement requires
at least1% raw-MSE improvement over every mean/base577/physical1156 comparator,
query-centroid-adjusted gene-profile r >=.10, and no r regression against
any defined static comparator, in every one of six strata. This is a first
paired endpoint pilot, not a replacement for the molecular program's existing
source-transfer requirements or evidence of benefit from joint supervision.

Exact paired development data are4bbb1eec9ede66211f1316b2841bb0037032ef975cd6c92d34aba0adb5fed744.
The pending specieswide feature pack will be checksum-pinned in protocol.json
before execution. Train151 genes/validate43 in each of three environments;
collapse guide pseudobulks equally within gene/context. Hidden64/state32,
nonlinear decoder32, dropout.2, batch32, seed731, AdamW lr.0005/decay.1,
gradient norm cap1, at most180 epochs with30-epoch patience and1,200-second
training cap. Each step samples1,024 RNA queries uniformly and observes all20
ADTs. Loss averages per-query fitting-SD-scaled MSE equally between modalities;
SD floor.05. Selection uses the same objective over all validation queries.
Controls encode128 RNA tokens chosen by control-only across-context variance
plus all20 ADTs, with equal modality weighting. Fitting-only feature statistics
and shared query amplitudes accompany the artifact. Six initial module checks
pass. This model has no calibrated likelihood, temporal dynamics or identified
single-cell transition. Protected and external test outcomes remain excluded.


### 2026-09-05 — First paired molecular model and a fixed species-wide graph

**Static feature extension completed.** New Frangieh/static union includes
18,893 human ENSG rows with1,156 dimensions. All18,063 RNA queries and237
action genes have explicit rows;16,189 queries and230 actions have peptides.
Missing proteins remain explicit. All10,231 old rows preserve their first577
ESM/GO dimensions bitwise. Neighbor aggregation uses a separate pinned23,879
translated-gene Ensembl116 universe; it no longer depends on requested query
rows. This deliberately changes physical features for1,385 old genes.
Full reconstruction, deterministic serialization and subset invariance pass.

Feature artifact
`data/derived/slp11-frangieh-static/ensembl116-goa2022-fixed-neighbor-v1/frangieh-extended-static-esm-go-fixed-physical-features.npz`,
SHA-256 `347fd1bf87d8fc3d0b447676082b4bcb64f021c9f12c7df4d1754dc262b2bf72`;
manifest `5fc95466a547b76b43f4a7223066c119c37abd3fef69a5bf5460ea3cfe245e9c`;
verification `25d3e1af2e348cb6fd66200464de3f227285d876559c02b9e88dd330783db7f4`;
fixed graph base `f4bbfe62b73cf6362170996fcf34200cea68da106d687d3c9e994e709e951f40`.
Successful build104.5 seconds, including72.7 seconds CUDA for13,800 new genes,
15,921 entity windows/13,832 unique peptide windows. One earlier73-second
extraction ended at a downstream wrong GO path; it is retained as failed
work, and a checksum-verified intermediate cache now prevents repeated
extraction on downstream retries. Thirteen focused checks pass.

**Paired static baseline completed: no general physical-feature gain.**
Fitting-only three-gene-fold CV chooses alpha from.1 through1,000,000 plus the
exact mean limit, separately for base577 and physical1156 in each context/head.
Only Control ADT passes the fixed1% MSE and r-nonregression rule:1.99% gain.
Other five strata fail; all194 development genes have explicit feature rows.
RNA CV often selects the mean limit or very strong shrinkage. Source graph
coverage alone has not supplied useful general RNA prediction here.
Report `results/slp11-transition/frangieh-specieswide-physical-ridge-v1/report.json`,
SHA-256 `af1fdb5c00e0bd9d974fb14b4c3ac33e4776093f994e590880cdb46c2d756af7`;
protocol `73eb94895863d010966e22d1668df968accf47c1d8c96d2da59e57ecf799d066`;
predictions `1e342a75e4a1cc67d6d0a6e3c1e4acefb95d7a51fad7a1bf47fcbff978c7abfe`.
CPU9.80 seconds; eight focused checks pass.

**Guide agreement and estimator diagnostic, fitting genes only.** In
Co-culture/Control/IFNg, gene-matched independent guide-side RNA landscape
r is .0455/.0242/.0476, versus shuffled -.0044/-.0008/-.0015. ADT r is
.1449/.0940/.1099. The deterministic split alternates sorted guide groups;
150/151/151 fitting genes have at least two groups. Smallest-side cell counts
below20 correspond to much larger errors; equal weighting of rare groups
amplifies sampling noise. This is agreement between different guide and cell
populations, not biological replicate evidence or a biological noise ceiling.
Report `results/slp11-transition/frangieh-guide-reproducibility-v1/report.json`,
SHA-256 `d521035114e79304edeebfd5784446ddd26c4d31cda27cdf1dc9fd9a0e6fbafc`;
protocol `ddfdea8bb2be50104cfb997cf86bc7e28c51004dcf1a13b46044a249596d0730`.

A separately frozen fitting-only diagnostic weights guide means by contributing
cells within the same guide sides. MSE improves13.44–30.47% in all six strata,
but IFNg ADT r falls .10993 -> .09875, failing the all-stratum rule (5/6 pass).
This estimator targets the sampled-cell population; the existing equal-guide
pilot data and decisions remain unchanged. Report
`results/slp11-transition/frangieh-guide-cell-weighting-v1/report.json`,
SHA-256 `8569ab235f25ec4e051db5b9c98c03f827960e2cf0709962c14cf497e570f9a9`;
protocol `c8ef00f5ba46c125959b60e91856ac244ca5208b7ff7b4acc2b670d6c6b35494`.

**Paired state model completed: reject,0/6 strata pass.** The predeclared
244,096-parameter model selects epoch10 and stops at40 under fixed patience.
The successful replay takes11.50 seconds including final checks. An earlier
run completed the same training but failed during target-free reference
materialization because a helper still expected a targets key. A focused
regression check and one-line fix remove that dependency; the deterministic
replay's checkpoint is byte-identical to the retained first checkpoint.
Neither architecture, objective, selection nor input data changes in replay.

| Environment | Modality | Model MSE | Landscape r | MSE improvement over physical ridge |
| --- | --- | ---: | ---: | ---: |
| Co-culture | RNA | .00212584 | .00552 | -.532% |
| Co-culture | ADT | .02546486 | -.05537 | -2.265% |
| Control | RNA | .00226308 | .00350 | -4.423% |
| Control | ADT | .01767453 | .10501 | -.848% |
| IFNg | RNA | .00201104 | .01105 | -2.881% |
| IFNg | ADT | .01900680 | .00720 | .559% |

Only Control ADT reaches r>=.10 and still fails its MSE comparison. Gene-paired
bootstrap1,000/seed731 intervals for its apparent gains over mean/base include
zero. RNA losses against comparators have wholly negative improvement
intervals. The model has learned no adequate unseen-intervention RNA map in
this small, noisy population. Joint supervision benefit is not established.

Model artifact
`results/slp11-transition/frangieh-paired-state-physical1156-seed731-v2/`;
protocol SHA-256 `4b611f9c3194a7383a2c3f2b4b484cab07944e24f507dfb2dead874a41242078`;
report `9aafc148a693a31a3f40c66a34375e213dab87fb9a200b9a02c1199e115579de`;
checkpoint `a3b57cea1755f70a144f7590d55a9ac11789633d06c8208a3ad39aee1a66b667`;
predictions `36ebe74677f7bb75e467bf8f225cc313417590772de356f98470d32a5e26b50b`;
reference `8b82e4781b73a721f995dd218ef341ea8324b87d3c9189bfe40644d436800e73`.
Independent scoring report
`results/slp11-transition/frangieh-paired-state-vs-static-scoring-v1/report.json`,
SHA-256 `d0c577e093198e9060a582cc5852b0db61246daa5772ae0c1e8451addc584b90`;
scoring protocol `6bda6a78d4115709c420ea2481cfc53f376871228ab11856294b7414470ecd3e`.

Portable CPU inference reloads source, weights, feature transforms, query
features and control normalizers from the artifact, accepting raw intervention
features and explicit control means. Three source-context profiles match CUDA
forecasts to2.38e-7 RNA/2.98e-8 ADT; query chunk drift1.49e-8; empty-action
identity exact. Verification SHA-256
`62bc52645e7ea1adcde545c33bae7d410579b52fee8f48f1b59b48e1a1ab9000`.
This engineering success does not imply molecular accuracy or OMF release
eligibility. No final holdout, Jurkat or application benchmark is opened.

**Next source-world experiment frozen.** Increase only fitting-derived response
query rank32 ->128 in the original v2 physical1156/state128 model, using exact
frozen v2 source fdb4555bd0f7c0a0786539da67048f6985f4ec2f36ef7aa45bd22c7c6bfbb2ef,
data55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c,
features2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7.
Hypothesis: the smaller response descriptor restricts learned query behavior;
the preceding matched ridge compression results motivate this fixed test.
Uniform rows, seed731, hidden/state128, batch64, basal64 tokens, lr.0005,
decay.1, dropout.2,180epochs/patience30/1,800-second cap. The existing all-source
NLL/.10-r/v2 nonregression rule and separate-centroid ridge gate both apply.
No feature/source/candidate is selected on HepG2 or Jurkat outcomes.


**Response-query rank128 result: reject.** The frozen v2 core selects epoch43
and completes in359.20 seconds. NLL/adjusted r are -.541842/.284365,
-.250626/.323769 and -.912958/.103958 in K562 essential/RPE1 essential/K562
GWPS. Every .02 ridge margin fails; essential-context NLL regresses against
response32, and GWPS adjusted r regresses. Independently centered r is
.290726/.310331/.110464; RPE1 still trails full ridge .315038. No all-source
improvement is established by widening this descriptor.
Root `results/slp11-transition/human-gwps-control-v2-response128-seed731-v1/`;
protocol SHA-256 `69ba515fda6148052b99008420738c92ca556a7d845d90aa4d724efe8e835812`;
report `9da205caef08c9ec317f496b16ab7ea393fa3c4dafee6c2fde038938db12724b`;
checkpoint `42171a72d9412b8a8fc1f97f7d6a44cf335cfb17a023535c32f2f30c07989985`;
summary `feae2fc83c0fe38f7a0bd81eead3838e4bc7e9115d73568d257fafbf6e9d1822`.
Empty-control mean/scale identity remains exact.

Repository verification at this stage:577 tests pass,21 skip,17 historical
PyTorch warnings,42.07 seconds. Newer graph/adapter additions receive their
focused checks separately. The paired-pilot figure is available as
`results/slp11-transition/figures/frangieh-paired-pilot-v2.png` and `.pdf`;
its intervals use the frozen gene-paired scoring report.

**A third human cell line is admitted to adaptive development.** The completed
HepG2 frozen diagnostic is permanently retired as confirmation evidence. A new
four-context snapshot appends its hash-train and hash-validation records to
the three Replogle source contexts, while excluding390 hash-test populations.
It is not permissible to claim a future HepG2 result is unseen-context
confirmation after training or selecting on this snapshot. Jurkat remains
unopened for future frozen context transfer.

Valid artifact `data/derived/slp11-human-four-context-v2/development.npz`,
SHA-256 `ffe158aaed370e48d384c2970211bd266ef287630cb5382d56c3f7d6083007cf`;
15,212 records by7,036 RNA queries,12,477 fitting/2,735 validation/zero test.
HepG2 contributes1,758 fitting and396 validation records,6,789 observed queries
and247 missing. Every inherited Replogle prefix retains exact dtype, shape
and logical bytes, including controls and split indices. Stable intervention
identities do not cross fitting/validation partitions. Target spaces are
explicitly context-indexed: Replogle author core-control-z summaries and
SLp-computed HepG2 per-GEM control-z means are not declared identical assays.

The724 Replogle independent control pseudobulks are unchanged. HepG2 has no
corresponding independent target-space control pseudobulks in this artifact;
its4,976 controls/56 GEM normalizer statistics remain separate and are not
fabricated into compatible uncertainty observations. This dataset is not a
drop-in replacement for the existing three-context exposure launcher. Future
training requires an explicit objective and uncertainty policy.
Protocol `93e7bca5c3a56c1bbc1d1daa88af1756e2d6efeb5731225bda00120218d3ff60`;
manifest `2a20d4b79e9af899ed7db2a101d39496f042ae9dd0287c3cde0e51abfcfbdebf`;
verification `930dc1bcf63793e3b72254a17d986780fde4dce0d4de69ab82490202796a709b`.
The initial v1 write changed inherited scalar label shapes and failed its
exact-prefix check. It is retained as FAILED_VERIFICATION and superseded by
v2, never used for training. Four focused checks and Ruff pass.

**Published-architecture comparison: first adapter invalidates BatchNorm use.**
Scouter author code is pinned at0cfddd000e19b72ff033ba67c8315f7bc3304932,
reproduction code at6f2c83e5a32505038060155ca8257fa094732e35 (MIT). The baseline
uses its full-panel control compressor/generator widths, separately fitted
per source context, replacing GenePT with the same static1156 features and
using our pseudobulk exposure NLL. This is explicitly an adaptation, not a
reproduction of published Scouter performance.

The32,420,284-parameter context models select epoch1 and fail badly under the
author-default BatchNorm configuration. Run31.515 seconds; NLL is
3.39535/2.51400/24.19367. A fitting-only train/eval diagnostic identifies an
adapter mismatch: one identical pooled control is repeated throughout each
context's training batches, so its control-encoder batch variance is exactly
zero while inference uses running statistics. Same-batch train/eval control
state RMS differences are3.06/3.37/16.67. All losses remain finite and weights
are unchanged by the diagnostic. This evidence does not isolate the learning
rate, scheduler or loss as the cause, and is not evidence against the published
single-cell method. A separately frozen correction will use the author's
LayerNorm option, all other numerical settings unchanged.
BN report `results/slp11-transition/human-gwps-scouter-adapted-physical-separate-context-seed731-v1/report.json`,
SHA-256 `8cb3b87567e3bb671335b3bff842154335338a752aae6299969588d4f0d6df2f`;
protocol `5bc8df736f98187b4491bd61cc67fb7fe0f9bcfd40c26f1f1c2cbf80e1248273`;
BN diagnostic `09f013b90d1dda834cdea5f1606f6127bb1842b2f53e8dc2b677d9665994d48b`.

**Explicit per-gene state pilot prepared.** The new application-neutral core
encodes measured basal RNA into16-dimensional gene states over a pinned
24,019-node graph, with83,264 directed normalized physical edges. Gene indices
locate interventions in supplied data, not trainable identity embeddings.
The unit action value means intervention presence, not measured100% knockdown.
Two sparse residual message steps update local state; a separate16-dimensional
global route permits responses beyond the physical graph. Nonlinear molecular
observations subtract the same basal-state observation, preserving exact
empty-intervention identity. This tests a joint architectural change, not an
isolated causal effect of graph messages, and not temporal dynamics.

Actual-graph GPU profiles select batch64 by the predeclared operational rule:
6.05 GiB reserved and316.5 examples/s versus batch32 at233.4 examples/s.
A32-epoch, patience10 pilot at lr.001/AdamW decay.1, state16/hidden64, no dropout,
uniform-row fixed v2 Gaussian scales is projected around1,148 seconds including
full validation, within1,800 seconds. No profile uses intervention outcomes.
The same source55def snapshot, mean/ridge/v2 comparisons and independently
centered landscape requirements apply. Frozen protocol and execution follow
focused runtime/reload review; this module has not yet shown biological gains.


**Scouter LayerNorm correction completed: stable, still reject.** The author's
supported LayerNorm option replaces BatchNorm in an isolated v2 adaptation;
widths, static features, pseudobulk endpoints, Adam lr.001/exponential.9
scheduler,180-epoch maximum, fixed patience and per-context caps are unchanged.
Identical fitting batches now have bit-exact train/evaluation states and
predictions; no BatchNorm buffers remain. This resolves the demonstrated
constant-control normalization mismatch.

Selected epochs20/9/20 give NLL -.502690/-.216914/-.895828, adjusted r
.229068/.266278/.083514 and independently centered r .243869/.287205/.085868.
All ridge-margin and v2 nonregression gates fail. Only GWPS avoids independent-r
regression against ridge, and it remains below.10. This negative result is
for the explicitly adapted comparator, not the original GenePT/single-cell
Scouter experiment. Runtime43.469 seconds,32,420,284 parameters per context;
20 focused checks and Ruff pass. Source reload error is at most5.25e-6.
Artifact root
`results/slp11-transition/human-gwps-scouter-adapted-layernorm-separate-context-seed731-v2/`;
report SHA-256 `627751d579f7779ada0f5f854583bfeb5527904ff120cff8198334e739679528`;
protocol `3370ed0919c94842d4f296d7f1ada546b68b792d4325b17518402070b0bb09bb`;
normalization diagnostic `b647d969f5b21c91275fed396124543fb3f2f8ca4003b029d95d6f7574359f43`;
verification `53b7cd90f28b330113192c5858968c795a3fe152d3e7d58fad2dd0d9667963a5`.


### 2026-09-05 — Explicit gene states and nonlinear baseline results

**Gene-state static577 pilot: reject.** On the pinned source-three human CRISPRi
snapshot (`55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c`),
the 24,019-node, two-message-step model completed 20 epochs and selected epoch
10 under the fixed patience rule. Training took 463.97 seconds on one RTX 4070;
there are 21,361 parameters. All three contexts fail the fixed advancement rule.

| Context | Gene-macro NLL | Training-centroid adjusted r | Independently centered r |
| --- | ---: | ---: | ---: |
| K562 essential | -0.487588 | 0.161257 | 0.160445 |
| RPE1 essential | -0.138994 | 0.157273 | 0.125992 |
| K562 GWPS | -0.906424 | 0.095262 | 0.097374 |

The model regresses against the frozen v2 candidate in every context. This
joint architectural test does not establish that physical message passing is
harmful: it also replaces the encoder, latent width and decoder, and omits
v2's response-derived query descriptors. Empty-action identity is exact;
target-free source reload differs by at most 7.15e-7. Seventeen focused tests
and Ruff pass. Artifacts are under
`results/slp11-transition/human-gwps-gene-state-base577-state16-seed731-v1/`.
Report SHA-256: `1eeff9809849c40fea25b733a167d77498e7a5aa88fb1f2d128f954846dddc5b`;
protocol: `4c676088aee64cbb5953d790fd74f67a7423042b04be704744d957ca3c577000`;
model: `a01d0b37af9ba7c9334c390f6010624ea560d01539093d6bd35b24eae52c573b`;
predictions: `bc312427b8dfe385fdcc4f493b2b49f476fd249f7fe6064e864fa80f910fdb59`.

**Next fixed hypothesis.** Adding the already frozen, fitting-derived 32 RNA
response descriptors and an availability flag to the same gene-state core
will pass the existing mean/ridge/v2 advancement checks. The new 610-dimensional
input preserves all original 577 features, graph edges, identities and maps
exactly. All 7,036 query nodes have descriptors; 5,740 of 8,358 unique action
nodes have them. Missing descriptors are zero with a false availability flag.
This modifies shared node/action/query representations together and is not an
isolated decoder test or purely static prior. No new response basis was fitted.
The graph SHA-256 is
`d0a23e4ee3569fd7ace543278ee576d6ffa098a280e863cc3bf09acde2d6c2d2`;
its verification is `92532775eec1c7a8fa4ff8eaa913eadcefd25171f9630e786032a0c948a825ca`.
Keep the same seed, batch 64, 32-epoch maximum, patience 10, optimizer, fixed
Gaussian scales and 1,800-second cap. No held-out context outcomes enter this run.

**Nyström RBF baseline: mixed gains, reject overall.** A 512-landmark RBF
feature map uses fitting-only standardization, median-distance bandwidth and
kernel eigensystem, with context-local output heads. Three global gene-hash
inner folds repeat all fitting transformations. Alpha is selected from the
fixed grid plus the exact mean limit; evaluation collapses identical genes
with equal construct weighting. This is a mean-prediction baseline, without
new likelihood or uncertainty claims. CPU execution took 30.5 seconds.

| Context | MSE improvement over full physical ridge | Independently centered r, kernel / ridge | Fixed decision |
| --- | ---: | ---: | --- |
| K562 essential | 3.895% | 0.268840 / 0.269269 | Fail correlation nonregression |
| RPE1 essential | 3.784% | 0.315091 / 0.315038 | Pass |
| K562 GWPS | -0.365% | 0.085910 / 0.084352 | Fail MSE improvement |

The fixed rule requires at least 1% lower MSE and correlation nonregression
in every context. It fails; the K562 essential correlation difference is
small and is not presented as established biological inferiority. The lower
MSE in two contexts nevertheless warrants retaining this stronger comparator.
Model reload, input hashes and identities verify; four focused tests pass.
Artifacts: `results/slp11-transition/human-gwps-nystrom-rbf512-physical-seed731-v1/`.
Report SHA-256: `ceff4ea924df07dd930b980929c9227a6719421673974fbc0c065b3deac1184e`;
protocol: `2e15d9518c81a83cad94f09e6b298aac2b1f3ef37c11407b5eb65f7019df98e6`;
predictions: `7446d670a1897287e62bf84f74d0f6bc8383a520d1e7b483f4e66753a0dc6da6`.
These are adaptive development comparisons, not untouched confirmation.


### 2026-09-05 — Four-context point baseline and static BP preparation

The fixed physical-feature ridge reproduces the original three source contexts'
float32 predictions exactly, while extending fitting to the retired HepG2
adaptive dataset. No Gaussian model or invented HepG2 sampling-control variance
is used. All 8,358 intervention genes have the pinned physical features.
Evaluation averages constructs equally within each gene, separately by context.

Ridge MSE improves over the fitting mean by 13.57%, 15.68%, 3.58% and 10.37%
in K562 essential, RPE1, K562 GWPS and HepG2, respectively. Independently centered
correlations are .26927, .31504, .08435 and .23638. Three of four contexts pass
the fixed 1% MSE improvement and r >= .10 rule; GWPS fails the correlation
threshold. HepG2 ridge MSE is .05787372 versus mean .06456706, on 361 validation
genes represented by 396 constructs. Its fitting set has 1,665 genes and 1,758
constructs. These HepG2 results are adaptive development, not confirmation.

Valid artifact root: `results/slp11-transition/human-four-context-physical-ridge-v2/`.
Report SHA-256: `b88fc44c76a99318942d783041352d588388a0473e57211fb4d360f833158a72`;
protocol: `605f2be4ef4afe8dbc905a977e36f05ef41985deea7db5323f6b631d0f26ecf0`;
predictions: `0c40ed63c336d5fb1795466693c733711150ec6de84d9fc21585f1d38fe57bc0`.
The v1 report is superseded because float32 residuals made a mathematically
constant mean comparator appear to have a tiny finite adjusted correlation.
Predictions are unchanged; v2 correctly reports undefined mean correlations.
Five focused tests and Ruff pass.

A matched source-three/source-four neural experiment is now specified: frozen
v2 architecture and response descriptors, identical initialization and 12,000
optimizer steps, explicit fitting-SD-standardized mean MSE, no early stopping
or checkpoint selection. The added-HepG2 hypothesis requires at least 2% lower
HepG2 MSE than both the matched source-three arm and within-HepG2 ridge, with
centered-correlation nonregression; each original source must avoid MSE and
centered-correlation regression. A separate all-context baseline rule will be
reported. The new objective and adaptive HepG2 fitting are explicit changes,
not a continuation of the previous Gaussian likelihood experiment.

**Archived biological-process descriptors prepared.** The same September 2022
GOA source yields an additional static BP128 representation and presence flag.
The vocabulary (8,145 direct terms) and SVD are fitted on 6,866 source-three
fitting intervention identities, without molecular response values. Coverage
is 92.19% of fitting interventions, 92.69% of validation interventions and
91.22% of RNA queries. The retained evidence categories exclude IMP, IGI, IEP,
HMP, HGI and HEP, negated annotations and post-cutoff dates; this does not imply
that every retained annotation is experimentally established or causal.
The existing MF/CC rights record remains unchanged; BP has its own scoped
rights record. No OMF admission is claimed.

Feature SHA-256: `b29cbd70f08e227cddfc013e66cd1032212c8cb62e6e25162965a57101cd1fac`;
basis: `cc8b8e16176623778b065c92c3eb22e5b28bdd40d6d84594c379c8bab7ae2d9e`;
verification: `f5252de88b70849b2860485dbe2f5eaa48163c198e91980986bb90def7ffb7ae`.
Build took 4.06 seconds on CPU2. Four focused tests and exact repeat/projection
checks pass. A fixed matched-ridge feature screen is the next small test;
no neural improvement is claimed from coverage alone.


### 2026-09-05 — Response-augmented gene states: reject and shelve this configuration

The fixed static577 + response32 + presence pilot completes 19 epochs and
selects epoch 9, taking 442.64 seconds on the RTX 4070. The 21,889-parameter
model improves upon static577 NLL in all contexts, but fails the unchanged
advancement rule in every context.

| Context | NLL | Training-centroid adjusted r | Independently centered r |
| --- | ---: | ---: | ---: |
| K562 essential | -0.509165 | 0.196777 | 0.207706 |
| RPE1 essential | -0.152152 | 0.174359 | 0.180334 |
| K562 GWPS | -0.909804 | 0.092267 | 0.112006 |

Response descriptors improve NLL over static577 by .02158, .01316 and .00338
in the table's context order. Nevertheless, essential-gene performance remains
substantially below ridge and v2. GWPS independently centered correlation
improves, while its training-centroid correlation regresses against static577.
The descriptors are fitting-derived quantitative features, not new static
biology or evidence of generalization to unmeasured readouts.

After checkpoint selection and terminal scoring, a separate diagnostic uses
only the first 128 fitting rows per context. Disabling the global route causes
prediction RMS changes .03949/.03672/.01557 in table order; disabling the local
route causes .00737/.00687/.00488. Both route removals worsen fitting NLL, with
larger changes for the global route. This establishes reliance within this
particular trained model, not causal biological pathways or held-gene benefits.
The audit did not select a checkpoint or use validation ablations.

Decision: retain the implementation and evidence, but shelve this 16-dimensional
physical-graph configuration. Its compute cost and current accuracy do not
justify further width/learning-rate sweeps. Continue the matched data-context
experiment and the separately frozen functional-feature screen instead.
Empty-action identity is exact; reload drift is at most 1.79e-7; 21 focused
checks and Ruff pass. CUDA is released.

Artifact root: `results/slp11-transition/human-gwps-gene-state-response32-state16-seed731-v1/`.
Report SHA-256: `5ded419bca65c9dff6b88d0c4c65897b8d18d4431c1278d7027236295f14cf7a`;
protocol: `8ee6ed850fd4fa9c9bc271394422efbcaa9dd81cfbb76778c769530aa7921e22`.
Route audit: `results/slp11-transition/human-gwps-gene-state-response32-state16-route-audit-v1/report.json`,
SHA-256 `74a8d58bab6b30a092454d1993b0076ad9b66748dabf5792f56e10f13f320769`.


### 2026-09-05 — Functional features improve two contexts but do not advance globally

The fixed BP128-plus-presence screen compares matched context-local ridge
models with physical1156 versus physical1285 inputs. Both choose alpha 10,000
in all three contexts through the fixed inner-fold procedure. The BP basis is
frozen and response-free; ridge statistics and targets are fitted within each
fold. Fitting collapses constructs equally per gene, explaining the small
differences between the matched physical arm and the historical row-fit ridge.

| Context | BP ridge MSE | MSE gain over matched physical | Centered r, BP / matched physical |
| --- | ---: | ---: | ---: |
| K562 essential | .02422074 | 1.488% | .280553 / .269423 |
| RPE1 essential | .05308906 | 1.532% | .324208 / .314099 |
| K562 GWPS | .01186866 | .314% | .086451 / .084916 |

Both essential contexts pass. GWPS fails the 1% improvement and r >= .10
requirements, so the overall decision is reject. Runtime 74 seconds on CPU2;
six focused tests and reload of all six models pass (maximum drift 1.42e-6).
Valid root: `results/slp11-transition/human-gwps-bp-ridge-source3-seed731-v2/`.
Report SHA-256: `8a3d1ba2265dc09bf6856c97c7a791775ef3282594beed269f708f353d895a0a`;
protocol: `14235464544f229cef047002732c3f2957fb9e95c439ca775eab3d3e688118e2`;
predictions: `f88efe29faccddbe93a7af1c3e95210b615d9235a3f9ad7d6f9de8530fec498f`.
The v1 attempt stopped at a comparator-schema assertion before numerical fitting.

A separately frozen test asks whether BP information complements the nonlinear
Nyström baseline. With the same 512-landmark design and fitting procedure, the
BP-augmented kernel selects alpha 1 in all contexts. It improves MSE by 1.169%
and 1.305% over the original kernel in K562 essential and RPE1, with centered r
improving to .279086 and .325863. Both are approximately 5% lower MSE than the
historical physical ridge. In GWPS the kernel MSE improves by only .484% and
centered r regresses to .077974. The fixed all-context rule therefore rejects
this candidate too; it does not authorize default replacement or a launch claim.

Runtime 31.64 seconds on CPU2; seven focused tests pass. New-model reload drift
is at most 3.37e-5. The only change to the shared original kernel helper is its
feature-width guard; frozen 1156-dimensional predictions remain within 1e-4.
Root: `results/slp11-transition/human-gwps-bp-nystrom-rbf512-seed731-v1/`.
Report SHA-256: `d8259c864460a21f9a13718b2190aad926ca58dc01409c0fab1220a6fbbd276c`;
protocol: `254355c38170b002ef00a112460bc5c2e6858cdae5c6b7be29655e3571f9a337`;
predictions: `1434a0c572728142dc91ac7b1ffb06ddd994badc1f040ef0a1b66f055f7e7725`.

The source frontier figure (`results/slp11-transition/figures/source-frontier-v1.png`
and PDF/JSON) reads pinned reports and plots MSE and independently centered
correlation relative to the same ridge baseline. It has been visually checked.
It displays point estimates, not uncertainty intervals. These adaptive results
show complementary strengths; none establishes consistent superiority.


### 2026-09-05 — Fixed-step context expansion and shared-context kernel results

The matched source3/source4 mean-objective pair completes exactly 12,000 updates
per arm from the same initialization. Frozen response descriptors, controls,
query panel, amplitude, architecture and optimizer are shared. Each arm selects
only its final checkpoint; validation is evaluated once after fitting. No
uncertainty head is exposed by the mean-only inference runtime.

| Context | Source3 MSE / centered r | Source4 MSE / centered r |
| --- | ---: | ---: |
| K562 essential | .02386436 / .291109 | .02389269 / .289411 |
| RPE1 essential | .05506963 / .285697 | .05437685 / .307313 |
| K562 GWPS | .01191498 / .099702 | .01191242 / .096951 |
| HepG2 adaptive | .06345705 / .183870 | .05703625 / .246547 |

Adding HepG2 fitting outcomes improves HepG2 MSE by 10.12% relative to the
matched source3 model. Its improvement over within-HepG2 ridge is 1.447%, below
the fixed 2% requirement. K562 essential regresses slightly in both metrics;
GWPS correlation regresses. RPE1 improves in both metrics. The adaptive rule
fails, and only K562 essential passes the separate standalone baseline rule.
This is a single-seed adaptive comparison; small regressions are not declared
statistically established biological harms. Fixed-update matching also changes
how optimization exposure is distributed among contexts when one is added.

Both arms finish in about 156 seconds including scoring and checksum-validated
fresh-process CPU replay. Replay drift is below 6e-7 and empty identity is exact.
The experiment's prepared v1 is superseded before fitting by v2, which requires
finite comparator correlations and actual nonempty reload parity. A final
stdout-only NumPy Boolean serialization error occurred after both complete
reports were written. All files and reports verify; no retraining was needed.
The current launcher echo path is corrected; frozen run sources are retained.

Pair report: `results/slp11-transition/human-source3-vs-four-context-mean-objective-seed731-v2/report.json`,
SHA-256 `34978aac3f366deccd927c3bda11cda1c4e1107ea388ed432d848cf71b02e010`;
protocol `9f2858397c8589d0c3149b968adddfb6e9deae1204622d2938057705bfdfb580`;
frozen reference `54cac4bc2e2ee02a6d78f812d5646cf3988154d5ae4f371265b24751f03c99b1`.

A seed-stability extension fixes seeds 732 and 733 for both arms and the
arithmetic mean of all three seeds before either new fit. The original rules
remain unchanged. Every seed and context will be reported; no member selection
is allowed. Extension protocol SHA-256:
`e63f50b8ca7c17bbc6893141ba5eede121c8fb30ad3d1e36184f9e80cf0b877a`.
No Jurkat, SL benchmark or protected holdout outcomes enter these experiments.

**Joint context-conditioned kernel: reject.** A separate baseline forms an RBF
basis from all 6,789 measured basal control queries and takes its tensor product
with 512 action-kernel coordinates. Gene-hash inner folds exclude each held
gene across every context. Ridge fits weighted raw outcomes; CV selection uses
context-local fitting-query SD. The same original physical features are used.
The final action basis has 511 retained directions and one inert padded column;
no eigenvalue threshold was relaxed. The shared alpha is 10.

MSE / independently centered r are .02465781/.275109 (K562 essential),
.05442182/.303515 (RPE1), and .01203163/.076039 (GWPS). MSE regresses against both
context-local kernel and physical ridge in every context, so this shared kernel
is not adopted. Runtime 54.42 seconds on CPU2; seven focused tests pass and
reload drift is at most 6.10e-5. The context-input runtime is an engineering
capability, not observed transfer evidence.

Report: `results/slp11-transition/human-gwps-joint-context-rbf512-physical-seed731-v2/report.json`,
SHA-256 `7ac0dd56c72596f6ec8278347e4a7236581beaf56599388ea0f4818f5f544d80`;
protocol `1210024ba4940e1028604bcfe5d8bc8770171c71d8862b50d99115a64b39de65`;
model `544a96e28a461037ae888dc6336e4703b7ba447eb50fccdca76b6e26a0519dbc`;
predictions `0fc2f5223dee0a510f2de90f140cb08ad9b3f519e50d6cee6de62737d9fe963d`.

### 2026-09-05 — Context expansion across seeds and direct cell-state data

The fixed 731–733 seed extension completes all four new 12,000-step CUDA fits.
Every individual seed fails both the adaptive and standalone rules. The
predetermined equal-weight three-seed ensembles give:

| Context | Source3 MSE / centered r | Source4 MSE / centered r |
| --- | ---: | ---: |
| K562 essential | .02355840 / .300433 | .02366163 / .297001 |
| RPE1 essential | .05331830 / .309303 | .05421131 / .309230 |
| K562 GWPS | .01180782 / .099836 | .01180041 / .101938 |
| HepG2 adaptive | .06322901 / .190077 | .05663613 / .256510 |

HepG2 passes its specific gate: source4 improves MSE by 10.427% over source3
and 2.138% over within-context ridge. K562 essential and RPE1 regress against
the matched source3 ensemble, so the adaptive rule still fails. Only K562
essential and HepG2 pass the standalone rule; GWPS gains .924% over ridge,
below 2%, while RPE1 is .671% worse. BP-kernel remains better in K562/RPE1
MSE, and in RPE1 centered correlation. No world-model winner is declared.

All four new fresh-process CPU replay checks agree within 7.2e-7 and preserve
exact empty-intervention identity. Ten focused checks and Ruff pass.
Ensemble report SHA-256 `5e3ad61e3a639f9ded5cad0e65ce41184eafa0ffbddbe6bd6e99ceb6bac5d850`;
predictions `4ca976498710d3a1678c8b4384fd3f1822da693a7b02e6830ba3cf5e5db902b7`;
directory `results/slp11-transition/human-source3-vs-four-context-mean-objective-ensemble731-733-v1/`.
The previously recorded protocol and advancement rules remain unchanged.

A subsequent descriptive paired bootstrap resamples intervention genes 2,000
times, seed 731, conditional on the frozen ensemble predictions. It reports
95% percentile intervals for improvement in equal-gene mean observed-query MSE:

| Context | Source4 versus source3, % [interval] | Source4 versus ridge, % [interval] |
| --- | ---: | ---: |
| K562 essential | -.438 [-1.424, .474] | 3.779 [1.254, 6.474] |
| RPE1 essential | -1.675 [-3.363, -.228] | -.671 [-3.102, 1.705] |
| K562 GWPS | .063 [-.214, .376] | .924 [.044, 1.843] |
| HepG2 adaptive | 10.427 [7.745, 13.257] | 2.138 [.715, 3.503] |

These intervals support a clear HepG2 improvement and RPE1 regression for this
adaptive population. They do not include training-seed, source, biological
replicate or adaptive-selection uncertainty, and are not corrected for multiple
comparisons. They do not change the fixed decision. Report SHA-256
`e1b4fc2df27bdcac2f88f9656d5ed40572f46060b2f57fbdf694e6c9b72a143e`,
`results/slp11-transition/human-context-ensemble-uncertainty-v1/report.json`.
The exact scoring source is retained beside the report; two mask/pairing tests pass.

**Direct paired-cell representation experiment: preparation complete.**
The Frangieh training/control adapter produces 51 shards with 103,862 paired
cells: 64,515 fitting intervention cells and 39,347 verified controls. It
excludes all 19,606 original validation cells before quantitative access;
original test cells were already outside the access allowlist. RNA comprises
18,063 stable queries and 320,301,640 sparse nonzero values. Protein contains
20 matched-isotype-normalized molecular channels. The deterministic barcode
reconstruction split has 93,397 training and 10,465 validation cells, drawn only
from original fitting/control populations. Three fitting guide means and all
three context control means reconstruct the earlier pseudobulk endpoint within
4.77e-7 RNA / 1.19e-7 protein. Total preparation takes 310 seconds on CPU.
Ten adapter checks pass. Manifest SHA-256
`e791b5cf35da96fa71951a4a240ed58b53e278d3c57e44066680abd3f386a9c7`,
`data/derived/slp11-frangieh/paired-singlecell-train-control-v1/manifest.json`.

The new `slp-1-1-cell-state-v1` core pools molecular values through supplied
static query features into a shared cell state. It uses no learned gene-ID
embedding. Its observation map is affine in state, so population averaging
commutes with molecular decoding. The nonlinear encoder must be applied to
individual cells before averaging. Five core checks pass, including missing
values, query permutations/chunking, exact zero-change control identity,
affine averaging and gradient propagation. Biological training is the next
experiment; these algebraic checks establish no reconstruction, forecasting,
temporal or combination capability. The planned latent ridge forecast is a
test of a learned output representation, not evidence of nonlinear intervention
dynamics. Original held-gene cells do not train the representation.

**Species-native yeast RNA addition.** Pinned CC BY 4.0 author molecular
summaries `FC_genotype.Rdata` and `ptb_summary.Rdata` are acquired from Zenodo
14062629. Their SHA-256 values are
`c210fe541b0b91bc6eead28aa2265065afceec763ade1abd682c58896299a240` and
`01c2d54ac838179be29694ed300cb17edac47dd4db23a4018407546e0651b165`.
A bounded two-pass R-object parser completes in 201.6 seconds at about .57 GiB
after general conversion exceeds the 6 GiB cap and is stopped. No author
fitness, SL, third-party comparison, or large raw R object is acquired.

The development adapter contains 3,419 records, 1,732 deletion actions and
6,340 externally mapped stable SGD queries, taxon 4932. Its control/NaCl
contexts contain 1,708/1,711 records; 1,687 actions occur in both. There are
2,789 fitting and 630 validation records, no test records and no overlapping
intervention genes. Its mask retains 19,411,942 observed targets and one
all-missing query. No absent value is treated as a measured zero. The source
exports `names`/`logfoldchanges`; the upstream DE call and precise transform
are unavailable, so the endpoint is explicitly
`author-logfoldchanges-unknown-upstream-transform`. There is no raw basal WT
state, SE, or control replicate to infer. This supports a source-specific
exploratory test, not calibrated cross-source molecular units.

Dataset SHA-256 `42f754425637bdf0413dbac6c36206737b5e402e04ba9732aa329cf2f1e702d5`;
verification `9e209e1e43405654ffeea4f363073c5a6690943e65ba1278ac26f3bbd77c704e`;
directory `data/derived/slp11-yeast-atlas-response/nadal-ribelles-control-nacl-development-v1/`.
Three focused parser/mask/partition tests pass. No yeast response is relabeled
as a human outcome, and no Jurkat or SL benchmark outcomes enter these steps.

Primary-methods clarification: the yeast paper describes mutant-versus-WT
Wilcoxon tests separately within control/stress and calls the effect log2 fold
change. The remaining uncertainty is the exact computational transform of the
archived numerical column, not the absence of a documented intended comparison.
The adapter's conservative value-space tag is retained.

The frozen Frangieh RNA query-feature reference contains 16,206 distinct rows
among 18,063 queries. There are ten duplicate groups containing 1,867 rows;
the largest contains all 1,849 completely zero raw static feature rows. Frozen
normalization maps these to the same nonzero vector. Of 1,874 queries lacking
protein features, 25 have other GO/physical information. A feature-generated
decoder cannot distinguish the remaining 1,849 queries in standardized units.
The first cell-state run retains this fixed panel; a separate static transcript
sequence extension is being prepared to test this representation limitation.

### 2026-09-05 — First direct paired-cell model and transcript feature repair

The first 802,050-parameter paired-cell state completes 20 epochs in 385.86
seconds of training, 415.78 seconds including forecast/scoring/reload. The
selected reconstruction checkpoint is epoch 20. Protocol SHA-256
`4540978da15d50881a1613a00ad013ebdb48e880a285ed8175ba0ed0818059d2`
fixes seed 731, batch 256, key/state/hidden dimensions 64/128/256, dropout .1,
AdamW .0005 with decay .01, 20% input denoising and equally weighted modality
MSE after fitting-cell standardization. There are 406 actual shard-aware updates
per epoch. Selection uses only the 10,465 within-fitting-gene validation cells.

Denoised reconstruction standardized MSE is .726345 RNA versus .752021 for the
training-mean predictor, a 3.414% gain that fails the fixed 5% RNA requirement.
Protein MSE is .132581 versus .995999, an 86.689% gain. Raw-unit gains are
9.03% RNA and 88.77% protein; these descriptive raw gains do not replace the
standardized advancement criterion.

After checkpoint freezing, equal-cell guide states are averaged equally by
gene/context. Fixed alpha-10000 latent ridge predicts changes from the same
1156D static action features as earlier baselines. The affine decoder anchors
these changes at measured context control means. No original validation gene
cells enter encoder or ridge fitting.

| Environment | RNA MSE / centered r | Protein MSE / centered r |
| --- | ---: | ---: |
| Co-culture | .00211750 / -.0020 | .0243601 / .12022 |
| Control | .00218360 / .02705 | .0174668 / .05667 |
| IFN gamma | .00196246 / .05279 | .0188979 / .09800 |

Only co-culture protein passes all fixed forecast requirements: MSE improves
2.713% over mean/base577, 2.172% over physical1156 and 4.338% over the prior
paired world, with centered r .12022. The remaining five strata fail. The
joint advancement rule therefore fails. This single adaptive development
result supports a specific protein-readout improvement, not a general world
model, new-context transfer or SL claim.

Trained CUDA states and nonempty predictions agree with fresh-process CPU
inference within 1.07e-6, with exact empty-intervention control identity.
Report SHA-256 `cada9a66568dda2340a95dd0bbd6b96bcc7af2ac76bf98f9b8f4e1d681bc182f`;
model `4e26a21287bf268df70b83dbc160e3eeb652ccd98491931bc06b008319a5c8bd`;
reference `4433edeffcf02f977fbabffd421a9ed7ee262f1c5924213a9be67c995ee8a6e4`;
predictions `a5cc6724ad55c5d3f2ad709be36a5fcbcb77e7255d7f79af29145556e2a24b96`;
directory `results/slp11-transition/frangieh-cell-state-ae-latent-ridge-seed731-v1/`.

A separate no-refit full-input reconstruction check gives RNA standardized
MSE .725915, a 3.471% improvement; protein .028339, a 97.155% improvement.
Thus selection-time denoising does not explain the weak RNA reconstruction.
Full-input diagnostic SHA-256
`e8e9b4c07ef1f5b59f75bf24518b28e71aa9b86ba4758c125e7f8a6e65608ec2`,
`results/slp11-transition/frangieh-cell-state-full-input-reconstruction-diagnostic-v1/report.json`.

**Transcript descriptor repair, static extraction complete.** Ensembl release
116 cDNA and ncRNA FASTAs pass source SHA-256 and BSD checksum verification.
The representative transcript is the longest per stable ENSG gene, with a
deterministic transcript-ID tie break. Features retain strand-specific 4-mer
frequencies, log length, ambiguous-base fraction and presence (259 dimensions).
No molecular outcome, gene symbol or textual description enters extraction.
The source-specific rights declaration covers only these transcript exports.

The pack has 85,410 species-native rows, including 85,308 present sequences
and 102 explicit missing union rows. It covers 17,961/18,063 Frangieh RNA
queries and all 237 metadata-listed intervention genes. Concatenation with the
frozen query features increases distinct rows from 16,206 to 17,986 and reduces
the largest unsupported equivalence group from 1,849 to 78. This establishes
representational coverage, not predictive benefit. Extraction takes 47.56
seconds on CPU; deterministic reserialization and six focused checks pass.

Pack SHA-256 `af165a97a0169dd7419e86ebdbc5fc3855dc7b868c7f774b817720d8cf3631d3`;
manifest `0ffa6d367cb5ebc2ef6915573d2a61b562725c1af674c48c9fd62340cbb9d057`;
directory `data/derived/slp11-human-transcript-sequence/ensembl116-kmer4-v1/`.
A matched query-feature-only cell-state comparison is now being prepared.
It leaves intervention features, protein features and outcome populations fixed.

**Yeast summary baseline: reject for joint training.** The CPU2 comparison
finishes in 200.27 seconds. Linear and rank-256 Nyström baselines select alpha
100,000 through three fitting-gene folds in both environments. Each environment
retains all 315 validation genes, including 20 with missing static features;
presence flags preserve that missingness without dropping rows.

| Environment | Mean MSE | Linear MSE / centered r | Nyström MSE / centered r |
| --- | ---: | ---: | ---: |
| Control | 70.53850 | 70.41511 / .023640 | 70.52219 / .011023 |
| NaCl | 68.15231 | 67.90530 / .027459 | 68.13220 / .014071 |

Every fixed Nyström advancement check fails. Ordinary profile correlation
near .643 is dominated by the shared response shape; it is not evidence of
perturbation-specific prediction. A subsequent fitting-only inspection finds
20.004%/18.560% of control/NaCl values have absolute magnitude above 20; the
fraction above 10 is essentially identical. Log cell count versus mean absolute
response has r=-.94824/-.94057. This strong dependence motivates checking raw
counts and the upstream estimator before any joint model uses these summaries.
No magnitude clipping or outcome-based row removal is used to improve a score.

The scoring-corrected v2 preserves v1 fits and predictions, fixes undefined
correlation for constant mean predictions and adds descriptive support strata.
Report SHA-256 `b51664cd056ea824277afc665d4a3209a3dce73d4fe2f62d87befaac906b4511`;
fit protocol `15729183c2f37dd5accb2a03a10d7d17dd493bc5427ed1776aa79378336dcfe7`;
scoring protocol `1bbd2767580f58d708a4c855a1ed0733b096f1552becf32d9ecdc6b6454dc00b`;
predictions `6700587531c4a1be4158d869ab44dd165238d812dd366c887e54ef827db3a03f`;
directory `results/slp11-transition/yeast-nadal-static-baselines-seed731-v2/`.
Target-free reload is exact, six focused tests pass and Ruff is clean.

### 2026-09-05 — Paired-cell PCA separates reconstruction from forecasting

A rank-128 PCA baseline fits the same 93,397 reconstruction-training cells,
with means and SDs bit-identical to the AE's frozen normalizer. RNA/protein
standardized coordinates receive weights 1/sqrt(2Q) per head. Seed-731
160-dimensional Gaussian subspace iteration uses three streaming covariance
passes and a final Rayleigh eigendecomposition, all float64 on CPU2. No dense
full-cell matrix is materialized. This is a fixed-assay baseline with learned
query loadings; it is not a query-feature world architecture.

The protocol is frozen before PCA and stats fitting; latent ridge and PCA are
both frozen before held-gene forecasting. The same 151 fitting genes/context,
1,399 exact guides, equal-guide aggregation, physical1156 action normalization,
fixed alpha10000 and measured controls are used as for the AE.

PCA full-input reconstruction-validation RNA standardized MSE is .704776,
a 6.282% gain over the training mean; raw MSE .0659574 gives a 14.42% gain.
Protein standardized MSE is approximately 1.06e-6 because a rank-128 basis can
retain virtually all variance of the 20-dimensional protein assay. This is
compression performance on supplied measurements, not protein forecasting.

| Environment | PCA forecast RNA MSE / centered r | PCA forecast protein MSE / centered r |
| --- | ---: | ---: |
| Co-culture | .00210968 / .006285 | .0243049 / .124067 |
| Control | .00216791 / .008310 | .0175257 / .056566 |
| IFN gamma | .00195337 / .037948 | .0189023 / .092214 |

Only co-culture protein passes the fixed six-stratum comparison, with gains
2.93% over mean/base577, 2.39% over physical1156 and 4.56% over the earlier
paired model. The joint rule fails. PCA improves MSE over the cell AE in four
of six strata by .23–.72%, while centered correlations are mixed. This rejects
the inference that better cell reconstruction by itself provides useful
intervention transfer. No model or rule is selected from this comparison.

Runtime 149.86 seconds on CPU2; target-free reload is exact; five focused
checks and Ruff pass. Report SHA-256
`e9afd49a315946b68a0862903eb67615d9291080a86c8613f82378110c8cff4f`;
protocol `b3ba1796d118a959a1c22291d7b62c0f914d4f33752f7b98a2de93344bbfba63`;
PCA/ridge payload `5070bdb09f9949132d4d610f6ba379d1e96537cd162554897916a8d83c2b2e26`;
predictions `bd81085f55b33050fe8670bf2f2cb062d32de9134c4b0100eabd6f1f452638c0`;
directory `results/slp11-transition/frangieh-paired-pca128-latent-ridge-seed731-v1/`.
Frozen AE comparison supplement SHA-256
`8c42fc5ec8287043ff796efba8abd56eba7efb74e9a27114e674fd0651aac522`.

The yeast endpoint diagnostic is retained at
`data/derived/slp11-yeast-atlas-response/nadal-ribelles-fc-endpoint-diagnostic-v1/report.json`,
SHA-256 `298b7eb3f7c48d5ad91a405e54d2c286a75a248a8fae8be19134413616016a45`.
Approximately 94–95% of extreme fold changes are negative. Their frequency
and magnitude are strongly related to cell count, consistent with a numerical
zero/pseudocount floor, although the missing estimator prevents proving this
mechanism. Two checks pass. A 2 MiB bounded HTTP prefix confirms the author's
separate-control/stress archive is gzip-compressed RDX3/XDR with S4 assay/count
slots. A reference-safe bounded parser and download-throughput profile are
being prepared; the full count archive has not yet been acquired.

### 2026-09-05 — Stop the paired-cell descriptor and residual-transition branch

The matched transcript-query comparison completes in 382.61 seconds total.
It appends the 259 static transcript features only to RNA queries; action
features remain physical1156 and protein queries remain the same 20 channels.
Transcript normalization fits the 151 fitting intervention genes only; constant
columns use unit scale and clipping is 10, matching the earlier reference
convention. All shared initial parameters match the primary model, with new
query-input columns initialized to zero. Initial forward drift is 7.15e-7.

RNA reconstruction standardized MSE .724703 improves 3.633% over the fitting
mean, still below 5%; protein .130894 improves 86.858%. Held-gene RNA MSE gains
over the primary cell model are only approximately .13%, .23% and .22% across
the three environments. Protein changes are small and mixed. Only co-culture
protein passes the original forecasting gate, so the overall rule fails.
The coverage repair is retained as a static resource, but there is no evidence
to advance this neural cell-state configuration on that basis.

Completed run v3 supersedes two prefit preparations. V1 omitted an explicit
primary-cell-model descriptive comparator. V2 stopped because the copied helper
resolved the repository root incorrectly, before shard matrix access or fitting.
V3 uses a checksum-matched helper with explicit path semantics. Frozen earlier
artifacts remain unchanged. Fresh CPU replay agrees within 7.15e-7 and empty
identity is exact. Ten focused checks and Ruff pass.

Report SHA-256 `1c8eacd25cbce223a01c0a91a887f2b3570fb6c444f44fc64eacf3c707a368ef`;
protocol `4ac9526fc3520b790a20da34c6d01253364e3a49799e8ace7f685f5766a2d17e`;
model `49a8a52c3caaa2405912e4f76cb416637a04a3e54790fc2b35f55aa057eb7792`;
reference `f4bd533bdd875b787773d521a4383626ce5df0df7409f958ecb683a32ec49519`;
predictions `dd496b731075b7dba430fc2b7a09d5904740197e2846b1651283179d25f11c87`;
directory `results/slp11-transition/frangieh-cell-state-transcript-query-ae-latent-ridge-seed731-v3/`.

A second controlled test freezes PCA128 and latent ridge, adding only a
181,248-parameter nonlinear action/state residual. The residual consumes raw
supplied control PCA state and normalized physical1156 action features, with
128 hidden units, LayerNorm/GELU/dropout .2 and a zero output layer. It starts
at exact PCA-baseline predictions and forces empty-intervention delta to zero.
The module has no gene or context-ID embedding. Fixed 1,000 updates use equal
context sampling, batch 32, AdamW .0005/decay .1 and gradient clipping at 1.
The decoded mean objective balances RNA/protein after fitting-gene query-SD
scaling with floor .05. The final checkpoint alone is scored.

This residual worsens MSE relative to PCA in every stratum: co-culture RNA/
protein by 5.62%/21.31%, control 5.87%/27.01%, IFN gamma 7.24%/17.75%.
All centered correlations are below .10. The fixed six-stratum rule fails.
Training takes 8.63 seconds, 23.34 seconds total. Fresh CPU replay is within
4.77e-7 and exact empty identity is preserved; three core checks and Ruff pass.
This rejects the specified residual transition, not nonlinear intervention
modeling in general. Further tuning of this small paired-cell branch is stopped.

Residual report SHA-256
`b28fbb989cb23ec49abc42978edb51967de4d9387c345ffb5e3593dd8415fc3b`;
protocol `f63052c2adbc36c6c229d043b7e805746edeb41ab60a6bfb45cbe4f0ce56d5d1`;
directory `results/slp11-transition/frangieh-pca128-residual-transition-seed731-v1/`.

A no-refit support diagnostic applies max(prediction,0) to every original
mean/static/paired/cell-state comparator because both processed Frangieh
measurement spaces are nonnegative. Gene IDs and target arrays align exactly;
the score checks the mathematical non-increase in squared error for each
method. Cell-state RNA negative fractions are 2.98%, 5.47% and 2.23%, mostly
very close to zero. The largest absolute MSE reduction is 6.27e-8 in control;
all six advancement decisions remain unchanged. This population-forecast
projection is not evidence of a valid nonlinear single-cell observation map.
Report SHA-256 `c62ddaa5778591f3c2a480b669eed506668ea7bd2c9caa91efa047bc996f61f8`,
`results/slp11-transition/frangieh-nonnegative-support-diagnostic-v1/report.json`.

The full repository suite now passes: 677 passed, 21 skipped, 17 historical
Transformer warnings in 39.62 seconds. Later source additions require their
own focused checks. No Jurkat, SL benchmark or protected test outcomes enter
these experiments.

### 2026-09-05 — Matched BP neural test fails; raw yeast counts acquired

Hypothesis: adding direct biological-process annotations to intervention
features improves the fixed human mean-objective transition beyond its matched
masked-feature control and BP ridge. Both arms use physical1156 plus BP128
and one presence channel, with the added columns zero in the control. Shared
initial parameters match; new action-input weights start at zero. BP statistics
fit 6,866 source3 intervention genes only (6,330 annotated). Query features,
controls, sampling and the mean objective remain fixed. Dataset SHA-256 is
`55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c`;
BP pack `b29cbd70f08e227cddfc013e66cd1032212c8cb62e6e25162965a57101cd1fac`.
Each arm trains 12,000 updates with seed 731, batch 64, AdamW .0005/decay .1,
gradient clip 1, and final-checkpoint-only scoring. CUDA fitting takes
125.44 seconds for control and 126.97 seconds for BP.

The fixed rule requires at least 2% MSE improvement over both masked control
and BP ridge in every context, centered correlation at least .10, and no
correlation regression against those comparators. BP results are:

| Human context | MSE | Independently query-centered r | MSE gain over matched control |
| --- | ---: | ---: | ---: |
| K562 essential | .023547257 | .304404 | 1.327% |
| RPE1 essential | .054595677 | .298346 | .897% |
| K562 genome-wide | .011865771 | .099592 | .409% |

All contexts fail. RPE1 is 2.84% worse than BP ridge; the BP kernel also
has lower MSE in both essential-gene contexts. This configuration is shelved.
The original finalizer incorrectly assumed four contexts. Both trained
checkpoints were already frozen; a superseding finalization package corrected
the target-free probe to two rows per source context before validation access.
There was no refit or decision-rule change. Fresh CPU replay agrees within
5.52e-7, with exact empty identity in all three contexts. Three focused checks
and Ruff pass. Report SHA-256
`5d7c14fff561f02e1a46c353c54fefda5aec8c778edaacea4c5bca5849441060`;
finalization protocol `ba2953a64417402eb1f22e6d9ec5d7ba4150ca4e4efa493946c4ec781a9ff85d`;
directory `results/slp11-transition/human-source3-bp-neural-mean-pair-seed731-v2-finalization-v1/`.

The full author-provided separate-environment yeast Seurat archive is acquired:
`data/sources/nadal-ribelles-2025-yeast-seus-split-v1/full-acquisition-v1/seus_split.RData`,
5,907,877,873 bytes, published MD5 `65bb56efd8120f32f65c044de5f040aa`,
SHA-256 `da99869c11d1a6c034454568098aa50bc3313cd4508dbd506d43241b0fb4695d`.
Four bounded connections completed network acquisition in 1,346.20 seconds;
allocation, acquisition and hashing took 1,472.80 seconds. Independent streamed
checksums, contiguous range coverage and atomic final publication were verified.
The separate CC-BY-4.0 rights scope covers this archive. This is source
acquisition, not a trained or admitted count model.

Bounded structural parsing takes 113.30 seconds and discovers raw RNA/counts
matrices of 6,951 queries by 326,438 control-environment cells (181,083,366
nonzeros) and 384,514 salt-stress cells (205,135,811 nonzeros). These environment
labels do not identify wild-type cells. Both ordered query rosters match.
The parser skips quantitative values during inventory; normalized RNA/data
and SCT assays are not eligible raw-count substitutes. Metadata replay and
stable-ID interpretation precede the count adapter. Inventory SHA-256
`c51fdbd303a9cce3253efa4a6ce78631bdb8f5097bac4c68def4d3c72a38808d`.
The earlier fold-change archive remains excluded from joint training because
its extreme endpoint values are strongly tied to cell count.

A separate shared human/yeast static GO basis repairs incompatible coordinates
in the former species-specific SVD packs. It uses direct molecular-function and
cellular-component annotations from the pinned 2022-09-19 GOA sources, excludes
NOT and perturbation-derived evidence, and fits 256 components with seed 731
and seven iterations. Species rows are weighted by inverse square-root species
population during fitting; output is unweighted binary annotation times the
shared basis. No quantitative outcomes are used. The pack has 30,862 taxon-keyed
rows: 6,983 SGD genes and 23,879 translated Ensembl genes, with eligible
annotation coverage 6,968 and 21,093 respectively. Identical annotation rows
across species give bit-exact vectors, while biological identities stay separate.
This establishes compatible static coordinates, not transfer performance.
Repeat serialization is byte-identical; two focused checks pass. Features
SHA-256 `fb673cf6053bb7bfe88c6b454cedb662646f7256f094abf9a6df1d2865f873f6`;
basis `718764f4ebb6ab9ac31dba65d7d6453525e04a98b999aa7dcfeb4c3a1ab62abd`;
directory `data/derived/slp11-shared-human-yeast-go/goa-2022-09-19-mf-cc-svd256-v1/`.

### 2026-09-05 — Count endpoint core and output-subspace diagnostic

The self-contained `modules/slp-1-1-count-moments-v1/` implements bounded
equal-cell log1p(CP10k) moments from raw integer counts. The caller fixes a
biological denominator mask, stable-query mapping and metadata populations.
Duplicate source rows resolving to one query are summed before normalization;
unmapped biological rows can remain denominator-only. Observed zero expression
is retained, while zero-library cells are explicitly excluded and counted.
Float64 sums and squared sums yield population means and unbiased cell
variances, with explicit missing support for empty or single-cell populations.
These variances are cellular dispersion, not independent-replicate uncertainty.
Seven focused checks and Ruff pass. Source SHA-256
`53344c00ad4a8615c796a2f41371efc46823eb59e82925e67ff2178d72d004c3`.
No source counts have yet entered this core; metadata eligibility and the actual
denominator remain to be frozen after source inspection.

A fitting-row diagnostic measures the actual human decoder subspace. Its
query encoder followed by linear mean projection restricts centered responses
to a shared rank at most 128. The BP neural model's subspace captures 31.91%,
31.22% and 18.62% of standardized fitting-gene landscape variance in K562
essential, RPE1 and K562 genome-wide respectively. Separate per-context rank128
PCA projections capture approximately 55.74%, 55.60% and 31.63%. The latter are
descriptive per-context ceilings, not one feasible shared decoder, and include
variation that may not be predictable from intervention features. This result
therefore motivates an alignment test; it does not prove the cause of held-gene
failure. BP and masked-control subspaces differ by less than .1 percentage point.
The adaptive-development NPZ is materialized, but only fitting rows enter
centering, projection or SVD. Diagnostic report SHA-256
`7692261501d6fcc986e13dfa7016f6ae241370a6d51b51458d31b321a664ae52`,
`results/slp11-transition/response-query-fitting-subspace-audit-v1.json`.
A pooled shared-basis comparison is the next discriminating check.

### 2026-09-05 — Shared response basis helps RPE1, fails the all-context rule

A feasible shared rank128 basis is fit to 9,973 gene/context molecular-delta
profiles from source3 fitting interventions only. Deltas are divided by the
same frozen per-query amplitude, and each context is weighted by the inverse
square root of its fitting-gene population. The SVD is uncentered to retain
mean responses; no context-specific basis or new centroid is added. Shared
query coordinates are components.T times sqrt(128), supplied as data. Basis
SHA-256 `a31bb27db40d542dfb541daccebe056415080b457803ec7eb222c303add0b1ee`;
protocol `df89a6f733775f6ff550bcfdc6d9addcb2f7e2823d326b71f7634b416f17f0bf`;
representation report `817c74653a248cd350fc659ab495072ada1b3efe43db5f3c2ad2879d0c532d25`.
Centered standardized fitting-landscape capture rises from 31.91/31.22/18.62%
to 44.58/51.00/20.63% in K562 essential/RPE1/genome-wide. This is a
representation diagnostic, not an intervention prediction result.

The new self-contained fixed-query transition removes the 168,960-parameter
query encoder while retaining the action/context encoders, transition and
trainable mean projection: 416,000 parameters total. Root review catches a
prefit initializer error in v1: directly constructing a 1,285-feature model
does not reproduce the matched BP initializer, which extends a 1,156-feature
model with 129 zero columns. V1 remains unfitted. Corrected v2 copies every
shared tensor from the actual pinned extension helper and verifies equality.
The frozen basis is the only intended numerical representation change.

The fixed diagnostic rule requires at least 1% MSE improvement over matched
learned-query BP and no centered-correlation regression in every context.
The stronger ridge/kernel frontier remains a separate advancement comparison.
Training uses the same 12,000 updates, seed 731, batch 64, equal-context/equal-gene
weights, AdamW .0005/decay .1, clip 1 and final-checkpoint-only scoring.

| Context | Fixed-basis MSE | Centered r | MSE gain vs learned-query BP | Diagnostic |
| --- | ---: | ---: | ---: | --- |
| K562 essential | .023277084 | .301695 | 1.147% | fails correlation nonregression |
| RPE1 essential | .052092519 | .324392 | 4.585% | passes |
| K562 genome-wide | .011862750 | .100150 | .025% | fails MSE threshold |

All-context decision: fail. K562 and genome-wide MSE are lower than the BP
kernel's, while RPE1 remains worse. Against BP ridge, K562 improves 3.90%,
RPE1 1.88%, genome-wide .05%; only K562 passes the 2% MSE/.10 correlation
advancement checks. Output alignment therefore contributes to RPE1 error but
does not resolve generalization across contexts. The residual limitation could
include action-to-coefficient prediction, regularization or unpredictable
measurement variation; this experiment does not distinguish them.

Fit time is 120.156 seconds; total 131.39 seconds. Actual trained-GPU probes
replay in a fresh CPU process from source/model/reference alone within 8.35e-7,
with exact empty identity. Seven focused checks and Ruff pass. Report SHA-256
`456ad9480652cf32f7e7be4bf1e3cffa753c76e67eb5082ef88dcf2057bb1ad8`;
protocol `d7d56c4e94d09dbb6e7210ed73de6ddc018209f1c01ecf6b45bfe11460ab728d`;
model `d073e4d66bb498dbbc2048f656b90da069318ff8a736860c03436e37a58cc693`;
predictions `919f9ef7a00b897bde76cf1972f75ab8115a00c6bf52228d67e0cb254e8247f3`;
directory `results/slp11-transition/human-source3-bp-fixed-response-basis-seed731-v2/`.

### 2026-09-05 — Freeze yeast count measurement and eligible populations

The complete Seurat metadata replay preserves serialized NA separately from
empty strings. The source-author WT rule is exact assignment_consensus2='WT',
with kogene agreement. There are 500 WT cells in the control environment and
458 in NaCl, spanning all 14 and 15 respective source batches. Batch B05 has
only two WT cells in each environment and NaCl B16 has three. Immediate
batch-WT subtraction is therefore not adopted: absolute normalized RNA moments
are retained, allowing later explicit treatment of control uncertainty.
Metadata report SHA-256
`1e2308fdb0182b34a7c0eacf361516c32298ac3581d4a858bafab940cd220946`.

The source-row audit identifies 6,683 exact current SGD mappings and 268
alias/dash candidates among 6,951 biological RNA rows. All 6,951 remain in the
per-cell denominator. Initial output queries use only the 6,683 exact mappings,
sorted by stable SGD ID; candidate mappings remain explicit denominator-only
rows. No artificial barcode row is present in the roster, consistent with
author processing documentation; native URA3 is retained. Audit report SHA-256
`b62ffdf7cebe4fbcfc760a379a8407705ecef21c42e6a5a363b07a7cc5434364`;
mapping `8e40587551a329b94e73989fe284240116645986e00cbd05fc2f1bd52bc01643`;
denominator roster `4dca4a469a1fea9703ad5282a17cdf0a25dfe3d2bf23a622d139eb75a98c8c80`.

Mutant actions require the bc- assignment prefix and an exact current SGD
mapping of kogene. Source assignment suffix variants are preserved, not treated
as separate genes. Global protected roles and development test actions are
excluded before selected quantitative count decoding. Selection retains
188,135 control-environment cells (155,520 fitting, 32,115 validation, 500 WT)
and 221,913 NaCl cells (183,556 fitting, 37,899 validation, 458 WT). Across
environments the selected action union has 2,013 genes. Control excludes
37,120 development-test, 92,474 protected-role and 8,709 exact-map-failure cells;
NaCl excludes 42,766, 109,766 and 10,069 respectively. Selection report SHA-256
`da1946b114b3462acf37b708aa8f1aa87f69de4063a32e67d3b858077c786e2b`.

Frozen measurement: per-cell ln1p(10000*count/sum_all_6951_RNA_rows), with
explicit zero-library exclusion and no minimum-cell or expression threshold.
Absolute sum/squared-sum moments are grouped by environment, batch and stable
genotype; WT is separate, clone is retained as source metadata. The intended
population counts before zero-library checks are 23,125 mutant gene/batch
groups in control and 24,344 in NaCl. Selected count extraction is in progress;
these are metadata counts, not a completed numerical corpus claim.

A self-contained streaming batch-ridge baseline is prepared in
`modules/slp-1-1-batch-ridge-v1/`. It fits a shared feature effect with unpenalized
fitting-derived source-batch intercepts; those intercepts are nuisance effects,
not WT states or predictions of unseen contexts. Its source SHA-256 is
`5d897f45ca1318ffe1d447cbafbb1732d0e428efa5f6a7b3dcfe4c32841c18c8`.
Three focused checks reproduce independent augmented weighted least squares,
verify constant-batch-shift invariance and reject invalid/unseen-batch inputs;
Ruff passes. No biological fit has yet used this module.

### 2026-09-05 — Yeast count corpus and complete intervention sequence features

Selected raw extraction completes in 280.72 seconds: control has 104,609,403
nonzeros and NaCl 118,410,604, all finite positive integer entries. All selected
410,048 cells have positive libraries. The source and selected-file hashes are
recorded in the extraction manifest SHA-256
`18c4b3e2f6cdfd33ce663f11a83cc2cdae65e3cad9ec56e9575cfd2783d9f148`.

Moment aggregation completes in 47.20 seconds, writing 29 shards totaling
5,095,669,986 bytes: 47,498 populations, comprising 38,978 fitting, 8,491
validation and 29 WT groups, with 6,683 queries. Shards contain float64
sums/squared sums, counts, support and source/stable identities. No batch
correction or WT subtraction is applied. Independent WT means, variances and
counts match bit-exactly. Nineteen focused checks and Ruff pass. Manifest
SHA-256 `70a49ecaeb271fc72ecc93ede207c59a816e74d1ae3133bbf3a2803cce5d8eba`;
protocol `601964f10c4649b95f56cd11ddec9af8b95646aa008aa6cc5f6ef440ca0a42c9`;
receipt `df3f49a6f37bc8220ff881cb5c50760e6874964c830fff591f29fc7a68b536a2`;
directory `data/derived/slp11-yeast-atlas-counts/nadal-ribelles-raw-rna-development-v1/`.

The independent WT-only diagnostic reads no mutant count columns. Median WT
library sizes are 891.5 and 980.5 counts. Weighted between-batch expression-mean
dispersion is .0127905 in control and .0134510 in NaCl. An independent-cell
sampling calculation attributes .0093786 and .0100330 to finite-cell sampling,
leaving signed excess .0034119 and .0034180. Thus batch estimates contain both
appreciable sampling noise and residual heterogeneity. This is descriptive:
cell independence ignores clone dependence, and batches are not established
biological replicates. No normalization or selection rule changes. Diagnostic
protocol SHA-256 `f9586500ece6561d02848e281b98c1ed738b57cbe2a5945c5a2a82f3fe6a7a46`;
WT reference `190dc64dd9ee8809f56f82b690265827376c72b36286e46e04b8aebee64fa1b5`;
directory `results/slp11-transition/yeast-wildtype-batch-diagnostic-v1/`.
One independent sampling-formula check and Ruff pass.

The shared static feature resource now covers every eligible yeast action.
The frozen original ESM-2 t6-8M recipe processes 1,395 missing vectors from
1,354 unique pinned peptides and 1,466 deduplicated windows, taking 14.83 seconds
of inference. Full-protein overlap-corrected pooling retains proteins up to
3,744 residues without truncation. A prior 72-entity profile takes 10.3 seconds
including model load; eight recomputed old vectors agree within 1.43e-6 maximum
absolute error. All 7,037 existing vectors remain bit-exact in the extension.

The superseding 6,744-row static577 pack combines ESM320, presence and the
shared MF/CC GO256 coordinates. All 2,013 actions have protein embeddings;
1,887 have shared GO support. Of 6,683 strict RNA queries, 6,410 have protein
embeddings and 5,987 have GO support; 273 non-ORF queries retain explicit missing
protein features. Twelve checks, Ruff, compilation and independent reload pass.
Extended ESM SHA-256 `dbcfbef8bb4b4c091ab43cacbc9da6700a91b6334896d209390703b9911a854f`;
static577 `81cda9469380c9efa000a40b2cd5e816a1d397ce777288fa53b0bcf26a55dc25`;
manifest `1eb86148cdec3a09a86b56fba3e915eddf610398265346410623afec455ce893`;
directory `data/derived/slp11-yeast-shared-static/current-sgd-strict-query-full-raw-actions-esm8m-complete-shared-go-v2/`.

### 2026-09-05 — Rebuilt yeast RNA: reproducibility and batch-ridge results

The fitting-only split-half diagnostic deterministically alternates barcode-hash
rank within each genotype/batch stratum with at least two cells. It includes
335,542 fitting/control cells from 34,515 strata; 4,492 singleton cells are
excluded from this diagnostic, and all 70,014 development-validation cells are
excluded before count reads. The underlying development corpus is unchanged.
Control A/B raw profile correlation is .9036 but independently query-centered
correlation is .0728; NaCl is .9213 and .1064. Centered A/B MSE is .053474
and .047722. The same 1,457 genes across environments have centered correlation
.0948. WT half-profile correlations are .9918/.9925. Shared batch, clone and
library structure means these measurements are not biological noise ceilings.
Runtime is 22.06 seconds; eleven checks and Ruff pass. Protocol SHA-256
`5c7ab40a72494178ca9cb535a60714ff02de2d1b8bf585790bc6c743af28c4fb`;
report `e8ed9915a92b861a1dd7bce5fb7c9f4a43edf69e512e3d7a74bd42867f77eab3`;
directory `results/slp11-transition/yeast-rna-fitting-split-half-v1/`.

A metadata-count-stratified supplement reuses the frozen fitting statistics.
For genes with at least 100 included cells, Control has 613 genes, centered
A/B correlation .0902 and MSE .009465; NaCl has 727 genes, .1307 and .008143.
For the same 590 genes with at least 100 cells in both environments, the
correlations are .0897/.1385 and cross-environment .1132. Sparse sampling
substantially increases MSE, but modest perturbation-specific reproducibility
persists in well-supported genes. No cohort or decision rule changes.
Supplement SHA-256 `82739f209b9983f8153048be6d2e872bd385e71c5eb173b18d738ecc098b7937`.

The batch-aware linear baseline uses context-separate fitting, three exact
gene-hash inner folds, unique-fitting-gene feature normalization and the fixed
ridge grid .1 through 1e6 plus the mean limit. Each gene has total weight one;
its batch populations receive weights proportional to their cell counts.
Pooled and batch-intercept arms use identical observations, features and
weights. Controls are not fitted as perturbations. Every final model is frozen
before development validation; all 346 validation genes per environment remain
in evaluation. The fixed diagnostic rule requires 2% MSE improvement over
pooled ridge and both mean limits, with batch-mean-subtracted centered
correlation at least .10 and no regression versus pooled ridge in each
environment. Batch means are fitting-derived nuisance effects, not WT states.

| Environment | Pooled ridge MSE | Batch ridge MSE | Batch mean MSE | Batch ridge residual centered r |
| --- | ---: | ---: | ---: | ---: |
| Control | .021932717 | .021551289 | .021552728 | .012045 |
| NaCl | .020496197 | .020231360 | .020232298 | .010976 |

Both arms select alpha 1e6 in both environments. The batch arm improves raw
MSE 1.74%/1.29% over pooled ridge but only .0067%/.0046% over batch means.
Raw independently query-centered correlations near .092/.099 mostly reflect
batch composition; after the same batch reference is removed from both
prediction and truth, they fall to approximately .01. Both fixed gates fail.
The extra nuisance parameters provide little evidence of intervention-specific
prediction. This is a point-baseline result, not a world-model or causal claim.

Runtime is 77.22 seconds, peak RSS 641,736,704 bytes. The runner streams one
moment shard at a time into sufficient statistics; it never stacks all
population targets in memory. Root review rejected an earlier unexecuted
full-stack implementation. Fresh artifact-only CPU prediction replay is exact.
Eight focused checks and Ruff pass. Protocol-v2 SHA-256
`2669369cca209616bb891f206ff3670854c9fabb5d041c00d4e1fcc4654fc53e`;
original report `e15c9b14dc37b4eae01ef1e5bc847860a2d39273c76c930cb12030e622488824`;
directory `results/slp11-transition/yeast-raw-count-batch-ridge-v1/`.

Root subsequently corrects a remaining constant-profile rounding artifact:
query centering first subtracts a reference gene row, avoiding accumulated
common-profile reduction residue. Frozen predictions and all MSEs are
bit-identical; every nonconstant correlation agrees within 1e-12 and every
decision is unchanged. Constant pooled means now have undefined perturbation
correlation. A 346-row common-profile regression check catches the former
error; nine focused checks pass. Superseding scoring-only report SHA-256
`291b8b34d9b03b0bedb6a40723cbab07b6fd2c094dbffdb5a5a644141454c128`,
`results/slp11-transition/yeast-raw-count-batch-ridge-roundoff-scoring-v1/report.json`.

### 2026-09-05 — Action-aligned basal-state input gap

Code inspection confirms that the current transition's action encoder receives
only static features. Its basal encoder averages 64 context-global query tokens;
there is no join between the intervention gene and that gene's measured
control abundance. Only 53 of 8,358 human source3 action genes happen to be
among those 64 tokens. Control-only sidecars now expose the missing quantity
without inventing zeros for unmeasured genes: 5,558 human actions are observed
on the fixed 6,789-gene control panel in all three source contexts; 1,956 of
2,013 yeast actions have exact WT abundance across the 29 batches, with 57
explicitly unobserved. Human values use log2 fixed-panel CP10k; yeast values use
the frozen all-6,951-row ln1p(CP10k) normalization. Units are not conflated.

Audit report SHA-256
`d8c205e34f2952d15f22f2256dab2f3ef19c7c589d44e930fcff021e8ebd8f4e`;
human sidecar `57957e763d9f284ae6770dca8c114c2805ccd439a4032bcaf3e6ba23fdf39de3`;
yeast sidecar `bb5e90e3c2491ab06abef44cdb235aab09756280dfe16451dbd9d250e526ad57`.
Three focused checks and Ruff pass. A matched human ridge test will compare
normalized measured abundance against a zero scalar with the same presence
flag in both arms. No perturbed outcomes enter the sidecar construction.

### 2026-09-05 — Corrected-count yeast neural pilot and three diagnostic decisions

The first neural pilot on the rebuilt yeast RNA endpoint completes exactly
12,000 updates in 131.81 seconds on the RTX 4070. It uses the frozen minimal
control-transition core, shared static577 actions and queries, 64 measured WT
basal tokens, and per-batch WT anchors. Fitting comprises 38,978 populations;
all 346 development-validation genes in each environment are scored once.
The fixed advancement rule requires 2% MSE improvement against pooled ridge,
batch ridge and both mean limits, with batch-reference-subtracted centered
correlation at least .10 and no regression against defined baselines.

| Environment | Neural MSE | Batch ridge MSE | Neural residual centered r |
| --- | ---: | ---: | ---: |
| Control | .022444462 | .021551289 | -.002914 |
| NaCl | .021143029 | .020231360 | -.005856 |

Every gate fails. Neural MSE is 4.14%/4.51% worse than batch ridge; this candidate
is rejected. Negative predictions comprise 5.57%/3.46% of entries and remain
unclipped. Gaussian aggregate means are not a validated cell generator.
Fresh artifact-only CPU replay differs by at most 7.15e-7; empty intervention
means are bit-exact. A final stdout serializer fails on numpy.bool_ after all
scientific artifacts are saved; its workspace fix does not rerun training or
validation. Five focused checks pass. Report SHA-256
`fb67323bcf6d3e405cbe8cb7ef0f60086e2b4a9cfbdf5e08194eb88f0270a8e7`;
model `88fd54046458663035ca5b4f05c483d4a0d1a13f99b29e7aa6f10fe43714324d`;
directory `results/slp11-transition/yeast-rna-world-transition-seed731-v1/`.

The matched human BP ridge basal-abundance experiment also fails all three
fixed 1% MSE gates. Control-to-basal MSE is .02422137 to .02421284 in K562,
.05309320 to .05308351 in RPE1, and .01186840 to .01186696 in genome-wide K562:
gains of .0352%, .0183% and .0122%. Centered correlations do not regress;
all selected alphas are 10,000. Explicit presence flags appear in both arms,
and no unobserved actions are dropped. The missing aligned scalar is therefore
not a substantial explanation for this ridge formulation's error. Runtime
288.5 seconds; fresh artifact replay is exact; three focused checks pass.
Protocol SHA-256 `350c09f9cb6196b50b300ac712b6131334c305f7f5b6641747371bb1e0a0db26`;
report `95d1f42727e75e5f384bf50da5304fe06bb6db3f9def59611c216b36e7932dc2`;
directory `results/slp11-transition/human-source3-bp-action-basal-ridge-seed731-v1/`.

A fitting-only yeast representation diagnostic fits rank-32 bases on two of
three gene folds and measures A/B reproducibility on the third. Positive
cross-covariance eigenvectors are compared with PCA on the fitting half-average.
Mean Control correlations are .0728 raw, .4314 PCA and .4346 cross-covariance;
NaCl is .1064, .5529 and .5564. The cross-covariance gains .0032/.0035 fail the
fixed +.02 rule in both environments. Projected A/B MSE falls 10.2%/11.0%,
but held shared-energy trace capture is lower than PCA. All six bases retain
32 positive eigenvalues. These are representation diagnostics, not intervention
forecasts; shared batch and clone effects remain. Runtime 13.47 seconds;
two focused checks pass. Protocol SHA-256
`741a653d715805c78e308a661cafbe047991aa5940bfea619d25e722b25309b9`;
report `4aebdcb26bcaa14026a3a44fbed83b6e49160580406715314167f1948b52d6c7`;
directory `results/slp11-transition/yeast-rna-fitting-crosscov-basis-rank32-v1/`.

The resulting next test asks whether a fitting-derived rank-32 response basis
improves prediction of the original full RNA measurements for unseen genes.
It retains full-panel MSE and batch-residual correlation as primary metrics;
better reconstruction alone cannot advance a candidate. A separate frozen
checkpoint audit will distinguish fitting error from generalization error.

### 2026-09-05 — Compatible human static pack and integrated verification

A human static577 pack now shares the exact ESM320, explicit protein presence,
and MF/CC GO256 coordinate system of the yeast pack. Its 24,031 taxon-9606
stable ENSG rows cover the 23,879 translated Ensembl-116 universe and the full
10,213-gene source identifier union. This broad static roster is distinct from
the 7,036-query numerical source3 panel. Protein features cover 10,061 source
genes and GO features 9,926; missing features remain explicit. Overlapping
protein vectors and the original GO vectors are bit-identical. Extra genes
are projected through the existing GO basis without refitting. No molecular
outcomes enter this pack, and compatible coordinates alone establish no
cross-species transfer result. Six focused checks pass. Features SHA-256
`20313e37d70d52253fa7b4b9b569b0fd504686a35be46b0607db1ab1c7484e54`;
manifest `857e34d73c45f94ef078cc1e2e271f91ec223d51e217f87d5411869457b3fe3c`;
directory `data/derived/slp11-human-shared-static/ensembl116-source3-esm8m-shared-go-complete-v2/`.

The integrated test suite passes: 745 passed, 21 skipped, 17 historical
Transformer warnings in 41.30 seconds. No release or SOTA claim follows from
these engineering checks or adaptive development experiments.

The response-basis diagnostic receives a no-refit scoring supplement. Its
original protocol subtracts separate fitting-fold query means; this supplement
also removes each held fold's own separate A/B centroid before applying the
saved bases. Control raw/PCA/cross-covariance correlations become
.07248/.41972/.42093; NaCl .10599/.54371/.54569. Thus the substantial projection
reproducibility persists, while cross-covariance gains only .00121/.00198 over
PCA and still fails the original +.02 rule. No basis, forecast or data selection
changes. One translation-invariance/constant-profile check and Ruff pass.
Supplement SHA-256 `32a5f3a85ce8f0c3809187f4d3f53a3552e7d7b798965633f6d8b27875557ae0`;
directory `results/slp11-transition/yeast-rna-basis-held-centering-supplement-v1/`.

### 2026-09-05 — Denoised forecasting fails; human exposure and cell-state work

The rank-32 yeast static-ridge forecast test retains all 346 validation genes
per environment and reconstructs the original 6,683-query measurements.
Fold-local PCA and positive cross-covariance bases use fitting split halves;
batch-intercept ridge uses the existing population weights and gene folds.
Both arms select alpha 100,000 in both environments. Control PCA/cross-covariance
MSE is .021546857/.021547388, only .0206%/.0181% better than batch ridge;
NaCl is .020230665/.020230590, gains .0034%/.0038%. Residual centered
correlations are .01702/.01613 and .01186/.01171. Every fixed 1% MSE and .10
correlation gate fails. Lower projected discrepancy does not improve full-panel
forecasting materially; this branch is stopped. Runtime 91.94 seconds, peak
RSS 714,797,056 bytes; artifact replay and 14 focused/baseline checks pass.
Frozen protocol SHA-256 `8a9776c8db21284798f0def3ee71a17e33ee7f8e6355b5aaed88a13b96335fd74`;
report `2b6e040d8c78dccb03feba54837246cf4fd97de566d4d558ccf601b1e1acc8ff`;
directory `results/slp11-transition/yeast-response-basis-static-ridge-rank32-v1/`.

A metadata-frozen 256-fitting-gene sample per context checks saved neural
checkpoints against their initializers and baseline fits. Every parameter
group changes. Human fixed-basis/learned-query fitting MSE is
.01843/.01894 in K562 essential, .03600/.03710 in RPE1, and .01067/.01063
in GWPS; each beats ridge on the same sample. Correct independently centered
correlations are .441/.433 versus ridge .475, .584/.570 versus .486, and
.151/.170 versus .264. Thus broad fitting-MSE underperformance does not explain
human failures. Yeast fitting objectives and MSE remain worse than mean/ridge.
These fitting samples do not identify a causal source of generalization error.
Root review finds and corrects a centering bug in audit-v2; audit-v3 supersedes
its correlations, leaving MSE, objective and parameter movement unchanged.
Five focused checks pass. Corrected report SHA-256
`97edded6d5ed4e08e1c6e4d16710e498a5b9484fefc17b5e416e675adf9e65c0`,
`results/slp11-transition/fitting-convergence-audit-v3/report.json`.

A human endpoint diagnostic accesses only 10,719 fitting rows. Spearman
correlation of log(1+filtered cells) with centered response norm is
-.4370/-.6605/-.8804 in K562 essential/RPE1/GWPS; on-target reduction versus
cell count is .0277/.0452/-.0058. On-target reduction versus response norm is
.1790/.0894/.1343, with exact observed action-query matches in
1,326/1,519/5,048 rows. Count-dependent dispersion could reflect sampling,
biology or selection; anticorrelation alone does not prove noise. No quality
exclusion follows. Measured efficacy is a post-intervention outcome and is
not an inference input. Bulk P1/P2 population labels are not stable guide IDs.
Report SHA-256 `4d659204f71b401e48695ecd3b8eb861bf3a1817abb0e53fa1e690e7772af945`;
protocol `243c74d7369ff74e29d850a1de2f054ea512bdd3815feae502514a5fbe1e9844`;
three checks pass, runtime 4.37 seconds.

Earlier models already used biological-plus-sampling/exposure variance in
Gaussian training. The next test applies those existing fitting/control-only
components to the later fixed-response-basis mean model, changing only its
loss precision. Precision is .8244632170073302 /
max(tau^2 + sigma^2 / n, .05^2), with one global fitting-derived scalar matching
the old weighted mean precision 51.3179990071. It is never renormalized per
minibatch. All 3 by 7,036 sampling components are marked control-identifiable;
counts remain absent from the mean/state. Component resource SHA-256
`d9acd063939535d819cdd70e1fcb9d26c38bf7ac04455958cdf3fc34bdb3425f`;
preparation report `358791078273166cc2877fe848686905336a74bcc42de120292b0f545b8c2a56`;
four focused checks pass. This is a prepared experiment, not a positive result.

K562 essential raw single-cell acquisition starts from official Figshare
file 35773219, version-1 release, under exact-file CC BY 4.0 rights. The source
is 10,661,879,995 bytes. A four-range profile transfers 256 MiB in 5.890 seconds;
projected remaining transfer plus hashing is 408 seconds within the 3,500-second
cap. The file is not yet declared complete here. No second large transfer starts.

In parallel, `modules/slp-1-1-count-latent-state-v1/` introduces an untrained
conditional Gaussian-state/NB2-observation prototype. A variational cell
encoder trains from integer counts; the prior receives only static actions
and measured controls. An analytic lognormal moment correction makes the
empty-action population mean equal the supplied control rate. Library exposure
enters observation likelihood and posterior inference, never the prior.
Six numerical checks pass, including an independent NB distribution oracle,
Gaussian-integrated means, masking, gradients and query/action invariance.
This prototype has no biological performance or portable release evidence;
raw-cell normalization and the training protocol remain to be fixed.

### 2026-09-05 — Exposure result and verified human single-cell source

The fixed-basis exposure-weighted neural experiment completes all 12,000
updates in 125.91 seconds and fails its all-context advancement rule. K562
essential MSE is .02324116 (gain .154%), RPE1 .05153812 (gain 1.064%), and
GWPS .01188233 (regression .165%) against the frozen fixed-basis mean model.
Centered correlations improve to .30991/.33379/.10572, respectively. Only
RPE1 passes the fixed 1% MSE/nonregressing-correlation rule. This changes
optimization usefully for response shape but is not a general mean-forecast
advance. Fresh artifact-only CPU replay differs by at most 5.36e-7; empty
means and repeat means are exact. Eight focused checks pass. Report SHA-256
`d79f010b8ca1cece4ec3c8b3395e843564818da36f1945c38863417dadc1f16d`;
receipt `f517bb2edbc66401de6464bb79b38749b9ab1185a6178ff40359bf76f790750d`;
directory `results/slp11-transition/human-source3-bp-fixed-response-basis-exposure-objective-seed731-v2/`.

The 10,661,879,995-byte K562 essential raw single-cell H5AD finishes downloading
and passes upstream MD5 `4f1122ce1c7f13299a68df6459a266d3` and local SHA-256
`3e5a63a9e892b21029bb55fca4e12517a49aad7af6c14133ca63d12cf68c6cee`.
Total acquisition time is 530.59 seconds including Windows preallocation,
transfer and whole-file hashes. It contains 310,385 cells, 8,563 unique ENSG
queries, 2,057 action genes, 2,273 exact guide-pair/population combinations,
48 GEM groups and 10,691 verified NT cells. Metadata routes 209,013 fitting
cells over 1,443 genes, 47,914 development-validation cells over 305 genes,
and 42,767 excluded test cells over 309 genes. A deterministic cell hash
within fitting genes/controls gives 197,804 reconstruction-training cells and
21,900 reconstruction-held cells. Guide pairs and GEM groups are preserved;
they are not asserted to be independent biological replicates.

Source path `data/sources/replogle-2022-k562-essential-singlecell-v1/K562_essential_raw_singlecell_01.h5ad`;
acquisition receipt SHA-256 `8f6d01ef1b47a848a7a97b57b91c1061336deeb528aedc182ef7e6a0f725b807`;
metadata audit `8b4eba21fe17f0082960bac9c66daf11ce9e3e9d1e3f1706ef30dba6c7d61270`;
routing sidecar `47c89c5082c0a9d4008c6b567407c530933a36fb7603621c37cbe913143f15ad`.

An allowlisted 8,192-row count profile takes 10.58 seconds, peaks at 1.10 GB
RSS and projects a 518-second shard build. Sampled counts are exact finite
nonnegative integers with no zero libraries. The retained-panel sums are
94.3%–99.4% of metadata UMI totals, consistent with upstream gene filtering.
The denominator remains the exact sum across the 8,563 retained columns;
metadata-total agreement is a diagnostic, not grounds to discard valid counts.
Profile-v2 preserves that clarification before full conversion. The four
role-specific shard paths keep fitting, controls, reconstruction-held and
development-validation cells distinct; excluded test rows are never selected.
Full conversion is still running at this entry.

Root's count-state core now factors its first control-encoder linear map and
encodes unique measured contexts once per optimizer step, avoiding repeated
cell-by-query-by-feature tensors. Eight numerical checks pass, including
gradient parity with explicit concatenation and repeated context encoding.
Its 516,129-parameter, full-8,563-query CPU forward/backward smoke takes .140
seconds at batch two and has finite gradients. Source SHA-256
`75df347a82151074c0ce6f4c732106e70ed17126aff07d017294894421d30bac`.
The prototype is still untrained. Its contract explicitly distinguishes fixed
posterior-panel encoding, arbitrary decoder queries, smoothed basal identity,
library-conditioned count factors and non-identifiable latent/noise components.

### 2026-09-05 — Raw-cell count-state pilot inputs and execution

K562 raw-cell conversion is complete. The canonical manifest is
`data/derived/slp11-human-k562-essential-raw-cells-v2/manifest.json`, SHA-256
`859b3fb0b0aeb830e25dce17e86edfc2d8ec3fcdbcec57beeeebf6d1a8faf685`.
Its 132 CSR shards contain 188,195 fitting intervention cells, 9,609 fitting
NT controls, 21,900 reconstruction-held cells and 47,914 development-validation
cells. No excluded test rows were selected; there are no zero libraries.
The original 900-second build cap stopped after 75 valid fitting shards. An
explicit 2,400-second execution amendment preserved and checksum-verified that
prefix; the remaining conversion finished in 235.46 seconds at 4.918 GB peak
RSS. This changed the operational allowance, not routing or the endpoint.

The sole random-access training pack contains 197,804 by 8,563 uint16 counts
(maximum 3,161), SHA-256
`e9bbfe69bd59cedf7131bd176632bb9fbd8dce59a0789ed7e18896ac34e4b511`.
It was built in 44.15 seconds at 1.148 GB peak RSS. Training gathers bounded
rows from this memory map rather than materializing a dense float32 corpus.
The static577 pack SHA-256 is
`6706f8867adedef8822897bc275ea90680584f84afd24771e4beb3c8ecf07659`;
the exact action/query roster is
`f2ee702a0714ca7f11f4fd2aa96f4c1825617c0e4f2bcdac42135cd0ba938d7b`.
It contains 8,752 species-native ENSG entities with ESM8M sequence features and
shared direct GO MF/CC coordinates. All 8,537 prior-pack overlaps are bit-exact.
There are 358 all-zero static rows, including 352 queries and seven actions;
the model cannot distinguish their static descriptions. Normalization uses
only the 1,443 unique fitting action genes.

All 48 experimental GEM groups have 109–265 reconstruction-training NT cells.
The control reference uses only those 9,609 cells and the explicit formula
`10000 * (pooled_query_count + .5) / (pooled_library + .5 * 8563)`.
Reference SHA-256
`c72d28e9eb6633fa237b11e0c16258d875eadaacf31e5b8b3def862150b36d13`.
Independent rebuilds of static features, the roster and control rates match
their original artifact bytes exactly.

The matched anchored static ridge is frozen before development evaluation:
`results/slp11-transition/k562-essential-count-anchored-static-ridge-seed731-v1/`.
Model SHA-256
`dbb669d2eb8d844ec9be7c88a2ed21f5592de434d1b2e916412bda4a52fe1cf3`.
Three fitting-gene folds select alpha 1,000 with MSE .0040148722, versus
.0043316485 for the anchored mean. The endpoint is the natural logarithm of
one plus each gene's equal-cell mean CP10k. The anchor uses that gene's GEM
cell proportions and the frozen smoothed control rates. All 8,563 queries
remain scored. Fitting takes 1.97 seconds on two CPU threads; copied-source
artifact reload is exact. Twelve focused checks pass.

Before biological fitting, an independent Poisson teacher smoke trains the
count-state core on 64 scalar action values and evaluates 63 interleaved held
values. Prior-only mean MSE is .44435 against control-only 16.06151 after 500
updates in 3.15 seconds; the empty-action mean remains exact. This verifies
simple action-conditioned interpolation, not biological or out-of-distribution
prediction. Report SHA-256
`56d85848160a9590c548d1560e0a6c81f9e42a7668742467ba62fedb19ccaaa9`.

The biological pilot hypothesis is that cell-level count-state learning
improves held-gene aggregate means over both matched anchored mean and ridge.
The fixed advancement rule requires at least 1% lower full-panel MSE than
both baselines, and anchor-subtracted independently query-centered correlation
at least .10 and no lower than ridge, over all 305 development genes. Accessible
modalities are fitting raw RNA counts, NT controls, static sequence/GO and
experimental metadata. Neither counts nor library totals enter the prior.
The fixed fit is 12,000 updates, batch 128 with 64 GEM-uniform controls and
64 gene/population-uniform intervention cells, seed 731, AdamW .0005 with .01
weight decay, clipping at one, beta-one normalized ELBO and final checkpoint
only. Protocol SHA-256
`a85d2ab7cb83760a818614f20ab28d2936c3604c4f9236293c18b355391b89e7`.
A target-free full-shape CUDA profile projects 353 seconds and uses 1.158 GB
allocated GPU memory. Execution has started; biological results are pending.
This is an adaptive K562 development experiment with a changed endpoint and
data representation, not a causal ablation of the earlier author-z model.

RPE1 essential raw single-cell acquisition is also complete in 447.77 seconds:
`data/sources/replogle-2022-rpe1-essential-singlecell-v1/rpe1_raw_singlecell_01.h5ad`,
8,700,873,216 bytes, upstream MD5 `6a2a9d0d2bf4ec147f4d1104043b268c`,
SHA-256 `9b05ef1f81526216fa008d677e9e0d03dce9a2f7a95499a4fb81e505e9d88ef1`.
Metadata contains 247,914 cells, 8,749 stable ENSG queries, 2,390 resolved action
genes, 56 GEM groups and 11,485 verified NT controls. Global routing gives
158,538 fitting cells over 1,666 genes, 39,014 validation cells over 360 genes,
38,419 excluded test cells and 458 unresolved-action cells. The unresolved
symbols RBM14-RBM4, PRSS50 and NEDD8-MDP1 remain excluded rather than being
assigned ambiguous stable identities. Reconstruction routing gives 142,601
training targets plus 10,350 controls and 17,072 held cells. Every K562 fitting
action is also a RPE1 fitting action under the same global gene split. Query
overlap is 7,226. Routing sidecar SHA-256
`10f3d313a5671122bde10a9bd586e3a2808d6f9b554f737ddcbbc28becc5e2f2`;
metadata report `323bece05ee1ccf2d51dad336ce3a671a47cc32f0a92db711be6683b5b3f668a`.
No RPE1 count values were read by this audit. A subsequent authorized build
will initially convert fitting, control and reconstruction-held rows only,
using the exact retained 8,749-column library denominator. RPE1 quantitative
validation, test and unresolved rows remain unselected.

The K562 count fit finishes all 12,000 updates in 413.43 seconds. The first
100-update loss average is 1.07548 and the final average 1.02882. Final
reconstruction loss is 1.02695 per query and KL is 15.95 per cell, with ten
latent units above the diagnostic KL threshold. Final lower-variance-bound
fractions are 2.04% for the prior and 1.73% for the posterior; upper-bound
fractions are zero. Every parameter group changes. These training diagnostics
do not establish forecast accuracy.

The frozen execution stops before development evaluation because its strict
CPU/GPU absolute-CP10k replay tolerance of 1e-5 is exceeded: maximum difference
3.0518e-5, relative difference 2.921e-7 and absolute ln1p difference 2.074e-7.
Repeated CPU predictions and empty-action identity are bit-exact. The original
failed numerical check is preserved. A separate continuation will verify an
explicit relative-CP10k/absolute-ln1p tolerance of 1e-6 and perform an isolated
CPU subprocess replay before scoring the unchanged checkpoint. This is a
numerical verification amendment, not model selection or changed biological
advancement criteria. Full focused repository suite at this point: 806 passed,
21 skipped and 17 historical transformer warnings in 52.47 seconds.

### 2026-09-05 — Count-state result and a matched molecular-mean continuation

The first K562 count-state pilot fails its fixed advancement rule. On all
305 adaptive development genes and 8,563 queries, neural MSE is .0037701751
with independently query-centered, anchor-subtracted correlation .1877687.
Static ridge achieves .0035845848/.2221895; the anchored mean MSE is .0039796800
and the pure control MSE .0043260864. Thus neural gains 5.26% over the mean
but regresses 5.18% against ridge. All predictions are nonnegative. This is
evidence of learned intervention signal, not a successful candidate.

The unchanged model SHA-256 is
`c7cc6a369f8b63d936c535f7cc59439fec38033202d4b98616b02270df74f3f8`,
reference `8020753e9e2597b08cb94c5351772be05986b286f61e0f7a26be26fbfabae4f6`.
An isolated artifact-only CPU subprocess passes the amended replay gate, with
maximum relative CP10k difference (unit floor) 2.921e-7 and maximum absolute
ln1p difference 2.074e-7. Repeated CPU and empty means are exact. The final
pre-development freeze is `FROZEN-BEFORE-DEVELOPMENT-V3.json`, SHA-256
`a8936f7e6a8f1ebed65a91ce4b91d8f375c5cf61b1fe300ddb1c8b681eb57208`.
The original failed strict tolerance is not retroactively marked passed.
Final report directory
`results/slp11-transition/k562-essential-count-latent-state-seed731-portable-finalization-v2/`,
report SHA-256 `62a8cb6a766ac3eb0b8767d8905178c407cfe87cc53e7e153954572e51470bbb`.
Forecasts were saved before the single development aggregation, SHA-256
`1f1b956abbe292b0f209df99c202073f62c49f9ba5cc8fe998289dcbce59c3b3`.
Only aggregate metrics were persisted; gene-bootstrap uncertainty is not
available from this report. No protected test or benchmark was opened.

Four fixed antithetic posterior draws on all 21,900 reconstruction-held cells
give negative ELBO 1.02366279 per query (reconstruction 1.02171012,
KL 16.72067 per cell). These cells come from fitting genes/controls and are
not independent biological replicates or held-intervention evidence.

The separately prefixed 128-gene fitting diagnostic selects genes by identity
hash before reading moments. Neural prior MSE/r is .0038934575/.3381267,
final-fitting ridge .0032204794/.5035930, and anchored mean MSE .0044650042.
All four profiles use the same gene-specific GEM control anchor. Predicted
total CP10k has minimum/median/maximum 9822.49/10016.86/10199.48. The 6.73-second
CPU-only diagnostic preserves its original scientific protocol; its v2
execution accepts the explicit amended replay freeze filename. Protocol
SHA-256 `419702cfc107087dce01360175c14a475b3360b9f0667d024e14c86f26c5da8d`;
directory `results/slp11-transition/k562-essential-count-prior-fitting-audit-v2/`.

A complementary, predeclared fitting-only variance diagnostic on those exact
128 genes neutralizes the intervention variance term without refitting. MSE/r
becomes .0038721202/.3469245, still substantially behind ridge. Weighted RMS
log-ratio terms are .0618242 from the mean and .0123350 from variance; forecast
ln1p RMS difference is .00476193. Variance contributes modestly, not a dominant
mass or variance failure. Report SHA-256
`c7a9243621143e9f6eccfd5c614e11edd53898d042c9d9c7458988b666cd389c`;
runtime 2.75 seconds on two CPU threads; three focused checks pass. Neither
fitting diagnostic reads reconstruction-held, development or test outcomes.
The fit gap motivates testing the molecular mean objective; it does not prove
a particular optimization or representation cause.

The next fixed hypothesis is that adding aggregate molecular-mean supervision
closes this prior fit gap while retaining cell likelihood. Two arms start from
the exact saved checkpoint with new AdamW optimizers, seed 1731, and 4,000
updates: count ELBO alone, or ELBO plus .1 times population-mean MSE divided
by the frozen final-fitting anchored-mean MSE. The auxiliary batch samples 16
fitting genes uniformly, mixes prior CP10k over their fitting-cell GEM
proportions before ln1p, and uses all 8,563 queries. Its dropout is disabled
while retaining gradients so it matches the inference mean. Cell sampling,
learning rate, decay and clipping match the previous fit. Both final models
and forecasts must freeze before the common development aggregation.

Advancement still requires at least 1% lower MSE than mean and ridge with
centered residual r at least .10 and no lower than ridge. It additionally
requires reconstruction-held negative ELBO no more than 1% worse than either
the original model or the matched count-only continuation. A new self-contained
helper implements only differentiable rate mixing and scaled molecular MSE,
SHA-256 `f9dc1fc1d7c6f1071f5bdb98e45a5140116cb583975bf3a76892814883989cd9`.
Four numerical checks pass, including an independent analytic gradient oracle.
The helper is a composite training objective, not a calibrated likelihood.

RPE1 raw-cell preparation meanwhile finishes in 250.03 seconds at 3.517 GB RSS.
The bounded reader closes each source-row-span memory map after copying only
selected rows. Canonical manifest SHA-256
`3d7ca31f945ffb193070eb463eaa328e374c9f12f3c0e3162a5e189f24d0fe9e`
in `data/derived/slp11-human-rpe1-essential-raw-cells-v1/` describes 85 CSR
shards: 142,601 fitting, 10,350 controls and 17,072 reconstruction-held cells.
The sole uint16 training pack is [152951,8749], maximum 3,061, SHA-256
`6df95d35bd725dd935e368859391a99fc7e82f2019b1700eabfc744c01481ba6`;
CSR parity is exact. Fitting moments are [1666,8749], SHA-256
`d15def86aead06b0bc75ab63c77513735ec7c57d65012bff72f3947bc654895c`;
control moments [56,8749], SHA-256
`5aceba5fb4874811aac797be14d1947a9fca866d11178d5f8fe2bdc534df6f61`.
All GEMs have 107–231 fitting controls. Development, test and unresolved count
rows remain unselected. Five focused RPE checks pass.

RPE1 raw static577 features cover 9,032 species-native entities, SHA-256
`621e1e9f0dffc740ef42382b1b2898f629edd5037e8a02d411e8d30e815ed816`.
The exact source query/action roster SHA-256 is
`b9e1b169c2be4ac756e94f465009dc5bef80d06bc0652950c3cf6916d26d1e56`.
All 7,517 K562-pack overlaps are bit-exact. There are 229 all-zero queries and
seven all-zero actions. Separate RPE1/K562 fitting-roster normalization
statistics retain float64 precision; their artifact SHA-256 is
`d397dfdb08973ccf9884d504a1279042cc470cba2ff5341c770443d0c7915951`.
All five static artifacts reproduce byte-for-byte; six focused checks pass.

### 2026-09-05 — Mean-auxiliary result, transfer limits and joint-training readiness

The matched count-state continuations complete from the same frozen K562
checkpoint. Both use seed 1731 and 4,000 updates; count-only takes 142.25
seconds and the population-mean auxiliary arm 227.95 seconds. Forecasts for
all 305 development genes and 8,563 queries were frozen before the single
development aggregation. The auxiliary arm reaches MSE .0037003113 and
independently query-centered, anchor-subtracted correlation .2113435, versus
.0037579845/.1888186 for the count-only continuation, .0039796800 for the
anchored mean and .0035845848/.2221895 for static ridge. Reconstruction-held
negative ELBO is 1.023829 per query for the auxiliary arm and 1.023354 for
count-only, within the fixed one-percent preservation limits. The auxiliary
model improves the mean but remains 3.23% worse in MSE than ridge and has lower
correlation, so the advancement rule fails. Portable CPU replay and exact
empty-action means pass for both arms. Model SHA-256 values are
`994ad65b310dae8e815a568ca3cece2d4922125fdf22f331f36d237016a58154`
for mean-auxiliary and
`b2de5c729647a8e40bfa134820dd0dcc4e07088fe67d7346a9f329ac919a248a`
for count-only; frozen forecast SHA-256
`899205bfbf61ea948ff1786086286e8ae50eda3fafc24e4ec580fe23a748331e`.
Report SHA-256
`3b14c241f7419eb004b03813089bee948e5c033b82d8bc27bc126d90cc4fb6fa`;
directory
`results/slp11-transition/k562-essential-count-latent-mean-aux-continuation-seed1731-v1/`.

Paired resampling of the 305 gene-level errors supports the mean-auxiliary gain
over the anchored mean: its 95% relative-MSE interval is [2.56%, 11.09%]. Its
gain over count-only is less certain, [-1.59%, 4.33%]. Against static ridge the
interval is entirely unfavorable, a relative regression of [.84%, 5.69%],
with an absolute candidate-minus-ridge MSE interval
[3.08e-5, 1.99e-4]. These intervals summarize the same adaptive development
genes and are not independent confirmation. Report SHA-256
`226fa63e6aa2951efc06436c7768ff48c0a4c78ec7ca14e89f8ade1bf6c1d9dc`;
directory
`results/slp11-transition/k562-count-mean-aux-paired-gene-intervals-v1/`.

The frozen K562 count checkpoint is also applied to an externally supplied
RPE1 control reference without fitting on RPE1 perturbation outcomes. Source-
only comparator predictions were frozen before RPE1 fitting moments were
opened. Subsequent descriptive scoring on RPE1 fitting moments gives, over all
1,666 genes and 7,226 common queries, MSE/correlation .0135802/.140215 for the
K562 count prior and .0130571/.129555 for the transferred K562 static ridge.
The prior's correlation is .00714 on the 223 RPE1-only action genes. Across the
1,523 RPE1-only queries it is .00740 and MSE .0134922, worse than the pure RPE1
control anchor at .0131716. An RPE1-fitted ridge is substantially better but is
an unequal, context-supervised descriptive reference. The checkpoint was never
fit to RPE1; this demonstrates functional control-context substitution, not
useful unfitted-context intervention transfer. No RPE1 development or test
counts were read. Report SHA-256
`bd22db75ad27d2dab5a71cebbaa9e64dc72437bd42a9065a8a32dc9db5b4fdd6`;
directory
`results/slp11-transition/k562-prior-rpe1-unfitted-context-fitting-score-v1/`.

Two fitting-only static-feature screens reject additional descriptors before
development access. A 71-dimensional dual-guide sequence-composition
descriptor changes three-fold gene-held OOF MSE from .0040148722 to
.0040560207 in K562 and from .0095232508 to .0096626124 in RPE1, regressions
of 1.02% and 1.46%. The fixed one-percent improvement rule fails in both
contexts, so no development model is fitted. Report SHA-256
`8bcd695ebf5c53c6cd81a4cfaba0e35a9e6a17d30b7ad0b97f21497759b8790c`;
directory
`results/slp11-transition/replogle-guide-composition-ridge-fitting-cv-v1/`.

The second descriptor uses only reconstruction-training non-targeting controls.
For each source it residualizes per-cell ln1p(CP10k) on log library within GEM,
then computes 64 leave-self-out correlations against one fixed static-derived
random anchor shared over the exact 7,226 common queries. Split-half reliability
passes its predeclared gate: all 8,563 K562 and 8,749 RPE1 queries are defined,
with median cosine .780733 and .900498. Independent full-source replay is
byte-identical. These stable coordinates still fail the fitting-only forecast
gate after appending a presence flag: K562 OOF MSE changes from .0040148722 to
.0040156010, a .018% regression, while RPE1 improves to .0094715688, only
.543%. No development model or neural feature tuning follows. Stability report
SHA-256
`6f90802b9090fefae5775002328170d8aee85206549fa29f669ff9937303f473`;
CV report SHA-256
`296275ac7c152ccf966a0e1be81208790964bce1600f44df8168af15a76175e2`;
directories `results/slp11-transition/human-control-coexpression-reliability-v1/`
and `results/slp11-transition/replogle-control-coexpression-ridge-fitting-cv-v1/`.

The outcome-free K562/RPE1 joint registry passes its identity and static-space
contract. Shared static577 contains 10,267 exact ENSG entities; all 7,517
overlap rows are bit-exact, K562's 1,443 fitting actions are a strict subset of
RPE1's 1,666, action roles have no conflicts, and the 48 plus 56 source/GEM
context identities are unique. Each source keeps its native query axis and
full-panel count denominator; no joint target matrix is constructed. Registry
SHA-256
`4de798e53a4d8149c200088e054caa4c9b71ecea91e6c00c68ecd3a6c938127c`;
report SHA-256
`d4a85c77139c4fe348dddc81ec0fdfb7d7ddd797dead4be4888ac26b8a13a3b8`.
The self-contained shared training step copies count core SHA-256
`75df347a82151074c0ce6f4c732106e70ed17126aff07d017294894421d30bac`
and molecular-mean helper SHA-256
`f9dc1fc1d7c6f1071f5bdb98e45a5140116cb583975bf3a76892814883989cd9`;
training-step SHA-256
`da544a7b969ddda4f6f4b44c77a8327c8be394746e6ea81ab012e56cc03a4062`.
It encodes one source's full control table once per step and keeps cell ELBO
and dropout-free population-mean losses separate while alternating native
panels. Four registry checks and five shared-training-step numerical checks
pass, but no joint biological training has run and no launch or transfer claim
follows from code readiness.

### 2026-09-05 — Shared-context training and response-rank diagnostic

The real panel adapter now loads both full native training packs in 7.41
seconds on two CPU threads. It reconstructs both saved control references
exactly: K562 has 197,804 cells, 8,563 queries, 1,443 fitting actions and 48
contexts; RPE1 has 152,951 cells, 8,749 queries, 1,666 fitting actions and 56
contexts. Adapter SHA-256 is
`a8f1ee3537041d20e1dda330c20ec0f73b3265ac63024eb3114ec1161d072c66`.
Five adapter checks pass, including identity alignment and read-only metadata.

The matched shared-context experiment freezes protocol SHA-256
`2b21d0aa4f609b5bc9d68632225b596783420efd0bd590aad4d3229d505a1954`
in `results/slp11-transition/human-essential-count-shared-context-seed731-v1/`.
Both arms start with seed 731 and the same shared static577 normalizer, train
12,000 count-only updates, reset the optimizer, then train 4,000 updates with
the fixed population-mean auxiliary objective. The joint arm alternates K562
and RPE1 equally, so each source receives half the exposure of the K562-only
arm at equal total compute updates. The fixed advancement rule requires at
least 1% lower K562 MSE than K562-only, at least 1% lower MSE than both mean
and static ridge in each source, centered residual correlation at least .10
and no lower than ridge in each source, and K562 reconstruction-held ELBO no
more than 1% worse than K562-only. Accessible modalities are fitting raw
counts, measured controls and static protein/GO features. Actual full-panel
mean-auxiliary steps take .03596/.03768 seconds in K562/RPE1; conservative
16,000-step projections are 575/589 seconds and peak reserved GPU memory is
2.51 GiB. Each arm has a fixed 1,500-second cap. The source snapshots and
query axes remain those pinned by the joint registry; rejected guide and
coexpression features are excluded. Training and subsequent frozen evaluation
are pending at this entry; no advancement decision follows from the profile.

A separate fitting-only diagnostic tests whether a rank-32 response map can
preserve full static ridge MSE within 1% in both contexts. It solves exact
rank-constrained regularized least squares, retaining an unpenalized intercept,
the existing three global gene folds, fold-local feature normalization and
alpha 1,000 previously selected in fitting CV. Predeclared ranks are 32, 64,
128, 256 and full. K562 OOF MSE is respectively .0038846644, .0039201181,
.0039693314, .0040080611 and .0040148722; RPE1 is .0092495505, .0093253380,
.0094254571, .0095072644 and .0095232508. Rank 32 improves full ridge by
3.24% and 2.87%, with gains in every fold, passing the sufficiency diagnostic.
This does not establish that the nonlinear count model's latent dimension is
adequate: the diagnostic learns native-panel output loadings and does not
require a feature-query decoder or forecast unmeasured queries. It argues
against enlarging the latent dimension solely to reproduce ridge's response
rank. No development or test outcomes are opened and no development model is
fitted by this diagnostic. Runtime is 7.47 seconds on two CPU threads.

Rank protocol SHA-256:
`2996c5f79694b6d4f879f2ba7e56e25bce48c685df5dde3515fe60ff905a67d2`;
report SHA-256:
`8adcd2b064d327a4d7b8df1ff82027263b535bf6334fe93fea674ec8c906e6b6`;
per-gene error SHA-256:
`6dedc60daa4e327482288698b57c689604c6a81f41b55aec4ad5712b73fa9180`;
directory `results/slp11-transition/human-essential-count-response-rank-audit-v1/`.
Three numerical tests pass, including a direct augmented-design SVD oracle,
rank-one recovery and full-rank parity. The preceding consolidated suite passes
884 tests with 21 skips and 17 historical warnings; these three new checks are
additional. These are native exploratory experiments, not admitted OMF releases.

### 2026-09-05 — Joint count model rejected; decoder conflict not supported

The frozen matched experiment completes 16,000 updates per arm in 562.98
seconds for K562-only and 585.44 seconds for joint training. Saved checkpoints
are unchanged through finalization: K562-only SHA-256
`49f782166291f8ae658f6895f92e1d1268b599a74fda50f206cacb039bf7afbb`
and joint
`cff1f691130bdf78bc574dfdcc449d5aeb8a3f81914f37ecf3f8c49686b87dff`.
A post-training four-row replay helper incorrectly returned a 16-row chunk;
an append-only finalizer corrects the slice. Its first continuation also used
a probe filename inconsistent with the isolated verifier; the second corrects
that operational mismatch. Neither correction changes weights, training or
evaluation rules. GPU and isolated CPU replay pass before both source-native
forecast bundles are frozen. Forecast-freeze SHA-256:
`9f6233f0e9c3cfc54f06cadc28c63dae2430733691c22ed9df4481f9fbbef523`.

Development evaluation uses 305 K562 genes over 8,563 queries and 360 RPE1
genes over 8,749 queries. Metrics below are absolute log1p-mean-CP10k MSE
and independently query-centered, control-anchor-subtracted profile Pearson.

| Source | Model | MSE | Residual Pearson |
|---|---|---:|---:|
| K562 | Anchored mean | .0039796800 | Undefined |
| K562 | Static ridge | .0035845848 | .2221895 |
| K562 | K562-only count state | .0036937959 | .2087663 |
| K562 | Joint count state | .0036797558 | .2038649 |
| RPE1 | Anchored mean | .0093290643 | Undefined |
| RPE1 | Static ridge | .0084534410 | .2652432 |
| RPE1 | K562-only count state | .0122427031 | .0783603 |
| RPE1 | Joint count state | .0090943861 | .2423476 |

Joint training improves mean MSE by 7.536% in K562 and 2.516% in RPE1,
but regresses ridge by 2.655% and 7.582%, with lower correlations in both.
K562 improves only .380% relative to matched K562-only, below the required
1%. The forecast gate fails. Four-draw antithetic reconstruction-held ELBO on
21,900 K562 cells is 1.023770603 for K562-only and 1.026722211 for joint,
a .2883% regression within the 1% preservation bound. Reconstruction does
not rescue the failed forecast decision. Joint fitting MSE/correlation is
.00324242/.40676 in K562 and .00670896/.58248 in RPE1; the larger RPE1
fitting-to-development gap remains a generalization limitation.

Evaluation completes in 29.3 seconds on two CPU threads, aggregating exactly
47,914 K562 and 39,014 allowlisted RPE1 development cells. No protected test
or unresolved RPE1 rows are selected. This opens RPE1 development counts for
adaptive molecular development; it is not untouched confirmation. Report:
`results/slp11-transition/human-essential-count-shared-context-development-evaluation-v2/report.json`,
SHA-256 `ca6438891609689cd2b00b0d9987f5ba44ff1270b1a3a3cb076b488a8bd25e07`.
Reconstruction diagnostic SHA-256:
`d773b0f0140fed2c28418c611ce5df06a9032fc3ba8a58f3e123c9a81e2a5074`.
The model remains rejected, with no SL benchmark, SOTA or launch claim.

The next fitting-only diagnostic freezes a local optimization hypothesis:
strong conflict requires median shared-query-loading gradient cosine below
-.25 and at least 75% negative batch cosines in each joint-training source.
Sixteen fixed batches per arm/source use seed 2831, 64 fitting controls plus
64 gene/population-balanced targets, and 16 unique fitting population genes.
Separate gradients use cell ELBO and the actual .1-weighted normalized
population-mean objective, preserving training dropout behavior. No optimizer
step occurs. Joint median cosine is .164729 in K562 and .314777 in RPE1;
all 16 cosines are positive in both contexts. K562-only also has entirely
positive samples, medians .217220/.357153. Joint auxiliary/count gradient
norm ratios have medians 2.88859/2.41526. The strong-conflict hypothesis is
rejected for these sampled checkpoint gradients; a split decoder is not
justified by that proposed mechanism. This does not exclude conflict earlier
in training or other representation limits. CUDA computation takes 2.25
seconds after loading, on fitting data only.

Gradient protocol SHA-256:
`2764e6ed824b36f0169acc00a699c0c2fcd63efafa56789a28a38bc57435a065`;
report SHA-256:
`7c5d6eb71d1660b283d8fc01424e0ef50bdfd52b0d1d142e573eb44030654818`;
directory `results/slp11-transition/human-essential-count-objective-gradient-audit-v1/`.

### 2026-09-05 — Rank-32 response model retained; static query decoder loses signal

A new fixed follow-up to the positive fitting-only rank diagnostic fits rank
32, alpha 1,000 on all fitting genes separately in K562/RPE1. Both models and
both source-native development forecasts are frozen before one access to the
existing development truth bundles. The rule requires at least 1% lower MSE
than full static ridge in both sources, centered residual Pearson at least .10
and no regression versus ridge. K562 MSE/correlation is .0034327136/.2487532
versus ridge .0035845848/.2221895; RPE1 is .0081592422/.285890 versus ridge
.0084534410/.2652432. Relative MSE improvements are 4.237% and 3.480%.
Every gate passes. The rank-32 model also improves the rejected joint count
model by 6.71%/10.28%. It is retained as the strongest measured-panel response
model in this comparison and becomes a stronger neural comparator. Runtime is
10.33 seconds on two CPU threads; three exact-formula/reload/query tests pass.

The model maps raw static action features into a 32-dimensional state with an
unpenalized intercept and fitted native-panel query loadings. These are
quantitative descriptors learned from fitting outcomes; they are not static
biological priors. No learned action IDs, new-query prediction, cell generator,
nonlinear dynamics or independent confirmation is established. No protected
test or SL benchmark outcomes are opened.

Protocol SHA-256:
`c6d854117b1da1fa37e3fea0b9c64b82ae37a8fcd4a451e4bf011c3e1487299f`;
report SHA-256:
`f1f97a9cb5d4b782db969f6ed0aa83a4ab39ea81b6534e575253252fe9bc49af`;
K562 model SHA-256:
`6267584a4a69dc30899b18d0c9660e0c73d2b8383a1e4911571295a1ea57ae44`;
RPE1 model SHA-256:
`ff864e96d02fb81b64baadc36c164de61a01d9e7d31a2609f78b64d48107be70`;
directory `results/slp11-transition/human-essential-count-response-rank32-seed731-v1/`.

Descriptive 10,000-resample paired-gene bootstrap intervals with seed 731 put
the rank-32 MSE improvement over ridge at [3.64%,4.92%] in K562 and
[3.11%,3.90%] in RPE1. These are conditional on the adaptive development
cohorts and fixed checkpoints, not biological-replicate, seed or context
uncertainty. The comparison report and checked PNG/SVG are in
`results/slp11-transition/human-essential-count-response-comparison-v1/`.

The unchanged models are packaged locally in `local-research-inference-v1/`
under the rank-32 result directory. The 12.73 MB bundle includes native
controls, static feature caches and self-contained inference source. It
requires explicit caller-supplied experimental-group weights, mixes control
rates before `ln1p`, and adds signed residual predictions without clipping or
renormalization. Seven feature/metadata probe rows per source reproduce the
frozen forecasts to maximum absolute error 8.88e-16. Isolated CPU CLI loading
works without the training corpus and preserves caller arrays. Seven focused
core/API tests pass. Bundle manifest SHA-256:
`0a181cc51fa29990e175b74e261687ac4af5a4796aeda8eeb0763d825252dbd5`.
This is a local research bundle, not an uploaded or admitted OMF release.

A separate fitting-only diagnostic tests recovery of the supervised query
loadings from static features. Within each of the three global fitting-gene
folds it derives rank-32 loadings only from the other two folds, scales each
loading coordinate by query-panel RMS, and trains a 577-to-256-to-32 GELU
decoder with a linear residual path. Both output layers start at zero; AdamW
uses learning rate .001, weight decay .01, seed 731, 2,000 updates and 1,024
uniformly sampled queries per update. There is no early stopping or sweep.
The frozen rule requires OOF MSE within 1% of the exact teacher in both sources.

K562 decoder OOF MSE is .0040238448 versus teacher .0038846644, a 3.583%
regression; RPE1 is .0095703299 versus .0092495505, a 3.468% regression.
Every fold regresses. Standardized descriptor reconstruction MSE averages
.32951/.31151. The rule fails, supporting a static-query representation limit
under this decoder and optimizer, rather than a universal impossibility claim.
The 174,656-parameter experiment takes 31.28 seconds and peaks at 81.1 MB
CUDA allocation. Three numerical tests pass; no development, reconstruction-
held or protected test values enter this experiment. Protocol SHA-256:
`e622b0319631c2178e1d947715cc78e081ca73c0dac874492920d06528468a1b`;
report SHA-256:
`c8d5defb29b890db5463af3c9ead4d1431b5fe4a285d7d2a910731251e85884c`;
directory `results/slp11-transition/human-essential-count-query-decoder-capacity-fitting-v1/`.

### 2026-09-05 — Matched count training with fitted response-query descriptors

The next experiment tests whether retaining the supervised query descriptors
can improve the molecular count forecast, after the static-feature decoder
failed to reproduce their predictive signal. Each source supplies 33 fitted
query coordinates: the retained rank-32 model's query loading plus its residual
intercept. Columns are divided by their native-panel RMS, without centering
or cross-source rotation. These quantitative descriptors come only from the
existing full-fitting response models. Static577 query features are appended
with either these 33 values or 33 zeros; action features receive exact zero
padding in both arms. Both count models therefore have the same 610-column
input width and parameter count. Counts, targets, control references and
sampling are unchanged. Source-native descriptors do not support unmeasured
query or new-context claims.

The sole feature adapter validates source, experimental context, query order,
rank and alpha. Real PanelData integration preserves the original quantitative
objects in both modes. Four focused checks pass. Feature-manifest SHA-256:
`8af044cedd683364ca789dd083f6c815740b8111fd59aec0aa326fdb734a27ba`;
adapter SHA-256:
`6a47563e97fd4cd710788917b35e40e5895abb5da340d66b8e53f5189f141807`;
feature directory
`data/derived/slp11-human-count-response-query33/rank32-alpha1000-full-fitting-v3/`.

Both arms alternate K562/RPE1 for 12,000 count updates and 4,000 mean-auxiliary
updates, with identical seed 731 initialization, sampling, optimizer reset,
learning rate, loss weights and final-only checkpoint selection. The fixed
rule requires response33 MSE at least 1% lower than both zero33 and the retained
rank-32 response model in each context, centered residual Pearson at least .10
and no lower than rank32, and K562 reconstruction-held ELBO no more than 1%
worse than zero33. Each arm has a 1,500-second limit on the RTX 4070.

The actual untrained full-panel artifact smoke passes every arm/context GPU
and isolated CPU replay before training: maximum absolute log1p discrepancies
are 3.48e-7/3.07e-7. Each arm has its own query reference. Actual mean-auxiliary
step profiles project 625.4 seconds for zero33 and 596.0 seconds for response33,
with peak reserved GPU memory 2.70 GB. Protocol SHA-256:
`d860cc04df1aa672106d399bdb71314a90f770b6ccfc2a5bb36af398a0950d61`;
runner SHA-256:
`d63d91e953744997f48e68449fc9486110b18e0f64f4c54a63a0c1324ce9fad5`;
run directory
`results/slp11-transition/human-essential-count-response-query33-seed731-v2/`.
Training is running at this entry. Both fitted models and all forecasts must
be fixed before the saved adaptive development truth is scored. Protected
test and SL benchmark outcomes remain unused.

### 2026-09-05 — Response-query count comparison completed and rejected

Both fixed 16,000-update arms complete: zero33 takes 587.69 seconds and
response33 580.54 seconds. All four trained arm/context GPU and isolated CPU
artifact replays pass, with maximum log1p discrepancy below 7.72e-7. Model
SHA-256 values are
`b66ffc9afef109ae48d53e40781fdd0b8ccee90522fc44665e38eaca309f80e1`
for zero33 and
`7120c198921a73abebcfa406aa8fe5d18dcc9a15ca13ddb82a0aedbe99217cf0`
for response33. Both models and all forecasts are frozen before development
truth access; forecast-freeze SHA-256 is
`b58c25e0edf2351caf1609c1ba339b8bbc4592f17b3a4d568de34f0e62172cd4`.

| Source | Model | Development MSE | Centered residual Pearson |
|---|---|---:|---:|
| K562 | Zero33 count control | .0036813121 | .1981368 |
| K562 | Response33 count candidate | .0036423909 | .2368486 |
| K562 | Retained rank32 | .0034327136 | .2487532 |
| RPE1 | Zero33 count control | .0092027190 | .2405053 |
| RPE1 | Response33 count candidate | .0090348808 | .2624321 |
| RPE1 | Retained rank32 | .0081592421 | .2858901 |

Response33 improves its matched count control by 1.057% in K562 and 1.824%
in RPE1, passing that component in both sources. It remains 6.108%/10.732%
worse than rank32, with lower centered correlation, so the overall fixed rule
fails. K562 reconstruction-held ELBO improves from 1.02680755 to 1.02299069
on 21,900 cells, a .3717% improvement. Reconstruction/query improves from
1.02490777 to 1.02074377, while KL/cell increases from 16.2679 to 19.2403.
Thus the descriptor does not merely trade away count reconstruction.

Fitting MSE gains of 11.65%/10.85% shrink to 1.06%/1.82% on development
genes. The descriptors recover some useful query information, but the
intervention-to-response generalization gap remains. The simpler rank32 model
is retained; this neural candidate is rejected without protected test or SL
benchmark access. Report SHA-256:
`b76647a1930137bf017b05a9f7b0a01b96ed1035a5bc64ad239e74cd7cf09c87`;
per-gene score SHA-256:
`e3f765169f18c577f286bbd840207cb9403f536acc6090eeb6e03087b642efc4`;
directory `results/slp11-transition/human-essential-count-response-query33-seed731-v2/`.
The consolidated suite passes 908 tests with 21 skips and 17 historical
warnings; the final 12 focused feature/inference/runner checks also pass.
This closes the matched experiment without a launch claim.

### 2026-09-05 — Combinatorial CRISPRi source acquired, outcomes not opened

The primary Adamson et al. Cell 2016 experiment, GEO GSE90546 sample
GSM2406677/10X005, supplies K562 dCas9-KRAB CRISPRi for the full factorial
set of three UPR sensors. Stable deposited identifiers are ATF6
ENSG00000118217, EIF2AK3/PERK ENSG00000172071 and ERN1/IRE1
ENSG00000178607. Their existing global split buckets are 11, 30 and 79:
the first two are fitting genes and ERN1 is validation. Every ERN1-containing
combination therefore remains validation; no action set maps to the protected
test role. The eight biological backgrounds are control, three singles,
three doubles and the triple. Drug contexts are tunicamycin 4 micrograms/mL
for six hours, thapsigargin 100 nM for four hours and DMSO for six hours.
There are 14,856 deposited assignment records, 14,820 intended assignments
and 13,516 intended author-labelled good-coverage singlets. These are metadata
counts, not expression-filtered admission decisions.

The official compressed count matrix is downloaded without decoding its
header or expression values: 199,715,348 bytes, SHA-256
`36392a38e727e5cf3c4c2eff4b0d19f4b16009926aabcc80d002256345c714b4`,
MD5 `a1a557368d0b860278b905d7c15f78c3`. GEO supplies matching content length
but no remote checksum. The source's 32,738 unique unversioned ENSG queries
overlap the current K562 panel at 8,365 identifiers; 198 current queries are
absent from Adamson. Only ATF6 is in the current K562 query panel; EIF2AK3
and ERN1 remain explicitly unmeasured there. All three have static protein
features. No absent query is filled by symbol matching or relabelled as an
observed measurement. This panel mismatch must be resolved in an explicit
evaluation protocol before applying a count model to this new source.

Metadata report SHA-256:
`bca82c2cf7bfa444e8c58a6fd012bb7a3a3cc344070768eff77d21a632644e05`;
manifest SHA-256:
`5032e0a546b6e4742150dd4e61ff70291657757e26c7f6df0cc3737f07fc2dac`;
directory `data/sources/adamson-2016-gse90546-epistasis-raw-count-v1/`.
Training use follows the recorded GEO public-data policy with citation;
redistribution remains conditional. This is preparation for a narrow
combination-state test, not broad SL or combination-generalization evidence.
The distinct 82-gene GSM2406681 screen is not merged into this source.

Primary references: [Adamson et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5315571/),
[GEO GSE90546](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546),
[GEO data policy](https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html).

### 2026-09-05 — Return to SLp-1: grounding the missing composition operator

The investigator requested an architectural reassessment rather than further
optimization of response-ridge and count-decoder variants. Executable inspection
of frozen `model/v1/world.py` and `modules/training/world.py` found that SLp-1's
shared modality/action Transformer is reusable as an inductive action-set
interface. Its strong supervised readouts are evidence for useful features;
they are not evidence that sequential state updates were learned correctly.
Static-relation consistency trained `T(T(0,A),B)` toward another model prediction.
Measured molecular doubles trained simultaneous endpoints or residual heads.
No audited v1 path trained `T(E(y_A observed),B)` against the measured double
endpoint. The poor sequential results therefore leave a specific untested
factorization, rather than establishing that all state-conditioned composition
is ineffective. We preserve frozen v1 and reconstruct its attention principle
in a new, self-contained 142,720-parameter module.

**Hypothesis and fixed rule, before biological training.** Observed-background
edge supervision will improve held-combination molecular prediction over a
capacity-matched simultaneous endpoint model and fixed additive/linear controls.
The three-seed mean must reduce pooled MSE at least 5% against every baseline,
not regress in any fold against the best pooled baseline, have positive
centered nonadditive correlation, and worsen at least 1% when conditioning
parents are swapped while preserving the correct observed additive reference.
This is a falsifiable first composition test, not an SL or temporal-dynamics
claim. All forecasts precede held-combination scoring; no adaptive epoch or
hyperparameter selection is performed.

**Exact accessible data.** Norman 2019 GEO GSE133344, Homo sapiens NCBI:9606,
K562 simultaneous CRISPRa, day-five RNA endpoint; existing per-cell full-library
log2(CP10k+1), core-control-standardized development payload SHA-256
`ab81e7ed07d7f111b3dfc964cece28a2db7de0dcf5975f6ff1a3bc2db0be683e`.
Only original fitting rows are used: 71 equally construct-weighted single
endpoints and 59 canonical double endpoints, with both observed constituent
singles for every pair. Three fixed pair folds contain 23, 19, and 17 doubles.
The common single-defined panel has 7,182 observed queries. Static ESM320,
protein-presence1 and GO256 descriptors come from the existing payload SHA-256
`7b3d78af66f013e2d1df3a3f98924707ed111bc795757753e82a5e8f495408b5`.
No global validation/test outcomes or SL labels enter this run. Rights remain
the recorded Norman GEO public molecular research rights. This is known-gene
composition interpolation in one context, with observed singles available.

The two arms use the same 32-state, 64-wide, two-layer attention core, identical
seeds 731–733, 1,000 AdamW updates, and equal single/pair class weighting. The
operator's pair class averages simultaneous, A-to-AB and B-to-AB losses. Each
fold's observation basis is fitted only to its fitting endpoints. A fixed
nonadditive readout adds the state-dependent increment difference
`f(z_A,B)-f(0,B)` (averaged over orders) to observed additive singles. Thus an
action-only offset yields exactly zero correction even if single predictions
are biased. Autonomous rollout is a separate secondary endpoint. The controls
include observed additive singles, a fitting mean nonadditive correction,
scalar-weighted additive response, symmetric single-state ridge, and the
matched endpoint attention arm.

Native CUDA profiling uses fitting data only: 30 updates take 0.75 seconds for
endpoint attention and 1.171 seconds for the observed operator, projecting
576 seconds for all 18 fits. The hard run limit is 2,700 seconds. Seven focused
loader/operator checks pass. WSL is available but its former disposable OMF
runtime and CLI are absent; no new OMF health or admission claim is made. This
authorized native research run is not an executor fallback or release.
The profile is retained under
`results/slp11-transition/norman-observed-composition-profile-v1/`.

### 2026-09-05 — Observed-background composition completed; primary rejected

The first execution retained one fitting endpoint checkpoint and stopped before
held scoring because fused PyTorch 2.11 evaluation attention differed between
CPU and CUDA by 4.60e-4 on a fitting-state probe. Disabling the fused MHA
fastpath reduced the difference to 1.13e-6. The corrected runtime was declared
before the new run; the aborted directory remains intact. No scientific
hyperparameter or advancement threshold changed.

All 18 fixed fits finish in 517.25 seconds on the RTX 4070. Maximum CPU/CUDA
state replay discrepancy across saved checkpoints is 1.91e-6. The data,
operator and CPU inference checks pass 13 focused tests. Every forecast is
fixed before held-combination scoring. The exact endpoint is mean squared
error on the common 7,182-query, core-control-standardized RNA panel, with
equal canonical-combination weight. Centered nonadditive correlation subtracts
the observed additive singles and separately removes the across-combination
mean of predicted and measured residuals before per-pair Pearson correlation.

| Fixed model/readout | Pooled MSE | Centered nonadditive Pearson |
|---|---:|---:|
| Observed additive singles | .01863243 | .00000 |
| Fitting mean residual correction | .01815324 | -.15874 |
| Scalar-weighted observed additive | .01659083 | .43537 |
| Symmetric observed-state ridge | .01615076 | .28563 |
| Matched endpoint attention, anchored difference ensemble | .01809699 | .25434 |
| Observed-state operator, anchored difference ensemble | .01778909 | .21780 |
| Conditioning-state swaps, same correct additive anchor | .01839678 | .15097 |
| Autonomous operator rollout ensemble, secondary | .01520345 | .48685 |

The primary operator improves its matched endpoint readout by 1.70% and
conditioning swaps worsen pooled MSE by 3.42%. It therefore uses some useful
background information. However, it is 10.14% worse than ridge, with regressions
against ridge in every fold, and fails the fixed advancement rule. The paired
59-combination descriptive bootstrap places its relative gain at
[-17.58%, -4.42%]; shared intervention genes and fitting folds make this a
conditional interval rather than independent biological uncertainty.

| Fold (held combinations) | State ridge MSE | Primary operator MSE | Secondary autonomous MSE |
|---|---:|---:|---:|
| 0 (23) | .01572963 | .01748818 | .01460008 |
| 1 (19) | .01896241 | .01955134 | .01693923 |
| 2 (17) | .01357810 | .01622663 | .01407979 |

The predeclared autonomous rollout ensemble has 5.87% less pooled MSE than
ridge and 18.40% less than observed additive singles. It remains worse than
ridge in fold 2. Individual seeds 731/732/733 have MSE
.01717545/.01652007/.01739727, all worse than ridge; averaging matters.
This is a promising secondary diagnostic, not a pass. A locked-weight
follow-up will compare autonomous composition with predicted additive singles
and direct simultaneous endpoints to distinguish learned interaction from
single-response prediction and ensemble denoising. It will not revise the
primary decision or tune these weights.

The 59 doubles span 42 genes; 39 held pairs have both parents present in other
fitting doubles, 16 have one parent without another fitting double, and four
have neither. All parents still have fitting single endpoints. This support
distribution further limits extrapolation claims. No original global held-gene
outcomes, external SL labels, or new species outcomes were evaluated.

Run directory:
`results/slp11-transition/norman-observed-composition-seeds731-733-v2/`.
Protocol SHA-256:
`d74c1a3605740cf7324a96620d2cef5fc193a0683c6f7109ec1e00309c9fa233`;
forecast SHA-256:
`c282bab72c6ae7d6708ecf833d27a92d0a02d23ca4325406623a799e20433831`;
report SHA-256:
`6582550971068ae2f56efea64435e016ded454bf9b2f06cd203c1c0e37a27f1b`.
The complete per-seed and per-fold metrics are retained in the report.
The standalone CPU wrapper requires the exact observed-single query axis,
value space and raw static features; artifact replay is not OMF promotion.

### 2026-09-05 — Locked rollout decomposition: tentative, unstable interaction signal

The follow-up changes no fitted parameters, epochs, seeds, features or primary
decision. A separate CPU script freezes all new forecasts from the existing
18 checkpoints before scoring. It asks whether the promising predeclared
autonomous result exceeds predicted additive singles and a matched direct
endpoint. This is a post-hoc mechanism audit, not an advancement test.

| Locked three-seed readout | Pooled MSE |
|---|---:|
| Observed singles projected into fitting response basis, additive | .01605801 |
| Endpoint model: predicted singles, additive | .01596753 |
| Operator: predicted singles, additive | .01607189 |
| Endpoint model: direct simultaneous endpoint | .01549159 |
| Operator: direct simultaneous endpoint | .01642505 |
| Operator: autonomous two-order rollout | .01520345 |

Autonomous composition improves its own predicted-additive baseline by 5.40%
and its simultaneous endpoint by 7.44%. Its improvement over the matched
endpoint model's direct simultaneous prediction is only 1.86%. It regresses
against predicted additive in fold 2 and against the matched direct endpoint
in fold 0. Descriptive gene-multiplicity-weighted bootstrap intervals are
[-0.21%, 11.03%] and [-3.95%, 7.01%] for these two principal contrasts.
They are conditional sensitivity summaries, not independent confirmatory
intervals. They include zero. Projecting observed additive singles alone also
removes substantial error, showing why raw additive comparison overstates the
evidence for emergent composition.

The fixed primary rejection stands. The evidence motivates retaining the
observed-state operator as an experimental factorization, with much stronger
composition support and an independent context needed before another emergence
claim. It does not justify copying the entire frozen v1 architecture, scaling
this pilot, or replacing the retained K562/RPE1 response model. The scientific
change is to ground state updates in measured perturbed backgrounds and judge
them against both additive and direct-endpoint alternatives. The broad
world-model/SL objective remains unfulfilled.

The audit script is `scripts/audit_slp11_compositional_rollout.py`; artifacts
are under the original run's `secondary-rollout-audit-v1/`. Report SHA-256:
`9daa9e86e5d70f813d582c05ea0a936947b86371cc2b38ebcc133ea73664e531`.
Autonomous CPU recomputation matches the earlier CUDA forecasts within 1.05e-6.
The separate public inference wrapper also reproduces the primary forecasts
on one actual held pair per fold for all nine operator checkpoints, maximum
expression-space discrepancy 8.76e-7. Checks and weight/basis/code hashes are
retained in `cpu-inference-replay.json` and `artifact-manifest.json`.

### 2026-09-05 — OMF 2 migration and a usable retained-model build

The project now uses pinned OMF 2.0.0, upstream revision
`75f002b4226b32dd428f5fec0efe9b950db0c6d5`, on Ubuntu 24.04/CPython 3.12.3.
The bootstrap installs upstream runtime/build locks with required hashes.
`scripts/omf2.sh` pins the runtime for both the CLI and dependency subprocesses.
Doctor reported ready, zero failures and catalog version 2. All 31 existing
concrete project/module/workload/evaluation/binding/policy resources checked
against the actual upstream registry validated; `omf.dev/v1alpha1` remains
correct. Unsupported OMF 1 policy keys were removed. Development source is
captured in archive mode, without requiring a commit before each run.

The active `experiment.yaml` now builds the retained rank-32 response model
through self-contained train/evaluate scripts in
`modules/slp-1-1-response-omf2/`. This is a migration of an existing numerical
model, not a newly competitive neural world model. The numerical acceptance
criterion was reproduction of the retained development scores and forecasts;
there was no architecture search or new biological hypothesis in this work.
Evaluation also retains the detailed report as a declared OMF artifact.

Inputs are derived from Replogle human K562/RPE1 essential-gene CRISPRi
measurements and the existing 577-component static ESM/GO descriptors. Training
has 1,443 K562 and 1,666 RPE1 intervention means; development has 305 and 360
interventions respectively. The query axes contain 8,563 and 8,749 genes.
The target is ln1p(mean CP10K), anchored to the original source/GEM control
mixture. This linear model uses intervention means rather than fitting individual
cell counts. Preparation checks intervention disjointness, query ordering and
metadata/truth identity. Separate captured training/development manifests now
record raw-input and panel SHA-256 values and sizes. No final holdout or SL
benchmark was opened.

The final candidate's DatasetSnapshot revisions are:

- Training: `sha256:3dc11e8787d67a6e295497c59a63d4787ff9eb31b1d0a4996a8a6785b84d4947`.
- Development: `sha256:d288986b0d41808fed0123c3251c09547176e90a9b753ad45533b75fb35257de`.

| OMF 2 model | K562 MSE | K562 centered r | RPE1 MSE | RPE1 centered r |
|---|---:|---:|---:|---:|
| Full ridge | .00358458479909 | .222189473422 | .00845344099081 | .265243194826 |
| Rank 32 | .00343271360076 | .248753210926 | .00815924214789 | .285890123781 |

These reproduce the prior results: rank 32 reduces development MSE by 4.24% and
3.48%. The full-ridge run completed successfully but its evaluation correctly
does not pass the rank-32 advancement criteria. The candidate and its captured
reproduction both pass. Baseline/candidate data revisions differ because source
receipts were added inside the captured directories after the initial baseline;
the molecular arrays and numerical implementation are unchanged. The dependency
lock also gained a verified CPython 3.11 Linux wheel hash before candidate capture.

Successful OMF runs:

- Full ridge: `01a07326-a1fb-79a8-8d6a-ea3359aadf1e`.
- Rank 32: `01a0732a-789a-7dab-84fe-264a28e89c2e`.
- Captured reproduction: `01a0732c-26ac-7624-81c8-d81718de6857`.

OMF measured 10.12 stage wall seconds/9.83 CPU seconds for full ridge and
5.78 stage wall seconds/5.90 CPU seconds for rank 32, excluding admission and
environment preparation. No GPU or paid remote compute was used. Reproduction
returns identical reported scores; this is numerical reproducibility, not a
claim that separate run artifact manifests have identical digests.

The candidate's model directory artifact is
`sha256:9b593d28f62f16343b68e6628c96c2370f74f9da82b37acce479d3dd0abd3f2f`.
OMF exported actual weights, inference code, dependencies and manifest under
`results/omf2-migration-v1/export-rank32/artifacts/model/`, together with captured
source, experiment definition, model card and evidence. The model directory is
approximately 4.77 MB. Export is not a dataset download or installed runtime.

`scripts/verify_slp11_omf2_export.py` exercised exported inference for every
existing development intervention in a fresh CPython 3.12 environment containing
NumPy 2.2.6 and threadpoolctl 3.6.0, with no OMF installation. It supplied only
static features and control anchors to the inference subprocesses. All
5,761,355 returned values match retained forecasts within 2.66e-15 (K562) and
7.99e-15 (RPE1), with exact query identity and matching weight digests. The
verification report is `results/omf2-migration-v1/standalone-replay-v2/report.json`.
The first CLI attempt exposed Conda Python 3.13 selection and correctly failed
dependency hashes; the launcher fixes that. The first standalone verifier exposed
symlink resolution selecting base Python; preserving the venv executable fixes it.
Neither failure was accepted as model evidence.

A real local release, `slp11-response-rank32-omf2-20260905`, was saved and read
back with `omf.release/v2` format and no aliases. Its revision is
`sha256:283f75a26c26433681a54fb9b9fcdf007806487d0c037f7c50e185d7e004c115`.
It is unpromoted and was not uploaded or deployed. The release records the
absence of vulnerability evidence; no substitute evidence was created.
The separate OMF ModelPackage service still needs tested artifact materialization;
this does not prevent the now-verified standalone directory inference.

Focused checks passed: nine model serialization, standalone inference, data
partition and factory-isolation tests. Two upstream schema tests explicitly skip
in the default Python environment lacking OMF dependencies; the actual upstream
registry independently validated all 31 concrete resources in a dependency-equipped
environment. Shell syntax and changed Python syntax checks passed. Logs, review
HTML/JSON, export, resource validation and release receipts are retained under
`results/omf2-migration-v1/`; installation diagnostics are in
`results/omf2-installation.log`.

The operating decision is to build one joint observation/transition model with
mechanism-aware interventions and assay-specific queried decoders. K562/RPE1
raw-cell populations and Norman observed single/double relations should train
the same state representation, with CRISPRi/a and measurement semantics preserved.
The chosen implementation scope is now in `MODEL_CARD.md`. Broader independent
combination and context coverage is needed for an emergence claim. This migration
makes the current model manufacturable and usable; it neither implements that
next shared neural model nor changes the evidence that SLp-1.1 is not yet SOTA.

### 2026-09-05 — STRING64 response backbone and canonical retrospective comparison

The public SLIM source was pinned at commit
`5a7e9ade5d0a6b6331e6dbc81181450605047bcc`. Its tracked
`gene_string_embeddings.v0.3.h5` contains 21,688 symbol-keyed, 64-dimensional
vectors. The exact file SHA-256 is
`789416877b8701ef6f800106d26bf7bb97ea8e72744e6ab93e24933a717f247d`.
The SLIM repository distributes this file under its root MIT license and asks
users to cite Hu et al., DOI 10.1038/s41540-026-00746-8; this does not assert a
separate upstream license for the Hu vectors. Stable Ensembl IDs are mapped
through the pinned Replogle source GTF, with zero vectors and an explicit
presence bit for missing symbols. Whole-roster coverage is 92.42% K562, 94.67%
RPE1 and 95.49% Norman; native fitting-action coverage is 99.65% and 98.32% in
K562/RPE1.

On the existing native development panels, three-fold fitting-only CV selected
rank 16 and alpha 1000 for the concatenated static577+STRING64+presence reduced-
rank response model in both contexts. Normalization is fitted inside each fold.
No held outcome selects rank, penalty, sign or feature policy.

| Native retained backbone | K562 MSE | K562 centered r | RPE1 MSE | RPE1 centered r |
|---|---:|---:|---:|---:|
| Earlier static577 rank 32 | .00343271 | .24875 | .00815924 | .28589 |
| static577+STRING64+presence rank 16 | **.00334073** | **.27656** | **.00795145** | **.32614** |

The comparator grid took 333.85 CPU seconds. The STRING-only arms also carry
signal, while the concatenated arm has the best MSE in both native contexts.
This rank-16 642-feature map is therefore the retained main response backbone.

For an executable published-method comparison, exact official GEARS archives
were downloaded from Harvard Dataverse and verified against both Dataverse MD5
and local SHA-256. Replogle K562/RPE1 are files 7458695/7458694 in GEARS V6,
DOI 10.7910/DVN/BD93JY, CC0 1.0. Norman is file 6154020 in PertNet historical
versions 2/3, DOI 10.7910/DVN/Q2ZV3E, CC0 1.0. Exact-scope rights records are
`rights/gears-replogle-filtered-canonical-cc0-1.0.yaml` and
`rights/gears-norman-canonical-cc0-1.0.yaml`. The split code is pinned to
cell-gears 0.1.2 commit `df09d7ae34e90f5ef25afa389daf7c5c589e710d`:
`simulation`, seed 1, 75% initial training-gene set, then a 90/10 simulation
split of that training set. Train/development aggregation read no test expression
bytes. All eight candidate artifacts and the scoring protocol were hashed before
the single test access.

The score below is the mean across conditions of Pearson correlation across all
measured genes between real and predicted control-referenced processed-expression
means. Predictions add the frozen training-control mean and clip to [0,14.99].
MSE is the mean all-gene profile MSE. This deterministic direct-mean protocol
does not construct SLIM's resampled synthetic cell population, so the values are
not a reproduction of the manuscript's stochastic scaffold or its reported
scores.

| Frozen canonical test model | K562 Pearson-delta | K562 MSE | RPE1 Pearson-delta | RPE1 MSE |
|---|---:|---:|---:|---:|
| Published-default SLIM, K10/lambda .1 | .47600 | .00586003 | .63630 | .01013786 |
| Fitting-CV SLIM | .50160 | .00569066 | .64582 | .00999647 |
| STRING64 reduced rank | .49288 | .00568595 | .65130 | .01006854 |
| static577+STRING64+presence reduced rank | **.51716** | **.00538725** | **.65333** | **.00942391** |

The canonical reduced-rank ranks are 32 for K562 and 16 for RPE1. On 273 K562
conditions, paired 10,000-replicate bootstrap intervals for the concatenated
model's Pearson gain are [.00464,.02658] versus fitting-CV SLIM and
[.02783,.05514] versus published-default SLIM; MSE-reduction intervals are
[.00013195,.00048412] and [.00029863,.00065552]. On 386 RPE1 conditions,
Pearson-gain intervals are [.00230,.01329] and [.01051,.02452], with MSE-
reduction intervals [.00031889,.00083190] and [.00045391,.00097548]. The
canonical model fit took 51.76 CPU seconds. Per-condition results, rosters,
model receipts and scoring protocol are under
`results/slp11-transition/gears-frozen-test-v1/`.

This test is retrospective because its underlying Replogle biology already
participated in SLp development. It supports a matched direct-mean advantage
over these frozen SLIM comparators. It is not prospective evidence, a general
leaderboard claim, or proof of state of the art.

The first shared joint model generation used static577 and 6,000 updates. Its
K562/RPE1 MSE and centered correlation were .0034458/.2451 and .0086551/.2676.
On Norman fold 0, autonomous-average MSE was .0156281: 16.12% below observed
additive and 5.53% below the prior-only forecast. Generation 2 adds STRING64
and its presence bit, uses a rank-16 response prior, and trained 20,000 updates
in 788.75 seconds on the RTX 4070 (810,882 parameters; 1,158.5 MB peak GPU).
Its K562/RPE1 metrics are .0033355/.2800 and .0081827/.3216, improvements of
3.20% and 5.46% in MSE over generation 1. Against the new retained linear
backbone, K562 MSE improves 0.16%, while RPE1 MSE regresses 2.91%.

Generation-2 Norman fold-0 autonomous-average MSE is .0151073, 8.38% below
its own predicted-additive forecast (.0164889). The observed-parent-average
readout is .0146551, 8.23% below its observed-parent-prior (.0159697).
Direct-two-action MSE is .0150115, slightly better than autonomous average, and
only one of three combination folds is represented. These are useful shared-
model development results, not stable composition evidence. Expanded joint
training is currently pending; no result or completion is claimed for it.

### 2026-09-05 — Correction: canonical STRING lookup and frozen retrospective scores

This entry supersedes only the canonical GEARS table and intervals in the
preceding STRING64 entry. The native-panel results and all earlier evidence
remain unchanged. Audit found that the first canonical runner treated presence
in a source-panel symbol roster as STRING availability. That omitted valid
vectors from the official symbol-keyed HDF5 for 3 of 737 K562 fitting genes,
17 of 1,041 RPE1 fitting genes, and 4 of 386 RPE1 test genes. Development
coverage was unaffected. There were no false-positive covered genes with zero
STRING vectors. The static577 and generated STRING-pack `entity_id` axes were
exactly equal (8,752 K562 rows and 9,032 RPE1 rows), but the implementation had
relied on the shared row index without asserting identity.

The corrected runner reads STRING64 directly from the official HDF5 by exact
canonical symbol, independently of the static roster. It maps static577 by the
explicit chain symbol to stable Ensembl ID to static row, rejects duplicate or
ambiguous identities, and defines the concatenated presence bit as actual
STRING coverage. Static coverage is 734/737, 82/82 and 273/273 in K562
train/development/test and 1,024/1,041, 116/116 and 382/386 in RPE1. STRING
coverage is 737/737, 82/82 and 273/273 in K562 and 1,041/1,041, 116/116 and
386/386 in RPE1. Thus HDF5-only genes receive their real STRING64 vector, zero
static577, and a one-valued STRING-presence bit.

Because canonical test outcomes had already been opened, this correction made
no new model choice. It reused exactly the previously frozen ranks and
regularizers: K562 concatenated reduced rank 32/alpha 1,000, STRING reduced
rank 16/alpha 100, fitting-CV SLIM K64/lambda 10 and published SLIM K10/lambda
.1; RPE1 used rank 16/alpha 1,000, rank 32/alpha 1,000, K32/lambda 10 and
K10/lambda .1 respectively. There was no feature-policy, sign, clipping or
test-conditioned adjustment. Predictions still add the training-control mean
and clip to [0,14.99]. The correction protocol truthfully records that test
outcomes were opened before this protocol.

| Corrected frozen canonical test model | K562 Pearson-delta | K562 MSE | RPE1 Pearson-delta | RPE1 MSE |
|---|---:|---:|---:|---:|
| Published-default SLIM, K10/lambda .1 | .476276 | .00585545 | .636390 | .01014863 |
| Fitting-CV SLIM | .501496 | .00569064 | .645608 | .01001188 |
| STRING64 reduced rank | .492983 | .00568554 | .651180 | .01010347 |
| static577+STRING64+presence reduced rank | **.516913** | **.00538925** | **.653012** | **.00943376** |

For the concatenated model versus fitting-CV SLIM, paired 10,000-replicate
condition-bootstrap 95% intervals for Pearson gain are [.004531,.026404] in
K562 and [.002322,.012966] in RPE1; MSE-reduction intervals are
[.00012974,.00048148] and [.00032024,.00084411]. Versus published-default
SLIM, Pearson-gain intervals are [.027393,.054611] and [.010343,.023795], and
MSE-reduction intervals are [.00029178,.00064801] and
[.00045089,.00098288]. Seed 731, 10,000 replicates, condition pairing, the 273
K562 and 386 RPE1 rosters, and all scoring equations are unchanged.

The source remains SLIM commit
`5a7e9ade5d0a6b6331e6dbc81181450605047bcc`; the official HDF5 SHA-256 is
`789416877b8701ef6f800106d26bf7bb97ea8e72744e6ab93e24933a717f247d`.
The Replogle source GTF SHA-256 is
`796bb1f1d36c75462fea32e87cc54c66e2ee7b60a2e6eed3b6e0c02e8df7908b`.
Static K562/RPE1 files are
`6706f8867adedef8822897bc275ea90680584f84afd24771e4beb3c8ecf07659`
and `621e1e9f0dffc740ef42382b1b2898f629edd5037e8a02d411e8d30e815ed816`.
The feature-input receipt SHA-256 is
`bf732052ce02b3da1e468fe532dc4769665ba286016b8c16395f6f64add073b2`.
It binds the source commit, official HDF5, mapping GTF, generated symbol/ID
packs and static577 inputs. The corrected model receipt SHA-256 is
`3840e3677a6fbf25d41201a5ec8b1fef049d09c5e699bc1af801d502bb2f586d`;
the development report is
`6945721a30783d1a79b7f559e04f8491cec09424e804b074b7fb02cf1390892b`,
and the corrected test report is
`0ad9656842fabad7425a41c7d1f2ba747b38152e6616ab3299d2b8a0d5fa2996`.
Artifacts are retained under
`results/slp11-transition/gears-response-models-v4-corrected-receipts/`
and `results/slp11-transition/gears-frozen-test-v3-corrected-receipts/`;
the erroneous artifacts remain intact for audit.

This remains a retrospective matched direct-mean comparison on source biology
that participated in earlier development. The correction supports no
prospective, general-leaderboard or state-of-the-art claim.

### 2026-09-05 — Five-context joint world model, Norman fold 1

The first completed expanded run trains a single 877,444-parameter population
state model across K562 essential CRISPRi, RPE1 essential CRISPRi, author K562
genome-wide CRISPRi, HepG2 CRISPRi and Norman K562 CRISPRa. The model uses 642
static action descriptors, four assay IDs, two mechanism IDs, a separately
masked control-expression context, and a value-bound observation encoder. The
control-expression context is distinct from the zero basal anchor of the
control-z endpoints. The empty-action transition remains exact identity.

This was a native Windows RTX 4070 run, seed 731 and Norman fold 1. It completed
20,000 updates in 790.84 seconds with about 2.4 GiB peak GPU memory. The five
contexts contributed 1,443 K562 essential, 1,666 RPE1, 7,438 genome-wide,
1,758 HepG2 and 111 Norman fitting populations. Development results at the
completed checkpoint are:

| Source and weighting | MSE | Profile Pearson | Independently query-centered Pearson |
|---|---:|---:|---:|
| K562 essential, intervention rows | .00332139 | — | .281796 |
| RPE1 essential, intervention rows | .00815964 | — | .320993 |
| K562 genome-wide, 1,613 population views | .01210323 | .096225 | .084511 |
| K562 genome-wide, 1,491 unique genes | .01176667 | .101895 | .090593 |
| HepG2, 396 population views | .05860674 | .273711 | .232930 |
| HepG2, 361 unique genes | .05642275 | .292248 | .248465 |

Population-view weighting represents every source/GEM population separately;
unique-gene weighting first averages views for each intervention gene. The
centered column removes the mean response across development interventions and
then each query profile mean, so it measures perturbation-specific structure
rather than shared endpoint level. Genome-wide MSE improves slightly over its
frozen prior under both weightings (.01212999 population-view and .01179906
unique-gene), while centered correlation is approximately unchanged. HepG2
MSE also improves slightly over its prior (.05869050 and .05657508), and its
centered correlation increases, while ordinary profile Pearson decreases.
These are adaptive development contexts, not new-context confirmation.

On the 19 held Norman fold-1 combinations, autonomous-average MSE is .01788604,
8.15% below its own predicted-additive forecast (.01947211) and 5.51% below
the direct-two-action forecast (.01893001). The observed-parent-average readout
is .01747486, 9.35% below its observed-parent prior (.01927705). Autonomous
centered nonadditive Pearson is .449887; observed-parent-average centered
nonadditive Pearson is .452169. These improvements are useful fold-1
composition evidence, but one fold cannot establish stable composition or
emergence across contexts.

The checkpoint SHA-256 is
`fd96ce7db8fae3a1550223374a306026acfaeb4acc002d3e22a60cb8af9b212c`;
the evaluation report SHA-256 is
`f6b4cbc7388a32258ab217149207392dd8f162104de1208ca21a475f729efa1d`.
The research export at
`results/slp11-transition/joint-world-expanded-fold1-research-export-v1/`
contains 20 payload files plus its manifest, including the model code, pinned
Linux requirements, adapters, priors and safetensor checkpoint. Its Linux
standalone portability check is pending. Runtime hash-lock and captured-source
ordering have been corrected for future training runs; this native run is not
an OMF-trained neural result. Norman fold 2 and the OMF fold-0 run are underway,
so no aggregate fold claim or completed OMF neural claim is made here.

### 2026-09-05 — Author-population-scored canonical comparison

The corrected frozen GEARS models were passed through SLIM's unchanged pinned
population builder and metric implementation. For each test perturbation,
`build_result_h5ad` samples fitting/control cells with seed 1, rescales each
gene to the frozen predicted mean, and clips synthetic cells to [0,14.99]. The
authors' evaluator then uses real controls and averages Pearson of expression
delta equally over the 273 K562 or 386 RPE1 test conditions. Cell-level H5AD
files were temporary; post-scaffold means, cell counts, query identities and
per-condition results are retained.

| Population-scored frozen model | K562 Pearson-delta | K562 MSE | RPE1 Pearson-delta | RPE1 MSE |
|---|---:|---:|---:|---:|
| Published-default SLIM, K10/lambda .1 | .476035 | .00585907 | .636037 | .01016149 |
| Fitting-CV SLIM | .501233 | .00569426 | .645247 | .01002488 |
| static577+STRING64+presence reduced rank | **.516688** | **.00539291** | **.652638** | **.00944712** |

Paired 10,000-replicate condition bootstraps use seed 731. Against fitting-CV
SLIM, the concatenated model's Pearson-gain 95% intervals are
[.004568,.026433] K562 and [.002304,.012975] RPE1; MSE-reduction intervals are
[.00012969,.00048137] and [.00031991,.00084358]. Against published-default
SLIM, Pearson intervals are [.027412,.054603] and [.010320,.023782], with MSE
intervals [.00029177,.00064804] and [.00045047,.00098224].

This supersedes the earlier hypothesis that SLIM's cell scaffold explains the
gap to its README values: population scoring changes the direct-mean results by
less than .0004 and does not recover .499 K562/.613 RPE1. Deterministic full-SVD
reconstruction matches the frozen SLIM means within 8e-15. The pinned authors'
default randomized PCA changes mean values only 3e-6 to 1.4e-5 across recorded
seeds 1/2/3. The repository does not contain the README-referenced
`manuscript-results/results/test_mean_scores.csv`, so the headline discrepancy
remains an upstream artifact or data-environment provenance gap; it does not
block the matched comparison reported here. The source biology and test
outcomes were accessed previously, and a training-only algebraic reconstruction
was performed after that access solely to validate frozen means; it did not
select or supply any scored prediction.

The parity report SHA-256 is
`f50481401a733d81c98c72e161143c7d3fbe7d177cb5490e049e73d998d8684e`.

### 2026-09-05 — Eight-context joint world model with held MCF10A environment

The context-transfer revision trained one 910,725-parameter population-state
model for 20,000 updates across K562 and RPE1 essential CRISPRi, K562
genome-wide CRISPRi, HepG2 CRISPRi, Norman K562 CRISPRa, and three MCF10A
CRISPR knockout environments (full-medium day 0, full-medium day 6 and TGF-beta
day 6). The model used 642 static descriptors, three mechanism IDs, five assay
IDs, observed molecular state and a separately masked control-expression
context. The immutable training snapshot was
`slp11-joint-world-context-transfer-v2-training-r4` (manifest SHA-256
`8d9beb1e77bbb4ef9ada2c48f8fd5396d1a1ab3425ca28e39073f67e11a27cab`).
The corrected GSE164996 population snapshot manifest SHA-256 was
`ac34a2bf8c26547cdcf559100c9f3f9edbab3ca2c7830f32b0d8aa44d1cc3c0b`.
It pools both deposited exact two-guide controls, retains author-defined
single-knockout calls without inventing unavailable vector slots, and excludes
every intervention in the global 1,492-gene validation roster.

The fixed protocol used seed 731, canonical combination fold 0, rank-16 frozen
linear priors, batch size 32, and a .05 reconstruction auxiliary loss. All
minimal-medium day-6 outcomes were physically absent from fitting. The native
RTX 4070 run completed in 994.17 seconds with 2,476.68 MiB peak GPU memory; its
step-20,000 checkpoint SHA-256 is
`18634c2765bc0636e1528dcaeaba4dd4543eee10bf6ec3e8a3b7e81e32ca0aee`.

The adaptive single-intervention development results were:

| Source and weighting | Joint MSE | Prior MSE | Joint centered Pearson | Prior centered Pearson |
|---|---:|---:|---:|---:|
| K562 essential, 305 rows | .00331249 | .00334074 | .282763 | .276556 |
| RPE1 essential, 360 rows | .00808808 | .00795168 | .320072 | .326121 |
| K562 genome-wide, 1,613 views | .01210236 | .01212999 | .086605 | .084861 |
| K562 genome-wide, 1,491 unique genes | .01176522 | .01179906 | .092601 | .090662 |
| HepG2, 396 views | .05844371 | .05869050 | .231821 | .226600 |
| HepG2, 361 unique genes | .05632339 | .05657508 | .247783 | .242137 |

The same canonical pair fold was withheld within each fitted combinatorial
context. Metrics below report MSE and independently query-centered
nonadditive Pearson; the latter removes shared endpoint level before measuring
perturbation-specific composition.

| Context (held pairs) | Direct two actions | Autonomous average | Observed-parent average | Predicted additive | Prior only | Zero response MSE |
|---|---:|---:|---:|---:|---:|---:|
| Norman CRISPRa (23) | .015661/.448680 | .014980/.474750 | .014722/.471795 | .016307/.424821 | .016324/.433422 | — |
| MCF10A full D0 (10) | .006106/.365264 | .005498/.496392 | .005919/.481572 | .006642/.498201 | .006634/.501102 | .004130 |
| MCF10A full D6 (10) | .007064/.355218 | .006292/.479066 | .006770/.436354 | .007650/.475873 | .007651/.478691 | .004659 |
| MCF10A TGF-beta D6 (7) | .010676/.402841 | .009029/.558255 | .010098/.554792 | .012435/.557165 | .012443/.560029 | .007119 |

Minimal-medium day 6 is a held environment: no outcome from this environment
entered fitting, calibration, checkpoint selection or sign selection. Forecasts
use the frozen full-medium day-6 model and prior with the minimal-medium control
profile. Its ten single interventions have joint MSE .00552547 and centered
Pearson .050358, versus prior-only .00551440/.048968 and zero-response MSE
.00303003. Pair results separate combinations observed in another MCF10A
environment from the canonical fold excluded in every environment:

| Minimal-medium pair subset | Pairs | Direct two actions | Autonomous average | Observed-parent average | Predicted additive | Prior only | Zero response MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seen in another environment | 17 | .009335/.543220 | .009589/.542715 | .005930/.529640 | .012210/.525960 | .012173/.528925 | .004402 |
| Globally held hash fold | 10 | .010879/.548690 | .009657/.560700 | .005721/.553671 | .012144/.557317 | .012099/.558283 | .004077 |
| All minimal-medium pairs | 27 | .009907/.543780 | .009614/.551867 | .005852/.540726 | .012186/.540128 | .012146/.542388 | .004282 |

Each pair cell reports MSE/independently query-centered nonadditive Pearson.
Observed-parent forecasts use measured single-parent states and therefore are
not autonomous forecasts. On the globally held combinations, autonomous
averaging improves direct-action MSE and centered correlation, but remains
worse than the observed-additive MSE (.008339) and zero-response MSE (.004077).
Single-intervention transfer is also worse in MSE than the zero-response
baseline. A 10,000-resample bootstrap with seed 731 clustered duplicate views
by canonical gene pair. Autonomous-minus-predicted-additive MSE differences
were -.001144 (95% interval [-.001399,-.000839]) for full D0, -.001358
[-.001665,-.001048] for full D6, -.003406 [-.003984,-.002830] for TGF-beta D6,
and -.002487 [-.002948,-.002050] for the globally held minimal-medium fold.
Thus the learned state update improves on its own additive forecast while both
remain behind the zero-response scale baseline in this endpoint. This is weak
new-context compositional evidence, not a state-of-the-art or general
cross-context claim. The corrected evaluation ran in 51.51 seconds, did not
access a protected benchmark test, and produced report SHA-256
`0f7b172b64f8aa8f56f6514ffe14c90ce74baf0193da6824aa43ec8a2db01fb3` and
prediction SHA-256
`dfe439a334fce5ff4af717ad98231b5e3ec97783015fcb78da19918fd0cc767c`.

### 2026-09-05 — Five-context fold-0 completion under OMF 2

OMF experiment run `01a07387-19fa-76f1-8e7a-bb9568b9d870` completed both
captured stages successfully on Linux with the local RTX 4070. The run used
seed 731, Norman fold 0, the five-context expanded training snapshot and the
captured `slp-1-1-joint-world-v1` module. Training completed 20,000 updates in
1,069.96 seconds with 2,404.47 MiB peak allocated GPU memory. The definition,
workload and evaluation revisions are respectively
`sha256:d97b4af9e15bf08fc09421fb34a711b21b5fe097f719637776268cc4772ce4e6`,
`sha256:0863c1b1f4165de4b9290eefe858b3d795d92faa590b00ad6b1641ef5ee243a1`
and
`sha256:b4c4f8923daeaa89ab88eccdcb9bd3935574f0ad361cbac650e72e4929bc7d26`.

The fold-0 development results are K562 MSE .00332378 and independently
query-centered Pearson .282516; RPE1 MSE .00819490 and centered Pearson
.319857; genome-wide K562 population-view/unique-gene MSE
.01209511/.01176228; and HepG2 population-view/unique-gene MSE
.05851243/.05640213. On 23 held Norman combinations, autonomous-average MSE is
.01496229 versus predicted-additive .01641067, a reduction of 8.83%.
Observed-parent-average MSE is .01443095 versus its observed-parent prior
.01596971, a reduction of 9.64%. This is adaptive development evidence, with
RPE1 still worse than its frozen prior MSE .00795168.

The captured checkpoint SHA-256 is
`1498753a778e2e439e0a09617dcb36d7dc1c531873179c9d9cddf6eb4a774b11`.
The stage report and predictions have OMF artifact digests
`sha256:e7b6145fa712314f6304f06f47cb6c4a7387118ffbf29fa879a64e10409b81b6`
and
`sha256:a633a8f522585a740df73b59dc0c5df09f5940d5150b40662601c5bbcb79a0fb`;
their materialized file SHA-256 values are
`97f83107eb9fe984bd91c5434adb0a510c1e2d6dc3a5d664f00466dab2e7b5f2`
and
`a223babb50c61ee1e7c5c45b8fe6a431aab4e80e7dc7b0eb44c89dcf857040d5`.
The supported `omf experiment export` command materialized model, captured
train/evaluate sources and evidence at
`results/slp11-transition/joint-world-expanded-fold0-omf2-export-v1/`;
its `evidence.json` SHA-256 is
`df289db824c5124015179431212a1eaa5fc134674d43b210f0e8585c18fb169d`.
OMF records `passed: false` because this exploratory EvaluationSpec has no
evaluator pass declaration or compatibility evidence. Both workload stages
nevertheless have terminal `succeeded` status; no release or deployment claim
follows from this research export.

### 2026-09-05 — Clean Linux inference portability

The fold-1 checkpoint was copied unchanged into the new research export
`results/slp11-transition/joint-world-expanded-fold1-research-export-v2/`.
Its checkpoint SHA-256 remains
`fd96ce7db8fae3a1550223374a306026acfaeb4acc002d3e22a60cb8af9b212c`.
The export replaces only the captured inference loader with the reviewed core
`safetensors.safe_open` loader and records original/replacement loader hashes
`f6506bdbdabce78148e3b9aab3829e13a5db3442247609a08720b800e5761c92`
and
`fb92d7025976962146e6b37b18c3f6a4f7b516c49be64256080b762ac706faee`.

The strict replay at
`results/slp11-transition/joint-world-expanded-fold1-portability-v5/` ran the
managed Linux Python with `-S`, admitted only its explicit site-packages and
bundle/script paths, limited Torch to four CPU threads, and confirmed that OMF
was not importable. All 20 payload hashes matched the export manifest, query
identities and support masks matched exactly, and empty actions preserved the
supplied observation exactly within each runtime. Across Windows and isolated
Linux, maximum prediction drift over all five contexts was
`1.4156103134155273e-7`; generated observation drift was at most
`4.336808689942018e-19`. The portability report manifest SHA-256 is
`7f0d05227e61a551fe83c3ea3a972f90f817981748193cdadb0660728ab57b7f`.

### 2026-09-05 — Five-context Norman three-fold aggregate

The three independently trained five-context models now cover every one of the
59 Norman held-combination development endpoints exactly once: fold 0 has 23,
fold 1 has 19 and fold 2 has 17. Stable source-row pair identifiers are unique
and nonoverlapping, all query axes match, and every fold uses the same 7,182
common measured queries. Fold 0 comes from succeeded OMF run
`01a07387-19fa-76f1-8e7a-bb9568b9d870`; folds 1 and 2 are the bounded native
seed-731 runs. The earlier three-context fold-0 model is excluded.

| Fold | Direct MSE / centered nonadditive r | Autonomous-average MSE / r | Observed-parent-average MSE / r |
|---|---:|---:|---:|
| 0 (23 pairs) | .0153351 / .461515 | .0149623 / .475228 | .0144310 / .497754 |
| 1 (19 pairs) | .0189300 / .393200 | .0178860 / .449887 | .0174749 / .452169 |
| 2 (17 pairs) | .0131854 / .448380 | .0136231 / .443319 | .0132181 / .422378 |
| pooled (59 pairs) | .0158734 / .434552 | .0155180 / .450436 | .0150617 / .442280 |

On the pooled conditions, direct two-action prediction reduces MSE by .000730
against its prior-only forecast; its centered nonadditive correlation gain is
.01462. Autonomous averaging reduces MSE by .001059 against its own
predicted-additive forecast and raises centered nonadditive correlation by
.04297. Observed-parent averaging reduces MSE by .001182 against its matched
observed-parent prior and raises centered nonadditive correlation by .02235.

Paired condition bootstraps use 10,000 replicates and seed 731. The 95%
intervals for autonomous MSE reduction and centered-correlation gain are
[.000252,.001897] and [.01457,.07224]. Direct MSE reduction is
[.000109,.001457], while its correlation interval [-.00849,.03905] crosses
zero. Observed-parent MSE reduction is [.000350,.002081], while its correlation
interval [-.01878,.06385] also crosses zero. Thus all three routes improve
pooled MSE, and autonomous composition has the clearest perturbation-specific
correlation evidence; route and fold variability remain material. These are
adaptive known-gene combination results, not independent confirmation of
emergence.

The aggregate uses only saved reports and predictions. Its report SHA-256 is
`37506a44448ec5b534d5ceb96b4bb8622628539a00c955d1d35f14f0d9277420`.

**Zero-response addendum.** Version 2 adds the exact all-zero forecast
appropriate to Norman's control-z target units; it does not change any
version-1 model metric or bootstrap. Its MSE is
.04045064/.05259486/.04066122 in folds 0/1/2 and .04442217 pooled,
substantially above every matched model and additive/prior forecast in version
1. The v2 report SHA-256 is
`e1e6c210542d1b7377a81b8666218fffbd5100549189a9773e550bb3b2dfebb9`.

### 2026-09-05 — Eight-context standalone inference bundle

The completed seed-731/fold-0 eight-context model was exported to the new local
research directory `results/slp11-transition/joint-world-eight-context-research-export-v1`. All 26 payload files were verified
against manifest SHA-256 `567540682b58795a88fce95ddb9466ed377aa5271fd949c0f95f8c59d67cc33b`.
The selected checkpoint is `step-020000.safetensors`, SHA-256
`18634c2765bc0636e1528dcaeaba4dd4543eee10bf6ec3e8a3b7e81e32ca0aee`.
The exporter retained the captured numerical core and weights and records an
explicit inference-loader replacement. The replacement uses the core
`safetensors.safe_open` API, avoiding an undeclared `packaging` import, and
selects the manifest checkpoint automatically when no checkpoint is specified.
An actual default-constructor load selected step 20,000 and loaded all eight
adapters. Earlier artifacts were not overwritten.

The isolated Linux worker ran with Python `-S`, only the explicit admitted
Linux dependency directory and standard library, and asserted OMF was not
importable. Native Windows and Linux each evaluated five target-free requests
per context: two empty-action backgrounds, single actions, and a two-action
set. All query IDs and support masks matched. Empty actions preserved observed
state exactly within each runtime. Across all eight native query panels,
maximum forecast drift was 2.15135514806e-07, below the fixed 1e-5 tolerance.
The independent synthetic input drift across NumPy versions was below 5e-19.
Replay report SHA-256 is `6f6cf29d697a5afabe6b34fcb4d6258265d76fbc8470071bcc3ba601c074a2b1`.

These checks exercise standalone inference without a training corpus or OMF
package. The eight-context weights were trained on native Windows CUDA; this
Linux replay is inference compatibility, not a second training reproduction.
The validated `experiment-context-world.yaml` provides an OMF 2 replay contract
with explicit evaluation-only CROP-seq inputs, but it has not been executed.
The completed five-context OMF execution/export is recorded separately above.
No external release was uploaded, promoted, or deployed.
