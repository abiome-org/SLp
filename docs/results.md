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
