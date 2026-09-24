# SL-Predict (2026 frozen MAE branch)

**Source:** [Jack Large, *Cold-start synthetic lethality prediction*](https://github.com/j8ckfi/sl-predict)
(repository commit `312974f`, April 2026) and its
[released 256-dimensional checkpoint](https://huggingface.co/potteryrage/sl-predict).
The source reports 0.714 ± 0.018 AUROC on a separate gene-disjoint Horlbeck
K562 task. That is a source result on different labels and negative sampling,
not an SLB score. The source's final classifier also uses expression,
pathway, PPI and surface features and SynLethDB positives.

**SLB adapter:** `scripts/models/sl_predict_2026/run.sh` extracts the frozen
masked-autoencoder gene representation and refits a LightGBM pair classifier
on SLB train. Input is DepMap Public 26Q1 `CRISPRGeneEffect.csv`, with the
same K562 and K562-derivative exclusions as the source. Each gene's profile
is z-scored over 1,206 cell lines, with missing values set to its mean. The
exact source encoder layers and released weights are used. For a pair, the
features are the sum and absolute difference of each of the 256 embedding
dimensions, plus raw Chronos coessentiality. The classifier uses the source's
LightGBM hyperparameters, with class balancing for SLB's measured-pair class
ratio and internal gene-holdout early stopping. That callback monitors
LightGBM's default binary log loss and AUROC and stops after 75 rounds without
improvement; the full-train refit has a floor of 50 trees even if the internal
holdout prefers fewer. The source and derived features, training matrix,
configuration and classifier have checked SHA-256 cache fingerprints. It is
human-only; other
species are tied at 0.500. Its source classifier weights, SynLethDB labels,
unfiltered STRING network and pathway files are not loaded.

The 26Q1 CSV was downloaded from the
[public 26Q1 mirror](https://huggingface.co/datasets/ChanghaoKan/crispr-depmap)
because the source repository's DepMap manifest endpoint now returns HTML.
Its SHA-256 is `e610a4cefb13a82b5b256b47eb08b63ff14843f8dbd0fb164bc0a32688e5b89e`;
the released checkpoint's SHA-256 is
`ed8840a654bb9591877d1f25007c07487f682d41c0a5c712264bfe154b0640cc`.
Both are verified by the extractor. The matrix has 1,208 original cell-line
rows and 1,206 after removing the two specified K562 IDs, which exactly
matches the checkpoint's input dimension. Derived embeddings, normalized
profiles and the refitted classifier are stored under ignored data/model
directories. The scored prediction files are in `results/models/slb1.3/`.

This is the paper's **MAE feature branch** under SLB supervision, not an
exact recreation of its final multi-feature classifier. The encoder learned
from single-gene profiles of some SLB-held-out genes; the family holdout
excludes them from *pair-label* training, not all unlabeled biological data.

| SLB-1.3 split | SLB | Human | Native human coverage |
|---|---:|---:|---:|
| Dev | 0.553 | 0.660 | 18,817 / 19,149 |
| Test | 0.575 (95% CI 0.547–0.595) | 0.724 | 20,040 / 20,210 |

The test human score is the benchmark's context/screen and ancestry-balanced
component. The flat human stratum scores 0.668, so the 0.724 should not be
read as a pooled AUROC over all 20,210 human pairs. K562 scores 0.596 on 43
positives, while other contexts vary considerably. This branch shows genuine
gene-family transfer on human data, although it cannot address either yeast
species.

To investigate the unexpectedly strong result, `diagnostic.py` fits the same
SLB LightGBM setup on the two feature groups separately. The diagnostics are
kept outside the ranked battery in `results/diagnostics/`:

| Human component | Dev | Test |
|---|---:|---:|
| MAE + raw coessentiality | 0.660 | 0.724 |
| MAE only | 0.647 | 0.688 |
| raw coessentiality only | 0.521 | 0.543 |

The MAE-vector features predict better than the scalar coessentiality feature;
the combination does best. This comparison does not isolate the effect of MAE
pretraining, because raw dependency-vector and random-encoder controls were
not run. The two test ablations were performed after the full branch's test
readout and are registered as exploratory, unranked results in the leaderboard.
