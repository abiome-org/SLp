# SLp-1.1 synthetic-lethality predictor v1

This inference-only module converts explicitly prepared pair features into a
research ranking score. The score is not a calibrated probability. The module
contains no benchmark loader, labels, training code, threshold, or biological
world-model behavior.

The bundle manifest schema is `slp.sl-predictor/v1`. It fixes widths of 1,081
retained baseline, 642 raw descriptors per gene, and 1,054 frozen world
features. Each seed/fold selects one frozen family and blend weight and receipts
both contained LightGBM files by SHA-256. V1 predicts from baseline and
baseline-plus-world inputs. V2 first appends symmetric descriptor sum,
absolute difference, and product, in that order, then appends world features to
the augmented arm.

`predict_fold` requires an explicit seed and fold. `predict_ensemble` is an
unweighted inference ensemble across selected serialized folds; benchmark CV
metrics do not describe that ensemble. Optional pair IDs are stable string
arrays `[B,2]`. Repeated or reversed unordered pairs remain separate request
rows, preserving repeated source views and context-specific requests.
