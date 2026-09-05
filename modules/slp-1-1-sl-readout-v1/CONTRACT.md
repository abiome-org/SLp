# SLp-1.1 supervised SL readout contract

The molecular world features are frozen before this module fits. This module is
a supervised synthetic-lethality readout and its results are not evidence that
synthetic lethality emerged without labels.

Each outer MuSL CV3 test fold contains pairs whose two stable gene IDs are absent
from that fold's SL-supervised training pairs. The world model may have learned
from molecular measurements of single-gene interventions involving those genes;
the supported claim is therefore cold-gene generalization with respect to the
SL labels, not complete intervention isolation.

The `fit` phase may read outer-training labels only. It freezes models,
predictions, tree counts, and the blend weight in a hash manifest. The `score`
phase verifies that manifest before reading outer-test labels and never refits or
selects a model, feature, threshold, fold, or blend from test results.
