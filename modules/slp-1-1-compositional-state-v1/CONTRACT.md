# Observed-state composition pilot v1

This native research module tests a capability that frozen SLp-1 did not train:
applying an action to a representation of an **observed** perturbed molecular
endpoint. It is not an OMF-admitted release.

`operator.py` depends only on PyTorch. `CompositionalStateOperator(Config())`
maps state `[B,32]`, static action descriptors `[B,2,577]`, and active-action
mask `[B,2]` to a next molecular state `[B,32]`. Two 64-wide attention layers
read a state token and ESM, presence, and GO tokens for each action. No gene-ID
embedding or position encoding is used. Active actions are exchangeable;
padding is ignored; an empty action set returns exactly its input state.
The output is a residual update with a zero-initialized output head.

The pilot's state coordinates are fold-fitting, uncentered rank-32 RNA SVD
coordinates with fitting RMS scaling. This is a measured-panel observation
model, not a species-wide or independently learned biological state ontology.
The core contains no source paths, labels, split logic, or application scores.

`data.py` separately loads the exact pinned Norman 2019 author-normalized-v2
development artifact, retains only its original fitting rows, rechecks global
intervention routes, aggregates replicate constructs equally, and makes three
fixed canonical-pair folds. Its numerical dependencies are NumPy only. All
71 observed singles remain available; 59 doubles are held out in groups of
23, 19, and 17. This is known-gene combination interpolation. It is not a test
of unseen intervention genes or cell contexts. Separate test-only artifacts
are rejected before opening. Static feature and source payloads are required
from the local artifact store and are not distributed with this module.

The external runner `scripts/run_slp11_compositional_operator.py` compares:

- observed-single additive, mean-residual, scalar-weighted additive, and
  symmetric-state ridge predictions;
- a capacity-matched SLp-1-inspired simultaneous endpoint attention model;
- the same model additionally trained on both observed single-to-double
  endpoint relations for each fitting combination.

All neural arms use identical initialization seeds and fixed optimizer updates.
Pair and single objective classes have equal weight. Within the operator's
pair class, simultaneous and two observed-parent edges have equal weight.
The primary forecast adds predicted nonadditivity to observed additive singles,
using the same fixed readout principle in both neural arms. Autonomous rollout
without measured single endpoints is reported separately. Cyclic state swaps
change only conditioning, preserving the correct additive reference, to test
whether background information contributes useful signal.

The measurements are from simultaneous CRISPRa endpoint assays. Single-to-double
edges are a conditional composition factorization, not observed chronological
transitions. No time dynamics, viability, cell-population generation, or emergent
synthetic lethality claim follows from this pilot. Every forecast is frozen
before any held-combination scoring. The protocol, source/code hashes, trained
safetensors, fold bases, forecasts, CPU replay checks and metrics are saved in
a new, non-overwritable results directory.

`inference.py` provides CPU-only `load(run_dir, fold, seed)` and
`predict(y_a, y_b, raw_features_a, raw_features_b)`. The two observed endpoint
vectors must follow the returned `query_ids` axis and use the stored experiment's
core-control-standardized value space. Raw action descriptors are normalized
using the saved fitting means/scales; no corpus is loaded for inference.
Importing the wrapper disables PyTorch's global fused MHA fastpath because
the tested 2.11 CPU/GPU implementations disagreed on a fitted checkpoint.
The standard path passed the run's artifact replay check. This explicit
runtime choice does not establish compatibility with arbitrary future Torch
versions or an OMF deployment adapter.
