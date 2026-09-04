# SLp-1.1

## Purpose

SLp-1.1 is a species-aware genomics world-model program. It learns conditional
distributions of molecular measurements after interventions. Synthetic
lethality is not a pretraining label or a world-model output type. It becomes a
separate downstream release test only after the molecular candidate and
decision rule are frozen, asking whether learned transition rules transfer to
two genes absent from quantitative trajectory fitting.

SLp-1 is frozen proof-of-concept evidence. Its fixed human gene universe,
hand-concatenated feature vector, shallow outcome-specific heads, experiment
runner, and benchmark development history are not architectural constraints on
SLp-1.1.

## Falsifiable first hypothesis

A query-decoder model trained on separately versioned human and yeast molecular
trajectories will predict perturbation-specific held-gene effects better than a
training-set mean baseline when the validation genes are absent from every
quantitative pretraining and reward trajectory.

The first candidate advances only when one frozen run satisfies all of these
conditions on molecular validation data:

- zero held-intervention-gene overlap and zero benchmark-label records;
- at least 0.02 nats per observed target and 2% Gaussian-NLL improvement over
  both the training-set mean and one fixed ridge baseline in a preregistered
  value space;
- perturbation-specific, training-centroid-adjusted Pearson correlation of at
  least 0.10;
- non-negative NLL delta for every represented species and protected source;
- no post hoc sign selection, source removal, threshold change, or checkpoint
  selection using an SL benchmark.

Failure rejects the candidate, not the data split. The external SL benchmarks
remain absent from the pretraining workload.

## Model interface

Each example contains three variable-size sets:

1. basal/context tokens describing molecular state, biological background,
   assay, protocol, dose, and time;
2. intervention tokens describing genetic or chemical actions;
3. readout-query tokens naming the molecular quantities to predict.

Tokens use released sequence-, protein-, annotation-, and measurement-derived
features. The model has no learned gene-ID embedding and no fixed gene list.
Context and action sets have no positional encoding, making their order
irrelevant. A cross-attention-only query decoder emits a mean and scale for
every sparse readout query instead of reconstructing one hard-coded expression
panel. Queries do not attend to each other, so a marginal prediction is exactly
invariant to panel membership, ordering, padding, and chunking.

Species is an explicit continuous feature block, not inferred from gene names.
Yeast observations retain yeast identifiers and experimental context. Orthology
and sequence similarity are auxiliary relations; yeast interaction scores are
never relabeled as human interaction measurements.

## Data boundaries

OMF `DatasetSnapshot` resources separately identify pretraining, molecular
validation, reward, and final-holdout corpora. Every snapshot carries rights,
content hashes, source versions, stable entity identifiers, species, assay and
protocol metadata, normalization provenance, and its accessible modalities.

The pretraining and reward snapshots may contain only molecular measurements.
Every quantitative trajectory involving a molecular-validation or final-holdout
intervention gene is excluded. Static sequence and annotation features may be
available for held genes because cold start means unseen intervention outcomes,
not absence of public molecular identity.

SL benchmark labels live in independent snapshots that no training or reward
workload references. A supervised SL decoder, if later added, is a distinct OMF
module and release line.

## Training and evaluation

Maximum-likelihood pretraining will mix sources with explicit per-source
sampling weights and report source- and species-level metrics. Molecular
reinforcement is currently disabled in the module contract. It may be opened
only after a deterministic continuation control exists; both continuations
must use identical reward records, token budgets and seeds, and every rejected
epoch must roll back before another update.

Selection metrics must remove the shared average perturbation shift and include
simple mean, additive, and linear baselines. The separate molecular evaluator
reports training-centroid-adjusted Pearson/cosine and common-panel centroid
accuracy by source and species. Scaling claims require fixed data mixtures
evaluated at multiple token budgets; parameter growth alone is not a scientific
result.

## Release status

No SLp-1.1 model is released. The repository contains the OMF project boundary,
corpus leakage auditor, workload and architecture contract only.

OpenModelFactory 1.0 is supported for its local lifecycle on Linux x86-64, not
this Windows checkout. It also passes JSON protocol state—but not a materialized
large model artifact—to the inference adapter. A portable SLp release therefore
requires an upstream artifact-to-adapter contract before promotion; absolute
run paths or model weights serialized into metadata are forbidden workarounds.
The same OMF revision also cannot pin a newly produced artifact for a dependent
stage in the same workload. Training and molecular evaluation therefore use two
separately admitted runs with the exact prediction-artifact digest pinned in
the second run until that upstream contract is fixed.
Only the built-in Linux-local binding is currently declared. Scaling beyond one
host also requires a tested `omf.executor/v1` provider; launching Modal from a
network-denied training module is not an executor integration.

## Historical evidence

The SLp-1 record remains in `docs/results.md`, `docs/model-card.md`, and
`model/v1/`. Those development results are retrospective and do not establish
SLp-1.1 performance.
