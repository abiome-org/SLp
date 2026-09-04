# SLp-1.1

## Purpose

SLp-1.1 is a species-aware genomics world-model program. It learns conditional
distributions of molecular measurements after interventions. Synthetic
lethality is not a pretraining label, a model output type, or a release gate.
It remains one downstream test of whether the learned transition rules transfer
to two genes absent from quantitative trajectory fitting.

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
- at least 2% Gaussian-NLL improvement over both the training-set mean and one
  fixed ridge baseline;
- effect Pearson correlation of at least 0.10;
- non-negative NLL improvement for every represented species;
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
irrelevant. A query decoder emits a mean and calibrated scale for every sparse
readout query instead of reconstructing one hard-coded expression panel.

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

Maximum-likelihood pretraining mixes sources with explicit per-source sampling
weights and reports source- and species-level metrics. A later stochastic
continuation may use only molecular reward snapshots. Each reinforcement epoch
is compared with its frozen pre-update checkpoint on the same molecular
validation objective and is discarded unless it improves without a protected
source regression.

Selection metrics must remove the shared average perturbation shift and include
simple mean, additive, and linear baselines. Scaling claims require fixed data
mixtures evaluated at multiple token budgets; parameter growth alone is not a
scientific result.

## Release status

No SLp-1.1 model is released. The repository contains the OMF project boundary,
corpus leakage auditor, workload and architecture contract only.

OpenModelFactory 1.0 is supported for its local lifecycle on Linux x86-64, not
this Windows checkout. It also passes JSON protocol state—but not a materialized
large model artifact—to the inference adapter. A portable SLp release therefore
requires an upstream artifact-to-adapter contract before promotion; absolute
run paths or model weights serialized into metadata are forbidden workarounds.
Only the built-in Linux-local binding is currently declared. Scaling beyond one
host also requires a tested `omf.executor/v1` provider; launching Modal from a
network-denied training module is not an executor integration.

## Historical evidence

The SLp-1 record remains in `docs/results.md`, `docs/model-card.md`, and
`model/v1/`. Those development results are retrospective and do not establish
SLp-1.1 performance.
