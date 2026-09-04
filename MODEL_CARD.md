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

The current typed sparse candidate separates identifiers from numerical model
inputs. Checksum-pinned entity, query and panel dictionaries provide
provenance, while the network receives feature vectors, presence masks and
ontology-type indices. Targets are record-local CSR query/value lists rather
than a dense record-by-global-readout tensor. Gaussian and negative-binomial
heads share the world representation but retain distinct likelihoods. Its OMF
training entry point performs deterministic bounded AdamW fitting from the
pretraining snapshot alone, weights scheduled records equally, and emits
timestamp-free bounded checkpoint bytes plus target-free validation-query
predictions. Protected validation truth is structurally absent from that run
and is joined only by the independent evaluator. A count-specific library-size
offset is not implemented, and the v1 target-free query omits context and
continuous covariates. This candidate cannot be selected until those relevant
likelihood/query contracts, the admitted biological training path, and all
frozen molecular baseline comparisons pass.

Species is an explicit continuous feature block, not inferred from gene names.
Yeast observations retain yeast identifiers and experimental context. Orthology
and sequence similarity are auxiliary relations; yeast interaction scores are
never relabeled as human interaction measurements.

## Data boundaries

OMF `DatasetSnapshot` resources separately identify pretraining, molecular
validation, and final-holdout corpora. Molecular reward is disabled and no
reward snapshot is part of the current protocol. Every snapshot carries rights,
content hashes, source versions, stable entity identifiers, species, assay and
protocol metadata, normalization provenance, and its accessible modalities.

The pretraining snapshot may contain only molecular measurements. Every
quantitative trajectory involving a molecular-validation or final-holdout
intervention gene is excluded. Static sequence and annotation features may be
available for held genes because cold start means unseen intervention outcomes,
not absence of public molecular identity. If molecular reward is introduced in
a future protocol, it must receive its own snapshot and the same held-gene
exclusions before the active corpus contract can be revised.

OMF 1.0 provides immutable resources, rights checks, declared-input
materialization, and lineage, but it does not provide actor-scoped snapshot
ACLs or filesystem isolation for the local executor. Dataset names, CLI actor
strings, and governed adapter behavior are not confidentiality boundaries. A
credible held-out claim therefore requires a custodian factory for the full
raw source, a physically separate clean training factory/store that never
contains held truth, a validation factory under a distinct OS/service identity,
and an independently controlled final factory opened only after candidate
lock. Until those boundaries exist, local role separation is integrity and
process evidence only.

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

The frozen context-only and TxPert mean/additive implementations operate on
separately pinned fitting and reference profile snapshots. TxPert effects never
cross a source/species stratum, and context-cold access requires each reference
intervention to have fitting outcomes in another context of that same stratum.
These v1 baselines emit point predictions only and remain evaluation-blocked
until a probabilistic scale is frozen. The feature-bilinear ridge baseline is
also blocked until released query and action feature vectors exist.

## Release status

No SLp-1.1 model is released. The repository contains the OMF project boundary,
corpus leakage auditor, outcome-blind held-intervention roster, typed sparse
world-candidate and OMF trainer, workload contracts, SGD stable-identity
normalizer, molecular point baselines, and target-separated molecular evaluator.
The sparse module can train and emit deterministic checkpoints and target-free
predictions, but it has not yet run on an admitted quantitative biological
corpus and therefore supplies no candidate-selection or performance evidence.

The exact SGD 2026-08-28 identity snapshot is admitted under CC-BY-4.0, and its
network-denied OMF normalization run produced immutable current-ORF, typed
external-relation, mapping-manifest and retired-quarantine artifacts. This is
identity provenance only. No biological outcome snapshot or model-fitting
permission follows from the mapping result.

The exact SGD S288C R64.5.1 translated-ORF protein source is also admitted as
a separately rights-bearing static-only snapshot. It covers all 6,613 current
ORFs by stable SGD ID. A deterministic 21-dimensional sequence-statistics
baseline has now been produced twice byte-identically and admitted as
`omf://abiome/slp/datasetsnapshot/slp-1-1-sequence-statistics-feature-block-v1@sha256:e9733974c551bca3af93c4cb488972f5167da5e7e3cf48ef5803348cd20d91e5`.
It is an intentionally weak static baseline, not a learned or frontier protein
representation.
The outcome-blind proteome action inventory and typed protein relations have
been composed twice into byte-identical, relation-closed identity payloads and
admitted as
`omf://abiome/slp/datasetsnapshot/slp-1-1-static-entity-universe-v1@sha256:de3efddf5a9e4f66496a1edda14b04de774e972bc7b9efd30964644de2a56cac`.
That source universe contains 5,187 SGD genes and 1,850 UniProtKB proteins;
711 genes exist only to close typed relations and are not action eligible.
The artifact uses `(ncbiTaxon, entityId)` identities and contains no numeric
features or train/validation/final assignments. The pretraining observations,
static sequence baseline, and outcome-blind held roster have now been composed
twice byte-identically into fitting-only corpus v1.2 and admitted as
`omf://abiome/slp/datasetsnapshot/slp-1-1-proteome-composite-corpus-v1@sha256:e91cad825b8a2e972da293902c630331a92ab664c5d14a95a65ff38090db6c48`.
The historical sparse corpus consumer still keys by bare ID and remains frozen;
a new world consumer must enforce composite joins before biological training.
These are static-data, identity, and corpus-construction boundaries, not
training or performance evidence.

The source-normalized yeast proteome pretraining observations, HIS3 basal
control, and protected molecular-validation observations are now separately
admitted as rights-bearing snapshots. The validation snapshot exists only in
the current custodian factory: its rights record forbids fitting, reward,
calibration and current-factory evaluation, but OMF 1.0 does not enforce those
documentary purpose restrictions. The molecular-final workload has never been
executed. A clean training factory and distinct validation/final service
identities and stores remain prerequisites for credible held-out evaluation.

A separate clean-training corpus-audit v1.4 contract is now implemented beside
the frozen earlier audits. Its interface contains no validation or
final quantitative input: it authenticates the exact composed optimizer corpus,
outcome-blind held roster, and protected-source inventories with a recipient-
and challenge-bound Ed25519 custodian authorization, then independently scans
all active actions for the held union. The v1.4 parser independently accepted
the production corpus structure and reproduced its counts and content digests,
but that compatibility check is not a signed authorization. The production
public trust anchor is intentionally unprovisioned pending an independent key
ceremony, and no physically separate training factory exists. Therefore the
v1.4 signed biological handoff is deliberately fail-closed and has not run.
Its signature is content authorization, not one-time-use enforcement,
source-to-corpus lineage, rights verification, or filesystem isolation; those
remain independent release gates. Existing world-trainer and evaluator
versions remain frozen and must not be silently adapted to the new corpus or
audit contract.

That normalization run used an empty dependency lock in an attested Python
3.12 environment. It establishes the mapping bytes and lineage, not a portable
runtime closure; release-eligible workloads still require retained, fully
hash-pinned wheels and offline dependency realization.

OpenModelFactory 1.0 is supported for its local lifecycle on Linux x86-64, not
this Windows checkout. It also passes JSON protocol state—but not a materialized
large model artifact—to the inference adapter. A portable SLp release therefore
requires an upstream artifact-to-adapter contract before promotion; absolute
run paths or model weights serialized into metadata are forbidden workarounds.
The same OMF revision also cannot pin a newly produced artifact for a dependent
stage in the same workload. Corpus audit, training, and molecular evaluation
therefore use separate admitted runs with exact prior artifact digests pinned
at each boundary until that upstream contract is fixed. The current trainer
also records a release blocker until factory policy independently proves that
its pinned audit artifact came from the admitted corpus-audit module and run.
OMF 1.0 policy cannot authorize individual snapshots or distinguish fitting
from evaluation access, and its local executor can read other host-accessible
factory files. Separate service identities, stores, and execution sandboxes are
therefore additional release blockers; custom unsupported policy keys are not
accepted as a workaround.
OMF 1.0 module compatibility tests also reuse a single `module-0` fixture
directory for every manifest named `module.yaml` without clearing stale
completion files. The v1.3 worker produced the expected validation-only output,
but the CLI first returned a stale prior-module failure. Collision-free module-
test identity is an upstream compatibility blocker; no duplicate-manifest or
runtime-state workaround is part of the release path.
Only the built-in Linux-local binding is currently declared. Scaling beyond one
host also requires a tested `omf.executor/v1` provider; launching Modal from a
network-denied training module is not an executor integration.

## Historical evidence

The SLp-1 record remains in `docs/results.md`, `docs/model-card.md`, and
`model/v1/`. Those development results are retrospective and do not establish
SLp-1.1 performance.
