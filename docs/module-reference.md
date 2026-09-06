# Module reference

Reference for retained research modules. Current training and usage are in
[development.md](development.md) and the [model card](../MODEL_CARD.md).
Module-local contracts are retained only where packaging or tests consume them.
Historical descriptions below describe their named module, not the current model.

- [slp-1-1-atlas-genotype-inventory: README](#slp-1-1-atlas-genotype-inventory-readme)
- [slp-1-1-batch-ridge-v1: CONTRACT](#slp-1-1-batch-ridge-v1-contract)
- [slp-1-1-cell-state-v1: CONTRACT](#slp-1-1-cell-state-v1-contract)
- [slp-1-1-cell-world-phenotype-v1: CONTRACT](#slp-1-1-cell-world-phenotype-v1-contract)
- [slp-1-1-compositional-state-v1: CONTRACT](#slp-1-1-compositional-state-v1-contract)
- [slp-1-1-control-transition-v1: CONTRACT](#slp-1-1-control-transition-v1-contract)
- [slp-1-1-control-transition-v2: CONTRACT](#slp-1-1-control-transition-v2-contract)
- [slp-1-1-control-transition-v3: CONTRACT](#slp-1-1-control-transition-v3-contract)
- [slp-1-1-control-transition-v4: CONTRACT](#slp-1-1-control-transition-v4-contract)
- [slp-1-1-corpus-audit: README](#slp-1-1-corpus-audit-readme)
- [slp-1-1-count-latent-continuation-inference-v1: CONTRACT](#slp-1-1-count-latent-continuation-inference-v1-contract)
- [slp-1-1-count-latent-inference-v2: CONTRACT](#slp-1-1-count-latent-inference-v2-contract)
- [slp-1-1-count-latent-state-v1: CONTRACT](#slp-1-1-count-latent-state-v1-contract)
- [slp-1-1-count-moments-v1: CONTRACT](#slp-1-1-count-moments-v1-contract)
- [slp-1-1-count-panel-data-v1: CONTRACT](#slp-1-1-count-panel-data-v1-contract)
- [slp-1-1-count-world-evaluation-v1: CONTRACT](#slp-1-1-count-world-evaluation-v1-contract)
- [slp-1-1-count-world-inference-v1: CONTRACT](#slp-1-1-count-world-inference-v1-contract)
- [slp-1-1-count-world-inference-v2: CONTRACT](#slp-1-1-count-world-inference-v2-contract)
- [slp-1-1-count-world-response-query-inference-v1: CONTRACT](#slp-1-1-count-world-response-query-inference-v1-contract)
- [slp-1-1-count-world-training-v1: CONTRACT](#slp-1-1-count-world-training-v1-contract)
- [slp-1-1-fixed-query-transition-v1: CONTRACT](#slp-1-1-fixed-query-transition-v1-contract)
- [slp-1-1-gene-state-v1: CONTRACT](#slp-1-1-gene-state-v1-contract)
- [slp-1-1-guide-composition-v1: CONTRACT](#slp-1-1-guide-composition-v1-contract)
- [slp-1-1-held-roster: CONTRACT](#slp-1-1-held-roster-contract)
- [slp-1-1-molecular-baselines: README](#slp-1-1-molecular-baselines-readme)
- [slp-1-1-molecular-eval: README](#slp-1-1-molecular-eval-readme)
- [slp-1-1-molecular-mean-objective-v1: CONTRACT](#slp-1-1-molecular-mean-objective-v1-contract)
- [slp-1-1-paired-state-v1: CONTRACT](#slp-1-1-paired-state-v1-contract)
- [slp-1-1-proteome-corpus-compose-v1: CONTRACT](#slp-1-1-proteome-corpus-compose-v1-contract)
- [slp-1-1-proteome-corpus-compose-v1: README](#slp-1-1-proteome-corpus-compose-v1-readme)
- [slp-1-1-proteome-inventory: README](#slp-1-1-proteome-inventory-readme)
- [slp-1-1-proteome-observation-prepare-v1: CONTRACT](#slp-1-1-proteome-observation-prepare-v1-contract)
- [slp-1-1-proteome-observation-prepare-v1: README](#slp-1-1-proteome-observation-prepare-v1-readme)
- [slp-1-1-proteome-protected-observation-prepare-v1: README](#slp-1-1-proteome-protected-observation-prepare-v1-readme)
- [slp-1-1-reduced-rank-response-inference-v1: CONTRACT](#slp-1-1-reduced-rank-response-inference-v1-contract)
- [slp-1-1-reduced-rank-response-v1: CONTRACT](#slp-1-1-reduced-rank-response-v1-contract)
- [slp-1-1-response-omf2: CONTRACT](#slp-1-1-response-omf2-contract)
- [slp-1-1-sequence-statistics-feature-block-v1: CONTRACT](#slp-1-1-sequence-statistics-feature-block-v1-contract)
- [slp-1-1-sequence-statistics-feature-block-v1: README](#slp-1-1-sequence-statistics-feature-block-v1-readme)
- [slp-1-1-sgd-map: README](#slp-1-1-sgd-map-readme)
- [slp-1-1-sl-predictor-v1: CONTRACT](#slp-1-1-sl-predictor-v1-contract)
- [slp-1-1-slim-baseline-v1: CONTRACT](#slp-1-1-slim-baseline-v1-contract)
- [slp-1-1-static-entity-universe-v1: CONTRACT](#slp-1-1-static-entity-universe-v1-contract)
- [slp-1-1-static-entity-universe-v1: README](#slp-1-1-static-entity-universe-v1-readme)
- [slp-1-1-training-corpus-audit-v1-3: README](#slp-1-1-training-corpus-audit-v1-3-readme)
- [slp-1-1-training-corpus-audit-v1-4: README](#slp-1-1-training-corpus-audit-v1-4-readme)
- [slp-1-1-training-corpus-audit-v1-5: CONTRACT](#slp-1-1-training-corpus-audit-v1-5-contract)
- [slp-1-1-training-corpus-audit-v1-5: README](#slp-1-1-training-corpus-audit-v1-5-readme)
- [slp-1-1-world-sparse: CONTRACT](#slp-1-1-world-sparse-contract)
- [slp-1-1-world-transition-v1: CONTRACT](#slp-1-1-world-transition-v1-contract)
- [slp-1-1-yeast-prepare: SOURCE_FORMAT](#slp-1-1-yeast-prepare-source-format)
- [slp-1-1-yeast-seurat-stream-v1: CONTRACT](#slp-1-1-yeast-seurat-stream-v1-contract)

<a id="slp-1-1-atlas-genotype-inventory-readme"></a>

## slp-1-1-atlas-genotype-inventory: README

Original location: `modules/slp-1-1-atlas-genotype-inventory/README.md`.

### SLp-1.1 atlas genotype identity inventory

This module extracts only the genotype identities shared between the exact
`ptbs.control` and `ptbs.nacl` frames in Zenodo record `14062629`'s pinned
`ptb_summary.Rdata`. It requires the exact nine-column frame contract and a
non-null integer `cell_number` greater than five for every row. The only frame
values the adapter accesses are `assignment_consensus2` and `cell_number`.

The pure-Python `rdata==1.1.0` parser converts each complete frame, so the audit
does not claim phenotype values were unparsed or uninterpreted by the library.
It records the narrower, testable boundary: the adapter never indexes,
inspects, uses, or emits the seven leverage/`Stucked` phenotype columns.

Candidates are the exact non-`WT` assignments present in both conditions. The
adapter removes one literal `bc-` prefix and performs one exact, case-sensitive
lookup against the pinned current-ORF map. It never normalizes case, resolves a
display symbol, follows a retired redirect, or selects among ambiguous current
targets. Retired status is established only by the separate immutable retired
quarantine artifact.

The held-roster inventory contains one record for each unique current SGD
CURIE. Separate evidence preserves the exact source assignment and every
quarantine classification. Neither artifact contains phenotype outcomes.

This small identity snapshot does not admit the 5.9 GB transcriptomic atlas or
authorize quantitative training. OMF 1.0 also cannot directly feed the output
artifact to held-roster; the exact inventory bytes need a separate
rights-bearing copied `DatasetSnapshot` admission with RunResult provenance.

The dependency lock makes package selection reproducible for CPython 3.12 on
Linux x86-64 with a manylinux2014/glibc-2.17 floor. It contains
`rdata==1.1.0` and all nine transitive packages with one verified distribution
hash each; it has no host path, editable source, VCS reference, or R runtime.
It is not an offline wheelhouse or a portable release closure: execution still
requires acquiring those exact distributions, and release portability remains
blocked until the dependency payload is retained and verified independently.
The module carries a small standard-library implementation of the documented
`omf.module/v1` request/result file protocol because OMF 1.0 does not inject
its controller SDK into non-empty isolated dependency environments. This is
part of the admitted code package, not a host interpreter or `PYTHONPATH`
dependency.

<a id="slp-1-1-batch-ridge-v1-contract"></a>

## slp-1-1-batch-ridge-v1: CONTRACT

Original location: `modules/slp-1-1-batch-ridge-v1/CONTRACT.md`.

### Batch-adjusted ridge v1

This numerical baseline fits complete molecular-response rows with a shared
linear feature effect and an unpenalized intercept for each observed source
batch. Its objective is `sum_i w_i ||y_i-X_i W-b_batch||² + alpha ||W||²`.
Sufficient statistics stream across bounded blocks, avoiding materializing all
population-by-query rows together. Positive finite alpha is required.

Inputs are caller-normalized features, responses, nonnegative statistical
weights and source batch labels. Only fitting rows may enter the statistics;
feature normalization, inner-fold selection and all weights must also be fitted
within each training fold. Every output query must be observed in every supplied
row. Missing molecular values must not be replaced with zeros for this module.

Batch intercepts are nuisance effects estimated from fitting interventions.
They are neither wild-type molecular states nor forecasts of unseen contexts.
Prediction for a batch absent from fitting fails explicitly. No gene identity,
gene embedding, application score or held-out outcome enters this module.

The returned batch-only means provide the corresponding no-feature baseline.
Readout comparisons must preserve identical populations, weights and feature
availability, and evaluate perturbation-specific patterns beyond shared means.
Stored coefficients, intercepts and batch labels suffice for inference.

Three checks verify equality with independently constructed augmented weighted
least squares, invariance of feature coefficients to batch-constant response
shifts, bounded-block accumulation, unseen-batch rejection and nonfinite-input
rejection without updating sufficient statistics.

<a id="slp-1-1-cell-state-v1-contract"></a>

## slp-1-1-cell-state-v1: CONTRACT

Original location: `modules/slp-1-1-cell-state-v1/CONTRACT.md`.

### Molecular cell state v1

This application-neutral experimental module encodes paired molecular cell
measurements using caller-supplied RNA and protein query features. It has no
learned identity embeddings, internal gene roster, benchmark handling, SL score,
intervention fitting, time variable or dose model. Query-feature dimensions are
fixed by configuration; query counts and ordering are external data contracts.

The encoder pools measurement-weighted feature keys and measured-panel summaries
into a shared state. Missing inputs are masked explicitly. At least one modality
must be observed for each cell. The reconstruction objective gives equal total
weight to RNA and protein, in externally fitted standardized measurement units.
Input denoising masks must remain separate from target-observation masks.

Each observation head is affine in latent state and generated from query
features. Consequently mean-state decoding equals mean decoded observations.
This avoids a nonlinear-decoder averaging assumption in a future pseudobulk
forecast. It does not make the latent variables biologically identifiable.

`observe_delta` adds a decoded state change to explicit measured controls. A zero
change gives exact control identity. No intervention transition has been fitted
by this module's implementation. Reconstruction of cells from already observed
measurements is not evidence of intervention forecasting or cell generation.

The intended first data source is a separately pinned paired-cell snapshot;
all numerical transforms and training inputs must be specified by its trainer.
Cell reconstruction validation within fitting intervention genes is distinct
from validation on unseen intervention genes. No biological run is complete yet.

<a id="slp-1-1-cell-world-phenotype-v1-contract"></a>

## slp-1-1-cell-world-phenotype-v1: CONTRACT

Original location: `modules/slp-1-1-cell-world-phenotype-v1/CONTRACT.md`.

### Single-intervention fitness observation baseline

This retained development baseline maps frozen molecular-world signatures and
action encodings to continuous DepMap gene effects through a context-query
factorization. It has no learned gene-ID embedding or SL-pair training target.
The corpus excludes the global molecular-held human interventions and preserves
the source's fitting-gene and fitting-cell restrictions. Feature normalization
uses fitting observations; unknown measurements are masked.

`phenotype.py` trains and exports the observation head. `forecast.py` generates
fitness profiles using one shared fitting-data reference for trained and random
heads. The later `v2` forecast artifacts correct the earlier model-dependent
reference. `profile_benchmark.py` in the separate SL application module reads
those forecasts with a fixed positive-cosine score.

This component predicts single fitness, without a learned persistent
double-intervention state. It is a baseline for the genomic fitness world,
not the current world-model architecture or an SL performance improvement.
Its full measured results are retained in `docs/results.md`.

<a id="slp-1-1-compositional-state-v1-contract"></a>

## slp-1-1-compositional-state-v1: CONTRACT

Original location: `modules/slp-1-1-compositional-state-v1/CONTRACT.md`.

### Observed-state composition pilot v1

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

<a id="slp-1-1-control-transition-v1-contract"></a>

## slp-1-1-control-transition-v1: CONTRACT

Original location: `modules/slp-1-1-control-transition-v1/CONTRACT.md`.

### Control-anchored aggregate molecular transition

This is a self-contained experimental numerical module. It consumes supplied
static action and query descriptors, measured basal molecular tokens, a control
mean and scale aligned to each requested query, and optional supplied assay or
perturbation-mode descriptors. It learns no gene-ID, assay-ID, mode-ID, species-
ID, or context-ID embedding.

The latent contract is

`state = encoded_basal_state + intervention_delta`.

The action encoder is a permutation-invariant set function. It combines the
mean encoded action with the average elementwise product over distinct action
pairs, giving symmetric pair capacity for multi-action records. Padding is
controlled only by a Boolean mask. Masked nonfinite entries are inert; a
nonfinite valid action is rejected.

The pair summary has its own bias-free projection. A single-action corpus
contains an identically zero pair summary, so its launcher must freeze that
projection and record it as untrained. Multi-action fitting can later enable
the projection without changing the empty-set identity.

The empty action set is an algebraic identity. A Boolean presence gate makes
both latent and molecular intervention deltas exactly zero. Empty rows select
the supplied control mean and control scale directly rather than asking a
learned network to reconstruct them. This applies to zero-width action tensors
and to padded tensors whose mask is entirely false. It does not depend on
seeing control records during fitting.

Query descriptors decode independently from one shared state. Reordering or
chunking queries in evaluation mode leaves corresponding outputs unchanged,
provided each chunk receives the same basal tokens and its aligned control
mean and scale. Basal tokens are separate from requested output queries so the
definition of state does not change with output chunk size.

The output is a diagonal Gaussian for aggregate molecular measurements. It is
not a time-dependent trajectory, mechanistic causal model, single-cell
generator, learned population distribution, or demonstrated combination
model. Symmetric pair capacity permits nonadditivity but does not establish it.
Mean action-embedding pooling also does not force a multi-action prediction to
equal the sum of individually learned molecular effects. A future compositional
revision should sum individually context-conditioned molecular deltas and add a
separately trained symmetric interaction residual; multi-action molecular
training is required before that residual can carry evidence.
No low-rank covariance or cell-level sampling model is included in this first
revision. The control identity anchors the supplied measurement baseline; it
does not infer an unseen context's control state.

The minimum useful experiment is a fixed-split comparison against the existing
transition candidate using the same training-only normalization, action/query
features, assay/mode features, control baseline, uncertainty inputs, optimizer,
and stopping rule. It must first pass exact held-out control reconstruction by
construction, then avoid regression on intervention-gene-macro molecular NLL
and centroid-adjusted profile correlation in every represented context. A
combination claim additionally requires separately held single- and multi-
action records, including strata with one and two held constituent genes.
Until that comparison runs, this module fixes a testable representational
defect but has no empirical performance claim and should not be called an
improved world model.

<a id="slp-1-1-control-transition-v2-contract"></a>

## slp-1-1-control-transition-v2: CONTRACT

Original location: `modules/slp-1-1-control-transition-v2/CONTRACT.md`.

### Minimal control-anchored transition revision

This revision isolates control anchoring from the capacity changes in
`slp-1-1-control-transition-v1`. Its action, basal-context, query, transition,
and mean decoder topology matches the original successful transition candidate:
one action encoder, one context encoder, one transition MLP, one query encoder,
and one bilinear mean decoder. It has no response-state network, pair branch,
learned scale branch, or identity embedding.

The latent contract is `state = basal_state + intervention_delta`. Empty and
fully masked action sets gate both latent and molecular intervention deltas to
exact zero and return the externally supplied control mean bit-for-bit. Masked
nonfinite padding is inert. Nonempty action sets use sum pooling and are
permutation invariant within floating-point tolerance.

Decoder amplitude is a positive query vector with shape `[Q]`. The API rejects
a batch- or context-indexed amplitude, preventing target-context perturbation
residuals from entering inference. A run may use unit amplitude or one pooled
per-query amplitude fitted only from source-context training outcomes. Record-
specific observation scale is a separate likelihood input and never changes
state or predicted mean.

This model is a diagonal Gaussian for aggregate molecular measurements. It has
no explicit multi-action interaction capacity, additive-composition guarantee,
time dynamics, single-cell generator, or inferred unseen-context control. An
unseen context requires a measured control baseline and control-only basal
descriptor on the same query panel. No empirical improvement is claimed until
the preregistered matched experiment runs.

`inference.py` is the portable fitted-checkpoint runtime. It requires the exact
ordered query roster and frozen control-panel mask, accepts raw static action
features, and reproduces the training feature and basal-token normalization.
The caller supplies the control molecular mean. A new context therefore needs
only its control-only fixed-panel descriptor; no perturbation statistic from
that context enters the shared decoder amplitude. The runtime returns no scale
when `measurement_scale` is omitted. A caller-provided positive measurement
scale is passed through as observation metadata and cannot change the molecular
mean or latent state.

<a id="slp-1-1-control-transition-v3-contract"></a>

## slp-1-1-control-transition-v3: CONTRACT

Original location: `modules/slp-1-1-control-transition-v3/CONTRACT.md`.

### Decoder of latent state changes

This numerical revision retains the v2 architecture, parameter shapes and
training-independent observation scale. It changes only the molecular decoder:
`mean = control_mean + amplitude * D(intervention_delta, query)`.
The learned latent relation remains `state = basal_state + intervention_delta`.
Because D is linear in the latent argument, this is algebraically
`control_mean + amplitude * (D(state, query) - D(basal_state, query))`.
Direct delta decoding avoids subtraction of two large decoded profiles.

A zero latent intervention delta must produce the control molecular mean for
both an empty action set and a nonempty action with learned zero effect. The
decoder does not receive an action-presence flag. Revision v2 instead decoded
the total state and gated its molecular effect by action presence; consequently
an unchanged latent state could produce a nonzero molecular change.

The revision has no new parameters, feature modality, learned identity or
uncertainty branch. It retains masked set encoding and a shared fitting-only
query amplitude. It does not establish biological accuracy, single-cell
generation, temporal dynamics or combination transfer. Existing v2 checkpoints
remain numerical evidence for v2, and must not be relabeled as v3 models.
A separately frozen training experiment is required before judging performance.

<a id="slp-1-1-control-transition-v4-contract"></a>

## slp-1-1-control-transition-v4: CONTRACT

Original location: `modules/slp-1-1-control-transition-v4/CONTRACT.md`.

### SLp-1.1 control transition v4

This revision changes only the molecular observation decoder relative to v3.
Action, control-context, transition, and query encoders retain the v3 topology.

The decoder is

`D(z,q) = linear(GELU(W_state z + W_query q + b))`

with 64 hidden units and no decoder dropout. The reported molecular change is
the shared per-query amplitude multiplied by
`D(basal_state + intervention_delta, q) - D(basal_state, q)`.

An empty or fully masked action set therefore produces exactly zero latent and
molecular change. The decoder consumes encoded static/query descriptors and
contains no learned gene identity, action-presence gate, outcome encoder, or
auxiliary loss. It is a nonlinear measurement-decoder experiment and does not
identify biological dynamics.

<a id="slp-1-1-corpus-audit-readme"></a>

## slp-1-1-corpus-audit: README

Original location: `modules/slp-1-1-corpus-audit/README.md`.

### SLp-1.1 corpus audit

This module is the mandatory, fail-closed boundary between three quantitative
corpora and any trainer or molecular evaluator. Inputs are exact copied,
revision-pinned OMF `DatasetSnapshot` objects for pretraining, molecular
validation, molecular final holdout, and the complete global held-intervention
roster. It also requires the two or more protected, outcome-blind
source-inventory DatasetSnapshots from which that roster was formed.

Corpus-audit v1.2 is deliberately reward-disabled. Its config and emitted
attestation both require `rewardEnabled: false`; a `molecularReward` input is an
error rather than an empty, copied, or relabeled placeholder. The module has no
path by which reward outcomes can enter this audit. Opening molecular reward
requires a new versioned audit and consumer contract after deterministic
continuation and rollback controls exist.

The audit verifies outer OMF resource and manifest identities, exact internal
file sets and hashes, corpus roles, rights, zero benchmark labels, bounded NPZ
identity arrays, and exact record-level source, species, and intervention
inventories. It independently recomputes every roster assignment from
`slp-1.1-yeast-global-held-v1\x00<SGD-CURIE>` and rejects role, hash, coverage,
mapping, count, or source-inventory drift. Every validation/final trajectory
must have its matching roster role. Every roster validation/final gene is
excluded from pretraining trajectories.

The protected inventories are parsed again from their exact JSONL bytes. Their
file hashes, record counts, QC sets, mapping identity, and outer DatasetSnapshot
identities must reproduce every source entry in `coverage.json`; their computed
QC-passing intersection must equal the roster exactly. Coverage metadata alone
is never accepted as proof of a global intersection.

A successful run emits exactly one file artifact,
`corpus-audit/corpus-audit.json`, using schema `slp.corpus-audit/v1.2`. Paths and
timestamps are absent, so identical immutable inputs produce identical bytes.
OMF v1 cannot safely pass this newly generated artifact into another stage of
the same workload. Training therefore happens in a later admitted run and pins
the exact previous-run artifact manifest; independent evaluation additionally
requires that file to be admitted and verified as the sole member of its own
rights-bearing `DatasetSnapshot`. No benchmark
input, sibling-stage path, or other same-workload handoff is permitted. Until a
factory policy independently proves the artifact's producing module and run,
the sparse trainer records that missing lineage proof as a release blocker.

OMF 1.0 workload manifests name dataset inputs as `dataset/<name>`. OMF resolves
each alias once during run admission, verifies current and pinned training
rights plus copied artifact bytes, and records the full immutable DatasetSnapshot
revision in the admitted run. A full `omf://...@sha256:...` URI in a stage input
is not materialized by OMF 1.0 and is therefore not used as a workaround.

Gene-set hashes are SHA-256 over canonical JSON arrays of sorted unique CURIEs
(`ensure_ascii=true`, compact separators, no trailing newline). Raw file hashes
cover exact bytes. OMF revisions and outer manifest digests retain their
`sha256:` prefix; internal and gene-set hashes are lowercase hexadecimal only.

<a id="slp-1-1-count-latent-continuation-inference-v1-contract"></a>

## slp-1-1-count-latent-continuation-inference-v1: CONTRACT

Original location: `modules/slp-1-1-count-latent-continuation-inference-v1/CONTRACT.md`.

### Count-latent continuation inference v1

`Predictor(artifact, arm, device="cpu")` loads one registered continuation arm
after checking the protocol, model, shared reference, and embedded numerical
core hashes. `arm` must be a key in `artifact-manifest.json["arms"]`.

`predict(action_features, gem_group_weights, query_indices=None)` accepts raw
static intervention features and nonnegative experimental-group mixture
weights. It returns the conditional-prior population mean in CP10k and
`log1p(CP10k)` units. An explicit all-false action mask gives the exact shared
control anchor.

The interface has no count, library-size, fitted aggregate-target, development,
test, or benchmark input. It is an aggregate molecular-mean approximation, not
a single-cell generator or identified biological mechanism.

<a id="slp-1-1-count-latent-inference-v2-contract"></a>

## slp-1-1-count-latent-inference-v2: CONTRACT

Original location: `modules/slp-1-1-count-latent-inference-v2/CONTRACT.md`.

### Count-latent prior inference v2

`Predictor(artifact, device="cpu")` loads a frozen K562 count-latent-state
artifact after verifying the protocol, model, reference, and embedded numerical
core checksums. The artifact must contain `protocol.json`,
`artifact-manifest.json`, `model.safetensors`, `reference.npz`, and
`source/count_latent_state.py`.

`predict(action_features, context_weights, query_indices=None, chunk_size=1024)`
returns the conditional-prior molecular mean in `log1p(CP10k)` units. Action
features have shape `[B, 577]`; context weights have shape `[B, 48]` and are
nonnegative with positive row sums; query indices select the frozen 8,563-query
axis. Empty action features return the exact frozen control anchor for the
specified context mixture.

Cell library size and observed outcome counts are intentionally absent from
the forecast API. They enter the negative-binomial fitting likelihood only and
cannot alter the predicted prior state or molecular mean. This module does not
load split assignments, development outcomes, final holds, or benchmark data.

The model is a conditional latent-state pseudobulk predictor. It is not a
single-cell generator, an identified dynamical system, or evidence that its
latent coordinates represent biological mechanisms.

<a id="slp-1-1-count-latent-state-v1-contract"></a>

## slp-1-1-count-latent-state-v1: CONTRACT

Original location: `modules/slp-1-1-count-latent-state-v1/CONTRACT.md`.

### Conditional count-state prototype

This self-contained, untrained numerical prototype represents an intervention-
conditioned Gaussian molecular state and a negative-binomial count observation
model. It makes no biological forecasting, cell-generation, temporal, causal,
cross-species, synthetic-lethality, or release claim.

Static feature vectors describe action genes and query genes. The prior also
consumes measured control rates with an explicit availability mask. It has no
learned gene identifiers, context identifiers or fixed query vocabulary. Actions
form a permutation-invariant set; arbitrary queried subsets share a latent
state. A posterior encoder consumes integer molecular counts for variational
training. It must never be used to produce an unseen intervention forecast.

The observation adapter supplies positive, externally smoothed control rates
in molecules per 10,000 source-denominator molecules. Its exact smoothing and
denominator must be specified in a biological experiment before fitting. The
core invents neither missing values nor a normalization panel. Zero-library
cells are ineligible; observed count zeros are genuine observations. The
library exposure is at least the sum of queried counts and enters observation
likelihood and posterior inference only. It never enters the prior or its
expected molecular rates. Conditional count means need not sum exactly to
the supplied library; this is an offset-based NB2 approximation, not a
multinomial or a coherent joint generator of the library and its constituent
counts. The observation factors ignore dependence induced by conditioning on
the library sum. Empty-action identity is relative to the supplied smoothed
control rates, including any explicitly declared positive pseudocount.

Let the control state be N(m0,V0), the intervention state N(m,V), and W the
queried loading matrix. The conditional log rate is log(basal) + W(z-m0) -
diag(W V0 W')/2. Its analytic population mean is basal * exp(W(m-m0) +
diag(W(V-V0)W')/2). Therefore empty interventions reproduce the supplied basal
mean exactly, while shared Gaussian variation induces cross-query dependence.
Learned dispersion describes additional conditionally independent NB2 noise;
latent variation and dispersion are not identifiable biological mechanisms.
Both prior and posterior log variances are bounded to [-8,4].

For efficient training, encode unique control contexts with `encode_context`
once per optimizer step and index their embeddings into `prior_from_context`.
Do not cache a learned context across optimizer updates. The posterior uses
a fixed full measurement panel; arbitrary mask or panel changes alter its
pooling scale and require separate validation. Decoder query chunking remains
valid. Feature-identical queries necessarily share loadings and dispersion;
the adapter must report static-feature missingness and collisions.

Training returns a one-sample negative ELBO: (sum of observed-query negative
log masses + Gaussian KL) / observed-query count, with beta=1. Reconstruction
and KL are reported separately. The caller specifies sampling, macro weights,
budget, deterministic seeds, and checkpoint selection, and must stop on any
nonfinite loss or prediction. Cell reconstruction alone is insufficient evidence
of intervention prediction; evaluate the prior without perturbed inputs.

Numerical checks cover the NB mass against an independent distribution API,
Gaussian-integrated population means, empty actions, query chunk/order and
action-set invariance, masking, integer-count units, and finite gradients.
Biological execution and portable artifact packaging remain separate work.

<a id="slp-1-1-count-moments-v1-contract"></a>

## slp-1-1-count-moments-v1: CONTRACT

Original location: `modules/slp-1-1-count-moments-v1/CONTRACT.md`.

### Raw count moments v1

Input is a bounded sparse block of eligible cells by source RNA features, with
integer population indices assigned from source metadata. Counts must be finite,
nonnegative integers. This module does not assign genotypes or train/test roles.

The caller supplies an immutable source-row-to-query map and denominator mask.
Multiple source rows mapping to the same stable query are summed before
normalization. Unmapped biological rows may contribute to the denominator while
remaining absent from output queries. Every queried row must be in the
denominator. Each retained cell has value
`ln(1 + 10000 * summed_query_count / sum_denominator_counts)`.

Zero-library cells are explicitly counted and excluded. Observed zeros in
positive-library cells remain in the estimand. Accumulation is float64 and
weights every eligible cell equally within its population. Output includes
population counts, means and unbiased cell variances; empty means and variance
with fewer than two cells are NaN with explicit support masks. Cell variances
describe dispersion among sampled cells, not biological replicate uncertainty.

There are no fitted parameters, ID embeddings, outcome-dependent filters or
application scores. Query mapping and population grouping remain caller-owned
provenance. Peak accumulator memory is two float64 population-by-query arrays,
plus one bounded sparse input block and returned summaries.

Checks compare against a dense independent normalization oracle, verify duplicate
query collapse before log transformation, denominator-only features, observed
zeros, empty/one-cell support, chunk/order invariance, and reject invalid counts.

<a id="slp-1-1-count-panel-data-v1-contract"></a>

## slp-1-1-count-panel-data-v1: CONTRACT

Original location: `modules/slp-1-1-count-panel-data-v1/CONTRACT.md`.

### Native count-panel data adapter

This self-contained NumPy adapter reads the versioned human essential-cell
training registry and its checksum-pinned members. It loads only the existing
fitting/control memory maps and fitting sufficient statistics. It does not
create splits, load held count shards, train models or evaluate applications.

Each panel retains its ordered query IDs, stable gene IDs, source-qualified
context IDs and full native count denominator. Counts remain in read-only
uint16 memory maps. Sampling copies bounded rows into float32 batches and
verifies their exact integer sums against their registered libraries.
Controls sample uniformly over GEM groups, then cells; targets sample uniformly
over genes, then exact populations, then cells. Population batches sample
unique fitting genes uniformly and retain their measured context proportions.

Fitting targets are ln1p of the equal-cell mean CP10k. Controls use the fixed
positive half-count pooled reference. The mean-loss scale is the full-fitting
MSE of the control-anchored mean predictor, computed before optimization.
The source schedule and relative loss weights belong to the workload.

The registry and adapter describe the present two human native panels; they
make no species-transfer or deployment claim. Replacing aligned features is
an explicit validated operation that must be recorded by a new workload.

<a id="slp-1-1-count-world-evaluation-v1-contract"></a>

## slp-1-1-count-world-evaluation-v1: CONTRACT

Original location: `modules/slp-1-1-count-world-evaluation-v1/CONTRACT.md`.

### Molecular population forecast evaluation

`evaluator.py` provides array-only numerical functions for an explicit external
evaluation. It does not load or train a world model. The separate experiment
script validates frozen forecasts and assembles source-native count summaries.

The endpoint is `ln1p(mean_cell(10000 * counts / full_native_library))`.
Cells contribute equally within an intervention gene; intervention genes and
queries contribute equally to reported MSE. Control predictions use the same
gene-specific experimental-group proportions, mixing rates before `ln1p`.

The perturbation-specific correlation subtracts each matched control profile,
then independently centers truth and predictions across the evaluated gene
cohort for each query. Pearson correlation is computed across queries for each
gene and averaged over defined genes. Undefined constant profiles remain
undefined and their count is reported. Subtracting an initial row before
centering prevents a repeated floating-point baseline from creating spurious
correlations. MSE uses the original absolute profiles.

The named two-source advancement function belongs to this experiment's external
evaluation; it is not world-model behavior. Its forecast checks alone cannot
establish the separate reconstruction-preservation requirement or release
readiness. Array shape/identity validation does not substitute for the caller's
source and split checks. Synthetic tests verify averaging order, numerical
centering, forecast contracts and the fixed forecast decision rule.

<a id="slp-1-1-count-world-inference-v1-contract"></a>

## slp-1-1-count-world-inference-v1: CONTRACT

Original location: `modules/slp-1-1-count-world-inference-v1/CONTRACT.md`.

### Registered-panel count prior inference

The artifact-local `Predictor` loads one frozen model arm and one registered
native measurement panel. Its inputs are raw static action features, an
optional Boolean action mask, context mixture weights and optional query
indices. It applies the artifact's saved fitting-action normalizer and returns
prior expected CP10k rates, their `ln1p` transform and Gaussian latent prior
parameters. No perturbed counts, library exposure or application labels enter
prediction.

Artifacts contain relative model, reference, numerical-source and protocol
files with checksums. The training packager copies the count numerical core
beside this loader; no repository-relative numerical import is required.
The reference supplies the complete measured native query panel and positive
smoothed control rates. A query subset changes decoding only; context encoding
continues to use the complete registered reference panel.

An empty intervention has the supplied context-weighted control mean. Mixture
weights combine rates before logarithms. Finite CPU/GPU differences require
the experiment's declared numerical tolerance; exact cross-device equality
is not promised. This version may normalize caller-owned float64 context
weights in place; callers needing to retain those values should pass a copy.

Successful local artifact replay establishes executable tensor-file inference.
It does not establish new-panel transfer, combination validity, calibrated
cell generation, an SL score, or portable OMF release materialization.

<a id="slp-1-1-count-world-inference-v2-contract"></a>

## slp-1-1-count-world-inference-v2: CONTRACT

Original location: `modules/slp-1-1-count-world-inference-v2/CONTRACT.md`.

### Count world inference v2

This self-contained adapter loads one frozen shared count-world arm and one
registered native measurement panel. It accepts raw static intervention
features and nonnegative weights over the saved control contexts, and returns
the prior expected CP10k mixture plus its `ln1p` transform.

Library size and perturbed counts are excluded from the forecast API. Native
query axes and control panels remain separate. Action normalization uses only
the persisted shared fitting-action statistics. Empty actions return the
corresponding weighted basal control mean within numerical precision.

Version 2 owns a private copy of caller-supplied context weights before
normalizing them. Prediction therefore cannot mutate an input array. Model,
reference, protocol, and embedded numerical-source checksums are validated
before inference.

<a id="slp-1-1-count-world-response-query-inference-v1-contract"></a>

## slp-1-1-count-world-response-query-inference-v1: CONTRACT

Original location: `modules/slp-1-1-count-world-response-query-inference-v1/CONTRACT.md`.

### Count-world response-query inference v1

The API accepts only raw static577 action features and explicit weights over
one artifact-native GEM axis. It appends exact zero33 action coordinates and
uses the persisted 610-wide fitting normalizer. Query features are arm-specific
and embedded in the immutable native-panel reference.

Outputs are expected prior molecular means. They do not constitute sampled
cells, count generation, unmeasured-query prediction, or new-context transfer.

<a id="slp-1-1-count-world-training-v1-contract"></a>

## slp-1-1-count-world-training-v1: CONTRACT

Original location: `modules/slp-1-1-count-world-training-v1/CONTRACT.md`.

### Shared molecular training step

This self-contained module packages the frozen count-state numerical core
and molecular-mean objective with a reusable training-step function. It does
not load datasets, assign gene roles, select checkpoints, fit application
scores or inspect benchmarks. Its outputs retain separate count likelihood
and population-mean losses.

The caller supplies one source-native measurement panel per step, explicit
control contexts, a cell batch and optionally a population batch. Different
panels can alternate while sharing parameters. Their static feature coordinates
must be aligned, their library denominators preserved, and their objective
weights fixed explicitly. Changing the posterior panel changes its pooling;
each resulting source/panel requires its own reconstruction and forecast
evaluation. A larger shared roster does not imply every panel measures it.
Every supplied native control panel must be fully measured and positive;
partial control support is rejected rather than filled from an unobserved
union panel. The population target panel is also fully observed.

The full control table is encoded once within a step and its graph is reused
by cell and population losses. This is valid for the included core's
deterministic context encoder. Never reuse a learned context across optimizer
updates. Population predictions disable dropout while retaining gradients.
All per-submodule training modes are restored even when validation rejects a population
batch. The caller performs backward, clipping, optimizer steps and guards
wall time/memory. All input masks describe actual measured support.

The cell factorization remains the original library-offset latent NB
approximation. Adding molecular mean supervision creates a composite objective;
it does not create a calibrated likelihood or coherent library generator.
The package contains no fitted model or biological performance claim.

<a id="slp-1-1-fixed-query-transition-v1-contract"></a>

## slp-1-1-fixed-query-transition-v1: CONTRACT

Original location: `modules/slp-1-1-fixed-query-transition-v1/CONTRACT.md`.

### Fixed Query Transition v1

This application-neutral module predicts queried molecular deltas from static
intervention features and supplied basal measurements. The output query
coordinates are fixed numerical inputs and are not learned gene identifiers.

The module does not fit a response basis, load datasets or define splits. Empty
action sets return the supplied control mean exactly. A fitted basis must carry
its own quantitative-data provenance and cannot be described as a static prior.

<a id="slp-1-1-gene-state-v1-contract"></a>

## slp-1-1-gene-state-v1: CONTRACT

Original location: `modules/slp-1-1-gene-state-v1/CONTRACT.md`.

### Gene-state molecular core v1

This self-contained module represents a caller-defined gene universe with
explicit per-gene basal and intervention states. Stable identity resolution is
external: rows of the static matrix, basal matrix, action-strength matrix and
sparse adjacency must describe the same fixed universe. The module contains no
learned gene-ID embedding or internal vocabulary.

`GeneStateCore.encode(static_gene_features[N,F], basal_rna[B,N],
basal_observed[B,N], action_strength[B,N], adjacency[N,N])` returns static,
basal-node, global-basal, global-action, global-delta, global-state,
initial-local-delta, two-step local-delta and local-state tensors. Static
features are projected once per encode call and reused. Nonfinite basal values
are allowed only where `basal_observed` is false. Every record needs at least
one observed basal gene.

The sparse graph is nonnegative and row normalized. `adjacency[i,j]` sends from
node `j` to node `i`; isolated all-zero rows are allowed. Two shared residual
message steps use `torch.sparse.mm` on a node-by-batch-state matrix. The local
intervention route therefore reaches at most two graph edges beyond directly
acted-on nodes. The separate global route deliberately permits responses at
disconnected genes, so decoded response locality is not claimed.

`observe(encoded, query_node_indices[Q], control_mean[B,Q], amplitude[Q])`
supports only physical RNA queries mapped to graph nodes by the caller. It
returns a nonlinear changed-minus-basal molecular delta and the control-anchored
mean. Amplitudes must be finite and positive. Empty action strength produces an
explicit zero delta and exact control identity. Query order and chunking do not
change the mathematical result. Version 1 fixes dropout at zero throughout so
the changed-minus-basal subtraction contains no stochastic mismatch.

This is an endpoint representation. It does not identify time dynamics,
causal graph edges, unseen assay components, or cell-level distributions.
Graph construction, source rights, stable-ID joins, normalization, splitting,
losses, calibration and evaluation remain outside the module. Dense basal,
action and local-state tensors scale as `B*N` and `B*N*state`; sparse message
passing avoids materializing a batch-expanded edge list but does not remove
that node-state memory cost.

`profile_synthetic_cuda()` is an explicit opt-in resource check with defaults
`N=24000`, `B=32`, `F=577`, state width 16 and 1,024 decoded queries. It never
runs at import or model construction and must be invoked only after GPU
coordination.

<a id="slp-1-1-guide-composition-v1-contract"></a>

## slp-1-1-guide-composition-v1: CONTRACT

Original location: `modules/slp-1-1-guide-composition-v1/CONTRACT.md`.

### Dual-guide composition v1

This module maps exact 20-nt A/C/G/T dual-guide sequences to a fixed
71-dimensional descriptor. It contains no guide ID, genomic coordinate,
outcome, inferred efficacy, or target-gene embedding.

Each guide contributes 32 reverse-complement-canonical overlapping 3-mer
frequencies. The pair descriptor concatenates their mean and absolute
difference, GC mean and absolute difference, homopolymer minimum and maximum,
base-entropy mean and absolute difference, and the minimum normalized Hamming
distance between `(A,B)` and `(A,reverse_complement(B))`. It is invariant to
guide order and to independently reverse-complementing either guide.

`aggregate_gene_descriptors` averages exact pair descriptors using supplied
fitting-cell frequencies. Every cell pair and action must join exactly;
unsupported pairs or genes raise an error rather than receiving zero features.

<a id="slp-1-1-held-roster-contract"></a>

## slp-1-1-held-roster: CONTRACT

Original location: `modules/slp-1-1-held-roster/CONTRACT.md`.

### Held-intervention roster contract

This module constructs an outcome-blind yeast intervention split from two or
more separately admitted identity-inventory artifacts. It never reads a
quantitative molecular value. Inputs retain `NCBITaxon:4932` and canonical
`SGD:S#########` identities; symbols and orthology substitutions are invalid.

Every inventory is a separate top-level stage input. At runtime the module
accepts only OMF's materialized copied `DatasetSnapshot` object with exactly
`resource`, `mode`, `path`, and `manifestDigest`. The resource URI must name the
`datasetsnapshot` kind and end in a literal SHA-256 revision; the artifact
manifest is independently SHA-256 pinned. The materialized directory must be
`.../inputs/<input-name>/<dataset-name>` consistently with that URI. Bare
paths, mutable revisions, alternate kinds, mounted data, extra fields, missing
paths, and inconsistent materialization paths fail.

Each input directory contains an `inventory.json` object with exactly:

```json
{
  "schema": "slp.intervention-identity-inventory/v1",
  "sourceId": "repository:immutable-accession",
  "sourceRelease": "immutable-version",
  "ncbiTaxon": 4932,
  "stableIdNamespace": "SGD",
  "identityMappingId": "sgd:immutable-mapping-release",
  "identityMappingSha256": "<lowercase SHA-256>",
  "inventoryFormat": "slp.intervention-identity-record/v1",
  "files": [
    {"path": "inventory-000.jsonl", "sha256": "<lowercase SHA-256>", "records": 1}
  ]
}
```

File entries are unique and path-sorted. Paths are canonical relative POSIX
`.jsonl` paths resolving to regular non-symlink files inside the artifact.
Every parent path component is also checked for symlinks. Digests and
record counts are recomputed. Every JSONL line has exactly:

```json
{"schema":"slp.intervention-identity-record/v1","interventionId":"SGD:S000000001","ncbiTaxon":4932,"qcPassing":true}
```

No outcome, score, label, abundance, expression, fitness, effect, p-value or
other quantitative readout field is permitted. Repeated identical records are
reported and collapsed; repeated identities with conflicting QC status fail.
The required identity-mapping ID and digest attest the pinned SGD mapping used
by the source adapter. Raw ORF spellings, including valid suffixed systematic
names, are mapped there; this roster module accepts only resulting SGD CURIEs
and does not implement a narrower ORF-name regex. Every protected inventory
must declare the exact same mapping ID and digest; otherwise canonicalization
drift could alter the intersection and roster construction fails.

The candidate set is the intersection of QC-passing identities across every
protected source. For each sorted CURIE, compute lowercase SHA-256 of the exact
bytes `slp-1.1-yeast-global-held-v1\x00<SGD-CURIE>`. Interpret the first 16 hex
digits as an unsigned integer and reduce modulo 100. Buckets 0–9 are
`molecular-final`, 10–29 are `molecular-validation`, and 30–99 are `pretrain`.
An empty or configured-undersized intersection fails; there is no reroll.
Production invocations also supply the independently reconstructed exact
intersection size, three role counts, and roster digest. These five frozen
expectations are all-or-none and must agree internally. Any mismatch is fatal;
the producer does not accept a larger population, rebalance roles, or reroll the
hash domain.

The output `held-intervention-roster.tsv` has no header and contains exactly
`SGD-CURIE<TAB>role<TAB>hash`, path-sorted by CURIE with LF line endings.
`coverage.json` attests the algorithm, roster SHA-256, source manifests,
coverage, QC failures, and source-specific exclusions. The module bounds source
count, files, records, manifest bytes, and JSONL line bytes before processing.

<a id="slp-1-1-molecular-baselines-readme"></a>

## slp-1-1-molecular-baselines: README

Original location: `modules/slp-1-1-molecular-baselines/README.md`.

### SLp-1.1 molecular baselines v1

This self-contained OMF module implements the `context-only` and
`txpert-mean-additive` point baselines frozen in
`evaluations/slp-1-1-molecular-comparison-protocol-v1.yaml`. It never reads or
emits benchmark labels.

Both inputs must be copied, revision-pinned OMF `DatasetSnapshot` objects whose
payload root contains `baseline.json`. `input.schema.json` freezes the manifest
and JSONL record contract. A profile identity is the tuple `(speciesTaxon,
sourceId, contextId, recordRole, perturbationId)`. Basal state is recognized
only by the explicit `recordRole: basal-control`; neither an empty action nor a
field such as `centeringGroup` is interpreted as basal. Readouts are sparse and
aligned. `null` means unobserved and is never converted to zero.

Every record is one already frozen context-by-perturbation molecular centroid,
declared by `profileLevel: context-perturbation-centroid-v1`. The module does
not aggregate replicates. Each snapshot instead binds the upstream aggregation
rules with `aggregationProtocolSha256`, and paired snapshots must use the same
digest. Duplicate natural keys are therefore ambiguous duplicate centroids and
are rejected.

For each observed readout the context-only prediction is its explicit matched
reference-context basal value. Training effects are fitting outcome minus the
explicit matched fitting-context basal. TxPert uses the mean exact intervention
set effect when it exists for the same species and source. Otherwise it sums
single-intervention means, replacing each unavailable constituent with that
species/source/readout's global fitting perturbation mean. Means are computed
only from the training snapshot and every exact, singleton, and global effect
table is local to one exact `(speciesTaxon, sourceId)` stratum. For pure
context-cold evaluation every reference intervention must have a quantitative
fitting outcome in another context in that same species/source stratum; global
fallback cannot silently convert that task into double-cold. The reference
manifest pins the exact training manifest checksum, and the selected frozen
task controls fail-closed gene and context exclusion checks.

The output contract is deliberately point-only. `slp.molecular-evaluation/v1`
requires `predictionLogScale`, but the protocol does not define a baseline
uncertainty estimator. The report therefore returns the machine-readable block
`prediction-log-scale-not-defined`; it does not invent a residual scale.
Feature-bilinear ridge is also reported as
`protocol-required-contract-blocked/feature-vectors-absent`, because these v1
artifacts contain identities and molecular values rather than action/query
feature vectors.

<a id="slp-1-1-molecular-eval-readme"></a>

## slp-1-1-molecular-eval: README

Original location: `modules/slp-1-1-molecular-eval/README.md`.

### SLp-1.1 molecular evaluator v2

This application-neutral OMF module is the only component allowed to combine
frozen predictions with held molecular truth. It never loads
synthetic-lethality labels and its output is diagnostic evidence, not the
`MODEL_CARD.md` advancement gate.

The evaluator consumes five distinct, copied, revision-pinned OMF
`DatasetSnapshot` inputs: fitting-only centering, evaluator-only held truth,
the target-free query, a passing reward-disabled `slp.corpus-audit/v1.2`, and the outcome-blind
held roster. Predictions and the exact model checkpoint are separate immutable
artifacts. File-valued artifacts use OMF's `.../payload/payload` materialization
semantics; the checkpoint bytes are hashed and must equal the prediction
manifest's `modelCheckpointContentSha256`. Predictions are one deterministic,
uncompressed tar artifact containing exactly `evaluation.json` followed by
`profiles-000.jsonl`; member names, order, type, size, digest, ownership,
permissions, and zero timestamps are checked before records are streamed.

The query and prediction contract uses exact canonical rows:

```text
profileId, speciesTaxon, sourceId, centeringGroup, perturbationId,
interventionIds, readoutIds, distributionTypes
```

Prediction rows add only `predictionParameters`, aligned one-for-one with the
ordered panel. Gaussian parameters are exactly `{mean, logScale}`;
negative-binomial parameters are exactly `{logMean, logInverseDispersion}`.
Any target, observed mask, undeclared field, missing/extra profile, changed
intervention, changed distribution, or changed/reordered readout panel is
fatal. `perturbationId` is derived from sorted intervention CURIEs and
`profileId` from the natural species/source/group/perturbation key, preventing
aliases or duplicate records from weighting metrics.

The producer prediction and protected truth each bind the exact query
DatasetSnapshot resource URI, its outer OMF manifest digest, and the raw
`query.json` digest. The workload renderer also
records the full query URI and outer manifest digest; the module checks those
against OMF's materialized input. Truth does not carry producer-facing corpus
fingerprints: it is independently admitted and joins the target-free query
only inside the evaluator.

This v2 roster contract is explicitly limited to SGD interventions in NCBI
taxon 4932. The evaluator recomputes every assignment digest, bucket, role,
role count, held-set hash, source coverage identity, and audit binding from the
frozen roster files. Mixed-species evaluation requires a new roster schema.
The independently admitted audit establishes quantitative intervention
isolation; absence of held IDs from centering is checked but never claimed as
sufficient evidence.

Every query readout requires outcome-blind fitting support from the minimum
number of distinct centering perturbations. Non-null truth is scored with the
declared Gaussian or negative-binomial NLL; null truth is excluded from scores,
while predictions must still cover the full preregistered panel. Metrics remain
stratified by species, source, and species-source, with perturbation-specific
centering. Undefined source/species correlations fail the diagnostic.

#### Frozen second-run workflow

Render only after the datasets, prediction, and checkpoint artifact are frozen:

```text
python modules/slp-1-1-molecular-eval/render_workload.py \
  --centering-dataset dataset/<centering> \
  --prediction-artifact sha256:<prediction-artifact-manifest> \
  --truth-dataset dataset/<truth> \
  --corpus-audit-dataset dataset/<audit> \
  --held-roster-dataset dataset/<roster> \
  --query-dataset dataset/<query> \
  --query-resource omf://.../datasetsnapshot/<query>@sha256:<revision> \
  --query-manifest-digest sha256:<outer-manifest> \
  --model-checkpoint sha256:<checkpoint-artifact-manifest> \
  --output workloads/generated/slp-1-1-molecular-eval-v2-<frozen-id>.yaml
```

The diagnostic does not establish improvement against frozen baseline NLL,
checkpoint-selection eligibility, benchmark performance, portable inference,
or release compatibility.

<a id="slp-1-1-molecular-mean-objective-v1-contract"></a>

## slp-1-1-molecular-mean-objective-v1: CONTRACT

Original location: `modules/slp-1-1-molecular-mean-objective-v1/CONTRACT.md`.

### Molecular population-mean auxiliary objective

This application-neutral helper mixes positive expected molecular rates over
measured contexts using supplied metadata weights, then applies log1p. It
compares this population mean with a measured fitting population's log1p mean
using equal-population, equal-query squared error. The normalization scalar
is fixed from fitting data before optimization. No benchmark, split loading,
gene vocabulary, thresholds or model-selection behavior is implemented here.

The caller fixes the source count denominator, population definitions,
sampling weights, assay support, scalar normalization and relative likelihood
weight. The rate mixture precedes log1p; averaging log-transformed context
rates is a different endpoint. Every query passed to this helper is observed.
It does not infer missing targets. Gradients pass through expected rates;
context weights describe the known population composition.

This is a composite molecular training objective when combined with a cell
likelihood. It is not itself a calibrated likelihood or a new generative
model. Improvements in aggregate prediction do not establish preservation of
cell-level likelihood or uncertainty; those require separate evaluation.

<a id="slp-1-1-paired-state-v1-contract"></a>

## slp-1-1-paired-state-v1: CONTRACT

Original location: `modules/slp-1-1-paired-state-v1/CONTRACT.md`.

### Paired endpoint state, revision 1

The self-contained numerical source is `paired_model.py`. It uses PyTorch;
it has no repository imports, data loaders or application-specific logic.

Fixed biological features describe action genes and RNA queries. Antibody
queries use fixed assay-component descriptors supplied by the caller. An
antibody panel basis is not a learned intervention-gene vocabulary, and does
not establish extrapolation to unmeasured antibody components.

Controls from RNA and protein have separate feature/value encoders. Each
observed modality contributes equally to the basal state. A shared latent
transition accepts a permutation-invariant sum of intervention tokens.
Separate nonlinear observation functions decode state changes relative to
the same basal state, preserving exact empty-action identity. Query outputs
are independent of other requested queries. Missing control entries are inert.

The model predicts molecular endpoint means. It has no calibrated probability
distribution, no time evolution, and no identified before/after cell coupling.
Supporting multiple action tokens does not establish combination accuracy.
Normalization, fitted feature statistics, shared query amplitudes and assay
identifiers are explicit caller inputs and must accompany a fitted artifact.

<a id="slp-1-1-proteome-corpus-compose-v1-contract"></a>

## slp-1-1-proteome-corpus-compose-v1: CONTRACT

Original location: `modules/slp-1-1-proteome-corpus-compose-v1/CONTRACT.md`.

### Proteome composite-corpus v1.2 contract

The module accepts exactly three immutable, copy-materialized OMF
`DatasetSnapshot` inputs: the fitting-only yeast proteome observations, the
sequence-statistics static feature block, and the outcome-blind held roster.
Their resource revisions, outer manifests, reconstructed OMF tree identities,
file sets, sizes, and SHA-256 digests are compiled into the module. Protected
outcomes, reward data, benchmark labels, mutable mounts, bare paths, and extra
inputs are not part of the interface.

Entity identity is always the ordered pair `(ncbiTaxon, entityId)`. The output
dictionary contains all 7,037 sequence-feature rows without changing their
float32 or presence bytes, followed in canonical composite-key order by the
species-specific experimental context. That context has an all-false feature
mask and canonical zero storage. Duplicate keys, unsorted keys, taxon swaps,
and joins through an identifier without its taxon fail.

The query dictionary contains only `query_entity_index` and
`query_readout_index`; there is no `query_id` or opaque ID lookup. Query rows
are strictly ordered by composite entity key and readout type. The source
readout CSR indices are mapped through the exact composite readout dictionary.
`trajectory-interventions.jsonl` contains canonical records with exactly
`schema`, `ncbiTaxon`, and `entityId`, never bare text identifiers. Every
active action exactly matches that set, and any molecular-validation or
molecular-final roster action ends composition.

The source float32 target payload is copied without numerical conversion. The
audit hashes the concatenated source and composed target bytes and requires
equality. Technical injection, well, plate, metadata-row, and matrix-column
values are retained only on the observation covariate axis with `audit`
access. They are not world-model features.

`featurePack` is `slp.static-feature-pack/v1`. Its blocks are contiguous and
ordered by offset, and each block binds its source DatasetSnapshot resource,
revision, outer manifest, tree digest, semantic digest, composite feature-key
set, and exact files. The pack SHA-256 is recomputed over canonical JSON with
only the `sha256` field omitted.

The output is canonical uncompressed USTAR rooted at `composite-corpus/`.
Every NPZ is uncompressed, member-sorted, timestamp-fixed, permission-fixed,
and reconstructed byte-for-byte by the validator. Archive traversal, links,
PAX metadata, undeclared members, object arrays, wrong dtypes or shapes,
duplicate records, partial feature masks, or non-finite targets fail closed.

<a id="slp-1-1-proteome-corpus-compose-v1-readme"></a>

## slp-1-1-proteome-corpus-compose-v1: README

Original location: `modules/slp-1-1-proteome-corpus-compose-v1/README.md`.

### SLp-1.1 proteome corpus composer v1

This self-contained OMF module turns the exact admitted fitting-only yeast
proteome observations and sequence-statistics features into the first
composite-keyed `slp.corpus/v1.2` bundle. It also consumes the outcome-blind
held roster solely to prove that no protected intervention entered the
optimizer corpus.

The bundle is a data boundary, not a model or evaluation result. It preserves
6,865,493 quantitative float32 target values byte-for-byte, exposes static
sequence features for 7,037 entities, and represents the experimental context
as one explicit species-native entity with missing static features. It contains
no validation/final outcomes, molecular reward, or synthetic-lethality labels.

See `CONTRACT.md` for the normative identity, lineage, deterministic-format,
and leakage rules. The historical corpus v1.1 and bare-ID consumers remain
frozen and are not silently upgraded by this module.

<a id="slp-1-1-proteome-inventory-readme"></a>

## slp-1-1-proteome-inventory: README

Original location: `modules/slp-1-1-proteome-inventory/README.md`.

### SLp-1.1 proteome identity inventory

This module reads only identity and sample metadata from Mendeley Data
`10.17632/w8jtmnszd9.2`. It verifies all four pinned raw-file hashes, but the
quantitative matrix contributes only its CSV header and first `Protein.Group`
field. Quantitative cells are never decoded, parsed, transformed, or emitted.

Deletion identifiers resolve only through an exact, case-sensitive systematic
name in the pinned SGD current-ORF artifact. Exact retired or merged names are
identified only through the separate pinned quarantine artifact and are never
redirected. Case itself is never a reason to reject an exact current mapping:
the pinned current map resolves `YAL043C-a` exactly. A mixed-case spelling such
as `YML009c` that has no exact current mapping remains unmatched; it is never
uppercased. Unmatched, retired, merged, and ambiguous rows remain in the audit
quarantine with their exact source spelling.

The intervention artifact implements `slp.intervention-identity-inventory/v1`
for the global held-roster module. The protein artifact assigns a UniProtKB
CURIE only after an exact `UniProtKB` / `UniProtKB ID` relation and retains all
current SGD ORF relations. Shared accessions remain one-to-many relations; the
module never selects a first gene.

Eligible metadata-row multiplicity is retained in the held-roster inventory:
the pinned source is expected to produce 4,623 records for 4,476 unique SGD
CURIEs (147 duplicate records). Here `qcPassing` means only that a row passes
the frozen source identity-admissibility rule; it does not assert proteomic
measurement quality.

OMF 1.0 cannot feed this directory artifact directly into the held-roster's
copied `DatasetSnapshot` input. A later step must separately admit and verify
the exact inventory bytes as a rights-bearing DatasetSnapshot while recording
this adapter RunResult and artifact digest as provenance.

This is an outcome-blind identity prerequisite, not a biological corpus,
training result, model, or held-roster assignment.

<a id="slp-1-1-proteome-observation-prepare-v1-contract"></a>

## slp-1-1-proteome-observation-prepare-v1: CONTRACT

Original location: `modules/slp-1-1-proteome-observation-prepare-v1/CONTRACT.md`.

### Contract

#### Inputs

The run accepts exactly four immutable copied DatasetSnapshots and two exact
file artifacts:

- raw proteome release `slp-1-1-proteome-raw-v2`;
- outcome-blind proteome intervention inventory;
- typed proteome protein-relation inventory;
- frozen cross-source held-intervention roster;
- SGD current-ORF JSONL artifact;
- SGD mapping-manifest artifact.

Names, revisions, outer manifest digests, artifact manifests, internal content
hashes, source release, taxonomy, mapping identity, raw file set, and internal
file hashes are fixed. Full `omf://abiome/slp/...` resource URIs, OMF request
objects, and materialized paths are checked literally. Symlinks, foreign
namespaces, mutable references, extra inputs, extra files, and legacy
file-artifact paths fail. The exact typed UniProtKB accession object and current
SGD relation targets are revalidated, including the frozen nuclear and
mitochondrial systematic-name forms in the current-ORF object set.

#### Partition and access

The module prepares only `pretrain`. It admits every exact current proteome KO
row except rows whose stable SGD action is a frozen molecular-validation or
molecular-final identity. This produces 3,811 records over 3,679 intervention
genes. All 537 validation rows, 275 final rows, 76 quarantined rows, and 389
analytical-QC rows are tokenized by the RFC CSV reader but never converted to
numbers, inspected as outcomes, validated as outcomes, or emitted.

The admitted intervention inventory intentionally lacks sample locators, so it
is never order-zipped to matrix columns. The module independently reconstructs
the raw sample mapping through the pinned current-ORF artifact and requires its
complete intervention multiset to reproduce the admitted inventory exactly.

#### Numerical protocol

Observed targets must be finite and strictly positive before the fixed
`log2` transform. `NA` means unobserved and is omitted from CSR. The output has
6,865,493 observed and 184,857 missing pretrain values. It retains absolute
log2 values in the frozen source space; additional centering is `none`.

Only 388 documented HIS3 controls are decoded for the separate basal profile.
The mean is calculated after log2 independently per readout, and at least 311
controls are required. Exactly 1,843 of 1,850 readouts pass. No KO or QC outcome
can affect this artifact.

NumPy 2.2.6 is hash-pinned in the module lock. Runtime Python implementation,
Python version, and NumPy version are embedded in both archives and the audit.
The basal archive also binds the full input provenance and a canonical digest
of the exact 388 control row/column locators.

#### Outputs and non-claims

`observation-corpus.tar` implements `slp.source-observation-archive/v1` with
stable identities, one assayed panel, source records, observed-value CSR, and
audit-only technical covariates. `basal-control.tar` implements
`slp.basal-control-profile/v1`. Neither is an admitted DatasetSnapshot until a
separate rights review and OMF admission. Neither is a model, feature pack,
benchmark, molecular metric, or SOTA result. Validators rederive archive
identity populations, shard uniqueness, action/trajectory equality, CSR
integrity, basal support masks, and count arithmetic. The final three-file
directory appears only through an atomic same-filesystem rename.

<a id="slp-1-1-proteome-observation-prepare-v1-readme"></a>

## slp-1-1-proteome-observation-prepare-v1: README

Original location: `modules/slp-1-1-proteome-observation-prepare-v1/README.md`.

### SLp-1.1 proteome observation preparation v1

This self-contained OMF module converts the exact non-imputed Mendeley
`w8jtmnszd9` version 2 yeast knockout proteome into a source-normalized,
pretraining-only sparse observation archive. It is not a world-model corpus:
it contains no learned gene identity, static feature vector, model query tensor,
sampling weight, architecture choice, or benchmark label.

The module independently rebuilds raw metadata ORFs against the pinned current
SGD mapping and requires the resulting intervention multiset to equal the
separately admitted identity inventory. It then excludes every frozen
molecular-validation and molecular-final row before numerical conversion.
Quarantined knockout and analytical-QC columns are also never converted to
numbers. Source-only exact current genes remain fitting-only and do not become
members of the protected two-source roster.

Every DatasetSnapshot is matched by its complete `omf://abiome/slp/...` URI,
revision, and outer manifest digest. Protein relations must retain their exact
typed UniProtKB declaration and resolve only to current CURIEs. The runtime is
closed over hash-pinned NumPy 2.2.6, and Python/NumPy versions are recorded in
both archives and the audit.

Targets are `float32(log2(x))` for finite, strictly positive observed MaxLFQ
relative intensities. Literal `NA` is omitted from CSR observations; there is
no pseudocount, imputation, zero substitution, or knockout-derived centering.
All 1,850 UniProtKB readouts and their exact typed SGD relations are preserved.
Raw filenames and raw ORF strings are not emitted into record shards; neutral
row-derived provenance IDs and stable SGD action CURIEs are used instead.

The separate basal artifact averages log2 values only over the 388 documented
HIS3-complemented biological WT controls. A readout is present only with at
least 311 observed controls. The profile is not subtracted from targets. Plate,
injection, and well indices remain audit-only.

The three outputs are regular files because OMF 1.0 cannot safely import a
producer directory artifact. `observation-corpus.tar` and `basal-control.tar`
use deterministic regular-file members; `preparation-audit.json` records exact
lineage, access boundaries, runtime, counts, and limitations. Both archives are
self-validated for canonical identities, global uniqueness, action/trajectory
equality, provenance, and numerical structure before the complete output
directory is atomically published. A later versioned composer must join an
admitted feature pack and produce `slp.corpus/v1.1`.

<a id="slp-1-1-proteome-protected-observation-prepare-v1-readme"></a>

## slp-1-1-proteome-protected-observation-prepare-v1: README

Original location: `modules/slp-1-1-proteome-protected-observation-prepare-v1/README.md`.

### SLp-1.1 protected proteome observation preparation v1

This self-contained OMF module normalizes exactly one protected yeast
proteome role per run: molecular validation or molecular final. It accepts no
pretraining corpus, reward, model, prediction, or benchmark input and emits
only the selected role's deterministic source-observation archive plus a
preparation audit.

The raw source snapshot contains every source column, so OMF 1.0 cannot claim
column-level storage isolation. The reviewed module behavior is the boundary:
it reconstructs the frozen role partition from stable SGD identities and
numerically converts only the selected role. Training and reward workloads
must never receive the raw source or either protected output.

The output remains an architecture-neutral `slp.source-observation-archive/v1`,
not an `slp.corpus/v1.1`, prediction query, evaluator truth bundle, model, or
performance result. Validation and final runs and DatasetSnapshots remain
separate.

<a id="slp-1-1-reduced-rank-response-inference-v1-contract"></a>

## slp-1-1-reduced-rank-response-inference-v1: CONTRACT

Original location: `modules/slp-1-1-reduced-rank-response-inference-v1/CONTRACT.md`.

### Reduced-rank response local research inference v1

The caller selects one saved native source and supplies weights over that
source's exact GEM control contexts. The adapter mixes positive control rates
in CP10k units, applies `log1p`, and adds the fitted signed response residual
without clipping or renormalization.

The primary numerical API accepts raw static features. A convenience method
looks up stable ENSG IDs in a frozen static-action cache; gene identity is not
a model parameter. Native query IDs and order are returned with every result.

This bundle predicts panel-specific molecular profiles. It is not a count
generator, a new-context model, an OMF release, or evidence for unmeasured
query transfer.

<a id="slp-1-1-reduced-rank-response-v1-contract"></a>

## slp-1-1-reduced-rank-response-v1: CONTRACT

Original location: `modules/slp-1-1-reduced-rank-response-v1/CONTRACT.md`.

### Reduced-rank response model v1

This module fits a rank-constrained, regularized linear map from raw static
intervention features to a measured molecular response panel. The intercept is
unpenalized. Feature normalization and the response basis are fitted only from
the supplied fitting records.

The returned state has no learned intervention-ID vocabulary. Its query
loadings are quantitative, panel-specific fitted descriptors. They do not
define unmeasured-query inference, static biological priors, a cell generator,
or identified molecular dynamics.

<a id="slp-1-1-response-omf2-contract"></a>

## slp-1-1-response-omf2: CONTRACT

Original location: `modules/slp-1-1-response-omf2/CONTRACT.md`.

### SLp response model OMF 2.0 script contract

`train.py` fits independent K562 and RPE1 reduced-rank response maps from raw
static577 intervention descriptors to the supplied native measured panels. A
rank of zero selects the full numerically supported feature rank. The output is
a portable directory containing both models, inference source, numerical core,
and a manifest.

`evaluate.py` predicts the supplied development genes by adding each model's
residual response to the supplied GEM-weighted basal anchor. It reports the same
gene-profile MSE and independently query-centered residual correlation used by
the retained development comparison. Inputs must be prepared without protected
test outcomes.

This vertical slice manufactures the retained panel-specific feature-linear
baseline under OMF 2.0. It is not the proposed observation-encoded world model,
does not infer unmeasured queries, and does not identify molecular dynamics.

The dependency lock targets Linux x86_64 on CPython 3.11 or 3.12. Its NumPy
2.2.6 hashes are restricted to the PyPI-published
`cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64` and
`cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64` wheels, respectively.
Other operating systems, architectures, Python implementations, and Python
versions are outside this module contract.

<a id="slp-1-1-sequence-statistics-feature-block-v1-contract"></a>

## slp-1-1-sequence-statistics-feature-block-v1: CONTRACT

Original location: `modules/slp-1-1-sequence-statistics-feature-block-v1/CONTRACT.md`.

### Normative sequence-statistics feature-block v1 contract

#### Boundary and inputs

The runtime input map is exactly `staticEntityUniverse`,
`sgdProteinSequences`, `sgdCurrentOrfs`, and `sgdMappingManifest`. The first
two values are copy-materialized OMF DatasetSnapshots; the latter two are
literal immutable artifact payloads. The module independently checks the
compiled resource revisions, outer manifests, tree identities, artifact
wrappers, payload sizes and SHA-256 digests. It then validates the canonical
inner manifests, JSONL records, relation closure, and SGD mapping identity.

The module has no held-roster, quantitative observation, reward, label,
benchmark, split, checkpoint, or learned-feature input. SGD header free text
and display names never enter the model-facing values. Species identity is
`NCBITaxon:4932`; `NCBITaxon:559292` is retained separately as S288C source-
strain provenance.

#### Sequence admission and relation semantics

The exact R64.5.1 gzip contains 6,722 records and covers all 6,613 current
ORFs. A current ORF must match its exact systematic name and have a peptide of
the form `M[ACDEFGHIKLMNPQRSTVWY]*\*`. Exactly one terminal stop is removed.
The 109 non-current records are excluded before feature construction. Their
source anomalies are audited rather than coerced: eight have internal stops,
two lack a terminal stop, and one does not start with methionine.

All 5,187 universe genes resolve directly to a current SGD peptide. Each of
the 1,850 UniProtKB protein rows resolves through the typed relation graph.
For each of the five one-to-many relations, every related peptide must be
byte-identical after terminal-stop removal. The common peptide is used and
all target identities remain in provenance. Averaging, selecting the first
target, and dropping ambiguity are forbidden.

#### Frozen numerical transform

Rows follow exact `(ncbiTaxon, entityId)` order from the 7,037-row universe.
Each row has 21 values:

1. stripped peptide length divided by 4096;
2. residue fractions in literal order `ACDEFGHIKLMNPQRSTVWY`, each using the
   stripped peptide length as denominator.

The result is IEEE-754 little-endian float32 in C row-major order. Length is
not clipped: the 4,910-residue REA1 peptide therefore produces a first
component greater than one. There is no fitting, centering, scaling, log
transform, PCA, learned parameter, ID embedding, or identifier-derived
numeric feature. `present.npy` has NumPy dtype `|b1`, the same `(7037, 21)`
shape, and is entirely true. Sequence provenance records include peptide
length and fixed-order residue counts, allowing the validator to recompute
every float byte independently.

#### Output

`sequence-feature-block.tar` is canonical uncompressed USTAR with exactly:

- `static-feature-block/entities.jsonl`
- `static-feature-block/excluded-non-current.jsonl`
- `static-feature-block/manifest.json`
- `static-feature-block/present.npy`
- `static-feature-block/sequence-provenance.jsonl`
- `static-feature-block/values.npy`

Members are path-sorted regular files with mode 0644; owner IDs and timestamps
are zero; names are empty; and links, traversal, PAX headers, native-endian or
object arrays, Fortran order, and trailing bytes are rejected. JSONL is
compact sorted-key JSON with one LF per row. The separate audit binds the
archive, manifest, every member, all four inputs, counts, composite entity-key
digest, frozen transform, five exact-consensus peptide hashes, access
boundary, and limitations. The builder validates both artifacts before
publishing the destination directory and never overwrites an existing one.

This is an outcome-blind deterministic baseline block, not a training corpus,
world model, learned protein representation, or performance claim.

<a id="slp-1-1-sequence-statistics-feature-block-v1-readme"></a>

## slp-1-1-sequence-statistics-feature-block-v1: README

Original location: `modules/slp-1-1-sequence-statistics-feature-block-v1/README.md`.

### SLp-1.1 sequence-statistics feature block v1

This self-contained OMF module builds the first species-aware static feature
block for the SLp-1.1 factory. It uses only the Python standard library and
hand-writes the frozen NumPy v1.0 representation.

The module is deliberately small in scientific scope: protein length and
amino-acid composition provide a deterministic weak baseline against which
later frozen protein-language-model, domain, and phylogeny blocks can be
tested. See `CONTRACT.md` for the normative data boundary and binary format.

The workload template uses OMF-supported `dataset/<name>` references for the
two snapshots and literal content-addressed references for the two mapping
artifacts. Running it requires a supported Linux OMF executor and a clean Git
state. The module must be admitted and tested before any biological run.

<a id="slp-1-1-sgd-map-readme"></a>

## slp-1-1-sgd-map: README

Original location: `modules/slp-1-1-sgd-map/README.md`.

### SLp-1.1 SGD stable-ID map

This module normalizes only the exact six-file SGD object-version set named by
`slp-sgd-map:2026-08-28-object-set-v1`. The production entry point accepts one
copied, revision-pinned OMF `DatasetSnapshot`; it then checks the exact file set,
byte counts, SHA-256 values, bounded line/record counts, and pinned README
markers before writing anything.

The outputs are relations, not a symbol resolver. Current ORF records preserve
the exact systematic name, including case and suffixes. Standard names and
free-text aliases are display metadata with `resolvesIdentity: false`.
External accessions are keyed by the exact `(value, source, type)` triple and
retain every asserted SGD target in sorted order. One-to-many relations are not
collapsed. Retired and merged rows—including five malformed physical rows in
the pinned upstream payload—are quarantined, and reported replacements are
evidence only.

`identityMappingSha256` is SHA-256 over canonical JSON `digestBasis`, which pins
the mapping-release ID, taxon, every verified raw payload digest, every emitted
payload digest/count, and the non-resolving policies. It is suitable for the
held-roster inventory field of the same name. The full mapping manifest records
the digest algorithm and basis so consumers can recompute it.

No benchmark labels, quantitative outcomes, lexical namespace inference,
case-folding, symbol lookup, automatic retired-ID redirect, or first-match
selection are present.

<a id="slp-1-1-sl-predictor-v1-contract"></a>

## slp-1-1-sl-predictor-v1: CONTRACT

Original location: `modules/slp-1-1-sl-predictor-v1/CONTRACT.md`.

### SLp-1.1 synthetic-lethality predictor v1

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

<a id="slp-1-1-slim-baseline-v1-contract"></a>

## slp-1-1-slim-baseline-v1: CONTRACT

Original location: `modules/slp-1-1-slim-baseline-v1/CONTRACT.md`.

### SLIM native-panel comparator

This module copies the bilinear closed-form algebra of SLIM commit
`5a7e9ade5d0a6b6331e6dbc81181450605047bcc`. It uses SLIM's default PCA
basis, training-perturbation mean bias, unstandardized intervention embeddings,
`K=10`, and ridge value `0.1`.

The adaptation supplies SLp's static577 intervention descriptors in place of
published STRING embeddings and fits control-anchored native-panel molecular
residuals in place of GEARS-normalized expression. It is therefore a matched
feature/native-panel comparator, not a reproduction of SLIM's published
canonical benchmark scores. It has no gene-ID parameters. Quantitative outcomes
from held development interventions are excluded from fitting.

The optional `K=32` arm is a declared development diagnostic. It is not used to
select or relabel the fixed `K=10` primary comparator.

`scripts/run_slp11_slim_cv.py` is the stronger matched-feature arm. It
standardizes every static577 column using fitting genes only and selects rank
and ridge strength by deterministic three-fold fitting-gene CV. Each fold fits
its own normalizer and PCA basis. The selected model is frozen before the
existing development arrays are loaded.

`scripts/prepare_slp11_string_features.py` prepares an optional independent
STRING64 feature source from the exact embedding file tracked by the pinned
SLIM repository. It maps stable Ensembl IDs through the source Replogle GTF,
retains explicit missingness, and contains no molecular outcome values.

<a id="slp-1-1-static-entity-universe-v1-contract"></a>

## slp-1-1-static-entity-universe-v1: CONTRACT

Original location: `modules/slp-1-1-static-entity-universe-v1/CONTRACT.md`.

### Normative static entity-universe v1 contract

#### Boundary

The input map is exactly `interventionInventory` and `proteinRelations`. Both
values must be immutable, revisioned, copy-materialized OMF DatasetSnapshots
whose resource revisions, outer manifest digests, inner manifest bytes, record
bytes, SGD mapping ID, and SGD mapping digest match the constants compiled into
this module. A held roster is deliberately absent: this operation cannot know
or emit pretraining, validation, final, reward, or benchmark assignments.

OMF 1.0 workload syntax accepts DatasetSnapshots only as `dataset/<name>`
references. OMF pins the current immutable revision during admission; the
module then independently requires the compiled resource URI and outer digest.
If the named resource advances or is revoked, execution fails rather than
silently accepting the new revision.

Input directories contain exactly the two files named by their source
contracts. Symlinks, traversal, non-regular files, extra files, malformed
CURIEs, conflicting duplicates, untyped accessions, cardinality drift, and
outcome-, label-, role-, split-, embedding-, or feature-like fields fail
closed. Repeated intervention records are allowed only when their
species-native identity and QC-admission state agree; they collapse to one
action entity. Protein-relation records and relation targets must be unique.

#### Identity semantics

The identity key is the ordered pair `(ncbiTaxon, entityId)`. Display symbols
never resolve identity, and the same CURIE in two taxa remains two entities.
No identifier is hashed, embedded, one-hot encoded, or otherwise converted to
a numeric model feature.

The production contract emits a relation-closed source universe of 7,037
entities: 5,187 SGD genes and 1,850 UniProtKB proteins. Exactly 4,476 genes are
action eligible and all proteins are readout-query eligible, preserving the
6,326 keys required by the current model interface. The 1,855 distinct typed
relation endpoints include 1,144 action genes and 711 `relation-support`-only
genes. Support-only genes close the source graph without becoming actions.
All 1,855 edges and all five two-target records are retained; targets are never
merged or selected.

Version 1 is deliberately bound to one source taxon, NCBI Taxonomy 4932. The
composite identity and digest are forward-compatible, but this version is not
evidence that the existing corpus or world-model consumers support multiple
taxa. Those consumers must migrate to composite joins before a multi-species
feature pack or corpus can be admitted.

#### Output

`entity-universe.tar` is an uncompressed USTAR archive containing exactly
`entities.jsonl`, `manifest.json`, and `relations.jsonl` beneath the
`static-entity-universe/` prefix. Members are path-sorted regular files with
mode 0644, zero timestamps and owner IDs, empty owner names, and no PAX
headers. Entity records are sorted by `(ncbiTaxon, entityId)`; relation records
are sorted by `(ncbiTaxon, proteinId)`. JSONL uses canonical compact JSON and
one LF per record.

The manifest uses `slp.static-entity-universe/v1` and binds both complete OMF
input identities plus their inner manifest and record-set hashes. The separate
audit binds the archive, manifest, entity-set and relation-set hashes and lists
the exact one-to-many mappings. It contains no measurements, labels, role
assignments, or numeric features. This artifact is an identity prerequisite,
not the static feature pack itself and not a training corpus.

Five semantic set digests are independently frozen. Each basis is reduced to
unique ASCII strings, sorted by ordinal byte value, written with one LF after
every item including the last, then SHA-256 hashed. The action and protein
bases are respectively `interventionId` and `proteinId`; an edge item is
`proteinId`, one TAB, then one `currentOrfRelations` ID. The full entity basis
is the union of all action, protein, and relation-target IDs under the single
bound source taxon. The authoritative composite-key basis is `ncbiTaxon`, one
TAB, then `entityId`; unlike the compatibility ID-only digest, it remains
unambiguous as the factory gains species. Their production digests are
compiled into the module and recomputed before any output is written.

<a id="slp-1-1-static-entity-universe-v1-readme"></a>

## slp-1-1-static-entity-universe-v1: README

Original location: `modules/slp-1-1-static-entity-universe-v1/README.md`.

### SLp-1.1 static entity universe v1

This self-contained OMF module turns two already-normalized, outcome-blind
proteome identity snapshots into one deterministic species-aware token
universe. It preserves SGD action identities, UniProtKB readout-query
identities, and every typed one-to-one or one-to-many relation without using a
held roster or any quantitative field. The relation-closed universe has 7,037
entities: 6,326 current model-eligible keys plus 711 SGD relation-support-only
genes that are not promoted to actions.

This v1 artifact is yeast-only. Its composite identity contract is the input
boundary for the planned corpus-v1.2 migration; the historical sparse-world
consumer still keys by bare entity ID and must not be used as evidence of
multi-species support.

The production input identities are compiled into the module. The workload
uses OMF 1.0's supported `dataset/<name>` grammar; OMF pins the immutable
revision at admission and the module rejects it unless its complete resource
URI and outer digest match the compiled contract. Running it still requires
the normal clean-Git, rights, validation, and admission protocol.

See `CONTRACT.md` for the normative byte and trust boundary.

<a id="slp-1-1-training-corpus-audit-v1-3-readme"></a>

## slp-1-1-training-corpus-audit-v1-3: README

Original location: `modules/slp-1-1-training-corpus-audit-v1-3/README.md`.

### SLp-1.1 clean-training corpus audit v1.3

This module replaces the three-quantitative-corpus v1.2 audit **for training
only**. It accepts one optimizer corpus, the outcome-blind held roster, at least
two outcome-blind protected-source intervention inventories, and a detached
Ed25519 authorization from the protected-data custodian. Validation and final
quantitative truth are structurally absent from the interface.

The audit authenticates the exact copied DatasetSnapshot resources and outer
manifest digests, then independently recomputes the corpus content digest,
roster and coverage hashes, inventory-manifest hashes, record-level active
actions, and the held validation/final union. It fails if any held intervention
is active in a fitting trajectory, if benchmark labels are present, if reward
is enabled, or if any signed identity differs from recomputed content.

The signature authorizes the exact composed optimizer corpus for one named
clean-factory identity and 256-bit challenge. It does not by itself prove
one-time consumption, freshness, source-to-corpus lineage, legal rights, or
filesystem isolation. Those remain coordinator, OMF-lineage, rights, and
separate-service controls. A future training consumer must independently
reverify the authorization rather than trust a Boolean field in this report.

The production public key is intentionally absent. Until an independent key
ceremony provisions the source-pinned trust anchor and a provenance-complete
composed corpus exists, every biological run must fail closed. Positive tests
use ephemeral keys outside the module. No private key or signing helper is
stored in this repository.

This module produces deterministic boundary evidence only. It trains no model
and supports no performance, novelty, release, frontier, or SOTA claim.

<a id="slp-1-1-training-corpus-audit-v1-4-readme"></a>

## slp-1-1-training-corpus-audit-v1-4: README

Original location: `modules/slp-1-1-training-corpus-audit-v1-4/README.md`.

### SLp-1.1 clean-training corpus audit v1.4

This module replaces the three-quantitative-corpus v1.2 audit **for training
only**. It accepts one optimizer corpus, the outcome-blind held roster, at least
two outcome-blind protected-source intervention inventories, and a detached
Ed25519 authorization from the protected-data custodian. Validation and final
quantitative truth are structurally absent from the interface.

The audit authenticates the exact copied DatasetSnapshot resources and outer
manifest digests, then independently recomputes the `slp.corpus/v1.2` content
digest, composite `(ncbiTaxon, entityId)` sets, static-feature-pack digest,
roster and coverage hashes, inventory-manifest hashes, and record-level active
actions. The immutable yeast roster is interpreted as taxon 4932 in memory;
the same textual ID in another taxon is not a held yeast action. It fails if
any held yeast intervention is active in a fitting trajectory, if benchmark
labels are present, if reward is enabled, or if any signed identity differs
from recomputed content.

The pretrain snapshot may contain the corpus files directly or exactly one
uncompressed `corpus-v1-2.tar` whose payload is under `composite-corpus/`.
Legacy `trajectoryGenes`, `entity_species_taxon`, and `query_id` layouts are
rejected rather than silently upgraded.

The signature authorizes the exact composed optimizer corpus for one named
clean-factory identity and 256-bit challenge. It does not by itself prove
one-time consumption, freshness, source-to-corpus lineage, legal rights, or
filesystem isolation. Those remain coordinator, OMF-lineage, rights, and
separate-service controls. A future training consumer must independently
reverify the authorization rather than trust a Boolean field in this report.

The production public key is intentionally absent. Until an independent key
ceremony provisions the source-pinned trust anchor and a provenance-complete
composed corpus exists, every biological run must fail closed. Positive tests
use ephemeral keys outside the module. No private key or signing helper is
stored in this repository.

This module produces deterministic boundary evidence only. It trains no model
and supports no performance, novelty, release, frontier, or SOTA claim.

<a id="slp-1-1-training-corpus-audit-v1-5-contract"></a>

## slp-1-1-training-corpus-audit-v1-5: CONTRACT

Original location: `modules/slp-1-1-training-corpus-audit-v1-5/CONTRACT.md`.

### Normative v1.5 contract

#### Inputs

The input map is exactly `pretrain`, `heldRoster`,
`custodianBoundaryAttestation`, and two to 64 sorted, unique
`protectedInventory*` entries. Every value is a copied, immutable, revisioned
`abiome/slp` DatasetSnapshot with the literal OMF materialization shape
`resource`, `mode`, `path`, and `manifestDigest`. Any other input name,
including molecular-validation truth, molecular-final truth, reward,
checkpoint, prediction, raw-source, or benchmark data, is forbidden.

`rewardEnabled` is literal false. `recipientFactoryIdentity` and
`challengeNonce` are required frozen values and must exactly equal the signed
recipient. Parser and allocation bounds are explicit and cannot be exceeded.

#### Authorization

The authorization snapshot contains exactly canonical `authorization.json`
and lowercase-hex `authorization.ed25519`. JSON uses sorted keys, compact
separators, ASCII escapes, and exactly one LF. The signature is Ed25519 over:

`abiome-org/SLp/training-corpus-handoff/v1 NUL || uint64be(length) || exact authorization.json bytes`

The statement binds the custodian key identity, recipient namespace, clean
factory signing identity, 256-bit challenge, safe protocol flags, and the exact
resource, outer manifest, and recomputable inner content identities of the
pretrain corpus, held roster, and every inventory. The verifier derives the key
ID from the raw 32-byte public key and checks both a compiled key ID and the
compiled SHA-256 of the 65-byte lowercase-hex public-key file. Runtime inputs
and configuration cannot supply a key or trust-store path. Rotation requires a
new immutable module version. The private key never enters Git or OMF.

The authorization is replay-resistant across corpora, recipients, and
challenges. Stateless verification cannot prove one-time use for the same
recipient and exact inputs; the coordinator must maintain a consumed
authorization-ID ledger. `issuedAt` is signed provenance, not a wall-clock
expiry rule.

#### Audit and output

The module verifies the signature before reading the large corpus, then scans
the full sparse action representation and canonical composite-key trajectory
JSONL. The pretrain corpus must use `slp.corpus/v1.2`, be molecular,
benchmark-free, reward-disabled, internally hash-consistent, and have no active
taxon-4932 validation/final roster member. Held static feature rows are allowed
when they never appear as active quantitative interventions. The same textual
identifier in a different taxon is a different entity and cannot collide with
or evade a composite-key consistency check.

The v1.2 entity archive contains exactly `entity_taxon`, `entity_id`,
`entity_type`, `entity_feature_value`, and `entity_feature_present`; the query
archive contains only entity and readout indices. The feature-pack SHA-256 is
recomputed from canonical pack JSON with its `sha256` member omitted. File byte
counts, digests, complete/absent feature rows, block key sets, query/context/
action taxa, and redundant corpus counts are recomputed. The pretrain snapshot
contains exactly two top-level regular files: an uncompressed
`corpus-v1-2.tar` with one `composite-corpus/` payload tree and canonical
`corpus-compose-audit.json`. Missing, extra, noncanonical, or mismatched files
are rejected.

The companion has an exact closed structure and binds the tar SHA-256 and byte
count, corpus manifest and input lineage, counts, composite entity identities,
feature pack, feature and target byte-preservation claims, zero protected
overlap, and false benchmark/reward declarations. Composed feature and target
hashes are independently reconstructed from the tar. Source-side hashes must
equal those reconstructed hashes, but cannot themselves be independently
recomputed because source arrays are deliberately absent from this boundary;
the output records `sourcePreservationIndependentlyRecomputed: false`.

The output directory contains exactly one canonical `corpus-audit.json` using
`slp.corpus-audit/v1.5`. It contains only the pretrain corpus identity,
outcome-blind roster/inventory identities and protected-set hashes, and
authorization identity. It contains no validation/final DatasetSnapshot
locator or quantitative truth identity. Re-running with identical inputs and
authorization bytes must produce identical output bytes.

The signature authenticates both bundle files and the final composed corpus
identity, not their entire scientific derivation. A biological run additionally
requires verified OMF lineage from exact
observation, static-feature, roster, and basal snapshots. The existing
`slp.corpus/v1.1` format does not carry all of that evidence and is rejected.
The v1.2 full-custodian audit, clean-training audit v1.3, and current world
trainer remain historical contracts and cannot consume v1.5 evidence. New
versioned consumers must reverify this authorization and its bound identities.

<a id="slp-1-1-training-corpus-audit-v1-5-readme"></a>

## slp-1-1-training-corpus-audit-v1-5: README

Original location: `modules/slp-1-1-training-corpus-audit-v1-5/README.md`.

### SLp-1.1 clean-training corpus audit v1.5

This module replaces the three-quantitative-corpus v1.2 audit **for training
only**. It accepts one optimizer corpus, the outcome-blind held roster, at least
two outcome-blind protected-source intervention inventories, and a detached
Ed25519 authorization from the protected-data custodian. Validation and final
quantitative truth are structurally absent from the interface.

The audit authenticates the exact copied DatasetSnapshot resources and outer
manifest digests, then independently recomputes the `slp.corpus/v1.2` content
digest, composite `(ncbiTaxon, entityId)` sets, static-feature-pack digest,
roster and coverage hashes, inventory-manifest hashes, and record-level active
actions. The immutable yeast roster is interpreted as taxon 4932 in memory;
the same textual ID in another taxon is not a held yeast action. It fails if
any held yeast intervention is active in a fitting trajectory, if benchmark
labels are present, if reward is enabled, or if any signed identity differs
from recomputed content.

The pretrain snapshot must contain exactly two top-level regular files:
`corpus-v1-2.tar` and `corpus-compose-audit.json`. The audit independently
parses the tar payload under `composite-corpus/`, recomputes its corpus,
feature, target, identity, and input-lineage digests, and checks the canonical
closed companion against those results. A tar alone, direct corpus files, an
extra file, or a mismatched companion fails closed. Legacy `trajectoryGenes`,
`entity_species_taxon`, and `query_id` layouts are rejected rather than
silently upgraded.

The companion proves that the producer asserted byte preservation and binds
that assertion to the composed bytes. Because the source arrays are not an
input to this clean-training audit, source-side preservation is explicitly
reported as not independently recomputed; it remains an upstream lineage gate.

The signature authorizes the exact composed optimizer corpus for one named
clean-factory identity and 256-bit challenge. It does not by itself prove
one-time consumption, freshness, source-to-corpus lineage, legal rights, or
filesystem isolation. Those remain coordinator, OMF-lineage, rights, and
separate-service controls. A future training consumer must independently
reverify the authorization rather than trust a Boolean field in this report.

The production public key is intentionally absent. Until an independent key
ceremony provisions the source-pinned trust anchor and a provenance-complete
composed corpus exists, every biological run must fail closed. Positive tests
use ephemeral keys outside the module. No private key or signing helper is
stored in this repository.

This module produces deterministic boundary evidence only. It trains no model
and supports no performance, novelty, release, frontier, or SOTA claim.

<a id="slp-1-1-world-sparse-contract"></a>

## slp-1-1-world-sparse: CONTRACT

Original location: `modules/slp-1-1-world-sparse/CONTRACT.md`.

### Typed sparse production contract

This module fits a fixed-epoch, application-neutral world model from one
`pretrain` `DatasetSnapshot`. It receives no molecular validation/final truth,
benchmark record, reward signal, label, or target-bearing evaluation input.
It is an engineering candidate, not biological or advancement evidence.

The other model input is a copied, revision-pinned
`molecular-validation-query` `DatasetSnapshot`. Its `query.json` and JSONL
records contain only canonical profile/context/intervention/readout identities
and aligned Gaussian or negative-binomial distribution types. The exact file
inventory is `query.json` plus its declared JSONL shards; targets, observed
masks, labels, benchmark fields, corpus-role fingerprints, and companion arrays
are structurally impossible. Static entity/species features and the small
readout ontology are resolved from the independently admitted pretrain feature
pack, without passing identifiers into the model.

This v1 query intentionally supplies no context or continuous covariates: those
inputs are explicitly missing during prediction. A future query revision must
add typed, unit-bearing, presence-masked context/covariates before claims that
depend on those values. Distribution-to-readout mapping must be unambiguous in
the admitted pretrain ontology or loading fails. The current evaluator-v2 query
revision is also explicitly yeast-only (`NCBI:txid4932`, systematic SGD
intervention IDs); cross-species evaluation requires a new admitted contract.

Both DatasetSnapshot inputs must have OMF's exact copied input shape, immutable
resource revisions, and outer manifest digests. Materialized paths must be
non-symlink directories shaped as
`.../inputs/<input-name>/<resource-name>`. The module also requires an exact OMF
artifact for a previously admitted, reward-disabled
`slp.corpus-audit/v1.2` and a copied,
revision-pinned DatasetSnapshot for the global held-intervention roster. Bare
JSON or an unbound filesystem path is not accepted. The frozen run config must
also name the exact admitted corpus-audit artifact-manifest digest; a different
materialization fails before training.

The admitted audit explicitly records `rewardEnabled: false` and binds
pretraining, molecular validation, and molecular final corpus identities plus
the roster provenance and population hashes. A reward snapshot, placeholder,
or identity is forbidden. The trainer independently verifies its pretrain identity, recomputes
the frozen roster assignment role from each identifier digest, checks the
validation/final corpus populations against roster hashes, requires the query
intervention domain to equal the complete nonempty validation roster, and
rechecks that no validation or final intervention occurs in a pretraining
quantitative trajectory. OMF prior admission is mandatory. Protected source
inventories are independently pinned in the audit and bind the recomputed
QC-passing intersection rather than relying on summary role strings.

#### Model and likelihood boundary

Stable CURIEs remain provenance only. `WorldBatch` receives numerical features,
explicit presence masks, species features, and small ontology type indices; it
contains no entity, query, or dictionary IDs. Parameter count is independent of
dictionary size. Context/action memory is permutation invariant, and each query
cross-attends only to shared encoded memory. A marginal prediction is therefore
exactly invariant to panel membership, ordering, padding, and chunking when
dropout is disabled.

Observed pretraining targets use bounded CSR arrays. Missing is distinct from
an observed zero. Each readout declares a Gaussian or negative-binomial
likelihood. Gaussian parameters are `{mean,logScale}`; negative-binomial
parameters are `{logMean,logInverseDispersion}`. Count-bearing biological use
remains blocked until a source-appropriate library-size offset contract is
frozen.

Sampling follows explicit source weights and deterministic
source → perturbation → replicate → record quotas. Each scheduled record first
means its observed typed NLL, then records receive equal optimizer weight.
Epoch schedules are seed-domain-separated. The canonical report contains only
pretraining optimization evidence and states that held truth was inaccessible.

#### Artifacts and release status

The module writes one deterministic timestamp-free checkpoint, one canonical
training report, and one deterministic uncompressed tar file containing exactly
`evaluation.json` and `profiles-000.jsonl`. The file-only transport avoids the
known OMF 1.0 directory-artifact importer defect while retaining an exact,
bounded internal file manifest. Prediction JSONL records duplicate the complete
canonical query identities and add only aligned typed `predictionParameters`;
they contain no targets, missingness mask, label, or target-derived inclusion.
`evaluation.json` binds the exact query resource, outer manifest, raw
`query.json` digest, and deterministic checkpoint file content SHA-256. A later
evaluator must independently pin the checkpoint and protected truth and verify
those joins before the predictions are evidence.

Checkpoint loading validates canonical headers, exact total and payload byte
sizes, dimension/layer/tensor/parameter bounds, tensor shapes and hashes before
model or payload allocation. Payloads use canonical raw little-endian float32,
not pickle or timestamp-bearing archives.

The empty dependency lock is retained only for this engineering milestone.
Every result states `environmentAttestedNotPortable: true` and
`releasePortable: false`. Release remains blocked by both a missing hash-pinned
offline wheelhouse and the OMF 1.0 artifact-to-inference-adapter gap. It is also
blocked until OMF policy independently proves that the pinned audit artifact
was produced and admitted by the corpus-audit module; digest equality alone is
not producer provenance. No network, absolute-path, import-path,
metadata-weight, or repository-relative workaround is used.

<a id="slp-1-1-world-transition-v1-contract"></a>

## slp-1-1-world-transition-v1: CONTRACT

Original location: `modules/slp-1-1-world-transition-v1/CONTRACT.md`.

### Experimental molecular transition module

This self-contained module runs native local CUDA development. It is not yet
an admitted OMF workload or a certified model release. Python 3.11.9 and the
CUDA 12.8 Torch runtime were exercised on one RTX 4070. No repository-relative
imports outside this module are required.

`transition_model.py` maps intervention feature sets and optional measured
basal molecular tokens to a latent state. Independently queried feature vectors
decode measurement means, diagonal uncertainty, and optional shared low-rank
Gaussian factors. It has no learned gene-ID vocabulary. The application-neutral
network receives no SL labels or scoring thresholds. Set inputs support
multiple interventions structurally; biological combination generalization
requires separate molecular data and evaluation.

`train.py` consumes the pinned yeast fitting composite. `train_human.py`
consumes a checksum-pinned human development NPZ with disjoint gene-grouped
training and validation indices and empty test indices. Each creates a fresh
output directory, saves its protocol before fitting, copies source, trains
with a bounded wall-clock allowance, and selects checkpoints by development
gene-macro Gaussian NLL. These choices do not establish independent performance.

The default protein-only decoder uses static features for intervention and
query genes. The optional response-basis decoder also requires training-derived
measurement descriptors for query genes. These are assay features, not static
sequence features: no claim is made that its response basis transfers to a new
unmeasured assay panel. Context-specific fitted references are currently
required; unseen-context inference is not yet established.

Artifacts are `model.safetensors`, `model-config.json`, `reference.npz`, copied
`source/`, `protocol.json`, and `report.json`. `inference.Predictor` reloads tensor
files without a corpus, database, or OMF runtime. Its `predict` accepts raw
action/query features, aligned references and scales, and optional measured
context; its `sample` draws from the specified joint Gaussian. Current
pseudobulk training supports uncertainty over aggregate molecular measurements,
not a validated single-cell population generator. OMF portable inference is a
separate untested integration.

Verification covers feature/query permutation and chunk invariance, missing
observations, exact low-rank Gaussian likelihood against a dense reference,
artifact reload, sampling covariance, grouped fitting-only calibration, and
source-specific preparation. Numerical revisions are identified by copied
source hashes, with new artifacts rather than overwritten experiments.

<a id="slp-1-1-yeast-prepare-source-format"></a>

## slp-1-1-yeast-prepare: SOURCE_FORMAT

Original location: `modules/slp-1-1-yeast-prepare/SOURCE_FORMAT.md`.

### Yeast preparation source contract

The module consumes one materialized OMF source snapshot directory containing
`source.json`, a pinned flat rights declaration, and one or more path-sorted
JSONL files. It never downloads data. The manifest must identify an immutable
source release, NCBI taxon 4932, the SGD namespace, molecular-only labels, the
SHA-256 of the rights file, and the SHA-256 and record count of every raw file.

Every JSONL record uses schema `slp.yeast-molecular-record/v1` and contains a
globally unique stable `recordId`, a source-stable `perturbationId`, taxon 4932,
declared modality, assay, protocol, endpoint, normalization, experimental
metadata, continuous species features, and canonically sorted context, action,
and query tokens. Entity identity is always an `SGD:S#########` CURIE; display
symbols are not accepted. Action covariates match the explicitly configured
`actionCovariateDim`. Queries identify a configured quantitative readout type,
numeric target, and observation mask. Every record's species features must
exactly equal the configured yeast vector recorded under taxon 4932 in the
output manifest.

Records must be strictly sorted by `recordId` across raw files. This permits
bounded streaming without a global in-memory sort. The output archive contains
deterministic `.npz` shards accepted by `slp-1-1-world`, a corpus manifest,
sorted intervention-gene inventory, and immutable provenance. The archive is a
preparation result, not an admitted training snapshot: unpack it, review the
rights/provenance report, then create and verify the role-specific OMF
`DatasetSnapshot` in a separate explicit step.

Each shard carries fixed-width `record_id`, `source_id`, `perturbation_id`, and
`action_curies` arrays. Padded `action_curies` are empty and every active entry
is aligned exactly with `action_mask`. The corpus intervention inventory is
constructed as the exact union of those active action CURIEs.

<a id="slp-1-1-yeast-seurat-stream-v1-contract"></a>

## slp-1-1-yeast-seurat-stream-v1: CONTRACT

Original location: `modules/slp-1-1-yeast-seurat-stream-v1/CONTRACT.md`.

### Yeast Seurat streaming contract

This numerical module inventories gzip RDX3/XDR serializations with pinned
`rdata==1.1.0` type codes. It retains small structure, symbols, S4 slots and
selected atomic ranges. Large atomic vectors are consumed in bounded chunks,
checksummed and discarded. A serialization reference points at the original
lightweight node and never causes a payload copy.

Only an S4 `dgCMatrix` reached through the RNA assay's `counts` slot is eligible
for later molecular processing. SCT, normalized `data`, and `scale.data` are
excluded. The module does not infer an assay from a truncated prefix.

Offsets refer to the uncompressed RDX3 byte stream. Gzip is replayed from byte
zero for pass two. CSC column selections are translated through the complete
`p` vector into ranges shared by `i` and `x`. No selected value range may be
chosen from an expression effect; eligibility must already follow from stable
intervention metadata, the global protected-gene partition, or control status.

The current 2 MiB source prefix ends within the first matrix's 181,083,366-entry
`i` payload. It cannot establish dimensions, dimnames, cell metadata, assay
identity, payload integrity, or a usable matrix. Full-file extraction remains
unexecuted and must first inventory these contracts.

Reference-bearing environments and external pointers preserve their reference
positions and lightweight structure. Unsupported weak references and bytecode
are rejected. This is deliberate: silently skipping one would shift the R
reference table and could corrupt every later symbol or slot interpretation.

<a id="historical-slp1-notes"></a>

## Historical SLp-1 development notes

Archived from `docs/model-card.md`. The frozen release card is
[model/v1/MODEL_CARD.md](../model/v1/MODEL_CARD.md).

# SL-Predict model outline

## Scientific claim

SL-Predict will model an intervention-conditioned distribution over cellular states. Synthetic lethality is scored from the predicted consequence of two perturbations, not represented as an edge in the transition model. A separate benchmark head may be fitted on training-fold SL labels, but results from that head will be reported separately from label-free transition scores.

## Inputs and state

- Basal state: a compact set of tokens derived from gene-expression, dependency, functional, interaction, Geneformer and State/ESM protein representations. Six non-SL graph views are represented by deterministic leading spectral coordinates. For cell-specific evaluation, a checksum-verified DepMap 24Q2 state adds 64 expression coordinates, 64 single-gene dependency coordinates and two availability flags. A gene-aligned variant also supplies the two perturbed genes' independently measured expression and dependency values. A frozen knowledge-graph block is available, but the provenance-safe external model zeros it because its exact training corpus is unresolved.
- A later perturbation-state candidate conditions transitions on a separate 128-dimensional basal-expression world state derived from 1,066 DepMap models. Source-cell mappings are exact for K562 and THP-1, averaged over five available engineered RPE1 derivatives and explicitly unknown for hESC. Fifty-percent context dropout preserves unknown-context inference.
- Action: one or more gene knockouts, dose, elapsed time and context tokens. The first benchmark is pan-cancer and uses explicit unknown-context tokens where context is absent.
- Next state: a stochastic latent distribution decoded into held-out dependency, expression and functional-relation measurements, including a 32-dimensional cross-study Perturb-seq expression state. Frozen auxiliary decoders expose raw single-gene tolerance and continuous signed double-knockout depletion without binary SL calls.

All gene features must remain available for genes absent from the supervised training fold. SL edges and features derived from the complete SL graph are forbidden in the transition path.

## Architecture

The initial model is a 6-layer, 384-wide transformer with six attention heads and a 1,536-wide feed-forward block. With protein features and two quantitative outcome channels it has 11,919,712 trainable parameters. A 12,782,560-parameter variant expands each biological relation view from 32 to 256 landmarks. A controlled 8-layer, 768-wide, 59,638,240-parameter candidate covers the requested 50–100M range; it advances only if held-out molecular-state objectives justify its extra capacity. Frozen Geneformer and State/ESM representations supply biological priors.

The transformer receives basal-state, context, time and action tokens. It emits deterministic memory plus Gaussian latent state parameters. The same transition function accepts single, simultaneous and sequential interventions, so `T(T(s, a), b)` and `T(s, {a,b})` are directly comparable.

## Training

1. Pretrain without SL labels by masked modality reconstruction, relation prediction and single-action state prediction.
2. Enforce action permutation invariance, sequential/simultaneous agreement and calibrated stochastic uncertainty.
3. Reinforcement-learn stochastic rollouts with a self-critical policy-gradient objective. Reward is held-out molecular-state likelihood, rollout consistency and uncertainty calibration; no test pair or test-derived feature enters the reward. Each reinforcement phase retains the pre-update parameters unless it improves a fixed label-free validation set.
4. Fit the transition and a compact decoder to measured single- and double-perturbation expression pseudobulks from seven local perturbation packs: Adamson, Dixit, Norman, two Replogle cell lines, Wessels/Satija THP-1 and Joung/Zhang hESC. All scenario-3 validation/test genes are excluded before fitting; the held-gene validation split and state loss use no SL labels. Study-balanced training upweights double perturbations, applies context dropout and permits validation-controlled self-critical rollout refinement.
5. In the basal-context candidate, first fit the context projection to predict single-gene CRISPR dependency from expression across the DepMap world pack, then fit the same held-gene perturbation-state objective with continuous source-cell state. The gene encoder and relation transformer remain frozen. Reinforcement updates are retained only if the fixed state validation objective improves.
6. Fit a context-aware decoder to library-size-normalized T0-to-endpoint guide depletion and the robustly normalized magnitude of SLKB's native continuous interaction score. Exact Feng benchmark pairs are excluded before normalization and target construction. The decoder sees quantitative measurements, not binary SL calls.
7. Compute label-free double-knockout scores from predicted interaction strength or perturbational expression non-additivity. Sparse viability residual, explicit expression-residual supervision and recovered-cell abundance targets were tested and rejected empirically. As a permitted secondary experiment, calibrate a linear or shallow benchmark head using training-fold SL labels only.
8. Fit measured cell state only from independently collected DepMap expression and single-gene CRISPR effects. The global state projection excludes all scenario-3 held-out genes, and unseen K562/Jurkat states do not enter normalization. An all-cell auxiliary objective predicts single-gene dependency from expression while excluding A549, K562, Jurkat and every held intervention; the relational backbone remains frozen. Fold-local ranking probes that consume benchmark training labels are reported separately from the transition score.
9. Freeze the final world model. Fit a 16,897-parameter tolerance decoder on raw DepMap gene effects and a separate 17,026-parameter interaction decoder on continuous SLKB depletion and interaction magnitude. The interaction decoder is selected on 388 pairs for which both genes are absent from fitting. Its fixed label-free score is negative predicted double-knockout depletion; no binary SL label trains or selects it.
10. Test decoder capacity only against the same molecular validation partition. A 165,633-parameter symmetric interaction decoder using joint, single-perturbation and gene-pair features was rejected before benchmark scoring because held-gene depletion Huber loss rose from 0.2693 to 0.3113 and correlation fell from 0.5240 to 0.3857.
11. Estimate repeated-screen reliability without SL labels. A same-size successor shrinks each context-specific depletion target 15% toward its gene-pair mean and retains the quantitative magnitude auxiliary task. Selection uses the sum of row and pair-mean Huber loss on the same both-genes-new validation partition. Pair-only, low-rank bilinear and tree alternatives are rejected at this partition without benchmark scoring.
12. Test whether a cross-modal DepMap signal supplies the missing conditional-dependency geometry. Symmetric expression-to-dependency correlations improve all held-gene continuous-depletion metrics, but a fixed one-coefficient residual lowers MuSL CV3 from 0.6271/0.6275 to 0.6257/0.6268 and is rejected before external evaluation. Direct relation residuals and rank-aware training are also rejected on molecular validation. Encoding the full cross-modal geometry in the gene token improves held molecular-relation loss only when introduced through a zero-initialized 153,600-parameter adapter; this adapter preserves state validation but worsens unseen-gene double-knockout depletion to 0.2940 Huber loss and 0.4416 correlation, so it is also rejected before benchmark scoring.
13. Test binary cross-dataset transfer without benchmark training labels. Four readouts are selected only by fixed both-genes-new validation on 51,013 SLKB pairs. The selected boosted full-state readout reaches 0.8116 AUROC within SLKB but only 0.5320 on MuSL CV3 after removing every external pair containing a test gene. Binary SL supervision is therefore rejected as a source-invariant readout target.
14. Residualize quantitative double depletion against independently measured raw DepMap single-gene effects in matched cell contexts. The residual decoder performs worse than a zero predictor on the fixed both-genes-new molecular partition and has negative Pearson and rank correlations, so it is rejected without benchmark scoring.
15. Treat the complete HAP1 map as continuous qGI supervision without binary hit calls. A compact decoder fitted to 1,017,515 pairs has only 0.0064 Pearson and 0.0063 Spearman correlation on 47,600 pairs whose two genes are absent from fitting, and its Huber loss is worse than predicting zero. It is rejected before MuSL scoring under the preregistered molecular advancement rule.
16. Test whether the failure is specific to the joint transition latent. Symmetric 66,625- and 535,153-parameter networks relearn gene representations from the frozen world encoding or all provenance-safe gene coordinates. The larger candidate still has only 0.0042 Pearson and 0.0052 Spearman correlation on the same both-genes-new HAP1 partition and is rejected before MuSL scoring.
17. Reconstruct the local SynLethKG input independently after removing explicit SL, synthetic-rescue and non-SL relations. A 128-dimensional TransE prior improves held relation reconstruction but misses the dependency checkpoint and a direct pair residual severely degrades unseen-gene continuous depletion. The prior and audit are retained for provenance; the model branch is rejected before MuSL scoring.
18. Expand the seven-source perturbational-expression target from 32 to 96 coordinates after a molecular-only rank audit. The richer endpoint improves the fixed context-aware state score from 1.86951 to 1.86344 without changing dependency loss. Its repeated-screen interaction decoder improves pair-mean Huber loss and Pearson correlation, but row Pearson correlation regresses 1.179%, narrowly violating the preregistered 1% transfer bound. The branch is rejected without MuSL scoring.
19. Combine the retained 32-dimensional and candidate 96-dimensional depletion predictions over a five-weight molecular grid. The automatic minimum-Huber selector fails because it chooses the rejected 96-dimensional endpoint alone. A separately frozen 25%/75% Pareto blend satisfies every continuous-transfer bound and is then evaluated once on MuSL CV3 without label-dependent fitting.

The executable model uses local DepMap correlation and biological-relation matrices as priors plus measured control-to-perturbed pseudobulks. The expanded corpus contains 7,665 rows over 206 common expression genes, including 404 training doubles. Its cell types and perturbation modalities remain heterogeneous, and the held-gene validation doubles still come from Norman; this limitation must remain explicit.

The retained compact scale is 11.92M parameters. A controlled 59.64M model improved the molecular proxy losses but reduced CV3 transfer to 0.7125/0.7050, so additional capacity is not justified by the present data.

## Evaluation

Every new molecular outcome corpus is first audited under deterministic
five-fold exact-action-set, composition-gene, intervention-gene, context,
source, perturbation-condition and compound source-plus-gene holdouts. The
composition-gene protocol permits matched single interventions for a held gene
but forbids that gene in any fitted multi-intervention example; the stricter
intervention-gene protocol forbids every fitted intervention involving it.
Source, context and condition identifiers are independent provenance fields.
Ineligible folds and absent fields are reported as failures. Molecular state
loss is compared with a cardinality-matched training mean and, for delta
targets, a matched-single additive predictor before any SL benchmark is
opened.

Primary evaluation is the released Feng et al. protocol, with pair holdout, one-new-gene and two-new-gene splits, random/expression/dependency negatives, class ratios 1:1 through 1:50, and both classification and ranking metrics. The intended confirmatory design selects models on inner training-only splits and evaluates outer folds once. The current development run inspected outer CV3 across representation and readout ablations, so its scores are development estimates; a final SOTA claim requires a newly locked evaluation or independent experimental benchmark.

The two-new-gene matrix is compacted into 24 protocols and evaluated with the authors' gene-wise ranking semantics. Label-free ranking scores are generated over all 9,845 genes; training pairs are removed before NDCG, recall, precision and MAP are calculated.

The headline endpoint is two-new-gene CV3. Feature access is reported for every result. Transition quality is evaluated separately with held-out modality reconstruction, action composition error and uncertainty calibration. Later tracks add intervention-isolated gene holdout, K562/Jurkat CDKO screens, context holdout and natural-prevalence ranking.

SLAMR scenario 3 supplies a second both-genes-cold ranking test in A549, Jurkat and K562. For external evaluation, the quantitative pretraining pack excludes the union of all mapped validation and test pairs from every fold before model fitting. The evaluator consumes no SLAMR labels. This prevents exact-pair reuse, although it does not remove every training perturbation involving a held-out gene.

A stricter intervention-isolated pack removes every quantitative trajectory containing any scenario-3 validation or test gene. Under this test the high exact-pair-isolated K562 and Jurkat rankings collapse for quantitative SLKB training alone. Inner both-new-gene outcome selection, a frozen-backbone control, a compact semantic context token, measured DepMap cell state, gene-aligned state and all-cell expression-to-dependency pretraining have all been tested. The all-cell objective improves isolated K562 recall but not MRR or cross-cell consistency; it is retained as an ablation.

Held-gene Perturb-seq expression prediction is the first state objective retained under this stronger test. Its label-free expression non-additivity reaches 0.1300 MRR / 0.3033 Recall@20 on K562, compared with 0.1099/0.2172 for the prior intervention-isolated outcome model and 0.1236/0.2644 for a fixed gene-description prior. It does not transfer cleanly to Jurkat (0.1102/0.2078) or generic Feng CV3 (0.4447 AUROC / 0.4587 AP). The result establishes context-specific fully intervention-cold emergence, not general SOTA. Fold-local supervised ranking probes remain explicitly secondary benchmark heads rather than emergent transition scores.

An exploratory native rollout score compares simultaneous double perturbation with the mean of both sequential intervention orders. It improves unknown-context K562 to 0.1335 MRR / 0.3167 Recall@20 without retraining; averaging the four provenance-defined K562 perturbation-study contexts reaches 0.1507/0.3435. The score is near random on generic Feng CV3 and was conceived after the primary external result. It is retained as an architectural diagnostic, not a locked headline score.

After that score was fixed, it was evaluated once on both official MuSL pan-cancer CV3 seeds, whose labels had not previously been inspected. The four-source mean reaches 0.5382 AUROC and 0.5310 average precision across ten folds, compared with 0.5133/0.5091 under the unknown context. This is the first independently locked evidence of above-chance two-new-gene transfer from the label-free state readout. It remains far below MuSL's author-reported 0.7895/0.8018 and therefore confirms emergence, not SOTA.

A separately reported supervised readout fits the fixed deterministic ensemble to each MuSL training fold. Across both official seeds it reaches 0.8377 AUROC and 0.8419 average precision, exceeding MuSL's released 0.7895/0.8018 table on the same CV3 protocol. This is protocol-specific development SOTA, not emergence: benchmark SL labels train the readout, and the test labels had already been opened for the preceding frozen-score evaluation. The transition model and its label-free result are unchanged.

Adding Wessels/Satija and Joung/Zhang combinatorial perturbations improves the seven-source model's unknown-context label-free MuSL development score to 0.5336 AUROC / 0.5220 average precision, but the former four-source contextual score degrades. Two experimental transfers reject a broad emergence claim. On a locked Harle et al. 2025 paralog screen, the original sequential score has Spearman rho -0.2531 with negative interaction strength and 0.4819 macro AUROC. On the retrospective Billmann/Costanzo HAP1 map, the seven-source score has 0.01489 macro AP at 0.01498 prevalence and 0.48805 macro AUROC. Neither result is sign-flipped after inspection.

The 11,968,864-parameter basal-context candidate fails general pair transfer. On MuSL CV3, averaging its sequential score over 32 deterministic DepMap state representatives gives 0.4711 AUROC / 0.4873 average precision, below both chance and the context-free seven-source model. Top-three and maximum context risk summaries fall further to 0.4537/0.4726 and 0.4353/0.4628. HAP1's exact basal expression state was reconstructed from the same checksum-pinned DepMap transform with `2.37e-7` maximum reference error. Conditioning on it changes HAP1 macro AUROC from 0.4881 to 0.5003 and macro AP from 0.01489 to 0.01548 at 0.01498 prevalence, but the prespecified within-query permutation test gives p=0.999.

The same candidate nevertheless recovers context-specific signal in a retrospective Sanger analysis. Across 24 cell lines with exact DepMap expression states, the intervention-cold sequential score reaches 0.6081 macro AUROC and 0.1947 macro AP, compared with 0.4718/0.1253 for the earlier context-free score on identical coverage. Pair-averaged correlation remains nonsignificant at rho 0.0747 (p=0.361), so the sequential score is not retained as a universal pair predictor.

HAP1 labels were allowed in two secondary, fold-local readout ablations. Every pair containing a MuSL test-fold gene was removed before fitting either auxiliary predictor. A static tree reaches 0.8348/0.8394 MuSL AUROC/AP, while a neural readout over both transition orders reaches 0.8352/0.8393; both are below the retained 0.8377/0.8419 readout. HAP1 supervision is not retained.

An external-only SLKB binary readout is also rejected. It reaches 0.8116 AUROC on fixed gene-cold SLKB validation but 0.5320 AUROC on MuSL CV3 under intervention-level exclusion, despite retaining more than 32,000 external pairs per fold. The discrepancy indicates incompatible label definitions rather than a lack of capacity to learn SLKB calls.

The frozen raw-tolerance decoder is accurate on held DepMap cell lines (dependency AUROC 0.9342; correlation 0.7167), but its hard tolerance gate lowers label-free MuSL to 0.3779/0.4220 and is rejected. The continuous double-knockout decoder is different: signed depletion reaches correlation 0.5240 on 388 both-genes-new quantitative validation pairs, while absolute magnitude remains weak at 0.0638. Using the fixed signed score gives 0.6227 AUROC / 0.6245 AP on MuSL CV3. After additionally removing every gene used to fit this decoder, Sanger gives pair-mean rho 0.4453 and macro AUROC 0.5987 on 114 pairs, while HAP1 gives macro AP 0.02034 at 0.01511 prevalence and macro AUROC 0.5771 on 927,801 pairs. No benchmark SL labels fit this decoder.

This is the retained emergent readout. It is broad retrospective evidence rather than independent confirmation because MuSL, Sanger and HAP1 outcomes had all been inspected in earlier experiments. It is also below label-free SOTA on MuSL. The strongest current claims are therefore: compact state modeling produces transferable above-chance SL rankings, and a separate fold-local supervised readout gives protocol-specific development SOTA at 0.8377/0.8419. Prospective confirmation and improvement of the label-free CV3 gap remain required.

Increasing only the continuous decoder's pair capacity did not close that gap. The 165,633-parameter symmetric successor was stopped at molecular validation and never evaluated on MuSL, Sanger or HAP1; its lower unseen-gene correlation supports retention of the 17,026-parameter readout.

Repeated-screen target shrinkage improves the same decoder's held-gene depletion correlation from 0.5240 to 0.5525 and MuSL CV3 from 0.6227/0.6245 to **0.6271 AUROC / 0.6275 AP**. Sanger pair-mean correlation also rises from 0.4453 to 0.4549, but Sanger cell-line macro AUROC falls from 0.5987 to 0.5847 and HAP1 macro AUROC falls from 0.5771 to 0.5645. The shrinkage checkpoint is therefore the pan-cancer development readout, while the original decoder remains the broader context-specific transfer model. Neither establishes label-free SOTA.

The cross-modal conditional-vulnerability residual improves unseen-gene quantitative validation correlation from 0.5525 to 0.5719 but lowers the fixed MuSL score to 0.6257/0.6268. This discordance reinforces the requirement that molecular advancement and benchmark transfer are separate tests; the residual is not retained and was not exposed to Sanger or HAP1 outcomes.

Dense quantitative supervision from HAP1 does not solve the cold-start readout. A 16,897-parameter decoder trained on more than one million raw-qGI pairs is indistinguishable from zero on a both-genes-new HAP1 partition. Allowing a half-million-parameter symmetric network to relearn gene representations from every safe input coordinate also remains indistinguishable from zero. Because both molecular tests were fixed before fitting, the branch was stopped without MuSL evaluation.

A representation-level version reaches better generic molecular-relation validation (0.01788 versus 0.01892) without degrading held perturbation-state reconstruction, but substantially worsens the later continuous double-knockout decoder. It therefore fails before any SL benchmark, showing that relation fidelity alone is not a sufficient world-model selection criterion.

An independently trained, label-excluded SynLethKG prior reinforces the same conclusion. It improves held relation loss to 0.01874 and reconstructs its own typed graph well, but does not improve dependency state and sharply worsens held-gene continuous-depletion residuals. The supplied ambiguous KG checkpoint remains unused; only the new audited embedding is retained as a reproducibility artifact.

A 96-dimensional Perturb-seq endpoint is the first post-retention candidate to improve the combined held molecular-state score, reaching 1.86344 versus 1.86951 at 32 dimensions with identical dependency loss. It is not benchmarked: the fixed downstream continuous decoder improves pair-level validation strongly but lowers row Pearson correlation from 0.55251 to 0.54600, exceeding the preregistered 1% regression limit. This result supports richer endpoint modeling as a molecular direction while preserving the 32-dimensional checkpoint as the current benchmark-facing model.

The molecularly frozen dual-resolution readout changes that narrow conclusion without replacing either world checkpoint. Blending 25% of the 32-dimensional and 75% of the 96-dimensional depletion predictions improves all four row/pair transfer criteria, then raises label-free MuSL CV3 from 0.62705/0.62747 to **0.63197 AUROC / 0.62823 average precision**. Exact-state HAP1 improves to 0.02027 macro AP / 0.57758 AUROC, and Sanger cell-line macro AUROC/AP improve to 0.59526/0.14269; Sanger's primary pair correlation falls from 0.45494 to 0.42834. The blend is retained for pan-cancer and cell/query-level development, while the original decoder remains the stronger pair-mean Sanger readout. Neither establishes label-free SOTA.

A nested 96-dimensional successor preserves the original 32 targets exactly in its data representation and assigns equal loss weight to that block and a new 64-dimensional residual block. It improves residual and combined held-state scores to 1.84930 and 1.86161, with dependency Huber unchanged, but the shared transition regresses on the retained block from 1.86951 to 1.87391. It fails the molecular preservation rule and is not fitted to interactions or evaluated on CV3. This isolates gradient interference, rather than target rank, as the next architectural problem.

The accepted single-checkpoint successor freezes that complete RL-refined path and adds only a 25,024-parameter residual endpoint. Its residual held score of 1.85729 gives a conservative combined score of 1.86340 while making the legacy endpoint and dependency behavior invariant. A molecularly selected zero-initialized continuous correction then improves MuSL CV3 to **0.62926 AUROC / 0.62948 AP**. Sanger pair correlation remains 0.45002 with 0.59593/0.13997 cell-line macro AUROC/AP, but HAP1 falls slightly to 0.01947/0.56255. This 11,998,016-parameter model is the compact single-checkpoint development readout; the two-world blend remains the higher-AUROC MuSL and stronger HAP1 readout. Neither is label-free SOTA.

A 2,337-parameter within-context ranking successor is rejected before benchmark evaluation. Its best eligible epoch improves held row Spearman by only 0.00004 and does not change pair Spearman; later epochs degrade ranking. The compact residual endpoint remains unchanged.

A sibling additive Perturb-seq checkpoint is excluded by its own strict-cold data audit. Transferring only its compositional hypothesis also fails: a nested-gene ridge over the complete predicted endpoint and additive residual worsens every locked continuous-depletion endpoint and stops before MuSL, Sanger or HAP1. The accepted continuous-depletion readout then fails fully intervention-isolated SLAMR despite exact A549, Jurkat and K562 basal states; Jurkat and K562 average 0.0482 and 0.0849 MRR. Expression-state rollout, not continuous depletion, remains the stronger K562 mechanism.

A new 16,897-parameter outcome head isolates single-action viability from composition. With the 11,998,016-parameter world frozen, it improves Huber by 40.59% on 366,375 observations jointly held across 1,847 genes and 202 cells, reaching 0.6655 Pearson correlation and 0.9202 dependency AUROC. Its fixed sequential conditional-dependency equation nevertheless gives only 0.3751 AUROC / 0.4254 AP on MuSL CV3. The sign is not reversed after inspection. The model therefore predicts unseen single interventions well, but its recursively transitioned latent is not calibrated as an input to the viability head.

The exact 128-dimensional basal-expression coordinate system is subsequently reconstructed to 2.37e-7 maximum error and used to supervise a context-shift model from 7,149 measured single interventions only. The held-gene source-macro loss improves substantially, but the result is not source invariant: RPE1 dominates the gain, four studies lose absolute Huber to zero, and Dixit has negative held cosine. The context transition is rejected before any benchmark use.

Normalizing every measured shift to unit direction and balancing sources raises held source-macro cosine to 0.3731. Four studies comfortably pass the fixed 0.10 minimum, while Dixit reaches 0.09914 on 16 held interventions. The model is rejected without rounding or benchmark exposure. This near miss indicates that source-scale normalization largely repairs context geometry, but current evidence does not establish all-source invariance.

Training-only single interventions from Wessels, Joung and GSE337988 repair that near miss without changing the original five-source selection set. Held source-macro cosine reaches 0.3806 and every source clears 0.10, including Dixit at 0.1425. The fixed symmetric dependency-change score is positive but insufficient on MuSL CV3 at 0.5988 AUROC / 0.6139 AP, below the retained 0.6320/0.6282 dual-world result. The context-direction head is retained as molecular evidence; its pair readout is rejected with no post-result variants or mixing.

Context-specific in-silico deletions from the cancer-continued Geneformer checkpoint do not improve this molecular endpoint. A frozen-head residual over 635 K562 and 759 RPE1 deletion deltas changes held macro cosine only from 0.380586 to 0.380733, missing its locked 0.01 gain by two orders of magnitude. The branch stops without SL evaluation.

Adjusted TCGA PanCancer mutation mutual exclusivity is reproducible across disjoint patient halves and is retained as a public relation artifact: 18.06 million supported pairs have 0.1689 Pearson, 0.1683 Spearman and 2.879-fold top-tail overlap. Encoding its 102,438-edge graph in the safe blank block nevertheless gives 0.0191825 held relation loss after 12 matched pretraining epochs, worse than the retained 0.018922. The world-model candidate is rejected before downstream or benchmark training.

Five source-specific LINCS-landmark decoders are retained only as a molecular endpoint. With the world frozen, their 104,480 parameters improve source-macro and held-double state prediction in both unknown and exact contexts. A sign-invariant geometric ridge then fails all locked continuous-transfer criteria, and a separately frozen simultaneous-versus-sequential score reaches only 0.5124/0.5242 on MuSL CV3. These failures preserve the accepted residual endpoint and prevent Sanger or HAP1 exposure for the new readout.

A DepMap vulnerability-state branch reaches the same boundary from another direction. The full 64-dimensional dependency landscape is only weakly predictable, and a low-rank action adapter damages relation and Perturb-seq preservation. Restricting the target to the first 16 variance-ordered dependency modes succeeds molecularly: generic held-gene Huber improves 5.76%, and the untouched scenario-3 set improves 16.16% with cosine 0.429. The world remains frozen and the 18,832-parameter endpoint is retained. Its zero-initialized continuous correction nevertheless rolls back to epoch zero because every trained epoch worsens both-genes-new depletion. No SL benchmark is opened for this endpoint.

An explicit pair-transition repair also fails before outcome fitting. A zero-initialized symmetric 4,368-parameter adapter is the only trainable module over 404 measured double-perturbation pseudobulks and is never invoked for single interventions. Its selection set contains four pairs with both genes absent from fitting. Epoch zero remains best at 0.54883 mean held-double Huber, and all 30 trained epochs worsen that value. Because molecular transfer does not improve, no continuous decoder or SL benchmark is opened. The present local corpus contains only 102 distinct fitting pairs across three sources, making diverse simultaneous-intervention measurements the clearest remaining data limitation.

A constrained four-scalar mixture of simultaneous, order-averaged sequential and additive single-intervention state predictions gives only a 0.23% held-double Huber improvement. The selected mixture uses no sequential component and modest additive correction, but misses the 2% molecular requirement. It is rejected before outcome fitting and reinforces the same data limitation without changing the retained model.

A 128-parameter diagonal calibration trained on 5,865 single perturbations also selects its identity initialization: every trained epoch worsens both the combined held expression score and held-double loss. Relation and dependency state are consequently unchanged, but no downstream evaluation is authorized. The missing mechanism is not a shared rescaling of frozen gene actions.

A 2,048-parameter low-rank action rotation is trained and selected using single perturbations only, leaving held double state unopened until afterward. It produces a negligible 0.038% held-single gain and slightly worsens held-double Huber. Relation and dependency state remain within tolerance, but the candidate stops before outcomes. Better single-action geometry alone does not yield the missing composition.

The released GEARS-Norman checkpoint is excluded because its learned perturbation and gene lookup tables were fitted without intervention isolation. A safe 3,888-parameter reconstruction of its explicit pair-fusion principle improves held-double Huber by only 0.012% and is rejected. The local pair-state corpus is now the limiting evidence: 102 fitting pairs across three sources and only four both-genes-new selection pairs, all from Norman. Further selection on these same pairs would not support a credible cold-start claim.

A public-data branch avoids that limitation by using no measured genetic double perturbations. Cancer-type- and mutation-burden-adjusted TCGA PanCancer mutual exclusivity is reproducible across deterministic patient halves. A 33,537-parameter decoder over frozen accepted gene actions reaches 0.17604 Pearson / 0.17418 Spearman on 749,700 relations where both genes were absent from fitting. Its independently measured Spearman fixes one low-weight rank fusion with the retained dual-world depletion score. Across both official MuSL CV3 seeds, the preregistered fusion improves label-free development performance from 0.63197/0.62823 to **0.64170 AUROC / 0.63501 average precision**. No double-perturbation data or SL label fits the decoder or its weight, and no post-exposure variant is evaluated. This becomes the strongest label-free repository readout, but is retrospective and below label-free SOTA.

Adding a training-gene-only 128-dimensional PCA of provenance-safe input state to that frozen decoder was rejected. Its two-held-gene correlations were 0.18963 Pearson and 0.18464 Spearman, below the preregistered 0.19604 and 0.19418 thresholds. No SL labels, double perturbations or benchmark scores entered fitting or selection.

A 768-wide, 256-state, eight-layer transformer tests the original scale hypothesis at 59.64M pretraining parameters. After the same 12 epochs it reduces held biological-relation Huber from 0.018922 to **0.013212**, a 30.17% improvement. Downstream dependency Huber also improves slightly to **0.385035**, but the combined held Perturb-seq score reaches only **1.865409**, a 0.22% gain rather than the preregistered 1% requirement. A single capacity-adjusted learning-rate correction worsens that score to 1.869647 and triggers its registered no-more-variants stop. The full scaled checkpoint is rejected, and no SL benchmark is opened; scale clearly helps relation pretraining but not the present single-perturbation objective.

Correction: those two downstream scaling runs were not single-perturbation-only. The loaded pack contains 404 fitting and 16 validation doubles, and the executed trainer included them. Only the relation-pretraining stage is double-free. A new matched compact-versus-scaled branch filters all 420 double rows before sampling and validation; its results are reported separately.

In that clean comparison the compact and scaled single-only scores are 0.756841 and 0.756322. The 0.0685% gain misses the registered 2% capacity threshold despite preserved dependency and relation state, so the scaled single-only world is rejected without SL evaluation.

The frozen scaled world nevertheless improves the established deterministic fold-local MuSL learner. Across both official CV3 seeds, the unchanged readout reaches **0.84845 AUROC / 0.85048 average precision / 0.76772 F1**, versus 0.83768/0.84186/0.75743 for the prior compact representation. The world receives no SL update and its molecular training is double-free; the readout itself uses each benchmark training fold, so this is supervised development rather than emergent label-free performance.

Six preregistered public-relation features add complementary supervised signal without altering the frozen world: split-half mean, disagreement and support for DepMap co-dependency and TCGA adjusted mutual exclusivity. The same ten-fold learner reaches **0.86403 AUROC / 0.86461 average precision / 0.78347 F1**, improving all three unaugmented scaled metrics. The relation sources contain no genetic double-perturbation measurements, and no feature or sign variant was tried after exposure. This was the strongest fully cold-start supervised development result before the expression-silencing extension below, but its fold-local MuSL labels preclude an emergent or label-free claim.

Public DepMap single-gene CRISPR profiles define an even more reproducible relation across two disjoint cell halves: 0.53100 Pearson, 0.48022 Spearman and 32.695-fold top-tail overlap over two million pairs. The source is admitted, but the accepted frozen actions do not expose it sufficiently for cold genes. A compact decoder reaches only 0.14895 Pearson / 0.06667 Spearman on 1,638,955 pairs where both genes were absent from fitting, below both preregistered 0.15 thresholds. No SL benchmark is opened for this branch.

Symmetric expression-to-partner-dependency correlation is another reproducible DepMap relation that uses neither SL labels nor genetic double perturbations. Across disjoint cell halves it reaches 0.47715 Pearson, 0.42156 Spearman and 31.015-fold top-tail overlap over two million pairs. Its fixed positive score is nevertheless anti-aligned with MuSL at 0.48369 AUROC / 0.50673 average precision. Weighting it by its independently measured reliability lowers the retained label-free fusion to 0.60641/0.60195. The relation remains an admitted molecular source, while its SL transfer is rejected without post-result sign or weight changes.

A direct functional-redundancy conjunction is stopped even earlier. Protein similarity, four safe GO/PPI spectral views and reproducible co-dependency fail three of four locked cross-view agreement criteria over two million pairs. No SL benchmark is opened, preventing sequence or graph resemblance from being rewarded merely because it is a plausible synthetic-lethality heuristic.

The scaled single-only world also fails to improve inductive TCGA relation recovery. An unchanged 66,817-parameter decoder reaches 0.15509 Pearson / 0.15642 Spearman on 749,700 two-unseen-gene pairs, below the compact decoder's 0.17604/0.17418. It stops before MuSL. The scaled representation's supervised CV3 advantage is therefore not evidence of better label-free tumor-relation encoding.

Pooling both disjoint TCGA patient halves as a denoised training target does not clear a cross-half replacement rule. The compact decoder improves half 1 and their mean, but reaches only 0.19516 Pearson / 0.19759 Spearman on half 0, below the locked 0.19823/0.20049 minima. The scaled decoder misses all six half-specific and pooled thresholds. Neither candidate is evaluated on MuSL, and no aggregation or model variant follows.

Lineage-adjusted expression silencing supplies another reproducible public relation without double perturbations: disjoint DepMap halves reach 0.23578 Pearson, 0.21142 Spearman and 9.13-fold top-tail overlap. Neither compact nor scaled frozen actions decode it for two unseen genes, so it is excluded from the label-free readout. Its three fixed relation features nevertheless raise the unchanged fold-local supervised scaled-world result to **0.86523 AUROC / 0.86560 average precision / 0.78684 F1** across both official MuSL seeds. This is the strongest supervised development result in the repository, but it uses MuSL training-fold labels and is not emergent.

A 41,729-parameter bounded integration then compresses half-0 silencing profiles through 256 non-held anchors and permits at most a five-percent action residual. The selected adapter preserves all accepted relation, dependency and Perturb-seq state metrics, but reaches only 0.06387 Pearson / 0.04773 Spearman on 1.78 million two-unseen-gene half-1 relations. It is rejected before continuous interaction fitting or SL evaluation. The source signal is useful to a fold-local learner but is not recovered as a small transferable action residual.

The official Costanzo yeast SGA map adds 958,232 human-mapped consensus interactions through strict Alliance one-to-one orthology without using a human double-perturbation screen. Reciprocal directions agree at 0.34348 Pearson and show 35.914-fold overlap in the strongest negative tail, but full-rank Spearman is 0.16861, below the preregistered 0.20 source criterion. Continuous cross-species supervision is rejected before model fitting; no tail-only redefinition or SL evaluation follows.

The public SE-600M and State Transition Tahoe checkpoints are now executable locally as a frozen expression-to-drug-transition pipeline. Exact released preprocessing normalizes the supplied log-expression values directly. On a separately fixed 64-drug, 48-cell panel, Tahoe transition features predict 717 fully held-out PRISM drug-cell measurements at **0.37829 Pearson / 0.23662 Spearman / 0.79890 bottom-quartile AUROC**. No SL labels or double perturbations are used. This source is admitted as pharmacologic state evidence, not as a gene-knockout or SL model.

Direct Tahoe translation to cold genes is rejected. Only 0.34% of MuSL CV3 pairs have direct target profiles, and the fixed spectral-safe gene-to-Tahoe projection has -0.08601 held-drug and -0.17371 target-isolated cosine.

A separate bridge uses public Replogle K562/RPE1 single-gene CRISPRi responses with frozen SE-600M protein tokens. Training-gene-only cross-fitted shrinkage confirms on untouched genes at 0.32644 source-macro cosine and 13.40% Huber improvement, so its 64 action coordinates are admitted as a molecular single-gene endpoint. Their independent pair relation is null, however, and a dependency-core residual fails preservation. MuSL remains unopened for every new branch, so the retained label-free and supervised benchmark results are unchanged.

The confirmed action coordinates add only 0.00266 Pearson and 0.00290 Spearman to the two-unseen-gene TCGA decoder, below the registered 0.01 gains, and exact full-dimensional State protein similarity does not recover a shared functional-redundancy relation. Both stop before MuSL. One separately locked label-free test of admitted expression-silencing vulnerability also fails: its fixed reliability-weighted fusion reaches 0.63442 AUROC / 0.62728 average precision, below the retained 0.64170/0.63501. No sign or weight variant follows. The headline models remain unchanged.

A leakage-audited fusion asks whether the confirmed SE/Replogle action coordinates complement the 59.7M world's broad full-transcriptome endpoint. Genes used to fit either representation are removed from confirmation. On the remaining 20 genes and 84 profiles across all five assays, a capacity-matched augmented endpoint improves unknown-context Huber by 1.28% but slightly worsens exact-context Huber and reduces macro cosine by 0.033/0.047. It is rejected before pair modeling, reinforcement learning or CV3; source-specific SE action coordinates are not a universal perturbation basis for these assays.

A compact shared gene-state encoder then jointly models independent co-dependency, tumor mutual-exclusivity and expression-silencing halves for 5,614 common genes. Although later epochs improve held-gene TCGA correlation beyond the accepted action decoder, expression-silencing transfer stays near zero and degrades the common molecular loss. The preregistered single-epoch selector retains epoch 1, which misses every all-source advancement rule. No source is dropped and no relation-specific epoch or SL evaluation follows.

Neither the compact nor 59.7M single-only transformer linearly retains the admitted SE/Replogle action on unseen genes: held-gene cosine is 0.18043 and 0.21763, respectively. The registered representation rule therefore stops action distillation and cloud training. Feeding those action coordinates directly into the admitted five-source context transition produces a promising macro gain, 0.380586 to 0.391687, but fails source preservation because Dixit decreases by 0.02193 against a 0.02 limit. No epoch substitution, architectural retry or SL evaluation follows.

SLMGAE's bundled 332-gene BC pathway and protein-complex matrices are not used. They are consumed in a benchmark pack beside explicit SL labels and lack an independently established mapping and primary provenance suitable for a label-free biological prior.

A source-preserving action sidecar is now admitted for molecular state prediction. Source-specific alpha-10 ridges, tangent corrections capped at 0.25 and gates chosen only from fitting-gene out-of-fold cosine raise held five-source macro cosine from 0.380586 to **0.391133**. The gate is zero where the action is not reproducibly useful, preventing the prior Dixit regression. This result uses public single perturbations only.

The corresponding fixed dependency-change score reaches only **0.59707 AUROC / 0.61097 average precision** on fully cold-start MuSL CV3. It is rejected without readout variants and does not change the headline label-free or supervised results.

Public interaction data are now locally reproducible from Horlbeck 2018 and SPIDR 2025, and an independent single-gene fitness panel is locally reproducible from Sanger Project Score. They are not training inputs to the retained model. Fixed audits reject Horlbeck for cross-cell rank transfer, SPIDR for current-universe coverage and Project Score co-dependency for cross-center rank transfer. DEMETER2 RNAi co-dependency is likewise excluded because it does not agree with CRISPR co-dependency. These negative controls prevent stable study-specific structure or benchmark overlap from being converted into apparent cold-start gains.

An auxiliary full-transcriptome single-intervention endpoint is admitted. Five source-specific 32-dimensional projections retain approximately 5,000 measured genes per Perturb-seq assay. Over the frozen 59.7M world, the endpoint reaches 0.20475 unknown-context and 0.19779 exact-context macro cosine on untouched interventions, with positive cosine in every source and no double-perturbation or SL supervision. Its exact-context pair geometry independently predicts DepMap co-dependency for 222 jointly held genes at 0.27808/0.26306 Pearson/Spearman on one cell half and 0.24617/0.22959 on the other.

This molecular advance does not replace the retained benchmark readout. Adding its one preregistered response-cosine feature lowers supervised MuSL CV3 from 0.86523 AUROC / 0.86560 average precision / 0.78684 F1 to 0.86427 / 0.86484 / 0.78424. The feature is rejected without post-result variants; the strongest label-free and supervised results remain unchanged.

A zero-initialized 33,280-parameter transition residual is also rejected: epoch 0 retains the lowest held full-transcriptome Huber. The admitted endpoint therefore remains an auxiliary molecular state decoder and does not alter the retained world checkpoint.

The endpoint also predicts a directed mutual partner-expression relation across Replogle K562 and RPE1. On 231 held pairs the equal-source prediction reaches 0.42458 Pearson / 0.39864 Spearman against observed single-perturbation responses, although only 22 untouched genes support that strict test. A fixed label-free MuSL fusion raises average precision to 0.63638 but lowers AUROC to 0.64049, so it is rejected without variants. The retained headline scores remain unchanged.

A preregistered full-transcriptome REINFORCE continuation does not advance the world model. Training 691,712 transition parameters improves exact-context Huber by only 0.16%, slightly worsens unknown-context Huber, and preserves the accepted relation, dependency and prior single-state losses within 0.15%. Although both endpoint cosines increase, the candidate misses the required one-percent Huber improvement in both contexts. It is rejected without MuSL evaluation or post-result variants; the original checkpoint remains the retained model.

GSE337988 adds a separately admitted single-gene endpoint. After averaging repeated conditions by target gene, an 82,752-parameter decoder over the frozen scaled world predicts 445 untouched genes at 0.20664 unknown-context and 0.20647 exact-DLD1 cosine, improving Huber by about 3.03% in both cases. Its pair geometry is null, however: observed and predicted response cosines remain below 0.026 Pearson or Spearman against either independent DepMap co-dependency half over 95,266 held-gene pairs. No MuSL score is opened, so the endpoint remains molecular evidence rather than an SL readout.

Adding the admitted five-source response profile to a fresh capacity-matched TCGA decoder is rejected before benchmark evaluation. On 7,875 patient-half pairs among 126 genes excluded from both endpoint and relation fitting, the action-only baseline reaches 0.15763 Pearson / 0.14393 Spearman and the augmented decoder reaches 0.15708 / 0.13911. Its slightly lower Huber does not compensate for lower rank and linear correlation. The response profile therefore remains molecular state evidence and is not substituted into the retained label-free TCGA fusion.

A six-source deterministic transcriptome continuation also remains below the advancement threshold. Updating 691,712 transition parameters raises endpoint cosine and preserves accepted relation, dependency and prior single-state losses within 0.09%, but reduces the six-source unknown/exact Huber by only 0.224%/0.200%. GSE337988 changes by less than 0.01%. The candidate is rejected, no reinforcement phase follows, and the original 59.7M checkpoint remains authoritative.

A 66.824M transition-specific variant adds one zero-initialized 768-wide transformer layer only to the action-transition path, leaving static gene encoding and every starting-checkpoint parameter frozen. It improves the five-source full-transcriptome endpoint by about 0.6% Huber but fails the registered six-source, GSE337988, Norman-source and relation-preservation criteria. The adapter is rejected before reinforcement learning or benchmark scoring; the 59.7M checkpoint remains authoritative.

The strongest supervised readout now uses a two-new-gene split nested inside every MuSL training fold to select weights for the unchanged ExtraTrees, LightGBM and neural components. With the 59.7M world and nine public-relation features frozen, it reaches **0.89339 AUROC / 0.89143 average precision / 0.81130 F1** across both official MuSL CV3 seeds. This is a protocol-specific supervised SOTA, not emergent world-model performance; the strongest label-free result remains 0.64170/0.63501 AUROC/average precision.

The first direct Feng CV3 evaluation of the authoritative single-only checkpoint does not show emergence. The frozen simultaneous two-action outcome-strength score reaches **0.46859 AUROC / 0.47745 average precision / 0.47674 trapezoidal PR-AUC** on the five official full-data balanced-random two-new-gene folds. No measured double perturbation, SL fitting, calibration, sign reversal or post-result readout variant was used. This negative result leaves the 59.7M checkpoint unchanged and separates its molecular and supervised value from zero-shot SL ranking.

A fixed LightGBM control over compact-world action products and differences does not improve the admitted unseen-gene TCGA decoder. Its independent patient-half result is 0.16832 Pearson / 0.16811 Spearman on 749,700 pairs of two held genes, below the neural baseline and locked thresholds. It is rejected without SL evaluation, so the label-free readout remains unchanged.

The public Tx1-70M checkpoint is admitted as an executable 70,996,993-parameter source, not as a replacement world model. Static and fixed 32-cell contextual Tx1 gene representations both fail the registered three-relation cold-gene test, principally because expression-silencing correlation remains below 0.019. They are excluded from endpoint fitting, reinforcement learning and SL scoring.

Tahoe-100M target annotations also map 197 inhibitor actions to 198 fixed-universe genes. A fixed 32-context projection fails on target-isolated held drugs at 0.04207 cosine and 0.03500 Spearman. The metadata remain an admitted public source, but no pharmacologic gene-action atlas is included in the model.

## Advancement rule

GPU scaling begins only after a compact feature-only baseline and an ablated transition model are evaluated on fixed folds. A larger model advances only if it improves pair-held-out molecular-relation and quantitative-interaction validation. Readout families and representation weights must be locked before the next confirmatory benchmark; outer-fold-weighted combinations from this development run are labelled exploratory.
