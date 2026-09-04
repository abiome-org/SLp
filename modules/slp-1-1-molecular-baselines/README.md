# SLp-1.1 molecular baselines v1

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
