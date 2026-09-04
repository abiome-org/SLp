# Typed sparse world-candidate contract

This phase-1 module validates `slp.corpus/v1.1`, constructs the corresponding
species-aware query-decoder network, and audits a deterministic sampling
schedule. It does not optimize parameters, select a checkpoint, or emit a
model artifact. A successful run is engineering evidence only and returns
`trainingImplemented: false`.

The OMF input must be one copied, immutable `DatasetSnapshot` with exactly
`resource`, `mode`, `path`, and `manifestDigest`. Both resource and manifest
are SHA-256 pinned. The materialized directory must be the non-symlink path
`.../inputs/corpus/<dataset-name>` matching the resource URI. Bare paths,
mounts, aliases, missing directories, additional fields, and path/resource
substitution fail closed.

## Separation of identity and model inputs

`corpus.json` references checksum-pinned entity, query, and query-panel
dictionaries. Stable CURIEs remain in those dictionaries and per-record
provenance. `WorldBatch` contains no entity, query, gene, or dictionary
identifier: the network receives only numerical features, explicit presence
masks, declared species features, and small entity/context/action/readout type
indices. Reordering or extending a dictionary therefore cannot change the
parameter count or a prediction when the referenced features are identical.

Context and action memory is permutation invariant. Queries attend only to
the shared encoded memory and never to one another. With dropout disabled, the
same marginal query is bitwise invariant to panel membership, ordering,
padding, and chunking.

## Sparse records and likelihoods

Each bounded `.npz` shard contains fixed-width context/action references and a
record-local query panel. Observed targets are represented by
`target_indptr`, `target_query_index`, and `target_value`; the module never
constructs a record-by-global-query target matrix. A readout explicitly
declares Gaussian or negative-binomial likelihood, unit, and whether absence
from the CSR list means an observed zero or an unobserved value. Missing
features and covariates must be stored as numeric zero with a false presence
mask.

Gaussian outputs are mean and log standard deviation. Negative-binomial
outputs are log mean and log inverse dispersion. The latter currently has no
library-size offset, so count-bearing biological training is blocked until a
source-appropriate offset contract is frozen.

## Leakage and sampling boundary

Every active species-specific action entity across all shards must equal the
sorted `trajectoryGenes` inventory exactly. Species-neutral chemical entities
do not enter that gene inventory. All context, action, and query entities must
be species-neutral or match the record taxon.

Sampling uses explicit positive source weights, exact deterministic quotas,
and then cycles source → perturbation → replicate → record with a fixed seed.
Identifiers determine grouping and audit provenance only; they are never
model features.
