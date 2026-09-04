# Typed sparse production contract

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

## Model and likelihood boundary

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

## Artifacts and release status

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
