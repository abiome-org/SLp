# Yeast preparation source contract

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
