# Proteome composite-corpus v1.2 contract

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
