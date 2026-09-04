# Normative static entity-universe v1 contract

## Boundary

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

## Identity semantics

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

## Output

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
