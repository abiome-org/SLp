# Normative sequence-statistics feature-block v1 contract

## Boundary and inputs

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

## Sequence admission and relation semantics

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

## Frozen numerical transform

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

## Output

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
