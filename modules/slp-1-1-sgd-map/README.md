# SLp-1.1 SGD stable-ID map

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
