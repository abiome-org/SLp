# SLp-1.1 proteome identity inventory

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
