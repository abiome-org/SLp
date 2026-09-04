# Normative v1.3 contract

## Inputs

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

## Authorization

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

## Audit and output

The module verifies the signature before reading the large corpus, then scans
the full sparse action representation and trajectory-gene list. The pretrain
corpus must be molecular, benchmark-free, internally hash-consistent, and have
no active validation/final roster member. Held static feature rows are allowed
when they never appear as active quantitative interventions.

The output directory contains exactly one canonical `corpus-audit.json` using
`slp.corpus-audit/v1.3`. It contains only the pretrain corpus identity,
outcome-blind roster/inventory identities and protected-set hashes, and
authorization identity. It contains no validation/final DatasetSnapshot
locator or quantitative truth identity. Re-running with identical inputs and
authorization bytes must produce identical output bytes.

The signature authenticates the final composed corpus bytes, not their entire
scientific derivation. A biological run additionally requires a
provenance-complete composition report and verified OMF lineage from exact
observation, static-feature, roster, and basal snapshots. The existing
`slp.corpus/v1.1` format does not carry all of that evidence; it is not by
itself sufficient for release. The v1.2 full-custodian audit and current world
trainer remain historical contracts and cannot consume v1.3 evidence. New
versioned consumers must reverify this authorization and its bound identities.
