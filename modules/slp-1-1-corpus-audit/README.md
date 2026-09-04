# SLp-1.1 corpus audit

This module is the mandatory, fail-closed boundary between three quantitative
corpora and any trainer or molecular evaluator. Inputs are exact copied,
revision-pinned OMF `DatasetSnapshot` objects for pretraining, molecular
validation, molecular final holdout, and the complete global held-intervention
roster. It also requires the two or more protected, outcome-blind
source-inventory DatasetSnapshots from which that roster was formed.

Corpus-audit v1.2 is deliberately reward-disabled. Its config and emitted
attestation both require `rewardEnabled: false`; a `molecularReward` input is an
error rather than an empty, copied, or relabeled placeholder. The module has no
path by which reward outcomes can enter this audit. Opening molecular reward
requires a new versioned audit and consumer contract after deterministic
continuation and rollback controls exist.

The audit verifies outer OMF resource and manifest identities, exact internal
file sets and hashes, corpus roles, rights, zero benchmark labels, bounded NPZ
identity arrays, and exact record-level source, species, and intervention
inventories. It independently recomputes every roster assignment from
`slp-1.1-yeast-global-held-v1\x00<SGD-CURIE>` and rejects role, hash, coverage,
mapping, count, or source-inventory drift. Every validation/final trajectory
must have its matching roster role. Every roster validation/final gene is
excluded from pretraining trajectories.

The protected inventories are parsed again from their exact JSONL bytes. Their
file hashes, record counts, QC sets, mapping identity, and outer DatasetSnapshot
identities must reproduce every source entry in `coverage.json`; their computed
QC-passing intersection must equal the roster exactly. Coverage metadata alone
is never accepted as proof of a global intersection.

A successful run emits exactly one file artifact,
`corpus-audit/corpus-audit.json`, using schema `slp.corpus-audit/v1.2`. Paths and
timestamps are absent, so identical immutable inputs produce identical bytes.
OMF v1 cannot safely pass this newly generated artifact into another stage of
the same workload. Training therefore happens in a later admitted run and pins
the exact previous-run artifact manifest; independent evaluation additionally
requires that file to be admitted and verified as the sole member of its own
rights-bearing `DatasetSnapshot`. No benchmark
input, sibling-stage path, or other same-workload handoff is permitted. Until a
factory policy independently proves the artifact's producing module and run,
the sparse trainer records that missing lineage proof as a release blocker.

OMF 1.0 workload manifests name dataset inputs as `dataset/<name>`. OMF resolves
each alias once during run admission, verifies current and pinned training
rights plus copied artifact bytes, and records the full immutable DatasetSnapshot
revision in the admitted run. A full `omf://...@sha256:...` URI in a stage input
is not materialized by OMF 1.0 and is therefore not used as a workaround.

Gene-set hashes are SHA-256 over canonical JSON arrays of sorted unique CURIEs
(`ensure_ascii=true`, compact separators, no trailing newline). Raw file hashes
cover exact bytes. OMF revisions and outer manifest digests retain their
`sha256:` prefix; internal and gene-set hashes are lowercase hexadecimal only.
