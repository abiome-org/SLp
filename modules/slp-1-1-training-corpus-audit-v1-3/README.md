# SLp-1.1 clean-training corpus audit v1.3

This module replaces the three-quantitative-corpus v1.2 audit **for training
only**. It accepts one optimizer corpus, the outcome-blind held roster, at least
two outcome-blind protected-source intervention inventories, and a detached
Ed25519 authorization from the protected-data custodian. Validation and final
quantitative truth are structurally absent from the interface.

The audit authenticates the exact copied DatasetSnapshot resources and outer
manifest digests, then independently recomputes the corpus content digest,
roster and coverage hashes, inventory-manifest hashes, record-level active
actions, and the held validation/final union. It fails if any held intervention
is active in a fitting trajectory, if benchmark labels are present, if reward
is enabled, or if any signed identity differs from recomputed content.

The signature authorizes the exact composed optimizer corpus for one named
clean-factory identity and 256-bit challenge. It does not by itself prove
one-time consumption, freshness, source-to-corpus lineage, legal rights, or
filesystem isolation. Those remain coordinator, OMF-lineage, rights, and
separate-service controls. A future training consumer must independently
reverify the authorization rather than trust a Boolean field in this report.

The production public key is intentionally absent. Until an independent key
ceremony provisions the source-pinned trust anchor and a provenance-complete
composed corpus exists, every biological run must fail closed. Positive tests
use ephemeral keys outside the module. No private key or signing helper is
stored in this repository.

This module produces deterministic boundary evidence only. It trains no model
and supports no performance, novelty, release, frontier, or SOTA claim.
