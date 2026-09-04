# Custodian trust anchor

This directory intentionally contains no production public key. Before this
module can execute a biological training-corpus audit, an independently
controlled key ceremony must add `custodian-ed25519-v1.pub` as exactly 64
lowercase hexadecimal characters plus LF and freeze its raw-key identity and
text-file SHA-256 in `attestation.py`.

The custodian private key must never enter Git, OMF configuration, a dataset,
an artifact, a workload, a log, or this repository. Key rotation creates a new
immutable verifier module version. Runtime inputs and config cannot replace the
compiled trust anchor.
