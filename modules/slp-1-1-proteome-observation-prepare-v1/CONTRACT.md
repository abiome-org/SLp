# Contract

## Inputs

The run accepts exactly four immutable copied DatasetSnapshots and two exact
file artifacts:

- raw proteome release `slp-1-1-proteome-raw-v2`;
- outcome-blind proteome intervention inventory;
- typed proteome protein-relation inventory;
- frozen cross-source held-intervention roster;
- SGD current-ORF JSONL artifact;
- SGD mapping-manifest artifact.

Names, revisions, outer manifest digests, artifact manifests, internal content
hashes, source release, taxonomy, mapping identity, raw file set, and internal
file hashes are fixed. Full `omf://abiome/slp/...` resource URIs, OMF request
objects, and materialized paths are checked literally. Symlinks, foreign
namespaces, mutable references, extra inputs, extra files, and legacy
file-artifact paths fail. The exact typed UniProtKB accession object and current
SGD relation targets are revalidated, including the frozen nuclear and
mitochondrial systematic-name forms in the current-ORF object set.

## Partition and access

The module prepares only `pretrain`. It admits every exact current proteome KO
row except rows whose stable SGD action is a frozen molecular-validation or
molecular-final identity. This produces 3,811 records over 3,679 intervention
genes. All 537 validation rows, 275 final rows, 76 quarantined rows, and 389
analytical-QC rows are tokenized by the RFC CSV reader but never converted to
numbers, inspected as outcomes, validated as outcomes, or emitted.

The admitted intervention inventory intentionally lacks sample locators, so it
is never order-zipped to matrix columns. The module independently reconstructs
the raw sample mapping through the pinned current-ORF artifact and requires its
complete intervention multiset to reproduce the admitted inventory exactly.

## Numerical protocol

Observed targets must be finite and strictly positive before the fixed
`log2` transform. `NA` means unobserved and is omitted from CSR. The output has
6,865,493 observed and 184,857 missing pretrain values. It retains absolute
log2 values in the frozen source space; additional centering is `none`.

Only 388 documented HIS3 controls are decoded for the separate basal profile.
The mean is calculated after log2 independently per readout, and at least 311
controls are required. Exactly 1,843 of 1,850 readouts pass. No KO or QC outcome
can affect this artifact.

NumPy 2.2.6 is hash-pinned in the module lock. Runtime Python implementation,
Python version, and NumPy version are embedded in both archives and the audit.
The basal archive also binds the full input provenance and a canonical digest
of the exact 388 control row/column locators.

## Outputs and non-claims

`observation-corpus.tar` implements `slp.source-observation-archive/v1` with
stable identities, one assayed panel, source records, observed-value CSR, and
audit-only technical covariates. `basal-control.tar` implements
`slp.basal-control-profile/v1`. Neither is an admitted DatasetSnapshot until a
separate rights review and OMF admission. Neither is a model, feature pack,
benchmark, molecular metric, or SOTA result. Validators rederive archive
identity populations, shard uniqueness, action/trajectory equality, CSR
integrity, basal support masks, and count arithmetic. The final three-file
directory appears only through an atomic same-filesystem rename.
