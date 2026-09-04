# SLp-1.1 proteome observation preparation v1

This self-contained OMF module converts the exact non-imputed Mendeley
`w8jtmnszd9` version 2 yeast knockout proteome into a source-normalized,
pretraining-only sparse observation archive. It is not a world-model corpus:
it contains no learned gene identity, static feature vector, model query tensor,
sampling weight, architecture choice, or benchmark label.

The module independently rebuilds raw metadata ORFs against the pinned current
SGD mapping and requires the resulting intervention multiset to equal the
separately admitted identity inventory. It then excludes every frozen
molecular-validation and molecular-final row before numerical conversion.
Quarantined knockout and analytical-QC columns are also never converted to
numbers. Source-only exact current genes remain fitting-only and do not become
members of the protected two-source roster.

Every DatasetSnapshot is matched by its complete `omf://abiome/slp/...` URI,
revision, and outer manifest digest. Protein relations must retain their exact
typed UniProtKB declaration and resolve only to current CURIEs. The runtime is
closed over hash-pinned NumPy 2.2.6, and Python/NumPy versions are recorded in
both archives and the audit.

Targets are `float32(log2(x))` for finite, strictly positive observed MaxLFQ
relative intensities. Literal `NA` is omitted from CSR observations; there is
no pseudocount, imputation, zero substitution, or knockout-derived centering.
All 1,850 UniProtKB readouts and their exact typed SGD relations are preserved.
Raw filenames and raw ORF strings are not emitted into record shards; neutral
row-derived provenance IDs and stable SGD action CURIEs are used instead.

The separate basal artifact averages log2 values only over the 388 documented
HIS3-complemented biological WT controls. A readout is present only with at
least 311 observed controls. The profile is not subtracted from targets. Plate,
injection, and well indices remain audit-only.

The three outputs are regular files because OMF 1.0 cannot safely import a
producer directory artifact. `observation-corpus.tar` and `basal-control.tar`
use deterministic regular-file members; `preparation-audit.json` records exact
lineage, access boundaries, runtime, counts, and limitations. Both archives are
self-validated for canonical identities, global uniqueness, action/trajectory
equality, provenance, and numerical structure before the complete output
directory is atomically published. A later versioned composer must join an
admitted feature pack and produce `slp.corpus/v1.1`.
