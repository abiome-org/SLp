# SLp-1.1 sequence-statistics feature block v1

This self-contained OMF module builds the first species-aware static feature
block for the SLp-1.1 factory. It uses only the Python standard library and
hand-writes the frozen NumPy v1.0 representation.

The module is deliberately small in scientific scope: protein length and
amino-acid composition provide a deterministic weak baseline against which
later frozen protein-language-model, domain, and phylogeny blocks can be
tested. See `CONTRACT.md` for the normative data boundary and binary format.

The workload template uses OMF-supported `dataset/<name>` references for the
two snapshots and literal content-addressed references for the two mapping
artifacts. Running it requires a supported Linux OMF executor and a clean Git
state. The module must be admitted and tested before any biological run.
