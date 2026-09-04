# SLp-1.1 atlas genotype identity inventory

This module extracts only the genotype identities shared between the exact
`ptbs.control` and `ptbs.nacl` frames in Zenodo record `14062629`'s pinned
`ptb_summary.Rdata`. It requires the exact nine-column frame contract and a
non-null integer `cell_number` greater than five for every row. The only frame
values the adapter accesses are `assignment_consensus2` and `cell_number`.

The pure-Python `rdata==1.1.0` parser converts each complete frame, so the audit
does not claim phenotype values were unparsed or uninterpreted by the library.
It records the narrower, testable boundary: the adapter never indexes,
inspects, uses, or emits the seven leverage/`Stucked` phenotype columns.

Candidates are the exact non-`WT` assignments present in both conditions. The
adapter removes one literal `bc-` prefix and performs one exact, case-sensitive
lookup against the pinned current-ORF map. It never normalizes case, resolves a
display symbol, follows a retired redirect, or selects among ambiguous current
targets. Retired status is established only by the separate immutable retired
quarantine artifact.

The held-roster inventory contains one record for each unique current SGD
CURIE. Separate evidence preserves the exact source assignment and every
quarantine classification. Neither artifact contains phenotype outcomes.

This small identity snapshot does not admit the 5.9 GB transcriptomic atlas or
authorize quantitative training. OMF 1.0 also cannot directly feed the output
artifact to held-roster; the exact inventory bytes need a separate
rights-bearing copied `DatasetSnapshot` admission with RunResult provenance.

The dependency lock makes package selection reproducible for CPython 3.12 on
Linux x86-64 with a manylinux2014/glibc-2.17 floor. It contains
`rdata==1.1.0` and all nine transitive packages with one verified distribution
hash each; it has no host path, editable source, VCS reference, or R runtime.
It is not an offline wheelhouse or a portable release closure: execution still
requires acquiring those exact distributions, and release portability remains
blocked until the dependency payload is retained and verified independently.
The module carries a small standard-library implementation of the documented
`omf.module/v1` request/result file protocol because OMF 1.0 does not inject
its controller SDK into non-empty isolated dependency environments. This is
part of the admitted code package, not a host interpreter or `PYTHONPATH`
dependency.
