# gier2020 (mouse AML multiplexed Cas12a epigenetic-regulator pairs, Gier et al. 2020)

**Citation.** Gier RA, Budinich KA, Evitt NH, et al. High-performance CRISPR-Cas12a genome editing for
combinatorial genetic screening. *Nat Commun* 11:3455 (2020). doi:10.1038/s41467-020-17209-1. PMC7359328.

**Files** (`data/raw/gier2020_mmus/`; `gier2020_mmus.fetch.tsv`; URL re-verified by sha256)

| file | sha256 |
|---|---|
| MOESM11.xlsx (Source Data; sheet "Fig 3", panel b: gene-pair expected/observed/p/q/diff) | 9cfe56c0b90cd6520f76e9710613708be150345ac46159e72ca08735ebc02893 |
Other supplements (crRNA libraries MOESM2-5, barcodes MOESM6, signatures MOESM7) not used.
License: CC BY 4.0.

**Design.** opAsCas12a dual-crRNA library against epigenetic regulators plus non-expressed
negative-control genes, murine MLL-AF9/Nras-G12D AML line RN2, depletion over time. Gene-pair
level: expected (additive) vs observed depletion, p, q, diff = expected - observed.
Species `mmus`, context `RN2`, mechanism `CRISPR-Cas12a`.

**Parser.** `eukaryotes_extra.gier2020()`; experimental x experimental pairs only; score = -diff.
Positive q < 0.1 and diff > 0; negative q > 0.25 and |diff| < median.

**Counts.** 210 pairs; **117 positives, 57 negatives**, 36 ambiguous (56% of labelled pairs positive).

**Quality check** (`gier2020_checks()`). Pairs of an experimental gene with a non-expressed control
gene cannot interact, yet they are "called" at 60% (97/163; median diff 0.89) versus 52% for
experimental pairs (median diff 0.87). The diff of experimental pairs is not distinguishable from the
control-gene pairs (AUROC 0.47). The systematic excess depletion is a double-cut/modelling artefact,
not genetic interaction. No replicate-level data are published.

**Recommendation: EXCLUDE.** Implausible hit rate (56%) and calls indistinguishable from
impossible (control-gene) interactions.
