# Single-intervention fitness observation baseline

This retained development baseline maps frozen molecular-world signatures and
action encodings to continuous DepMap gene effects through a context-query
factorization. It has no learned gene-ID embedding or SL-pair training target.
The corpus excludes the global molecular-held human interventions and preserves
the source's fitting-gene and fitting-cell restrictions. Feature normalization
uses fitting observations; unknown measurements are masked.

`phenotype.py` trains and exports the observation head. `forecast.py` generates
fitness profiles using one shared fitting-data reference for trained and random
heads. The later `v2` forecast artifacts correct the earlier model-dependent
reference. `profile_benchmark.py` in the separate SL application module reads
those forecasts with a fixed positive-cosine score.

This component predicts single fitness, without a learned persistent
double-intervention state. It is a baseline for the genomic fitness world,
not the current world-model architecture or an SL performance improvement.
Its full measured results are retained in `docs/results.md`.
