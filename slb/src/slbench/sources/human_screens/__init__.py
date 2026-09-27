"""Human combinatorial screens added in the 2026-09 expansion, one module per source.

Each module exposes `load()` (MEASUREMENT_SCHEMA rows via `finalize`) and, where the data allow it,
`replicates()` (per-replicate / per-map GI used by the inclusion checks). The inclusion evidence for every
source (replication, cross-study, fitness confounding, licence) is in data/interim/new_human/<key>.{json,md}
and summarised in the SLB README; sources that failed are listed in build.EXCLUDED_SOURCES.
"""
