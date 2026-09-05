# Raw count moments v1

Input is a bounded sparse block of eligible cells by source RNA features, with
integer population indices assigned from source metadata. Counts must be finite,
nonnegative integers. This module does not assign genotypes or train/test roles.

The caller supplies an immutable source-row-to-query map and denominator mask.
Multiple source rows mapping to the same stable query are summed before
normalization. Unmapped biological rows may contribute to the denominator while
remaining absent from output queries. Every queried row must be in the
denominator. Each retained cell has value
`ln(1 + 10000 * summed_query_count / sum_denominator_counts)`.

Zero-library cells are explicitly counted and excluded. Observed zeros in
positive-library cells remain in the estimand. Accumulation is float64 and
weights every eligible cell equally within its population. Output includes
population counts, means and unbiased cell variances; empty means and variance
with fewer than two cells are NaN with explicit support masks. Cell variances
describe dispersion among sampled cells, not biological replicate uncertainty.

There are no fitted parameters, ID embeddings, outcome-dependent filters or
application scores. Query mapping and population grouping remain caller-owned
provenance. Peak accumulator memory is two float64 population-by-query arrays,
plus one bounded sparse input block and returned summaries.

Checks compare against a dense independent normalization oracle, verify duplicate
query collapse before log transformation, denominator-only features, observed
zeros, empty/one-cell support, chunk/order invariance, and reject invalid counts.
