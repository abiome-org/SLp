# Molecular population-mean auxiliary objective

This application-neutral helper mixes positive expected molecular rates over
measured contexts using supplied metadata weights, then applies log1p. It
compares this population mean with a measured fitting population's log1p mean
using equal-population, equal-query squared error. The normalization scalar
is fixed from fitting data before optimization. No benchmark, split loading,
gene vocabulary, thresholds or model-selection behavior is implemented here.

The caller fixes the source count denominator, population definitions,
sampling weights, assay support, scalar normalization and relative likelihood
weight. The rate mixture precedes log1p; averaging log-transformed context
rates is a different endpoint. Every query passed to this helper is observed.
It does not infer missing targets. Gradients pass through expected rates;
context weights describe the known population composition.

This is a composite molecular training objective when combined with a cell
likelihood. It is not itself a calibrated likelihood or a new generative
model. Improvements in aggregate prediction do not establish preservation of
cell-level likelihood or uncertainty; those require separate evaluation.
