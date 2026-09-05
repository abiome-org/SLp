# Molecular population forecast evaluation

`evaluator.py` provides array-only numerical functions for an explicit external
evaluation. It does not load or train a world model. The separate experiment
script validates frozen forecasts and assembles source-native count summaries.

The endpoint is `ln1p(mean_cell(10000 * counts / full_native_library))`.
Cells contribute equally within an intervention gene; intervention genes and
queries contribute equally to reported MSE. Control predictions use the same
gene-specific experimental-group proportions, mixing rates before `ln1p`.

The perturbation-specific correlation subtracts each matched control profile,
then independently centers truth and predictions across the evaluated gene
cohort for each query. Pearson correlation is computed across queries for each
gene and averaged over defined genes. Undefined constant profiles remain
undefined and their count is reported. Subtracting an initial row before
centering prevents a repeated floating-point baseline from creating spurious
correlations. MSE uses the original absolute profiles.

The named two-source advancement function belongs to this experiment's external
evaluation; it is not world-model behavior. Its forecast checks alone cannot
establish the separate reconstruction-preservation requirement or release
readiness. Array shape/identity validation does not substitute for the caller's
source and split checks. Synthetic tests verify averaging order, numerical
centering, forecast contracts and the fixed forecast decision rule.
