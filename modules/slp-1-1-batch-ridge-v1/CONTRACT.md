# Batch-adjusted ridge v1

This numerical baseline fits complete molecular-response rows with a shared
linear feature effect and an unpenalized intercept for each observed source
batch. Its objective is `sum_i w_i ||y_i-X_i W-b_batch||² + alpha ||W||²`.
Sufficient statistics stream across bounded blocks, avoiding materializing all
population-by-query rows together. Positive finite alpha is required.

Inputs are caller-normalized features, responses, nonnegative statistical
weights and source batch labels. Only fitting rows may enter the statistics;
feature normalization, inner-fold selection and all weights must also be fitted
within each training fold. Every output query must be observed in every supplied
row. Missing molecular values must not be replaced with zeros for this module.

Batch intercepts are nuisance effects estimated from fitting interventions.
They are neither wild-type molecular states nor forecasts of unseen contexts.
Prediction for a batch absent from fitting fails explicitly. No gene identity,
gene embedding, application score or held-out outcome enters this module.

The returned batch-only means provide the corresponding no-feature baseline.
Readout comparisons must preserve identical populations, weights and feature
availability, and evaluate perturbation-specific patterns beyond shared means.
Stored coefficients, intercepts and batch labels suffice for inference.

Three checks verify equality with independently constructed augmented weighted
least squares, invariance of feature coefficients to batch-constant response
shifts, bounded-block accumulation, unseen-batch rejection and nonfinite-input
rejection without updating sufficient statistics.
