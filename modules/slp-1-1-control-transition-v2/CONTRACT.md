# Minimal control-anchored transition revision

This revision isolates control anchoring from the capacity changes in
`slp-1-1-control-transition-v1`. Its action, basal-context, query, transition,
and mean decoder topology matches the original successful transition candidate:
one action encoder, one context encoder, one transition MLP, one query encoder,
and one bilinear mean decoder. It has no response-state network, pair branch,
learned scale branch, or identity embedding.

The latent contract is `state = basal_state + intervention_delta`. Empty and
fully masked action sets gate both latent and molecular intervention deltas to
exact zero and return the externally supplied control mean bit-for-bit. Masked
nonfinite padding is inert. Nonempty action sets use sum pooling and are
permutation invariant within floating-point tolerance.

Decoder amplitude is a positive query vector with shape `[Q]`. The API rejects
a batch- or context-indexed amplitude, preventing target-context perturbation
residuals from entering inference. A run may use unit amplitude or one pooled
per-query amplitude fitted only from source-context training outcomes. Record-
specific observation scale is a separate likelihood input and never changes
state or predicted mean.

This model is a diagonal Gaussian for aggregate molecular measurements. It has
no explicit multi-action interaction capacity, additive-composition guarantee,
time dynamics, single-cell generator, or inferred unseen-context control. An
unseen context requires a measured control baseline and control-only basal
descriptor on the same query panel. No empirical improvement is claimed until
the preregistered matched experiment runs.

`inference.py` is the portable fitted-checkpoint runtime. It requires the exact
ordered query roster and frozen control-panel mask, accepts raw static action
features, and reproduces the training feature and basal-token normalization.
The caller supplies the control molecular mean. A new context therefore needs
only its control-only fixed-panel descriptor; no perturbation statistic from
that context enters the shared decoder amplitude. The runtime returns no scale
when `measurement_scale` is omitted. A caller-provided positive measurement
scale is passed through as observation metadata and cannot change the molecular
mean or latent state.
