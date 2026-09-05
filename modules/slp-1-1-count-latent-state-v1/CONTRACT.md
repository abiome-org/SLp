# Conditional count-state prototype

This self-contained, untrained numerical prototype represents an intervention-
conditioned Gaussian molecular state and a negative-binomial count observation
model. It makes no biological forecasting, cell-generation, temporal, causal,
cross-species, synthetic-lethality, or release claim.

Static feature vectors describe action genes and query genes. The prior also
consumes measured control rates with an explicit availability mask. It has no
learned gene identifiers, context identifiers or fixed query vocabulary. Actions
form a permutation-invariant set; arbitrary queried subsets share a latent
state. A posterior encoder consumes integer molecular counts for variational
training. It must never be used to produce an unseen intervention forecast.

The observation adapter supplies positive, externally smoothed control rates
in molecules per 10,000 source-denominator molecules. Its exact smoothing and
denominator must be specified in a biological experiment before fitting. The
core invents neither missing values nor a normalization panel. Zero-library
cells are ineligible; observed count zeros are genuine observations. The
library exposure is at least the sum of queried counts and enters observation
likelihood and posterior inference only. It never enters the prior or its
expected molecular rates. Conditional count means need not sum exactly to
the supplied library; this is an offset-based NB2 approximation, not a
multinomial or a coherent joint generator of the library and its constituent
counts. The observation factors ignore dependence induced by conditioning on
the library sum. Empty-action identity is relative to the supplied smoothed
control rates, including any explicitly declared positive pseudocount.

Let the control state be N(m0,V0), the intervention state N(m,V), and W the
queried loading matrix. The conditional log rate is log(basal) + W(z-m0) -
diag(W V0 W')/2. Its analytic population mean is basal * exp(W(m-m0) +
diag(W(V-V0)W')/2). Therefore empty interventions reproduce the supplied basal
mean exactly, while shared Gaussian variation induces cross-query dependence.
Learned dispersion describes additional conditionally independent NB2 noise;
latent variation and dispersion are not identifiable biological mechanisms.
Both prior and posterior log variances are bounded to [-8,4].

For efficient training, encode unique control contexts with `encode_context`
once per optimizer step and index their embeddings into `prior_from_context`.
Do not cache a learned context across optimizer updates. The posterior uses
a fixed full measurement panel; arbitrary mask or panel changes alter its
pooling scale and require separate validation. Decoder query chunking remains
valid. Feature-identical queries necessarily share loadings and dispersion;
the adapter must report static-feature missingness and collisions.

Training returns a one-sample negative ELBO: (sum of observed-query negative
log masses + Gaussian KL) / observed-query count, with beta=1. Reconstruction
and KL are reported separately. The caller specifies sampling, macro weights,
budget, deterministic seeds, and checkpoint selection, and must stop on any
nonfinite loss or prediction. Cell reconstruction alone is insufficient evidence
of intervention prediction; evaluate the prior without perturbed inputs.

Numerical checks cover the NB mass against an independent distribution API,
Gaussian-integrated population means, empty actions, query chunk/order and
action-set invariance, masking, integer-count units, and finite gradients.
Biological execution and portable artifact packaging remain separate work.
