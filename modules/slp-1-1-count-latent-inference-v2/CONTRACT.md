# Count-latent prior inference v2

`Predictor(artifact, device="cpu")` loads a frozen K562 count-latent-state
artifact after verifying the protocol, model, reference, and embedded numerical
core checksums. The artifact must contain `protocol.json`,
`artifact-manifest.json`, `model.safetensors`, `reference.npz`, and
`source/count_latent_state.py`.

`predict(action_features, context_weights, query_indices=None, chunk_size=1024)`
returns the conditional-prior molecular mean in `log1p(CP10k)` units. Action
features have shape `[B, 577]`; context weights have shape `[B, 48]` and are
nonnegative with positive row sums; query indices select the frozen 8,563-query
axis. Empty action features return the exact frozen control anchor for the
specified context mixture.

Cell library size and observed outcome counts are intentionally absent from
the forecast API. They enter the negative-binomial fitting likelihood only and
cannot alter the predicted prior state or molecular mean. This module does not
load split assignments, development outcomes, final holds, or benchmark data.

The model is a conditional latent-state pseudobulk predictor. It is not a
single-cell generator, an identified dynamical system, or evidence that its
latent coordinates represent biological mechanisms.
