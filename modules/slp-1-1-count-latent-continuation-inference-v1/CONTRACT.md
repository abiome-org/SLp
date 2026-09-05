# Count-latent continuation inference v1

`Predictor(artifact, arm, device="cpu")` loads one registered continuation arm
after checking the protocol, model, shared reference, and embedded numerical
core hashes. `arm` must be a key in `artifact-manifest.json["arms"]`.

`predict(action_features, gem_group_weights, query_indices=None)` accepts raw
static intervention features and nonnegative experimental-group mixture
weights. It returns the conditional-prior population mean in CP10k and
`log1p(CP10k)` units. An explicit all-false action mask gives the exact shared
control anchor.

The interface has no count, library-size, fitted aggregate-target, development,
test, or benchmark input. It is an aggregate molecular-mean approximation, not
a single-cell generator or identified biological mechanism.
