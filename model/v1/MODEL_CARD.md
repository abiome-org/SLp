# World model v1

`model/v1` is the immutable, data-free implementation of the current
intervention-conditioned cellular world model. It accepts encoded genes,
optional basal context, and a set of interventions; it produces a stochastic
latent endpoint state. It does not load datasets, train itself, select a
checkpoint, or assign an SL/fitness score.

## Contract

- Input gene actions are latent vectors produced by `SLPredict.encode`.
- `transition` preserves the legacy single and paired-action checkpoint API.
- `transition_set` and `rollout` are the forward-compatible action-set API.
- `WorldPrediction` exposes a mean and log standard deviation; application
  modules decide how to decode or score it.

## Versioning

Never change files in this directory to alter the behavior of a released
checkpoint. Create `model/v2` with a new model card, compatibility tests, and
a distinct checkpoint manifest instead. Bug fixes that change numerical
behavior are model changes and therefore require a new version.
