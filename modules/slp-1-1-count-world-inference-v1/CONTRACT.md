# Registered-panel count prior inference

The artifact-local `Predictor` loads one frozen model arm and one registered
native measurement panel. Its inputs are raw static action features, an
optional Boolean action mask, context mixture weights and optional query
indices. It applies the artifact's saved fitting-action normalizer and returns
prior expected CP10k rates, their `ln1p` transform and Gaussian latent prior
parameters. No perturbed counts, library exposure or application labels enter
prediction.

Artifacts contain relative model, reference, numerical-source and protocol
files with checksums. The training packager copies the count numerical core
beside this loader; no repository-relative numerical import is required.
The reference supplies the complete measured native query panel and positive
smoothed control rates. A query subset changes decoding only; context encoding
continues to use the complete registered reference panel.

An empty intervention has the supplied context-weighted control mean. Mixture
weights combine rates before logarithms. Finite CPU/GPU differences require
the experiment's declared numerical tolerance; exact cross-device equality
is not promised. This version may normalize caller-owned float64 context
weights in place; callers needing to retain those values should pass a copy.

Successful local artifact replay establishes executable tensor-file inference.
It does not establish new-panel transfer, combination validity, calibrated
cell generation, an SL score, or portable OMF release materialization.
