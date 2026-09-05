# Paired endpoint state, revision 1

The self-contained numerical source is `paired_model.py`. It uses PyTorch;
it has no repository imports, data loaders or application-specific logic.

Fixed biological features describe action genes and RNA queries. Antibody
queries use fixed assay-component descriptors supplied by the caller. An
antibody panel basis is not a learned intervention-gene vocabulary, and does
not establish extrapolation to unmeasured antibody components.

Controls from RNA and protein have separate feature/value encoders. Each
observed modality contributes equally to the basal state. A shared latent
transition accepts a permutation-invariant sum of intervention tokens.
Separate nonlinear observation functions decode state changes relative to
the same basal state, preserving exact empty-action identity. Query outputs
are independent of other requested queries. Missing control entries are inert.

The model predicts molecular endpoint means. It has no calibrated probability
distribution, no time evolution, and no identified before/after cell coupling.
Supporting multiple action tokens does not establish combination accuracy.
Normalization, fitted feature statistics, shared query amplitudes and assay
identifiers are explicit caller inputs and must accompany a fitted artifact.
