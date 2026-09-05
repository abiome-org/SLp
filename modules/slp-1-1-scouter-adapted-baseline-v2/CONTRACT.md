# Scouter adapted LayerNorm baseline v2

This is a single engineering correction to the v1 adapted Scouter baseline.
It selects the author implementation's supported LayerNorm option and disables
BatchNorm in every hidden layer. All widths, SELU activations, fixed static
action-feature sum, full output panel and training contract remain unchanged.

The correction addresses the observed incompatibility between BatchNorm and
this study's context-local input: every training row in one model receives the
same pooled control profile, giving exactly zero across-batch control variance.
With dropout disabled, LayerNorm has identical train and evaluation behavior
for the same inputs. This engineering identity is tested directly.

The architecture and MIT attribution otherwise follow Ouyang Zhu and Jun Li's
Scouter implementation and paper, DOI `10.1038/s43588-025-00912-8`. Selecting
LayerNorm is an author-supported API option, but this pseudobulk, static-feature,
exposure-likelihood experiment is an adaptation and not a reproduction of the
published quantitative result.

The v1 limitations remain: the decoder is tied to a fixed molecular panel,
models are context-local, pooled controls replace sampled cells, and singleton
training cannot identify combination interactions. No learned gene identifier
or mutable gene vocabulary is used.
