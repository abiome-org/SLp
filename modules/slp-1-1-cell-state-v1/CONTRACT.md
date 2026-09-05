# Molecular cell state v1

This application-neutral experimental module encodes paired molecular cell
measurements using caller-supplied RNA and protein query features. It has no
learned identity embeddings, internal gene roster, benchmark handling, SL score,
intervention fitting, time variable or dose model. Query-feature dimensions are
fixed by configuration; query counts and ordering are external data contracts.

The encoder pools measurement-weighted feature keys and measured-panel summaries
into a shared state. Missing inputs are masked explicitly. At least one modality
must be observed for each cell. The reconstruction objective gives equal total
weight to RNA and protein, in externally fitted standardized measurement units.
Input denoising masks must remain separate from target-observation masks.

Each observation head is affine in latent state and generated from query
features. Consequently mean-state decoding equals mean decoded observations.
This avoids a nonlinear-decoder averaging assumption in a future pseudobulk
forecast. It does not make the latent variables biologically identifiable.

`observe_delta` adds a decoded state change to explicit measured controls. A zero
change gives exact control identity. No intervention transition has been fitted
by this module's implementation. Reconstruction of cells from already observed
measurements is not evidence of intervention forecasting or cell generation.

The intended first data source is a separately pinned paired-cell snapshot;
all numerical transforms and training inputs must be specified by its trainer.
Cell reconstruction validation within fitting intervention genes is distinct
from validation on unseen intervention genes. No biological run is complete yet.
