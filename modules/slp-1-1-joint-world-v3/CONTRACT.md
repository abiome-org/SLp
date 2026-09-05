# Joint population world model v3 numerical core

`SharedWorldModel` maps a masked molecular population observation to a fixed
set of latent state slots, applies an exchangeable set of statically described
interventions, and decodes arbitrary queried molecular coordinates. The public
`Config`, `encode`, `transition`, `decode`, `forward`, and `reconstruct`
interfaces are unchanged from v2. Feature width, state-slot count, mechanism
count, and assay count remain configuration values. The module contains no
learned gene identity or source vocabulary.

The v3 transition retains multi-head set attention and adds a small residual
branch conditioned jointly on each current state slot and the masked sum of
action tokens. The sum is divided by `sqrt(width)`, a fixed architectural
scale, and is never divided by active-action count. It therefore preserves the
difference between one and two active interventions while remaining invariant
to action order. Masked action padding contributes exactly zero. An entirely
empty action set returns the input state bit-for-bit, and the final transition
projection is initialized to zero so the initial neural delta is exactly zero.
The final projection receives finite, nonzero gradients on the first training
step; deeper transition parameters begin receiving gradients after that
projection moves away from zero.

Observation encoding remains invariant to permutations of aligned query,
value, and mask entries. It binds query features multiplicatively to response
relative to basal. When control context is enabled, its unscaled
log2(1+CP10K) value is bound through `tanh(context / 4)`. This fixed scale
avoids near-immediate saturation at ordinary expressed-gene values while
preserving the explicit context mask. Missing values never enter a token.

`forward` returns only the neural decoded change. A trainer or inference bundle
is responsible for the observed anchor, frozen linear action prior,
normalization, context adapters, and serialization. The queried decoder emits
population residual values in its selected assay head; it is not a cell-level
count likelihood or uncertainty distribution.

Training mixes measured-parent observations with stop-gradient model-generated
parents, ramping to 50% of selected parent-conditioned updates. Generated
parents use basal state, static action descriptors and frozen priors; they do
not read the measured parent endpoint. The final observation and decode query
union is generated, while the remaining molecular queries are not consumed by
that update. A batch-centered direction loss removes the common perturbation
effect before comparing directions.

Small source priors use leave-descriptor-group-out shrinkage. Fitting combination
rows estimate a bounded template-saturation scalar, allowing a generic response
to be shared rather than repeated per action. The source adapter records this
scalar. A direct set subtracts saturation times (active count minus one) times
the source mean; a continuation from a perturbed background subtracts saturation
times active count times that mean. Empty action sets preserve the observation.
