# Joint population world model v4 numerical core

`SharedWorldModel` maps a masked molecular population observation to a fixed
set of latent state slots, applies an exchangeable set of statically described
interventions, and decodes arbitrary queried molecular coordinates. The public
`Config`, `encode`, `transition`, `decode`, `forward`, and `reconstruct`
interfaces are unchanged; `transition_chain` adds an explicit ordered operator
composition without changing parameters. Feature width, state-slot count, mechanism
count, and assay count remain configuration values. The module contains no
learned gene identity or source vocabulary.

The v4 transition retains multi-head set attention and a small residual
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

Training retains direct set prediction and measured-parent continuation. For a
scheduled fraction of parent-selected pair rows, v4 instead encodes basal once,
applies the first action in latent state, and applies the remaining action to
that resulting latent state. Both action orders are sampled. The endpoint is
decoded against the original basal state and receives the same frozen combined
action prior and learned repeated-template saturation used by a direct pair.
Gradients flow through both transition calls; no RNA endpoint is decoded and
re-encoded between them. A one-action latent chain is exactly the ordinary
single transition. A batch-centered direction loss still removes the common
perturbation effect before comparing directions.

The latent chain preserves intervention-dependent state that the v3
decode-to-RNA/re-encode route could discard. It does not turn simultaneous
endpoint combinations into measured temporal trajectories. It is a training
and inference consistency path over the learned operator, and claims remain
limited to the held-combination endpoint evaluations.

Small source priors use leave-descriptor-group-out shrinkage. Fitting combination
rows estimate a bounded template-saturation scalar, allowing a generic response
to be shared rather than repeated per action. The source adapter records this
scalar. A direct set subtracts saturation times (active count minus one) times
the source mean; a continuation from a perturbed background subtracts saturation
times active count times that mean. Empty action sets preserve the observation.
