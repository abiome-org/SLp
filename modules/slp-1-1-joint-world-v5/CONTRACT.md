# Joint population world model v5 numerical core

`SharedWorldModel` maps a masked molecular population observation to a fixed
set of latent state slots, applies an exchangeable set of statically described
interventions, and decodes arbitrary queried molecular coordinates. The public
`Config`, `encode`, `transition`, `decode`, `forward`, and `reconstruct`
interfaces are unchanged. `transport_action` moves a learned residual between
encoded molecular-prior anchors without adding parameters. Feature width, state-slot count, mechanism
count, and assay count remain configuration values. The module contains no
learned gene identity or source vocabulary.

The inherited transition retains multi-head set attention and a small residual
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

For direct and measured-parent routes, training encodes the observed RNA once
as `z0` and separately encodes observed RNA plus the complete frozen prior as
`zPriorFinal`. The learned update is transported as
`zPriorFinal + (z0-z0) + (transition(z0, action)-z0)`. The decoder residual is
measured relative to `zPriorFinal`, and the target remains endpoint minus
observed minus total prior. Thus known prior RNA is present in latent state
rather than being added only after decoding. An inactive action returns the
input state exactly.

For a scheduled fraction of parent-selected pair rows, v5 encodes basal once,
encodes the first-action and full-pair prior RNA anchors, and transports the
state from basal to the first anchor and then from the first to the full anchor.
Both orders are sampled. The full prior retains the existing learned
repeated-template saturation; the first-action anchor uses its ordinary single
prior. Gradients flow through both transports and all encoded anchors, with no
RNA decode/re-encode between actions. No learned parameter is added.

Latent transport preserves intervention memory relative to known molecular
prior state. It does not turn simultaneous endpoint combinations into measured
temporal trajectories. Claims remain limited to held-combination endpoints.

Small source priors use leave-descriptor-group-out shrinkage. Fitting combination
rows estimate a bounded template-saturation scalar, allowing a generic response
to be shared rather than repeated per action. The source adapter records this
scalar. A direct set subtracts saturation times (active count minus one) times
the source mean; a continuation from a perturbed background subtracts saturation
times active count times that mean. Empty action sets preserve the observation.
