# Joint population world model v1

`SharedWorldModel` encodes a masked molecular population observation, applies an
exchangeable set of statically described actions, and decodes arbitrary queried
molecular coordinates. It contains no gene-ID or source vocabulary. Mode and
assay IDs distinguish CRISPRi from CRISPRa and ln1p mean CP10K from normalized
control-z observations. IDs are fixed as CRISPRi `0`, CRISPRa `1`, ln1p mean
CP10K `0`, Norman control-z `1`, author GWPS control-z `2`, and separately
normalized HepG2 control-z `3`. The last two share a mechanism but retain
different assay heads and original endpoint units.

The transition is an exact identity for an empty action set and its final update
projection starts at zero. Its parameters receive gradients on the first update;
the encoder and decoder remain ordinarily initialized and trainable. `forward`
returns only the neural residual change. The trainer is responsible for adding
the frozen reduced-rank action prior and observed-response anchor. Rank and
regularization are serialized per context; current count priors use rank 16
and the Norman prior uses rank 32.

By default the observation token includes a multiplicative binding between its
static query representation and its scaled response relative to basal. This
retains gene-response cross-moments through masked set pooling: permuting whole
query/value pairs is still invariant, while assigning a response to a different
gene can produce a different state. `bind_observation_values` records this
numerical variant in the serialized configuration.

With `control_context=True`, the encoder additionally receives the control
population's log2(1+CP10K) profile and its measurement mask. These values remain
separate from a control-z endpoint's zero basal anchor. Count-space requests
derive this context from their explicit basal anchor; GWPS/HepG2 adapters carry
the pooled source control context, and callers can override it with an aligned
profile. Missing context is explicitly masked. It is not an observed outcome
of the requested intervention. The configuration preserves compatibility with
earlier three-channel checkpoints.

This module models population endpoints. Its outputs are residual values, not a
negative-binomial cell distribution, paired-cell trajectory, or performance
claim. Static feature width, query count, and action count are configurable.

Portable model directories contain `config.json`, `normalizer.npz`, context
adapters under `adapters/`, frozen reduced-rank priors under `priors/`, and
safetensor world states under `checkpoints/`. `JointWorldBundle.predict` accepts
raw static action features, an action mask, a full native-panel basal array and
an optional observed background. It returns the observed background plus the
summed frozen action prior and the scaled neural state update. Empty-action rows
return the observed background exactly. The standalone CLI accepts `actions`,
`action_mask`, `basal`, and optional `observed` arrays and writes `predictions`
and `query_ids`. Optional `control_context_values` and `control_context_mask`
must be provided together and align with the context's full query panel.
`prediction_supported` marks queries jointly measured in that source's fitting
populations; `supported_query_mask(context)` exposes the same Boolean vector.
Values outside this support are numerical placeholders, not supported forecasts.
