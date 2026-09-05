# SLp-1.1 control transition v4

This revision changes only the molecular observation decoder relative to v3.
Action, control-context, transition, and query encoders retain the v3 topology.

The decoder is

`D(z,q) = linear(GELU(W_state z + W_query q + b))`

with 64 hidden units and no decoder dropout. The reported molecular change is
the shared per-query amplitude multiplied by
`D(basal_state + intervention_delta, q) - D(basal_state, q)`.

An empty or fully masked action set therefore produces exactly zero latent and
molecular change. The decoder consumes encoded static/query descriptors and
contains no learned gene identity, action-presence gate, outcome encoder, or
auxiliary loss. It is a nonlinear measurement-decoder experiment and does not
identify biological dynamics.
