# Decoder of latent state changes

This numerical revision retains the v2 architecture, parameter shapes and
training-independent observation scale. It changes only the molecular decoder:
`mean = control_mean + amplitude * D(intervention_delta, query)`.
The learned latent relation remains `state = basal_state + intervention_delta`.
Because D is linear in the latent argument, this is algebraically
`control_mean + amplitude * (D(state, query) - D(basal_state, query))`.
Direct delta decoding avoids subtraction of two large decoded profiles.

A zero latent intervention delta must produce the control molecular mean for
both an empty action set and a nonempty action with learned zero effect. The
decoder does not receive an action-presence flag. Revision v2 instead decoded
the total state and gated its molecular effect by action presence; consequently
an unchanged latent state could produce a nonzero molecular change.

The revision has no new parameters, feature modality, learned identity or
uncertainty branch. It retains masked set encoding and a shared fitting-only
query amplitude. It does not establish biological accuracy, single-cell
generation, temporal dynamics or combination transfer. Existing v2 checkpoints
remain numerical evidence for v2, and must not be relabeled as v3 models.
A separately frozen training experiment is required before judging performance.
