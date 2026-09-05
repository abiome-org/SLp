# Observed-state transition v1 contract

This numerical module retains the control-anchored v3 forecast topology. Its
inference `forward` accepts static action features, query features, a supplied
control mean, a shared training-derived amplitude, observation scales, and a
control-only basal descriptor. It cannot accept an observed perturbed response.

During fitting only, `training_loss` encodes the masked standardized molecular
change with learned query-feature keys. It subtracts the encoder's zero-input
state, so a response equal to the supplied control has an exact zero posterior
delta. The fixed loss is forecast Gaussian NLL plus `0.1` times posterior
reconstruction Gaussian NLL plus `0.1` times normalized latent-state MSE. The
posterior state is detached only in the latent matching term; reconstruction
trains the response encoder and shared molecular decoder.

The model predicts a diagonal aggregate Gaussian. The auxiliary encoder is a
representation-learning device, not evidence of time dynamics, single-cell
generation, causal state recovery, or calibrated transfer to unseen contexts.
Advancement requires held-intervention molecular results under a frozen rule.
