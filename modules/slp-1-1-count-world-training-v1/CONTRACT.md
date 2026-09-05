# Shared molecular training step

This self-contained module packages the frozen count-state numerical core
and molecular-mean objective with a reusable training-step function. It does
not load datasets, assign gene roles, select checkpoints, fit application
scores or inspect benchmarks. Its outputs retain separate count likelihood
and population-mean losses.

The caller supplies one source-native measurement panel per step, explicit
control contexts, a cell batch and optionally a population batch. Different
panels can alternate while sharing parameters. Their static feature coordinates
must be aligned, their library denominators preserved, and their objective
weights fixed explicitly. Changing the posterior panel changes its pooling;
each resulting source/panel requires its own reconstruction and forecast
evaluation. A larger shared roster does not imply every panel measures it.
Every supplied native control panel must be fully measured and positive;
partial control support is rejected rather than filled from an unobserved
union panel. The population target panel is also fully observed.

The full control table is encoded once within a step and its graph is reused
by cell and population losses. This is valid for the included core's
deterministic context encoder. Never reuse a learned context across optimizer
updates. Population predictions disable dropout while retaining gradients.
All per-submodule training modes are restored even when validation rejects a population
batch. The caller performs backward, clipping, optimizer steps and guards
wall time/memory. All input masks describe actual measured support.

The cell factorization remains the original library-offset latent NB
approximation. Adding molecular mean supervision creates a composite objective;
it does not create a calibrated likelihood or coherent library generator.
The package contains no fitted model or biological performance claim.
