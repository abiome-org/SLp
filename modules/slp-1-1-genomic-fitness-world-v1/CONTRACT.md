# Genomic fitness world

This module learns persistent functional state and intervention-conditioned
continuous fitness observations. `encode(context, assay)` represents an observed
basal molecular context. `intervene(state, actions)` returns a new state.
`observe(state, query_action)` predicts the quantitative effect of that next
intervention. Empty or masked actions are identity operations. Gene descriptors
are sequence/annotation/network features and frozen RNA-world simulation
signatures; no learned gene-ID lookup is used.

The `capacity` architecture adds actions in a common 256-dimensional functional
state and decodes one shared nonlinear viability landscape. The landscape has
linear, signed quadratic and nonlinear residual terms. The effect of B after A
is `V(z+A+B)-V(z+A)`; the double effect is `V(z+A+B)-V(z)`. This makes path
composition an architectural identity and exposes a genuine state observation
decoder. The earlier `conditional` architecture is retained as development
history. Their weights and architecture names are recorded in each artifact.

The two native endpoints are human DepMap raw CRISPR gene effect (taxon 9606)
and yeast deletion natural-log relative fitness (taxon 4932; fixed floor .05
before logarithm). Different molecular context coordinates and assay scales
have separate encoders; gene encoding, transition and query networks are shared.
Human observations train single interventions. Costanzo NxN deletion measurements
train yeast single effects and both orders of conditional double-mutant effects.
An endpoint double mutant constrains cumulative fitness, not elapsed time or
an experimentally observed sequential trajectory. Higher-order rollouts are an
API capability, not validated higher-order biological accuracy.

Training uses no human gene-pair SL labels, published yeast epsilon values or
p-values. A continuous interaction residual is computed from the three raw
fitness values to train conditional dynamics. Self-supervised reconstruction
of the 642 static descriptor coordinates preserves action information. The
separate application module implements SL scores and any supervised SL decoder.

`prepare.py` streams the compressed source, samples source rows with seed 731
independently of outcomes, maps exact systematic yeast IDs to native SGD IDs,
and excludes every mixed training/held fitness pair. Feature normalization and
context normalization use fitting observations. Some fitness-held genes have
RNA pretraining; original globally molecular-held genes are separately retained.
Data snapshots and real payloads remain outside Git.

Run `verify.py` for immutable state, empty-action identity, transition gradient,
conditional response and serialization checks. `train.py` uses the fixed
development objective recorded in training.json, writes real safetensors and
normalizers, and stops within its explicit local time limit. `fitness_inference.py`
loads those payloads for descriptor-driven inference without repository imports.
