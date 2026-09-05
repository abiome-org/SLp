# Control-anchored aggregate molecular transition

This is a self-contained experimental numerical module. It consumes supplied
static action and query descriptors, measured basal molecular tokens, a control
mean and scale aligned to each requested query, and optional supplied assay or
perturbation-mode descriptors. It learns no gene-ID, assay-ID, mode-ID, species-
ID, or context-ID embedding.

The latent contract is

`state = encoded_basal_state + intervention_delta`.

The action encoder is a permutation-invariant set function. It combines the
mean encoded action with the average elementwise product over distinct action
pairs, giving symmetric pair capacity for multi-action records. Padding is
controlled only by a Boolean mask. Masked nonfinite entries are inert; a
nonfinite valid action is rejected.

The pair summary has its own bias-free projection. A single-action corpus
contains an identically zero pair summary, so its launcher must freeze that
projection and record it as untrained. Multi-action fitting can later enable
the projection without changing the empty-set identity.

The empty action set is an algebraic identity. A Boolean presence gate makes
both latent and molecular intervention deltas exactly zero. Empty rows select
the supplied control mean and control scale directly rather than asking a
learned network to reconstruct them. This applies to zero-width action tensors
and to padded tensors whose mask is entirely false. It does not depend on
seeing control records during fitting.

Query descriptors decode independently from one shared state. Reordering or
chunking queries in evaluation mode leaves corresponding outputs unchanged,
provided each chunk receives the same basal tokens and its aligned control
mean and scale. Basal tokens are separate from requested output queries so the
definition of state does not change with output chunk size.

The output is a diagonal Gaussian for aggregate molecular measurements. It is
not a time-dependent trajectory, mechanistic causal model, single-cell
generator, learned population distribution, or demonstrated combination
model. Symmetric pair capacity permits nonadditivity but does not establish it.
Mean action-embedding pooling also does not force a multi-action prediction to
equal the sum of individually learned molecular effects. A future compositional
revision should sum individually context-conditioned molecular deltas and add a
separately trained symmetric interaction residual; multi-action molecular
training is required before that residual can carry evidence.
No low-rank covariance or cell-level sampling model is included in this first
revision. The control identity anchors the supplied measurement baseline; it
does not infer an unseen context's control state.

The minimum useful experiment is a fixed-split comparison against the existing
transition candidate using the same training-only normalization, action/query
features, assay/mode features, control baseline, uncertainty inputs, optimizer,
and stopping rule. It must first pass exact held-out control reconstruction by
construction, then avoid regression on intervention-gene-macro molecular NLL
and centroid-adjusted profile correlation in every represented context. A
combination claim additionally requires separately held single- and multi-
action records, including strata with one and two held constituent genes.
Until that comparison runs, this module fixes a testable representational
defect but has no empirical performance claim and should not be called an
improved world model.
