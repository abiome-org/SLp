# Gene-state molecular core v1

This self-contained module represents a caller-defined gene universe with
explicit per-gene basal and intervention states. Stable identity resolution is
external: rows of the static matrix, basal matrix, action-strength matrix and
sparse adjacency must describe the same fixed universe. The module contains no
learned gene-ID embedding or internal vocabulary.

`GeneStateCore.encode(static_gene_features[N,F], basal_rna[B,N],
basal_observed[B,N], action_strength[B,N], adjacency[N,N])` returns static,
basal-node, global-basal, global-action, global-delta, global-state,
initial-local-delta, two-step local-delta and local-state tensors. Static
features are projected once per encode call and reused. Nonfinite basal values
are allowed only where `basal_observed` is false. Every record needs at least
one observed basal gene.

The sparse graph is nonnegative and row normalized. `adjacency[i,j]` sends from
node `j` to node `i`; isolated all-zero rows are allowed. Two shared residual
message steps use `torch.sparse.mm` on a node-by-batch-state matrix. The local
intervention route therefore reaches at most two graph edges beyond directly
acted-on nodes. The separate global route deliberately permits responses at
disconnected genes, so decoded response locality is not claimed.

`observe(encoded, query_node_indices[Q], control_mean[B,Q], amplitude[Q])`
supports only physical RNA queries mapped to graph nodes by the caller. It
returns a nonlinear changed-minus-basal molecular delta and the control-anchored
mean. Amplitudes must be finite and positive. Empty action strength produces an
explicit zero delta and exact control identity. Query order and chunking do not
change the mathematical result. Version 1 fixes dropout at zero throughout so
the changed-minus-basal subtraction contains no stochastic mismatch.

This is an endpoint representation. It does not identify time dynamics,
causal graph edges, unseen assay components, or cell-level distributions.
Graph construction, source rights, stable-ID joins, normalization, splitting,
losses, calibration and evaluation remain outside the module. Dense basal,
action and local-state tensors scale as `B*N` and `B*N*state`; sparse message
passing avoids materializing a batch-expanded edge list but does not remove
that node-state memory cost.

`profile_synthetic_cuda()` is an explicit opt-in resource check with defaults
`N=24000`, `B=32`, `F=577`, state width 16 and 1,024 decoded queries. It never
runs at import or model construction and must be invoked only after GPU
coordination.
