# SLp-1.1 protected proteome observation preparation v1

This self-contained OMF module normalizes exactly one protected yeast
proteome role per run: molecular validation or molecular final. It accepts no
pretraining corpus, reward, model, prediction, or benchmark input and emits
only the selected role's deterministic source-observation archive plus a
preparation audit.

The raw source snapshot contains every source column, so OMF 1.0 cannot claim
column-level storage isolation. The reviewed module behavior is the boundary:
it reconstructs the frozen role partition from stable SGD identities and
numerically converts only the selected role. Training and reward workloads
must never receive the raw source or either protected output.

The output remains an architecture-neutral `slp.source-observation-archive/v1`,
not an `slp.corpus/v1.1`, prediction query, evaluator truth bundle, model, or
performance result. Validation and final runs and DatasetSnapshots remain
separate.
