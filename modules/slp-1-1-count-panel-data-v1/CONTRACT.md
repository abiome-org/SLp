# Native count-panel data adapter

This self-contained NumPy adapter reads the versioned human essential-cell
training registry and its checksum-pinned members. It loads only the existing
fitting/control memory maps and fitting sufficient statistics. It does not
create splits, load held count shards, train models or evaluate applications.

Each panel retains its ordered query IDs, stable gene IDs, source-qualified
context IDs and full native count denominator. Counts remain in read-only
uint16 memory maps. Sampling copies bounded rows into float32 batches and
verifies their exact integer sums against their registered libraries.
Controls sample uniformly over GEM groups, then cells; targets sample uniformly
over genes, then exact populations, then cells. Population batches sample
unique fitting genes uniformly and retain their measured context proportions.

Fitting targets are ln1p of the equal-cell mean CP10k. Controls use the fixed
positive half-count pooled reference. The mean-loss scale is the full-fitting
MSE of the control-anchored mean predictor, computed before optimization.
The source schedule and relative loss weights belong to the workload.

The registry and adapter describe the present two human native panels; they
make no species-transfer or deployment claim. Replacing aligned features is
an explicit validated operation that must be recorded by a new workload.
