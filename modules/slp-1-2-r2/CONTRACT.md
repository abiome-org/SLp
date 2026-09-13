# SLp-1.2-r2 implementation contract

This module implements the next inductive experimental predictor. Its model and
ordinary fitting path have passed cloud numerical checks.
No r2 research model has been trained. An admitted source subset is prepared;
Costanzo remains quarantined pending source permission. The staged execution record lives in
[`docs/development.md`](../../docs/development.md).

`model.py` contains a 118,652,163-parameter default transformer. Canonical gene
identities select external, frozen sequence and annotation features; there is
no learned gene-ID table. Context and basal observations condition unordered
interventions. Each requested measurement attends to conditioning and itself,
independently of other queries. Numeric outputs are location and log scale;
human SL has a direct logit. Context caches are inference-only and reject changed
inputs, weights, device or precision. Full correlated-cell generation is outside
this initial implementation.

`records.py` enforces native species, target and outcome semantics, exact pair
resolution, duplicate conflicts and parent lineage. Fold fitting views exclude
every human intervention involving an outer or inner held gene. Pretraining
admits quantitative measurements; adaptation admits human data only. Static
observational features have a separate provenance contract.

`data.py` provides the reference record reader, shared feature joins, train-only
scales and a resumable sampler. Sampling cycles distinct conditions, then
experimental units inside a condition, then measured coordinates inside a unit.
This reference implementation holds records in memory. `packed.py` provides
memory-mapped quantitative records and `population.py` keeps molecular endpoint
matrices dense, storing action metadata once per population. Both apply the
human gene mask before fitting scales; quarantined templates cannot enter fitting.

`train.py` provides processed-score Student-t and binary SL objectives, full
backbone optimization, finite update/time limits, and atomic optimizer/RNG/sampler
checkpoints. `evaluate.py` keeps average precision and trapezoidal PR-AUC separate
and selects only from complete, declared inner-fold benchmark evidence.

Twenty-five focused local tests and the real-data CUDA readiness runs have
passed. The latest run exercised all 60 inner/outer exposure masks, both fitting
phases, direct feature baselines and an 807 MB bundle with real basal profiles.
The earlier check verified exact optimizer continuation, SL backbone gradients
and context caching. Mixed human/yeast molecular and DepMap data have also been
exercised.
The 1,424,919,131-byte full optimizer checkpoint was restored from R2 and
continuation reproduced the complete saved state bit for bit. Costanzo is
excluded from the launch corpus; its admission is not a launch prerequisite.
Research training remains on hold. Native RunPod execution must be
described as native RunPod; it is not an admitted OpenFoundry executor merely
because the project uses OpenFoundry.

`fit.py` supports mixed- or human-only quantitative pretraining, human SL
adaptation with optional human replay, and direct feature baselines. Retained
pretraining points are immutable hard links alongside the latest optimizer
checkpoint. `score.py` requires matching checkpoint, fold, static features and
basal-input checksums and preserves every official row. Outer scoring requires
an explicit flag. The feature MLP has 2,928,009 parameters; the five-parameter
sequence-similarity baseline is not an externally curated paralogy annotation.

The finite campaign packet is produced by `scripts/slp12_r2_campaign.py`; that
script only writes plans and recipes. Checkpoint publication/restoration uses
`scripts/slp12_r2_checkpoint_store.py` with exact expiring object capabilities.
The packet generator and checkpoint controller do not launch research compute.
`campaign.py` validates by default. Its explicit research execution path runs
inner comparisons, selects within each outer fold, refits from scratch and only
then accesses that fold's official test labels. The local ticket broker supplies
exact expiring file capabilities and copies only the recovery journal to the Mac.
Stage archives preserve optimizer state, retained-checkpoint aliases and actual
source; completed fold caches are removed only after R2 verification.
