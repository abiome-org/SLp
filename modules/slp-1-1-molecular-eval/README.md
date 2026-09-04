# SLp-1.1 molecular evaluator v2

This application-neutral OMF module is the only component allowed to combine
frozen predictions with held molecular truth. It never loads
synthetic-lethality labels and its output is diagnostic evidence, not the
`MODEL_CARD.md` advancement gate.

The evaluator consumes five distinct, copied, revision-pinned OMF
`DatasetSnapshot` inputs: fitting-only centering, evaluator-only held truth,
the target-free query, a passing `slp.corpus-audit/v1.1`, and the outcome-blind
held roster. Predictions and the exact model checkpoint are separate immutable
artifacts. File-valued artifacts use OMF's `.../payload/payload` materialization
semantics; the checkpoint bytes are hashed and must equal the prediction
manifest's `modelCheckpointContentSha256`. Predictions are one deterministic,
uncompressed tar artifact containing exactly `evaluation.json` followed by
`profiles-000.jsonl`; member names, order, type, size, digest, ownership,
permissions, and zero timestamps are checked before records are streamed.

The query and prediction contract uses exact canonical rows:

```text
profileId, speciesTaxon, sourceId, centeringGroup, perturbationId,
interventionIds, readoutIds, distributionTypes
```

Prediction rows add only `predictionParameters`, aligned one-for-one with the
ordered panel. Gaussian parameters are exactly `{mean, logScale}`;
negative-binomial parameters are exactly `{logMean, logInverseDispersion}`.
Any target, observed mask, undeclared field, missing/extra profile, changed
intervention, changed distribution, or changed/reordered readout panel is
fatal. `perturbationId` is derived from sorted intervention CURIEs and
`profileId` from the natural species/source/group/perturbation key, preventing
aliases or duplicate records from weighting metrics.

The producer prediction and protected truth each bind the exact query
DatasetSnapshot resource URI, its outer OMF manifest digest, and the raw
`query.json` digest. The workload renderer also
records the full query URI and outer manifest digest; the module checks those
against OMF's materialized input. Truth does not carry producer-facing corpus
fingerprints: it is independently admitted and joins the target-free query
only inside the evaluator.

This v2 roster contract is explicitly limited to SGD interventions in NCBI
taxon 4932. The evaluator recomputes every assignment digest, bucket, role,
role count, held-set hash, source coverage identity, and audit binding from the
frozen roster files. Mixed-species evaluation requires a new roster schema.
The independently admitted audit establishes quantitative intervention
isolation; absence of held IDs from centering is checked but never claimed as
sufficient evidence.

Every query readout requires outcome-blind fitting support from the minimum
number of distinct centering perturbations. Non-null truth is scored with the
declared Gaussian or negative-binomial NLL; null truth is excluded from scores,
while predictions must still cover the full preregistered panel. Metrics remain
stratified by species, source, and species-source, with perturbation-specific
centering. Undefined source/species correlations fail the diagnostic.

## Frozen second-run workflow

Render only after the datasets, prediction, and checkpoint artifact are frozen:

```text
python modules/slp-1-1-molecular-eval/render_workload.py \
  --centering-dataset dataset/<centering> \
  --prediction-artifact sha256:<prediction-artifact-manifest> \
  --truth-dataset dataset/<truth> \
  --corpus-audit-dataset dataset/<audit> \
  --held-roster-dataset dataset/<roster> \
  --query-dataset dataset/<query> \
  --query-resource omf://.../datasetsnapshot/<query>@sha256:<revision> \
  --query-manifest-digest sha256:<outer-manifest> \
  --model-checkpoint sha256:<checkpoint-artifact-manifest> \
  --output workloads/generated/slp-1-1-molecular-eval-v2-<frozen-id>.yaml
```

The diagnostic does not establish improvement against frozen baseline NLL,
checkpoint-selection eligibility, benchmark performance, portable inference,
or release compatibility.
