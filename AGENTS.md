# Operating the SLp Open Model Factory

## Objective

Build a perpetual, reproducible factory for species-aware genomics world
models. The primary scientific question is whether intervention-conditioned
molecular-state modeling transfers to unseen intervention genes and contexts.
Synthetic-lethality prediction is a downstream stress test, not the world
model's training target or identity.

SLp-1 is frozen proof-of-concept evidence. Do not treat its architecture,
9,845-gene universe, feature concatenation, decoder family, data mixtures,
thresholds, or benchmark results as defaults for SLp-1.1.

## Begin every substantial run

1. Read root `MODEL_CARD.md` and the latest section of `docs/results.md`.
2. Inspect bounded OMF state with `omf doctor`, `omf agent context`, and
   `omf agent capabilities` on a supported Linux host.
3. State one falsifiable hypothesis, one fixed advancement rule, the accessible
   modalities, and the exact data snapshots before allocating compute.
4. Prefer the smallest run that can reject the hypothesis.

Git holds code and versioned configuration. OMF artifact stores hold datasets,
checkpoints, model payloads, and releases. `.omf/` is untracked runtime state;
never edit or commit it.

## Scientific boundaries

- Prevent leakage before optimizing performance. For intervention-isolated
  claims, exclude every quantitative trajectory involving a validation or test
  intervention gene, not only exact pairs.
- Keep pretraining, molecular validation, molecular reward, molecular final
  holdout, and benchmark data in separate OMF `DatasetSnapshot` resources with
  separate rights and access boundaries.
- Never use benchmark test labels for architecture, feature, target, sign,
  calibration, checkpoint, early-stopping, or hyperparameter choices.
- Select the world model and label-free readouts only on fixed molecular
  objectives. Open an external benchmark after the candidate and decision rule
  are frozen.
- Keep world-state predictions, quantitative molecular decoders, label-free
  application scores, and fold-local supervised readouts in separate modules
  and reports.
- Compare against mean, additive, linear, degree, feature, and published
  baselines on identical splits. Report every fold, seed, source and species,
  not only averages.
- Treat retrospective evidence as retrospective. A state-of-the-art claim
  requires a locked protocol and untouched or prospective confirmation.
- Pin or checksum raw inputs, splits, feature packs, modules, runtimes, and
  checkpoints. Record exclusions, populations, endpoints, normalization, and
  accessible feature modalities.
- Remove systematic average perturbation effects in evaluation and measure the
  perturbation-specific landscape; ordinary control-referenced correlation is
  insufficient evidence.

## Species-aware data policy

- Preserve stable source identifiers and NCBI taxonomy IDs. Never merge genes
  across species by display symbol.
- Keep yeast measurements species-native. Orthology and sequence similarity
  are auxiliary relations, not permission to relabel a yeast phenotype as a
  human outcome.
- A cross-species claim needs within-species held-gene results and a separately
  defined transfer test. Aggregate gains cannot conceal regression in one
  species.
- Static sequence, protein, annotation, and phylogeny features may cover held
  genes. Quantitative intervention outcomes for held genes may not enter
  fitting or reward.
- Every imported dataset needs verified training rights. Redistribution rights
  are separate. A rights file with `trainingAllowed: false` is a deliberate
  quarantine, not an invitation to bypass admission.

## OMF work loop

1. Add or verify rights-bearing data snapshots with `omf data add` and
   `omf data verify`.
2. Put executable behavior in a self-contained `modules/<name>/` directory,
   semantic stage graphs in `workloads/`, metrics in `evaluations/`, physical
   placement in `bindings/`, and authorization in `policies/`.
3. Validate the module, workload, evaluation, executor preflight, and clean Git
   state before a run. Never fall back silently to another executor.
4. Run the corpus audit before training. A failed audit ends the run.
5. Inspect terminal run state, outputs, lineage, immutable evaluation results,
   and the pinned baseline before making another change.
6. Append the hypothesis, protocol, resource revisions, hashes, compute,
   source/species metrics, interpretation, and decision to `docs/results.md`.
   Update `MODEL_CARD.md` only when supported intent or claims change.
7. Correct OMF knowledge with a superseding immutable assertion; do not rewrite
   prior evidence.

## Engineering rules

- OMF module source must be self-contained because admission packages only the
  module code root. Do not depend on repository-relative imports outside it.
- New numerical behavior creates a new versioned module or immutable module
  revision. `model/v1/` remains frozen historical code.
- The world module must remain application-neutral: no benchmark names, split
  loading, SL scoring, benchmark fitting, or application thresholds.
- A training workload must not reference benchmark snapshots. Benchmark
  evaluation is a separate explicit workload after lock.
- Gene identity is data provenance, not a learnable shortcut. New world models
  may not require a learned ID embedding or fixed gene vocabulary.
- Keep the repository lean. Prefer bounded, shard-streaming formats and focused
  modules over monolithic runners, accumulating flags, or copied frameworks.
- Preserve deterministic seeds and add focused checks whenever split,
  exclusion, normalization, rights, or benchmark logic changes.
- Do not commit real data, results, checkpoints, weights, caches, credentials,
  `.omf/`, or `ontology/`. Tiny synthetic contract fixtures are allowed only
  under `data/fixtures/`.
- Keep `docs/` limited to `litreview.md`, `model-card.md`, and `results.md`.
- Write like a scientist: name dataset, organism, population, endpoint,
  uncertainty, and limitation. Avoid claims implied only by scale.

## Release rules

- Promotion requires passing molecular evaluation, compatibility evidence,
  complete lineage, current rights, vulnerability evidence, signatures,
  policy approval, and an independent approver. Never fabricate or self-approve
  evidence.
- Use immutable artifact paths and never overwrite a release.
- Review exact local and destination paths before upload. Never put a token in
  a command, manifest, log, or tracked file.
- OMF 1.0 does not materialize a large model artifact into its inference
  adapter. Until a tested upstream contract closes that gap, training evidence
  may be retained but no SLp-1.1 model may be promoted as portable or
  deployable. Do not use absolute run paths or metadata-embedded weights.

## Compute

Local compute is one RTX 4070. Run locally without asking only when the job
should finish within one hour. For longer work, use an admitted remote executor
within the assigned allowance or ask first. Stop scientifically invalid runs
as soon as the fixed advancement rule cannot pass.

## Completion

A change is complete only when its OMF resources and module contracts validate,
focused tests pass, leakage constraints are checked, outputs are reproducible
from pinned inputs, and `docs/results.md` states what the evidence does and does
not support. Code-only changes need the narrowest practical syntax and smoke
checks.
