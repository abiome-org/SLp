# Building SLp with Open Model Factory 2

## Objective and working style

Build an SL predictor targeting above-SOTA scores across the combined benchmark
suite, with CV3 (both genes withheld) as the primary objective. Achieve this by
learning perturbation consequences in human and nonhuman cells. World-model
quality must contribute to human SL generalization; generic molecular loss is
not the final selection target.

The user explicitly requires both outer-test genes to be absent from all SL
label fitting and all human perturbation fitting, across sources, in that fold.
Human observational/static features remain available subject to provenance;
nonhuman perturbations remain species-native pretraining data. Post-training
and application selection use human data only. Use fold-specific exposure
masks; withholding SL edges alone is insufficient. Architecture and selection
for the next model are derived in `docs/development.md`.

The user explicitly requests a Bitter Lesson design: prefer one general
conditional experimental predictor and end-to-end human SL adaptation. Treat
biological knowledge as admissible data rather than mandatory biological
modules. Four-condition simulations and interaction equations are useful
diagnostics, not a required bottleneck for SL prediction. Scale data coverage,
representation learning and compute based on human inner-CV3 transfer.

User intent and host instructions take precedence over this guide. Carry
already authorized work through implementation, training and verification.
Resolve routine choices autonomously and give regular, concrete status updates.
Organize work around a coherent model and usable artifacts. Evaluation supports
model development; do not make a new hypothesis document, approval ceremony or
standalone gate a prerequisite for each routine improvement. Use development
feedback to improve the model and report that reuse honestly.

SLp-1 remains frozen historical evidence. Reuse ideas supported by its code and
results, not its fixed gene universe, feature layout or benchmark conclusions.
Do not modify `model/v1/` or overwrite historical results and releases.

## OMF 2 workflow

Read `MODEL_CARD.md` and the latest section of `docs/results.md` before substantial
work. The exact upstream runtime is pinned in `omf-version.json`.
Run `bash scripts/bootstrap_omf2.sh` on Linux with Python 3.11/3.12 to install it.
From the project root, use the launcher to pin child Python processes as well:

```sh
bash scripts/omf2.sh doctor
bash scripts/omf2.sh agent context
bash scripts/omf2.sh agent capabilities experiment.run
bash scripts/omf2.sh experiment run experiment.yaml --candidate rank32
bash scripts/omf2.sh experiment list
bash scripts/omf2.sh experiment review <run-id> --baseline <baseline-id>
bash scripts/omf2.sh experiment reproduce <run-id>
bash scripts/omf2.sh experiment export <run-id> --to <new-model-directory>
```

Global `--project` and `--actor` options precede subcommands. The actor defaults
to the configured owner. Use the existing identity; never create a substitute
to work around a denial. Existing `omf.dev/v1alpha1` resource manifests remain
valid in OMF 2; do not invent a v2 resource API version.

The active `experiment.yaml` names captured scripts, data inputs, artifacts,
metrics, candidates and compute limits. Scripts accept explicit paths and need
no OMF imports. Capture uncommitted source using the configured `archive` mode.
Place hash-pinned dependency locks inside each captured source directory.
Keep source self-contained: admitted packages cannot rely on sibling repository
imports or absolute run paths. Use explicit training and evaluation input lists.
Custom stage graphs may continue using `modules/`, `workloads/`, `evaluations/`,
`bindings/`, `rights/` and `sources/` where appropriate.

Use exact executors and report the capabilities actually exercised. Do not
silently substitute an executor. Interrupted work is managed with `omf operation
reconcile` or `omf operation cancel`; do not edit runtime records by hand.

## Model and data design

- Share observation, intervention and measurement learning in a general
  backbone. Keep quantitative and SL target semantics and evaluation distinct;
  permit end-to-end SL gradients during human-only post-training.
- Use inductive sequence/annotation representations without learned gene-ID
  lookup requirements. Represent intervention mechanism explicitly, including CRISPRi
  versus CRISPRa. Respect assay-specific measurement and normalization semantics.
- Treat unpaired single-cell observations as populations; do not fabricate paired
  cellular trajectories. Simultaneous endpoint combinations are not time courses.
- Preserve stable source IDs and NCBI taxonomy IDs. Keep yeast and other species
  native; orthology is an auxiliary relation, not permission to relabel outcomes.
- Retain dataset rights and intended roles. Training permission and redistribution
  permission are separate. A quarantined source is not available for fitting.
- Group interventions appropriately for the claim. For intervention-cold results,
  all quantitative outcomes involving held genes remain excluded from fitting.
  Known-gene combination tests must be described as such.
- Use development data for training choices. Keep designated final holdouts and
  external benchmark test labels separate until the intended final evaluation.
  Report retrospective results as development evidence, not independent SOTA.
- Compare useful baselines on matched data and splits. Measure perturbation-
  specific and nonadditive responses, not only generic control-referenced
  correlation. Report meaningful regressions and source/species differences.

## Artifacts and release

Keep new large datasets off the user's Mac. Durable source storage is private
Cloudflare R2 at `s3://abiome-artifacts/slp/`; its binding and checksum-pinned
source inventory are in `storage/`. Use `scripts/cloud_data.py` to request
publisher-to-Cloudflare transfers. Local calls carry only metadata. Prepare and
cache training data on cloud compute disks; a RunPod volume is a working cache,
not the sole durable copy. Preserve other prefixes in the shared Abiome bucket.
Bucket presence does not establish training admission or CV3 eligibility.

Git holds source and configuration. Artifact stores hold datasets, checkpoints
and model bundles. `.omf/` is generated, untracked state: use OMF commands for
migration, backup and recovery; never manually edit or commit it. Keep real
payloads, credentials, caches, `results/` and `ontology/` out of commits. Tiny
synthetic contract fixtures belong under `data/fixtures/`.

OMF 2 separates saving a release from selecting it. Follow the current project
policy and real runtime requirements; do not fabricate reports, signatures or
approvals. Existing v1 releases must be recreated from their recorded runs before
new promotion/deployment. Export only to a new destination; review exact local
and remote paths before uploading and never overwrite a release.

Ordinary-script experiments materialize file/directory artifacts for evaluation
and export. A standalone SLp inference bundle must include the actual weights,
configuration, code and dependency contract. The separate OMF ModelPackage
service adapter still needs verified large-artifact state materialization;
do not claim that serving integration based only on successful experiment export.

## Compute and completion

Local compute includes one RTX 4070. Run authorized jobs locally when they should
finish within an hour; use an admitted remote executor within its assigned
allowance or ask before longer work. Keep bounded, resumable training runs and
stop invalid or unusable jobs. Optimize the model, not the number of experiments.

Verify changed numerical behavior, data routing, serialization and interfaces
with focused tests and real artifact replay. Avoid redundant tests or full-suite
reruns after narrow changes already verified. Record what was trained, inputs,
versions, compute, results and limitations in `docs/results.md`; update the model
card when intent or supported capability changes. Keep one root `MODEL_CARD.md`. Use `docs/development.md` for engineering,
`docs/module-reference.md` for retained module details, `docs/litreview.md` for
references and `docs/results.md` for evidence. Avoid duplicating project status
in per-module prose. Preserve local contract files that artifact exporters or
focused tests consume. Original code and weights are MIT; data keep source terms.
