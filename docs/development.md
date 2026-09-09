# Developing SLp

The [README](../README.md) covers installation and inference. The
[model card](../MODEL_CARD.md) is the maintained scientific description;
[results.md](results.md) retains experiment evidence. Historical module details
are consolidated in [module-reference.md](module-reference.md). Frozen SLp-1
source and its card remain in `model/v1/`.

## SLp-1.2 joint pretraining

`modules/slp-1-2` is a self-contained PyTorch implementation of a shared set
transformer. Molecular observations, individual genetic interventions and
RNA/protein/fitness queries enter the same learned backbone. A mean and
observation-variance objective trains measured endpoints; computational flow
matching additionally trains the single-cell distributional path. There is no
frozen simulation bridge, additive latent action constraint or SL classifier.
Static descriptors remain available alongside optional learned entity indices.
The main continuation mixes 10% no-intervention examples: real control cells
with held-out query coordinates, molecular population controls, and zero-effect
fitness references. Cell query anchors use population controls, never the held
cell coordinates being reconstructed. The opening segment lacked this mixture;
its replacement is recorded as a new continuation segment, preserving weights
and optimizer state.

The first campaign uses the checksum-pinned 1.1 data release. It does not yet
upgrade the sequence-descriptor backbone or add new human combination screens.
Held intervention genes are excluded jointly across molecular and fitness
sources. Population scales are recomputed from the resulting training split.
Development panels are retrospective; no independent benchmark claim follows.

Install the hash-pinned Linux CUDA dependencies in
`modules/slp-1-2/requirements-linux-cu128.lock`, then run:

```sh
python modules/slp-1-2/prepare.py --root . --output data/derived/slp12-corpus
python modules/slp-1-2/train.py --root . --output results/slp12/run-1 \
  --config modules/slp-1-2/config.json --max-hours 24
python modules/slp-1-2/train.py --root . --output results/slp12/run-1 \
  --config modules/slp-1-2/config.json --resume results/slp12/run-1/checkpoint-0001000
python modules/slp-1-2/export.py --run results/slp12/run-1 --output results/slp12/bundle-1
```

Checkpoints include optimizer state and per-rank Python/NumPy/Torch/CUDA and
sampler state. Two resumable checkpoints are retained; best inference weights
are separate. `--stop-after-steps` saves after a bounded number of additional
steps without changing the planned learning-rate schedule. Resumption requires
the same model, corpus receipts and rank count. Changed segment configurations
and source are captured explicitly rather than overwriting original run receipts.
`torchrun` supports DDP, while the initial campaign uses one GPU.

Fitness has a separate batch size because its short sequences permit many more
experiments per device pass. Paired-cell shard index and reuse count are saved
with sampler state. `benchmark.py` measures the full architecture on synthetic
maximum-sized token arrays; its weights are discarded and its timings carry no
biological meaning. `tests/test_slp12.py` checks numerical contracts.

The authorized RunPod campaign has a $50 total ceiling and a $5 reserve. The
initial allocation is one secure RTX 4090 for at most 26 hours, including setup
and export, with a 50 GB network volume. `scripts/slp12_runpod.py` records its
exact resource IDs under ignored `data/slp12-campaign/` and starts a local
termination guard. `pod_guard.py` independently uses the provider-injected
pod-scoped GraphQL credential. The account key is read locally from `.env` and
is never sent to the training pod. The trainer has its own checkpointing time
limit. Copy and verify model artifacts locally before removing the volume.
Record native RunPod execution honestly; OMF replay/export is a separate step.

Final evaluation records source-specific errors, intervention ablation, yeast
interaction residuals against measured-single additivity, and empirical flow
endpoint energy distance and moments on single-cell panels. These distribution
panels contain unpaired cells and carry finite-sample uncertainty.

The pinned OMF local executor requires real network namespaces, which the
RunPod container does not expose. After collecting the bundle, run the separate
CPU workflow on a compatible Linux host:

```sh
python3 scripts/slp12_omf_replay.py \
  --bundle results/slp12-joint-142m-r1-bundle \
  --output results/slp12-joint-142m-r1-omf
```

The helper selects the x86 or ARM Linux CPU lock, captures the exact experiment
definition, preserves real OMF run/export receipts and performs zero optimization.
This campaign uses the existing local `omf-tests` Colima VM. OMF replay on that
host is distinct from both native CUDA training and macOS CPU replay. It does
not establish OMF ModelPackage service deployment support.

The standalone API and measurement units are documented in the
[bundle contract](../modules/slp-1-2/CONTRACT.md). Human combination-fitness
predictions remain extrapolations until supervised by appropriate human data.

The completed first campaign's local inference bundle is
`results/slp12-joint-142m-r1-bundle`. It contains the selected update-27,000
weights, while `results/slp12-joint-142m-r1/checkpoint-0040000` retains the final
optimizer and sampler state for continuation. Validate standalone CPU inference
with:

```sh
python results/slp12-joint-142m-r1-bundle/replay.py \
  --bundle results/slp12-joint-142m-r1-bundle
```

The bundle's `inference.py` also accepts `--bundle`, `--input` (a normalized
batch NPZ), `--output` (a new NPZ path), and optional `--sample`. Its shipped
`example-input.npz` is a real molecular example. The `World.fitness` API accepts
native stable gene IDs and explicit species/context, as specified in the
contract. Consult the final results before choosing an output: mean prediction
uses intervention information, but this candidate has weak yeast interaction
prediction and mixed endpoint-distribution results. Public download pointers
still refer to the 1.1 release.

## Evaluate a base before human post-training

The 40,000-update campaign produced a mixed human/yeast pretrained base. It did
not implement a distinct post-training phase. Human-only post-training is the
next phase; yeast remains pretraining material and a diagnostic. Keep the base
artifact immutable and write adapted models to new destinations.

`base_evaluation.py` separates panel preparation from checkpoint scoring. The
versioned `base-evaluation.json` specifies sampling and readout budgets. Prepare
once, then reuse the exact checksummed panels across compatible checkpoints:

```sh
python modules/slp-1-2/base_evaluation.py prepare --root . \
  --output results/my-base-panels
python modules/slp-1-2/base_evaluation.py score \
  --bundle results/slp12-joint-142m-r1-bundle --panels results/my-base-panels \
  --output results/my-base-scores
python modules/slp-1-2/transfer_probe.py \
  --bundle results/slp12-joint-142m-r1-bundle --panels results/my-base-panels \
  --output results/my-human-probe
```

These commands default to CPU with four Torch threads. They never optimize the
base or allocate cloud resources. Outputs must be new directories. Reports
include source snapshots, input/weight hashes, predictions and individual case
errors. Panel scoring rejects vocabulary, corpus, exclusion-roster and
normalization mismatches. Different model widths/depths can share panels when
those input contracts match. A changed data contract needs a deliberate matched
evaluation adapter; do not silently compare separate panel draws.

The report asks three questions:

1. **Does the frozen mean head predict held interventions?** Compare it with
   fitting-perturbation means, empirical control means and source references.
   Human fitness uses the exact fitting-gene mean for each context. Molecular
   baselines are source-pooled sampled means, not context-matched estimates;
   their coordinate counts and fallback coverage are explicit. Cell means use
   mean transformed observations, distinct from transforming a population mean.
2. **Does intervention identity matter?** Replace the intervention with a
   different gene set from the same source, preserving cardinality, mechanism,
   context, measurements and queries. Donors span panels because many cell
   batches contain only one intervention. Unavailable swaps are excluded and
   counted. Positive wrong-minus-correct MSE means the correct identity helps.
   The masked-action comparator separately measures action presence. Report
   RNA/protein/fitness separately, with within-panel cell pseudobulk errors.
3. **Are human outcomes easier to learn from the frozen representation?** A
   temporary ridge readout sees 128, 512, 2,048 or 8,192 unique human gene/context
   labels. Compare pretrained query features, a matched random frozen backbone,
   and static action descriptors plus context with the same nested labels and
   ridge rule. Standardization and the intercept use adaptation rows only. Both
   adaptation and evaluation genes were excluded as interventions during base
   fitting, and these two gene groups are disjoint. Their static descriptors and
   appearances as measured/query genes can be known. Parameter digests verify
   the base stays frozen.

Human source results and human label efficiency guide post-training choices.
Yeast results diagnose pretraining behavior; they do not contribute to a pooled
human promotion score. There is no automatic pass threshold. Compare actual
benefits, regressions, compute and eventual human post-training results.

This first panel uses the existing development pool, already reused for
checkpoint selection. Its transfer probe measures new-gene adaptation within
the human quantitative task, not a new SL task or an independent test. One
partition and one random initialization give a development comparison; they
do not establish a scaling law or isolate the contribution of yeast pretraining.
The report retains individual errors for matched follow-up analysis; rows and
gene pairs can share interventions, so ordinary row-wise confidence intervals
would overstate independent evidence. The first report supplies point estimates.
Generation and nonadditivity remain separate required capabilities to assess:
the existing `evaluate.py` implements their endpoint-distribution and yeast
interaction diagnostics. Neither endpoint MSE nor a successful linear probe
establishes those capabilities. Human combination fitness needs human evidence.

## Downloads and local layout

Run commands from the repository root with Python 3.11/3.12. Install
`requirements-inference.txt` in a virtual environment; select a CPU or CUDA
PyTorch wheel appropriate to the host. Linux also needs a working OpenMP runtime
for LightGBM (for example `libgomp1` on Ubuntu).

The model and prepared-data releases are pinned to immutable Hugging Face
revisions in `artifacts.lock.json`. Downloads verify those exact snapshots.

```sh
python scripts/fetch_artifacts.py model
python scripts/fetch_artifacts.py data
```

The model needs approximately 121 MB and the prepared data release approximately
11.17 GB. The helper verifies the revision-pinned inventory and each file's
SHA-256. Existing matching files are reused; changed files cause an error before
downloads begin. It never silently overwrites an experiment or edited receipt.
Transient transfer failures are retried with byte-range resumption where the
server supports it. Every completed file still must match its pinned checksum.
Use a separate checkout if existing local artifacts conflict. To check an
already populated checkout, add `--verify-only`.

Model files go to
`results/slp11-transition/cellular-genomic-world-predictor-v2/`. Data paths mirror
the original checkout. Input manifests retain their exact bytes and historical
preparation receipts, including original machine path strings; current fitting
loaders use the explicit input directories below. The data release supplies
prepared fitting inputs, not every upstream raw archive or historical run.

`rights/*.yaml` uses LF line endings on all platforms because prepared manifests
hash these files. The published data inventory and source notices distinguish
molecular fitting, functional development, application benchmarks and evidence.
An input file can contain excluded rows: the molecular index determines which
rows the trainer may use. Do not feed every downloaded NPZ into fitting.

## World APIs

`functional_predict.SLpWorld` loads and verifies the standalone bundle. Its
`predict_pairs` method accepts supported human gene symbols or stable IDs and
returns a ten-decoder mean, individual decoder scores, a separate label-free
excess-fitness-loss score and world-derived features. The ensemble is intended
for subsequent research inference. Reported benchmark scores use each held-out
fold's own decoder.

The joint API takes measured context and static descriptors:

```python
state = world.encode(
    observed, basal, query_descriptors,
    fitness_context=fitness_context,
    modality=modality, scale=scale,
    assay=6, taxon=9606, mechanism=2,
)
changed = world.intervene(state, action_descriptors, action_mask)
prediction = world.decode(changed)
```

`observed` and `basal` are B by Q; query descriptors are Q by 702; actions are
B by A by 642; action masks are Boolean B by A. Human fitness contexts have 128
coordinates in the recorded DepMap context basis. `decode` returns molecular
predictions and continuous fitness effects. Query order, units, static feature
bases and assay codes must match the recorded preparation. Arbitrary arrays of
the right shape are not valid biological inputs.

`world.molecular.generate` samples processed RNA/protein observations from its
learned observation distributions. `world.functional` exposes the shared
viability function and conditional fitness. For new human genes,
`world.actions_from_descriptors(raw)` runs actual molecular simulations before
functional action encoding. Nonhuman joint interventions require species-native
molecular signatures. Stable identifiers index cached descriptors, not a learned
gene-ID embedding.

The self-contained API specifications packaged with model artifacts remain in
the [molecular](../modules/slp-1-1-cell-world-v1/CONTRACT.md),
[functional](../modules/slp-1-1-genomic-fitness-world-v1/CONTRACT.md) and
[application](../modules/slp-1-1-cell-world-sl-v1/CONTRACT.md) modules. These files
are export inputs; other module prose is consolidated in the reference.

## Train from the prepared release

These are the recorded r1 settings. Outputs must be new directories. Numerical
reproducibility depends on the runtime and device; the artifact records exact
native and Linux dependency locks. Time limits can stop a run before its target
update count, so inspect `latest.json` before using a checkpoint path.

```sh
python modules/slp-1-1-cell-world-v1/train.py --source-root . --index data/derived/slp11-cell-world-training-v5 --output results/my-molecular-state --stage state --steps 12000 --schedule-steps 16000 --seconds 2400
python modules/slp-1-1-cell-world-v1/train.py --source-root . --index data/derived/slp11-cell-world-training-v5 --output results/my-molecular-world --initialize results/my-molecular-state/checkpoint-012000 --stage generative --steps 10000 --warmup 500 --learning-rate 0.0001 --seconds 2400
python modules/slp-1-1-genomic-fitness-world-v1/audit.py --data data/derived/slp11-genomic-fitness-world-v1 --output results/my-fitness-audit.json
python modules/slp-1-1-genomic-fitness-world-v1/train.py --data data/derived/slp11-genomic-fitness-world-v1 --output results/my-functional-world --architecture capacity --interaction-weight 100 --steps 20000 --batch 1024 --max-seconds 3000
```

The prepared functional corpus includes signatures from the released molecular
checkpoint. These commands reproduce the two recorded fitting stages; training
a different molecular model does not automatically regenerate those signatures.
Use the functional `prepare.py` and application simulation tools when changing
the molecular component, and record the newly generated corpus separately.

Functional development loss selects the checkpoint. Human SL labels are absent
from both world trainers. The application decoder is a separate operation:

```sh
python modules/slp-1-1-cell-world-sl-v1/functional_decoder.py --phase fit --functional results/slp11-transition/cellular-genomic-world-predictor-v2/functional_model --genes data/derived/slp11-genomic-fitness-world-v1/genes.npz --contexts results/slp11-transition/genomic-fitness-world-contexts-v1/contexts.npz --roster data/derived/slp11-musl-world-pair-roster-v1 --labels data/models/MuSL/processed_data/data/CV3_bins_32/fold_data --output results/my-sl-decoder
python modules/slp-1-1-cell-world-sl-v1/functional_decoder.py --phase score --labels data/models/MuSL/processed_data/data/CV3_bins_32/fold_data --output results/my-sl-decoder
```

Fitting uses training labels and writes fixed predictions before scoring reads
test labels. The public benchmark has already been inspected historically;
rerunning it is reproduction, not untouched confirmation. The released
evaluation reports preserve every fold and matched control.

## Open Model Factory 2

The runtime is pinned by [omf-version.json](../omf-version.json). On Linux or WSL:

```sh
bash scripts/bootstrap_omf2.sh --diagnostics
bash scripts/omf2.sh doctor
bash scripts/omf2.sh agent context
bash scripts/omf2.sh agent capabilities experiment.run
```

`experiment-cellular-genomic-world.yaml` describes the exercised artifact replay
and export. It also needs its separately prepared Linux readout-runtime ZIP;
that environment artifact is not bundled into the biological-data release.
Follow the runtime preparation recorded in the result ledger before running
that historical OMF experiment. Standalone inference uses installed dependencies
and does not need this ZIP or an OMF database.

OMF replay verifies trained artifacts and performs zero optimization. The
ordinary-script training commands above perform actual fitting. An OMF export
is distinct from a production ModelPackage deployment. Keep `.omf/` generated
and untracked; never edit runtime records by hand.

## Collaboration and Git ownership

Use a focused branch and pull request. Describe the resulting behavior, changed
scientific assumptions, validation and artifact revisions. Preserve frozen
`model/v1/` and previously published releases. New artifacts get new paths;
update `artifacts.lock.json` only after verifying their remote contents.

| Tracked in Git | Stored outside Git |
|---|---|
| Original source, focused tests, tiny synthetic fixtures | Real observations and prepared arrays |
| Dependency locks and experiment/configuration files | Checkpoints, optimizer states and decoder payloads |
| Source/rights descriptions and artifact revision pointers | Per-run reports, predictions and generated archives |
| Maintained docs and the scientific results ledger | OMF state, ontology workspace, virtual environments and caches |

Run `python scripts/audit_repository.py` to check tracked payloads and broken
local Markdown links. Stage named source/config paths; inspect
`git diff --cached --stat` and `git diff --cached --check` before committing.
Do not use `git add -f` to add ignored research payloads. A fixture must be tiny,
synthetic, and documented under `data/fixtures/`.

For model changes, run the relevant module checks and artifact replay. For
publication tooling, use `python -m unittest tests.test_release_tools`. Append
new scientific evidence to `docs/results.md`; keep the README focused on usage.

## Publishing an artifact revision

`scripts/prepare_hf_release.py` records the explicit r1 source list and builds
hash/size inventories without copying payloads. It is release-specific; change
the version and source list for subsequent releases rather than overwriting r1.
`scripts/publish_hf.py` validates a plan, supports resuming identical remote
files, and rejects replacement of existing release payloads. Only explicitly
named mutable overview cards can change.

```sh
python scripts/prepare_hf_release.py --output results/my-publication-plan
python scripts/publish_hf.py results/my-publication-plan/model-plan.json --receipt results/my-publication-plan/model-receipt.json
```

The second command is validation only. Add `--execute` to publish a reviewed
plan using an already authenticated Hugging Face account. Review exact source
and destination paths and component licenses first. Tokens belong in the
credential store, never in a command, plan, repository or report. Upload receipts
stay under ignored `results/`; immutable remote revisions go in the Git lock.

## SLp-1.2-XL pretraining scale-up

`modules/slp-1-2/config-xl.json` prepares a 24-layer, width-1024, 16-head model
with 344,953,859 parameters under the current vocabulary. It retains the shared
molecular/fitness architecture, descriptor contract, admitted corpus and global
intervention exclusions. The schedule is 60,000 updates, with runtime bounded
by the full-model hardware profile and the remaining funded budget.
The 142M run took 40,000 updates. Human adaptation and external SL benchmarking
are deferred while the shared predictor is scaled. Internal development checks
remain part of training.

`benchmark.py --config modules/slp-1-2/config-xl.json --gene-count 30779`
measures molecular and fitness training shapes with the actual larger model.
`--batch-sizes`, `--fitness-batch-sizes` and `--activation-checkpointing` expose
memory/throughput choices. Synthetic profile weights are discarded. These
measurements estimate execution cost, not biological performance.

The local RunPod helper accepts `--state` and `--plan`; the collector accepts
`--state`, `--run-name`, `--ops`, `--backup-seconds` and `--keep-checkpoints`.
Use `--keep-checkpoints 2` for XL to retain two verified resumable backups within
local disk capacity; the default zero preserves historical unlimited retention.
Use a fresh state directory
and run name for XL so that the completed first campaign stays intact. The
allocation check includes earlier SLp spending and conservative storage charges
under the original $50 total cap, plus a $5 reserve. It separately checks shared
account credit against other running pods without changing those resources.
When another job is expected to finish soon, an explicit
`other_pods_reserve_hours` can budget its remaining runtime separately; the
default reserves the entire allocation. A local five-minute credit check
terminates only this campaign's pod below $5, preserving its persistent volume
and most recent checkpoint for recovery. This does not increase the $50 cap.
A `run_job.py --config ...` launch can start fresh or use `--resume` for an exact
checkpoint continuation. Every allocation still needs a verified local guard,
scoped pod guard and collection plan. Current CLI 2.12.0 does not expose the
`--terminate-after` flag described in some RunPod examples; use the exercised
guards, not an invented CLI flag.
