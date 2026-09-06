# Developing SLp

The [README](../README.md) covers installation and inference. The
[model card](../MODEL_CARD.md) is the maintained scientific description;
[results.md](results.md) retains experiment evidence. Historical module details
are consolidated in [module-reference.md](module-reference.md). Frozen SLp-1
source and its card remain in `model/v1/`.

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
