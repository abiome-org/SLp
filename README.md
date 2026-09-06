# SLp-1.1

SLp-1.1 is a **cellular and genomic world model** with RNA/protein generation,
persistent genetic interventions and a shared nonlinear viability landscape.
It is pretrained on molecular observations, human single-gene fitness and
species-native yeast double-deletion fitness. SL is decoded downstream.

On official MuSL CV3, the trained world plus its SL decoder reaches
**.7878 AUROC / .7806 AP**, versus .7341/.7297 for the same decoder over an
untrained functional component. The trained world wins all ten folds; its mean
AUROC gain has a gene-bootstrap interval [.0361,.0723]. The fixed excess-loss
readout uses no human SL labels and reaches .5546 AUROC. See
[MODEL_CARD.md](MODEL_CARD.md) for the complete controls and pretraining mix.

The local complete predictor is
`results/slp11-transition/cellular-genomic-world-predictor-v2/`. With the dependencies in
its `requirements.lock` available:

```python
import sys
from pathlib import Path
bundle = Path("results/slp11-transition/cellular-genomic-world-predictor-v2").resolve()
sys.path.insert(0, str(bundle))
from functional_predict import SLpWorld
model = SLpWorld(bundle, device="cuda")
result = model.predict_pairs([("BRCA1", "PARP1"), ("BRCA2", "PARP1")])
print(result["sl_score"])
print(result["label_free_excess_fitness_loss"])
```

The scores are research predictions, not calibrated clinical probabilities.
The joint `encode → intervene → decode` API returns molecular and fitness
consequences. `model.molecular` retains RNA/protein generation;
`model.functional` exposes native conditional fitness and viability decoding.
The implementation and reproducible benchmark entrypoints are in
[the SL application module](modules/slp-1-1-cell-world-sl-v1/CONTRACT.md).
The functional training implementation and data contract are in
[the genomic world module](modules/slp-1-1-genomic-fitness-world-v1/CONTRACT.md).
`experiment-cellular-genomic-world.yaml` retains and replays the complete artifact
with OMF 2. Source, checkpoints and exact results are recorded in the model card
and result ledger.

The earlier molecular-only decoder can still be reproduced with its original
bounded command using the indexed molecular data and benchmark snapshots:

```powershell
python modules/slp-1-1-cell-world-sl-v1/run.py --root . --bundle results/slp11-transition/cell-world-v1-generative-research-export-v1 --output results/my-world-sl
```

That historical command simulates the molecular world and controls, trains fold-local decoders, scores both
benchmark suites, verifies saved predictions, and exports a named-gene predictor.



SLp builds molecular world models for genetic intervention research. The current
[cellular world model](modules/slp-1-1-cell-world-v1/CONTRACT.md) learns molecular
state, intervention dynamics, and RNA/protein observation distributions in one
14.12-million-parameter network. Synthetic lethality is a downstream application.

It is trained on individual human cells, paired RNA/protein measurements, human
single/combination perturbation populations, and native yeast responses. Read
[MODEL_CARD.md](MODEL_CARD.md) for the actual pretraining mix, capabilities and
limitations, and [docs/results.md](docs/results.md) for measured evidence.

## Molecular world model

The model encodes variable molecular panels into 32 latent state slots, applies
mechanism-conditioned genetic actions, and decodes molecular consequences.
A conditional latent flow and sparse RNA/continuous protein observation heads
support cell generation. Both state and observation distributions are learned
from molecular data. Gene descriptors replace a fixed learned gene vocabulary.

The core has no fitted linear response prior or SL classification head.
The earlier joint population models and supervised readouts remain available as
historical baselines; their performance is not attributed to this model.

```python
from inference import WorldModel

world = WorldModel("path/to/exported-model", device="cuda")
state = world.encode(observed, basal, query_descriptors,
                     modality=modality, scale=scale,
                     assay=6, taxon=9606, mechanism=2)
changed = world.intervene(state, action_descriptors, action_mask)
prediction = world.decode(changed)
generated = world.generate(state, action_descriptors, action_mask, seed=731)
```

`observed` is B by Q, query descriptors are Q by 702, and action descriptors are
B by A by 642. Masks are Boolean. Native units, query order, static descriptors
and assay scales must match the [input contract](modules/slp-1-1-cell-world-v1/CONTRACT.md).
Call `intervene` on the returned state to compose a different intervention.
`generate` produces processed human RNA/protein measurements, not sequencing
counts or biological time trajectories.

## Train and evaluate

Biological payloads are ignored local artifacts. With the existing pinned local
sources and the declared CUDA runtime, create a new fitting index:

```powershell
python modules/slp-1-1-cell-world-v1/prepare.py --root . --output data/derived/my-cell-world-index
python modules/slp-1-1-cell-world-v1/train.py --source-root . --index data/derived/my-cell-world-index --output results/my-cell-world-state --stage state --steps 12000 --schedule-steps 16000 --seconds 2400
python modules/slp-1-1-cell-world-v1/train.py --source-root . --index data/derived/my-cell-world-index --output results/my-cell-world --initialize results/my-cell-world-state/checkpoint-012000 --stage generative --steps 10000 --warmup 500 --learning-rate 0.0001 --seconds 2400
```

The trainer captures source, input hashes, configuration and sampled source
counts. It saves actual safetensors plus optimizer/RNG continuation state.
`--initialize` starts from a compatible molecular checkpoint while adding the
observation heads; the recorded generative model follows 12,000 initial state/
dynamics updates. The source weights and all objectives are in the module.

```powershell
python modules/slp-1-1-cell-world-v1/evaluate.py --checkpoint results/my-cell-world/checkpoint-010000 --source-root . --index data/derived/my-cell-world-index --output results/my-cell-world-evaluation
python modules/slp-1-1-cell-world-v1/export.py --checkpoint results/my-cell-world/checkpoint-010000 --index data/derived/my-cell-world-index --output results/my-cell-world-export
python modules/slp-1-1-cell-world-v1/make_example.py --index data/derived/my-cell-world-index --output results/my-cell-world-request.npz
python results/my-cell-world-export/replay.py --bundle results/my-cell-world-export --request results/my-cell-world-request.npz --output results/my-cell-world-replay
```

Use the actual final checkpoint recorded in `latest.json` if the time bound ends
training before the requested update count. All output destinations must be new.
`requirements-native.lock` records the exercised Windows training packages.
The hash-locked Linux inference and training environments are separate files.
Focused interface checks run with `python modules/slp-1-1-cell-world-v1/test_model.py`.

## Open Model Factory 2

OMF 2.0.0 is pinned to upstream commit
`75f002b4226b32dd428f5fec0efe9b950db0c6d5` in [omf-version.json](omf-version.json).
On Linux/WSL with Python 3.11 or 3.12:

```sh
bash scripts/bootstrap_omf2.sh --diagnostics
bash scripts/omf2.sh doctor
bash scripts/omf2.sh experiment run experiment-cell-world-replay.yaml --candidate generative-world
bash scripts/omf2.sh experiment list
bash scripts/omf2.sh experiment export <run-id> --to results/my-omf-world-export
```

The cell-world replay definition imports the recorded trained bundle, checks
molecular inference and generation against the native CPU reference, and exports
actual weights through OMF. This is artifact replay, with zero optimization
steps. It is distinct from the recorded native CUDA training and from a deployed
OMF ModelPackage service. The launcher pins OMF and child runtime paths.

Historical workflows remain executable:

- [experiment.yaml](experiment.yaml): retained reduced-rank response baseline.
- [experiment-joint-world.yaml](experiment-joint-world.yaml): earlier joint population training.
- [experiment-context-world.yaml](experiment-context-world.yaml): eight-context population model.
- [SL readout](modules/slp-1-1-sl-readout-v2/CONTRACT.md): separate supervised benchmark research.
- [SL predictor](modules/slp-1-1-sl-predictor-v1/CONTRACT.md): historical readout inference.
- `model/v1/` and [docs/model-card.md](docs/model-card.md): frozen SLp-1 evidence.

Git contains source and configuration. `data/`, `results/` and `.omf/` contain
local data, artifacts and runtime state. Do not commit those payloads or modify
OMF runtime records manually. Operator instructions are in [AGENTS.md](AGENTS.md).
