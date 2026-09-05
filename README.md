# SLp-1.1

SLp develops species-aware molecular world models for genetic intervention
research. The intended model represents observed molecular state, applies
interventions, and predicts RNA, protein and combination responses. Synthetic
lethality is a downstream application.

Read [MODEL_CARD.md](MODEL_CARD.md) for implemented capabilities and measured
performance, and [docs/results.md](docs/results.md) for the evidence ledger.

## Build and run with OMF 2

OMF 2.0.0 is pinned to upstream commit
`75f002b4226b32dd428f5fec0efe9b950db0c6d5` in
[omf-version.json](omf-version.json). On Linux/WSL with system Python 3.11 or 3.12:

```sh
bash scripts/bootstrap_omf2.sh --diagnostics
bash scripts/omf2.sh experiment run experiment.yaml --candidate full-ridge
bash scripts/omf2.sh experiment run experiment.yaml --candidate rank32
bash scripts/omf2.sh experiment list
bash scripts/omf2.sh experiment review <rank32-run-id> --baseline <full-ridge-run-id>
bash scripts/omf2.sh experiment reproduce <rank32-run-id>
bash scripts/omf2.sh experiment export <rank32-run-id> --to results/exported-response-model
```

Run these commands from the SLp project root. The launcher pins both OMF and the
runtime `PATH`, including for child dependency environments. Each export
destination must be new. The bootstrap installs the exact upstream code and
hash-locked dependencies without modifying existing OMF records manually.
Project configuration archives uncommitted source; a separate commit is not a
prerequisite for development.

The active [experiment.yaml](experiment.yaml) manufactures the strongest retained
molecular-response baseline through ordinary scripts. Training and evaluation
receive separate captured datasets. The model artifact is a directory containing
both native-panel models, inference code, dependency lock and manifest.
This workflow is the runnable OMF 2 integration; it is not a claim that the
linear baseline is the final world-model architecture.

Biological inputs are ignored local artifacts. To reconstruct the two input
snapshots from the existing pinned corpus, use:

```powershell
python scripts/prepare_slp11_omf2_data.py
```

This preparation uses the existing fitting moments, static features and
previously used development measurements. It does not open protected test data.
The numerical model remains unchanged: rank-32 response prediction improves
K562/RPE1 development MSE over full ridge by 4.24%/3.48%.

## Standalone model inference

OMF exports the model directory under `artifacts/model`. From that directory,
install its requirements on the declared Linux/Python platform, then run:

```sh
python inference.py --model . --context k562 --input request.npz --output prediction.npz
```

`request.npz` contains `features` (N by 577 raw static descriptors) and
`basal_anchor` (N by the context's query count, in the original ln1p response
space). Output contains `predictions` and stable `query_ids`. The Python API is
`load_bundle(path).predict(context, features, basal_anchor)`. Models verify their
weight hashes and require explicit context controls. No corpus or OMF service is
needed by the exported inference code.

## Model build direction

SLp-1.1 is being consolidated around a shared observation encoder, a
mechanism-aware action/state transition, and queried molecular decoders. The
observation encoder must participate in inference and accept observed perturbed
states. CRISPRi and CRISPRa are explicit mechanisms. Unpaired single-cell assays
supply population states, not invented cell-to-cell trajectories.

The available training base includes 350,755 K562/RPE1 fitting/control cells,
static protein/annotation descriptors, and measured Norman single/double
endpoints. RNA count likelihoods and standardized endpoint likelihoods retain
their native measurement semantics while training the same state transition.
Composition is learned within that model instead of attached as a separate
small correction after training. The detailed build direction and its current
implementation status are in the model card.

Current count, response and composition branches remain reproducible research
artifacts. None has established SOTA or broad emergent SL capability. The
strongest recent autonomous combination result is pooled and unstable across
folds. Development feedback is used to improve the model; fresh cases are needed
for independent claims.

## Project layout and compatibility

- `experiment.yaml` and `modules/slp-1-1-response-omf2/`: active captured-script
  training, evaluation and portable baseline inference.
- `modules/`, `workloads/`, `evaluations/`, `bindings/`, `rights/`, `sources/`:
  compatible `omf.dev/v1alpha1` resources and retained numerical components.
- `model/v1/`, historical `src/` and `docs/model-card.md`: frozen SLp-1 evidence.
- `data/`, `results/`, `.omf/`: local data/artifact/runtime state, not source commits.

OMF 2 uses v2 release manifests and separates saving from promotion. Old releases
must be recreated from recorded runs before new deployment. Script experiment
export supports file and directory model artifacts. The separate ModelPackage
service adapter still lacks verified large-weight state materialization, so
exported standalone inference and OMF service deployment are distinct claims.

The canonical external artifact home remains
[potteryrage/SLp](https://huggingface.co/potteryrage/SLp). Migration does not publish
or overwrite an external release. See [AGENTS.md](AGENTS.md) for the current
operator guide.
