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

The baseline [experiment.yaml](experiment.yaml) reproduces the retained 577-feature
rank-32 molecular-response model through ordinary scripts. Training and evaluation
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

## Shared molecular world model

The implemented [joint module](modules/slp-1-1-joint-world-v2/CONTRACT.md) combines
a masked observation encoder, a mechanism-aware action/state transition, and
queried molecular decoders. Inference can start from a measured perturbed state
or roll forward from the model's own predicted state. CRISPRi, CRISPRa and Cas9
knockout are explicit mechanisms. Unpaired single-cell assays supply population endpoints.

The eight-context training corpus includes K562/RPE1 essential-gene screens,
Norman single/double endpoints, the K562 genome-wide screen, HepG2, and three
MCF10A knockout environments from GSE164996.
Static protein/annotation features and public STRING embeddings describe genes
without learned gene-ID lookup tables. Separate assay heads preserve native
count-derived and standardized endpoint units. The encoder additionally accepts
the control population's expression profile; a zero-centered endpoint does not
replace that biological context. The objective models population mean response,
not a single-cell count likelihood.

The stronger 642-feature reduced-rank response backbone beats matched SLIM
comparators on both canonical retrospective test sets. The shared neural model
improves autonomous Norman combination MSE by 6.39% over its predicted-additive
output across all three folds (59 held pairs). The eight-context extension also
improves combination MSE over additive prediction in each trained MCF10A
environment, although retaining the control state has lower absolute MSE.
MCF10A intervention prediction and held-medium transfer need further improvement. These support continued use of the architecture;
they do not establish broad SL performance or emergent causal understanding.
The model card reports exact comparisons and the remaining RPE1 tradeoff.

Native CUDA training from the prepared eight-context corpus:

```powershell
python modules/slp-1-1-joint-world-v2/train.py --data data/derived/slp11-joint-world-context-transfer-v2-training-r4 --output results/my-joint-world --steps 20000 --save-every 5000 --seed 731 --fold 0 --batch-size 32 --observation-queries 512 --decode-queries 512 --learning-rate 0.0002 --response-rank 16 --residual-penalty 1 --max-seconds 2700
```

The output destination must be new. The trainer captures source, normalizers,
query adapters, frozen linear priors, and safetensor checkpoints. Joint-model
inference uses a different request contract from the historical baseline:

```powershell
python results/my-joint-world/inference.py --model results/my-joint-world --checkpoint step-020000.safetensors --context k562 --input request.npz --output prediction.npz
```

Requests contain `actions` (B by A by 642 raw descriptors), Boolean
`action_mask`, and `basal` (B by native query count). Optional `observed`
supplies the starting perturbed state. Optional `control_context_values` and
`control_context_mask` supply an aligned log2(1+CP10K) control profile. Output
contains `predictions`, stable `query_ids`, and `prediction_supported`; empty-action rows preserve the
starting observation exactly.

[experiment-joint-world.yaml](experiment-joint-world.yaml) declares the equivalent
completed five-context OMF 2 captured-script workflow with physically separated
training and development inputs. [experiment-context-world.yaml](experiment-context-world.yaml)
is the validated eight-context replay configuration, with MCF10A evaluation
inputs separate from training. Its hash-pinned Linux CUDA dependencies can be installed and checked with
`bash scripts/bootstrap_slp_joint_linux.sh`. Native CUDA execution and OMF
execution are recorded separately in the results ledger.

The completed eight-context research bundle is
`results/slp11-transition/joint-world-eight-context-research-export-v1/`. Its
26 payload files passed hash verification and standalone Windows/Linux replay
across all eight contexts, with maximum prediction drift 2.16e-7 and no OMF
import available. The bundle's Python API is `JointWorldBundle(model_directory)`;
its manifest selects the completed checkpoint automatically. This is a local
research model, not an externally published release.

## Project layout and compatibility

- `experiment.yaml` and `modules/slp-1-1-response-omf2/`: active captured-script
  training, evaluation and portable baseline inference.
- `experiment-joint-world.yaml` and `modules/slp-1-1-joint-world-v1/`: verified
  five-context OMF training, composition, and standalone inference.
- `experiment-context-world.yaml` and `modules/slp-1-1-joint-world-v2/`: generalized
  eight-context training, held-medium evaluation, and portable inference.
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
