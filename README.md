# SLp

SLp is a research codebase for cold-start synthetic-lethality prediction: ranking a gene pair when one or both genes were absent from synthetic-lethality training labels. The project studies whether an intervention-conditioned cellular world model can transfer to unseen gene pairs, and reports label-free transition scores separately from benchmark-supervised readouts.

The current compact model is a 6-layer, 384-wide transformer with 11.92M trainable parameters. The strongest current label-free pan-cancer development result is 0.6320 AUROC / 0.6282 average precision on MuSL CV3; a separate fold-local supervised readout reaches 0.8377 / 0.8419. These are development results, not a prospective state-of-the-art claim. See [the model card](docs/model-card.md) for the complete protocol and [results](docs/results.md) for the experiment record.

## Codebase map

```text
src/
  training/
    world_model.py       model, objectives, fitting, and ablations
    run_modal.py         Modal GPU training and evaluation entry point
    sl_predict.py        feature, split, graph, and baseline preparation
    depmap.py            DepMap expression and dependency preparation
    perturbseq*.py       perturbational-expression state construction
    slkb.py              quantitative double-knockout outcome preparation
    kg.py                provenance-safe knowledge-graph preparation
  benchmarks/
    sl_predict.py        Feng-style classification and ranking metrics
    musl.py              MuSL two-new-gene evaluation
    slamr.py             SLAMR scenario-3 ranking evaluation
    sanger.py            context-specific Sanger evaluation
    hap1.py              HAP1 preparation, scoring, and evaluation
    main.rs              small deterministic cold-start split utility
docs/
  model-card.md          claims, data provenance, architecture, and limits
  litreview.md           literature review
  results.md             chronological scientific results
data/                    local inputs; never committed
results/                 local artifacts; only .gitkeep is committed
ontology/                local research graph; never committed
```

## Environment

Python 3.11 is the working runtime. The core local dependencies are PyTorch, NumPy, pandas, SciPy, scikit-learn, h5py, and Modal; particular preparation paths additionally use `rdata` or `safetensors`. GPU jobs are defined with pinned packages in `src/training/run_modal.py`.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install torch numpy pandas scipy scikit-learn h5py modal rdata safetensors
```

Raw datasets and generated arrays are intentionally absent from Git. Restore the expected inputs under `data/`, then build the relevant packs under `results/sl_predict/`; every script resolves paths from the repository root.

## Training

Prepare the benchmark features and splits required by the selected experiment:

```bash
python src/training/sl_predict.py prepare_spectral_safe
python src/training/sl_predict.py benchmarks
```

The retained intervention-isolated Perturb-seq model is trained on a Modal GPU:

```bash
modal run src/training/run_modal.py --spectral-safe-intervention \
  --pretrain-epochs 12 --perturb-epochs 10 --rl-epochs 3
```

Run a named checkpoint evaluation through the same entry point, for example:

```bash
modal run src/training/run_modal.py --musl-interaction-shrinkage-model MODEL_NAME
```

Use `modal run src/training/run_modal.py --help` for the full set of controlled training and ablation flags. Local preprocessing commands expose their own `--help` output.

## Evaluation rules

- Treat two-new-gene cold start as the primary target.
- Remove every intervention involving validation or test genes from quantitative pretraining when claiming intervention-level isolation.
- Select transition models and label-free decoders only on molecular validation objectives; do not inspect benchmark test labels for selection, calibration, sign choice, or stopping.
- Report label-free scores, benchmark-supervised readouts, retrospective analyses, and prospective confirmation as distinct claims.
- Record the exact split, seed, input checksums, checkpoint checksum, feature access, and mean plus fold-level metrics in `docs/results.md`.

## Repository policy

The public repository contains only lean, relevant source and scientific documentation. Datasets, checkpoints, experiment products, caches, the local ontology, and third-party model copies remain untracked. See [AGENTS.md](AGENTS.md) before making changes.
