# SLp

SLp is a research codebase for cold-start synthetic-lethality prediction: ranking a gene pair when one or both genes were absent from synthetic-lethality training labels. The project studies whether an intervention-conditioned cellular world model can transfer to unseen gene pairs, and reports label-free transition scores separately from benchmark-supervised readouts.

The current compact model is a 6-layer, 384-wide transformer with 11.92M trainable parameters. The strongest current label-free pan-cancer development result is 0.6320 AUROC / 0.6282 average precision on MuSL CV3; a separate fold-local supervised readout reaches 0.8377 / 0.8419. These are development results, not a prospective state-of-the-art claim. See [the model card](docs/model-card.md) for the complete protocol and [results](docs/results.md) for the experiment record.

## Codebase map

```text
model/
  v1/                    immutable, data-free world-model contract and checkpoint tools
modules/
  training/              molecular-only training entry points
  decoders/              application decoders (fitness, SL, drug response)
  evaluation/            read-only benchmark guardrails and adapters
src/
  training/
    world_model.py       compatibility layer for legacy training objectives and ablations
    run_modal.py         Modal GPU training and evaluation entry point
    sl_predict.py        feature, split, graph, and baseline preparation
    depmap.py            DepMap expression and dependency preparation
    perturbseq*.py       perturbational-expression state construction
    validate_generalization.py  fail-closed molecular generalization gate
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

`model/v1` simulates molecular state only. It does not load a dataset, inspect
a benchmark split, or score synthetic lethality. Application decoders remain
outside the model so an SL score cannot silently become part of the world
model. A released checkpoint is identified by its versioned model code and
SHA-256 manifest; behavior changes require a new model version.

## Environment

Python 3.11 is the working runtime. The core local dependencies are PyTorch, NumPy, pandas, SciPy, scikit-learn, h5py, and Modal; particular preparation paths additionally use `rdata` or `safetensors`. GPU jobs are defined with pinned packages in `src/training/run_modal.py`.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install torch numpy pandas scipy scikit-learn h5py modal rdata safetensors
```

Raw datasets and generated arrays are intentionally absent from Git. Restore the expected inputs under `data/`, then build the relevant packs under `results/sl_predict/`; every script resolves paths from the repository root.

## Hugging Face artifacts

The public artifact repository is [potteryrage/SLp](https://huggingface.co/potteryrage/SLp). It is currently an empty artifact home; repository creation alone is not a model or data release. Git remains source-only, while cleaned, redistributable data products and released weights belong on the Hub under immutable versioned paths:

```text
checkpoints/vN/RELEASE/       world-model weights and checkpoint manifest
decoders/vN/RELEASE/          application-decoder weights and metadata
data/CORPUS/VERSION/           cleaned arrays plus schema and provenance audit
```

Every checkpoint upload must include the generated SHA-256 manifest. Every data upload must identify its upstream sources, licenses, transformations, schema, split construction, exclusions, and file checksums. Do not upload restricted raw inputs or third-party weights whose licenses do not permit redistribution.

After an artifact exists locally, upload it directly without moving it into Git, for example:

```bash
hf upload potteryrage/SLp results/sl_predict/RELEASE/world_model.pt checkpoints/v1/RELEASE/world_model.pt
hf upload potteryrage/SLp results/sl_predict/RELEASE/world_model.manifest.json checkpoints/v1/RELEASE/world_model.manifest.json
```

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

Training commands now perform molecular validation only. They do not open
Feng, MuSL, SLAMR, Sanger, or HAP1 by default. Use a dedicated evaluation flag
only after the checkpoint and its application decoder are locked.

Feature packs created before this boundary remain readable for historical
reproduction, but they do not contain a separate relation-pretraining pool.
Regenerate the feature family before a new generalization experiment so model
pretraining uses only PPI and independent random relation pairs, never pair
membership from a benchmark split.

Use `modal run src/training/run_modal.py --help` for the full set of controlled training and ablation flags. Local preprocessing commands expose their own `--help` output.

## Evaluation rules

- Treat two-new-gene cold start as the primary target.
- Remove every intervention involving validation or test genes from quantitative pretraining when claiming intervention-level isolation.
- Select transition models and label-free decoders only on molecular validation objectives; do not inspect benchmark test labels for selection, calibration, sign choice, or stopping.
- Report label-free scores, benchmark-supervised readouts, retrospective analyses, and prospective confirmation as distinct claims.
- Record the exact split, seed, input checksums, checkpoint checksum, feature access, and mean plus fold-level metrics in `docs/results.md`.

### Hard molecular generalization gate

Before an SL benchmark can be opened, run the molecular outcome pack through
the five-fold generalization gate:

```bash
python3 src/training/validate_generalization.py \
  results/sl_predict/perturbseq_world.npz
```

The default gate requires every fold of seven distinct protocols: exact
action-set holdout; composition-gene holdout, where matched singleton outcomes
remain available but held genes never occur in a training composition; full
intervention-gene holdout; context holdout; source/study holdout;
perturbation-condition holdout; and compound source-plus-gene holdout. It also
reports a cardinality-matched mean and a matched-single additive baseline.

This command is intentionally fail-closed. A pack must provide a finite
`target`, scalar `target_semantics=perturbation_delta`, distinct `source` or
`source_id`, `context_id`, and either an explicit experimental condition ID or
finite modality-plus-duration metadata. Source, context, and experimental
condition are not aliases. Every required fold must meet the default minimums
of 128 training rows, 32 test rows, 16 distinct test action sets, and 8 test
genes. The JSON audit is written under `results/generalization/`, including the
input SHA-256, exclusions, baseline coverage, and the exact reason for every
ineligible fold. Failing a gate is a data-support result, not permission to
weaken or silently redefine the protocol.

## Repository policy

The public repository contains only lean, relevant source and scientific documentation. Datasets, checkpoints, experiment products, caches, the local ontology, and third-party model copies remain untracked. See [AGENTS.md](AGENTS.md) before making changes.
