# SLp

SLp is a species-aware world model for genetic intervention research. It encodes
molecular observations, applies genetic interventions, and decodes molecular
and fitness consequences. Synthetic-lethality prediction is a downstream
application of the learned state.

[Model on Hugging Face](https://huggingface.co/potteryrage/SLp) ·
[Data and source terms](release/THIRD_PARTY_NOTICES.md) ·
[Model card](MODEL_CARD.md) · [Development guide](docs/development.md)

## SLp-1.1

The release combines a 14.12M-parameter generative molecular world with a
1.00M-parameter functional world. Molecular pretraining uses human and yeast
RNA observations and paired human RNA/protein measurements. Functional
pretraining uses human single-gene effects and species-native yeast
single/double-mutant fitness. The components are trained in stages; human SL
labels train only the separate downstream decoder.

The API supports molecular encoding, intervention composition, RNA/protein
generation, continuous fitness prediction and named-gene SL scoring. The
[model card](MODEL_CARD.md) describes the architecture, exact pretraining mix,
evaluation protocols and limitations.

On official MuSL CV3, the world plus downstream decoder reaches **0.7878 AUROC /
0.7806 AP** across ten folds. The matched untrained functional component reaches
0.7341/0.7297; direct descriptors reach 0.7986/0.8030. These are retrospective
research results. Scores are not calibrated clinical probabilities.

## Get started

Use Python 3.11 or 3.12. Linux CPU and Windows CUDA inference have been exercised.
For CUDA, install the matching PyTorch wheel for your system before the remaining
requirements.

```sh
git clone https://github.com/abiome-org/SLp.git
cd SLp
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -r requirements-inference.txt
python scripts/fetch_artifacts.py model
```

The download is approximately 121 MB. It includes inference source, actual
weights, normalizers, observed contexts, gene descriptors and all ten SL decoders.
The training corpus is not needed for inference. Exact revisions and checksums
are recorded in [artifacts.lock.json](artifacts.lock.json).

```python
import sys
from pathlib import Path

bundle = Path("results/slp11-transition/cellular-genomic-world-predictor-v2").resolve()
sys.path.insert(0, str(bundle))
from functional_predict import SLpWorld

world = SLpWorld(bundle, device="cpu")  # or "cuda"
result = world.predict_pairs([("BRCA1", "PARP1"), ("BRCA2", "PARP1")])
print(result["sl_score"])
print(result["label_free_excess_fitness_loss"])
```

`world.molecular` exposes molecular state and generation. `world.functional`
exposes continuous viability and conditional fitness. Joint `encode`,
`intervene` and `decode` methods connect both components. See the
[API and training guide](docs/development.md) for inputs and examples.

## Work on the model

Published artifact revisions are listed in [artifacts.lock.json](artifacts.lock.json).
The model is available; the prepared-data upload awaits publication approval.
Once the data revision is populated, `python scripts/fetch_artifacts.py data`
downloads the inputs for the [training and evaluation commands](docs/development.md).
Open Model Factory 2 records experiments and artifact replay; its exact runtime
is pinned in [omf-version.json](omf-version.json).

| Location | Purpose |
|---|---|
| `modules/`, `src/`, `scripts/`, `tests/` | Source, self-contained model modules and verification |
| `experiment*.yaml`, `workloads/`, `evaluations/`, `bindings/`, `policies/` | Versioned experiment and execution configuration |
| `sources/`, `rights/`, `schemas/` | Source descriptions, rights receipts and data schemas |
| `MODEL_CARD.md` | Maintained scientific description of SLp-1.1 |
| `docs/` | Development guide, module reference, literature and results ledger |
| `model/v1/` | Frozen SLp-1 source and model card |
| `data/`, `results/`, `.omf/`, `ontology/` | Ignored local payloads and runtime state; tiny synthetic fixtures are the exception |

Use branches and pull requests for collaborative changes. Keep datasets,
checkpoints, caches and credentials out of Git; publish versioned artifacts to
Hugging Face and update the artifact lock. The development guide explains the
checks and publication workflow.

## License

Original SLp code and weights are [MIT licensed](LICENSE). Biological datasets,
annotations and bundled third-party components retain their
[source terms and attribution](release/THIRD_PARTY_NOTICES.md).
