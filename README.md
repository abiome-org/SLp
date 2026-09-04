# SLp Open Model Factory

SLp is being rebuilt as an OpenModelFactory (OMF) project for perpetual,
evidence-preserving development of species-aware genomics world models.
SLp-1 remains a frozen proof of concept. SLp-1.1 starts from a new data and
model contract rather than extending the old benchmark-shaped implementation.

Read [`MODEL_CARD.md`](MODEL_CARD.md) first. It defines the first falsifiable
hypothesis, cold-gene boundary, model interface, advancement rule, and current
release blockers.

## What changed

The active SLp-1.1 path is OMF-native:

```text
omf.yaml                         project identity
bindings/                        executor placement, separate from work intent
policies/                        admission and promotion policy
rights/                          dataset rights declarations
sources/                         versioned source-admission plans
schemas/                         versioned corpus contracts
modules/slp-1-1-corpus-audit/   shard integrity and leakage firewall
modules/slp-1-1-world/          fresh species-aware query-decoder model
workloads/                       auditable stage graphs
evaluations/                     molecular-only advancement gates
MODEL_CARD.md                    supported intent and limits
```

Git holds code and versioned intent. OMF's ignored `.omf/` directory holds
local metadata, identity, run state, and content-addressed artifacts. Real
datasets, checkpoints, and experiment products remain outside Git.

The historical `model/v1`, `src/`, `modules/training`, `docs/model-card.md`, and
`docs/results.md` paths reproduce or explain SLp-1. They are not the starting
point for SLp-1.1 architecture selection.

## SLp-1.1 model shape

The new model consumes variable-size sets of basal/context tokens and
intervention tokens, then decodes sparse molecular readout queries. It has no
learned gene-ID table, fixed human gene universe, fixed expression landmark
panel, or synthetic-lethality output. Context and interventions are
permutation-invariant. Species, assay, protocol, action, dose, time, and
readout identity are explicit inputs.

Yeast data remains yeast data. Costanzo SGA and future yeast expression,
fitness, chemical-genomics, and regulatory measurements are trained with yeast
identifiers and context. Sequence and orthology can align representations, but
the pipeline does not relabel a yeast interaction as a human measurement.

## OMF setup

OpenModelFactory 1.0 supports its local lifecycle on CPython 3.11/3.12 on
Linux x86-64. This Windows checkout can run the Python unit tests and schema
validation, but it is not a supported OMF executor host.

On a Linux host:

```bash
git clone https://github.com/abiome-org/OpenModelFactory.git /tmp/OpenModelFactory
git -C /tmp/OpenModelFactory checkout ef26eea2cb694596f7680a4bce400371738cbb4b
python3.11 -m venv .venv
.venv/bin/pip install /tmp/OpenModelFactory
.venv/bin/omf --project . bootstrap
.venv/bin/omf --project . doctor
```

Run the synthetic corpus-audit loop before importing biological data:

```bash
.venv/bin/omf --project . --actor slp-researcher data add \
  data/fixtures/slp11-pretrain --name slp-1-1-fixture-pretrain --mode copy \
  --rights rights/fixture-cc0.yaml
.venv/bin/omf --project . --actor slp-researcher data add \
  data/fixtures/slp11-validation --name slp-1-1-fixture-validation --mode copy \
  --rights rights/fixture-cc0.yaml
.venv/bin/omf --project . --actor slp-researcher data add \
  data/fixtures/slp11-reward --name slp-1-1-fixture-reward --mode copy \
  --rights rights/fixture-cc0.yaml
.venv/bin/omf --project . --actor slp-researcher run \
  workloads/slp-1-1-audit-smoke.yaml --binding bindings/local-linux.yaml
```

For real training, import three separately governed snapshots using the exact
names `slp-1-1-pretrain`, `slp-1-1-molecular-validation`, and
`slp-1-1-molecular-reward`, apply
`evaluations/slp-1-1-molecular.yaml`, and run
`workloads/slp-1-1-pretrain.yaml` only after preflight succeeds.

The world module's empty dependency lock currently means its executor image
must already provide compatible PyTorch and NumPy. This is acceptable for
contract development only; a released workload requires a hash-pinned runtime.
Only a Linux-local binding is declared. Large distributed runs require a real,
tested `omf.executor/v1` plugin; the historical Modal launcher is not treated as
an OMF executor and is not called from inside a training module.

## Data contract

Each snapshot contains `corpus.json`, a stable-CURIE intervention-gene list,
and digest-addressed shards. Production shards are bounded `.npz` files so the
trainer can stream them one at a time. Required arrays are documented by
`modules/slp-1-1-world/trainer.py`; every example carries context, action,
query, species, target, and validity tensors.

The audit stage runs before compute allocation to the model. It verifies shard
bytes and rejects any benchmark labels or any validation intervention gene
present in pretraining or molecular reward. Static public features for held
genes are allowed and recorded separately from quantitative trajectories.

## Validation

```bash
python -m unittest tests.test_slp11_corpus_audit
python -m unittest tests.test_slp11_architecture
python -m unittest discover -s tests
```

The first advancement gate is molecular-only: at least 2% validation NLL
improvement over both a training-set mean and fixed ridge baseline, at least 0.10 perturbation-effect
Pearson correlation, non-negative improvement in every represented species,
and zero leakage or benchmark-label records. External SL benchmarks stay
closed until the candidate and downstream decision rule are frozen.

## Release boundary

No SLp-1.1 release exists yet. OMF 1.0 does not currently materialize a large
training artifact for its independently captured inference adapter. Until that
upstream contract supports portable model payloads, SLp-1.1 may produce
immutable training and evaluation evidence but must not be promoted as a
deployable model. Absolute run paths and weights embedded in metadata are not
acceptable workarounds.

The canonical artifact home remains
[`potteryrage/SLp`](https://huggingface.co/potteryrage/SLp) until an explicitly
versioned replacement is chosen. Existing paths are never overwritten.
