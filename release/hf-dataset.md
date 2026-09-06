---
license: other
license_name: source-specific-terms
license_link: https://huggingface.co/datasets/potteryrage/SLp-1.1-data/blob/main/THIRD_PARTY_NOTICES.md
task_categories:
- other
language:
- en
tags:
- biology
- genomics
- perturbation
- synthetic-lethality
- world-model
size_categories:
- 1M<n<10M
---

# SLp-1.1 prepared research data

Version r1 contains the prepared inputs used to train SLp-1.1's molecular and
functional world components, plus separate downstream-decoder inputs and
evaluation evidence. [Model](https://huggingface.co/potteryrage/SLp) ·
[Code and reproduction](https://github.com/abiome-org/SLp) ·
[Scientific model card](https://github.com/abiome-org/SLp/blob/main/MODEL_CARD.md).

The file hierarchy mirrors the source checkout. Download with
`python scripts/fetch_artifacts.py data` from a cloned source repository.
`artifacts.lock.json` pins this release to an immutable HF commit. The helper
verifies SHA-256 checksums and refuses to overwrite changed local files.
`inventory.json` is the complete file list, with byte sizes, roles and source
terms. Model inference only requires the separate model download.

| Group | Contents |
|---|---|
| Molecular fitting | `data/derived/slp11-cell-world-training-v5/` and all referenced fitting arrays: sparse paired cells, memory-mapped RNA counts and population responses |
| Functional fitting/development | `data/derived/slp11-genomic-fitness-world-v1/`: 4,182,121 observed human fitting effects across 843 contexts and 1,818,947 yeast fitting pairs; development observations are stored separately |
| Application benchmark | Selected MuSL folds, roster and descriptor inputs; these labels are for the separate downstream decoder, never world pretraining |
| Evidence | Fixed contexts, per-fold decoder outputs/controls and molecular/functional evaluation reports |
| Rights | Original source receipts and additional source-scope notices |

Human identifiers retain NCBI taxonomy 9606; yeast identifiers retain 4932.
Human cell RNA is processed log1p(CP10K), not raw sequencing reads. Yeast RNA
targets are population means of per-cell log1p(CP10K). Human fitness targets are
raw DepMap gene effects. Yeast binary records retain relative single/double
fitness; the trainer applies its recorded floor and natural logarithm.
Other population assays retain the units in their manifests.

This is a prepared-input release, not a mirror of every upstream raw dataset or
every historical SLp experiment. The molecular index enforces its own fitting
rows even when a source file also contains held rows. Development and benchmark
files must not be added to fitting. Historical local paths in receipts describe
where preparation occurred; executable loaders use explicit repository-relative
input directories. The development guide gives the supported training commands.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source-specific licenses,
citations, adaptations and exclusions. MIT applies to original SLp code and
weights, not to these third-party biological observations. The GEO components
retain the NCBI public-data-policy designation and submitter rights.
