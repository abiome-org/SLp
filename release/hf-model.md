---
license: other
license_name: mit-weights-with-source-data-terms
license_link: https://github.com/abiome-org/SLp/blob/main/release/THIRD_PARTY_NOTICES.md
tags:
- biology
- genomics
- world-model
- synthetic-lethality
- pytorch
datasets:
- potteryrage/SLp-1.1-data
---

# SLp

SLp models molecular state and the consequences of genetic interventions.
Synthetic-lethality prediction is a downstream application of the learned world.

| Release | Artifact path | Description |
|---|---|---|
| SLp-1.1 r1 | `checkpoints/v1.1/SLp-1.1-r1/` | 15.12M-parameter molecular and functional world; RNA/protein generation, continuous fitness decoding and ten downstream SL decoders |
| SLp-1 | `checkpoints/v1/SLp-1/` | Frozen historical proof of concept |

[Source and usage](https://github.com/abiome-org/SLp) ·
[Model card](https://github.com/abiome-org/SLp/blob/main/MODEL_CARD.md) ·
[Prepared data](https://huggingface.co/datasets/potteryrage/SLp-1.1-data)

SLp-1.1 r1 includes actual weights, inference source, normalizers, biological
contexts, a descriptor registry and downstream decoder weights. Use the
revision-pinned download command in the source repository; inference does not
require the training corpus. The original SLp code and weights are MIT.
Bundled biological data retain their source terms; see the release notices.
Historical artifacts and their terms are preserved.

On ten official MuSL CV3 splits, SLp-1.1's downstream decoder has mean AUROC
0.787830 and AP 0.780602. Its matched untrained functional component reaches
0.734081/0.729673, while direct descriptors reach 0.798579/0.802964.
The molecular and functional worlds are trained on quantitative observations;
human SL labels train only the separate fold-local decoder. These retrospective
results are research evidence, not a prospective SOTA claim or clinical validation.

The pretraining mix combines human CRISPRi/CRISPRa RNA, paired RNA/protein,
native yeast RNA populations, human single-gene fitness and yeast double-mutant
fitness. See the model card for exact populations, units, sampling proportions,
exclusions and species-specific results.
