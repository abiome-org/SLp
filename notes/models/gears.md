# gears (and CPA, scGPT / Geneformer perturbation heads, additive baseline, Boolean models): NOT RUN

battery: none; species: (human only, in principle); needs: combinatorial Perturb-seq training data.

## Why these were not scored on SLB
- **GEARS** (Roohani et al. 2023, Nat Biotechnol; github.com/snap-stanford/GEARS, MIT) predicts the
  transcriptional outcome of single and double perturbations; its genetic-interaction module ("magnitude",
  "synergy" etc.) is learned from observed double perturbations. The released models and every published
  GI use are trained on Norman et al. 2019 (K562 CRISPRa, 131 gene pairs). Under the SLB contract a
  combinatorial readout is a GI label, so any GEARS with a working GI head is leaky, and Norman's pairs are
  CRISPR *activation* of a small set of TF/cell-cycle genes, not loss-of-function; essentially no SLB pair
  is in it. Trained only on single perturbations (e.g. Replogle 2022 K562 genome-wide, which would be a
  permitted input), GEARS has no signal to learn non-additivity and its double predictions collapse to
  near-additive, i.e. predicted GI ~ 0 for every pair.
- **CPA** (Lotfollahi et al. 2023) composes perturbation embeddings additively in latent space; same
  situation: GI terms are learned only from combinations.
- **scGPT / Geneformer perturbation heads**: the scGPT perturbation fine-tune uses the GEARS data loaders
  and Norman/Adamson data (combinations = leaky); Geneformer "in silico deletion" of two genes has no
  published or validated GI read-out. Gene embeddings from these foundation models are single-gene
  features and are covered by models-features / models-features-fm.
- **Linear additive baseline** (Ahlmann-Eltze et al. 2024, "deep learning-based predictions of gene
  perturbation effects do not yet outperform simple linear baselines"): predicts double = sum of singles, so
  its GI is exactly 0 for every pair: a constant score, SLB 0.500 by construction.
- **Boolean / logic network combination predictors** (e.g. MaBoSS-based or CellNOpt drug/knock-out
  combination screens): model-specific small networks (tens of nodes) with no genome-scale gene coverage
  and no generic code path from a gene pair to a prediction; not applicable to SLB's genome-wide pairs.

## What would make this runnable
A loss-of-function combinatorial Perturb-seq in SLB-train families only (none public), or accepting a
leaky run on Norman 2019 (coverage of SLB pairs ~0). Neither is worth GPU time now.
