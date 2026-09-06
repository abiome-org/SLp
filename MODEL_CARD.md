# SLp-1.1 cellular world model

SLp-1.1 learns molecular state, genetic-intervention dynamics, and distributions
over RNA/protein observations. The current model has **14,118,917 trainable
parameters**. Its training targets are molecular measurements. Its molecular core has no
synthetic-lethality training target or fitted linear response backbone. A separate
SL decoder now reads its simulated single- and double-intervention consequences.

The implementation is [cell-world-v1](modules/slp-1-1-cell-world-v1/CONTRACT.md).
The earlier population models and supervised SL readouts remain historical
baselines. The following SL scores are measured on this cellular world itself.


## Synthetic-lethality capability

The complete pipeline is **molecular state -> genetic interventions -> predicted
molecular consequences -> SL decoder**. The 14.12M-parameter core stays frozen.
The SL decoder receives 1,080 coordinates derived from molecular simulations in
K562, RPE1 and HepG2; it receives no direct raw gene descriptors or legacy SL
classifier outputs. An additional fixed response-cosine decoder needs no SL labels.

| Official MuSL CV3, ten cold-gene folds | AUROC | Average precision |
|---|---:|---:|
| Cellular world + trained SL decoder | **.668168** | **.668591** |
| Untrained world + same SL decoder protocol | .634180 | .623050 |
| Direct descriptors + same SL decoder protocol | .798579 | .802964 |
| Cellular world, fixed label-free response score | .540000 | .532153 |
| Untrained world, fixed label-free response score | .517388 | .518348 |

The pretrained world improves the world-only SL decoder mean by .033989 AUROC and
.045541 AP over the untrained control. It improves AUROC in eight of ten folds.
The descriptor-only classifier is stronger than the current world-only decoder.
The gene-cluster interval for the world-minus-random AUROC difference is
[-.001690,.069390], including zero; the mean gain alone is not conclusive attribution.
The SL decoder is fitted to training-fold SL labels; the label-free row is not.

The fixed label-free decoder also scores MuSL CV1/CV2 at .531706/.533703 AUROC.
In SLAMR scenario 3, its mean reciprocal rank is .208321 in Jurkat and .126011
in K562, versus .086240/.069912 for the untrained-world control. Exact coverage,
all folds and direct descriptor controls are in the result ledger.

The measured capability is SL prediction from the molecular world,
with both a trained downstream decoder and an executable no-SL-training readout.
The pooled label-free CV3 AUROC is .54014 with a descriptive gene-bootstrap 95%
interval [.51874,.56141]. Its world-minus-random interval includes zero; the
mean gain alone is not conclusive pretraining attribution for that label-free
score. These retrospective benchmarks support the stated predictive capability,
not an unrestricted claim about causal reasoning or leading every SL benchmark.

The complete research predictor is
`results/slp11-transition/cell-world-sl-predictor-v2/`, including the actual world
weights, ten SL decoders, three molecular contexts and descriptors for 7,683 human
genes. `SLPredictor.predict_pairs` accepts exact gene symbols or stable IDs;
`predict` accepts biological descriptors for additional supported genes. The
ensemble is for subsequent inference; benchmark scoring uses each test fold's
own decoder. The molecular generation API remains available as `predictor.world`.

All 30 saved world/control decoders reproduce 22,175 test occurrences exactly.
On 32 descriptor-to-score requests, Linux CPU and Windows CUDA produce identical
ensemble SL scores; maximum molecular-feature drift is 1.72e-5. This verifies
standalone research inference, separately from a service deployment.

## Training status

Training is complete: 12,000 state/dynamics updates followed by 10,000 joint
generative updates, seed 731, batch size 16, on one RTX 4070. The final stage
took 1,541.59 seconds and used 710.14 MiB peak allocated CUDA memory.

Final checkpoint SHA-256:
`9ecffff8d83e41878da638847d77a950077d2270e2cf6dc951d3e9e0309c4a15`.

## Architecture

| Component | Implemented shape and role |
|---|---|
| Observation encoder | Variable RNA/protein query panels; 32 latent slots of width 256; three attention blocks |
| Genetic action encoder | Static gene descriptors and explicit intervention mechanism; exchangeable action sets |
| State dynamics | Four conditional attention blocks; persistent latent state across successive interventions |
| Generative flow | Four conditional blocks; rectified flow over molecular latent residuals |
| Molecular decoder | Queried response means and reconstruction variance, conditioned on measured control context |
| RNA observation model | Bernoulli detection plus positive lognormal processed expression |
| Protein observation model | Continuous Gaussian observations in the native assay space |

The core is trained on molecular observations. Frozen ESM2-8M protein vectors,
shared Gene Ontology features, and available STRING descriptors supply 642 static
gene features. Protein measurements retain their antibody barcode identity in
60 fixed features. These are not learned gene-ID embeddings or forced RNA/protein
identity matches. There is no fixed learned gene vocabulary.

Both human (NCBI 9606) and yeast (NCBI 4932) retain native identities. CRISPRi,
CRISPRa and knockout have separate mechanism codes. Assay conditioning preserves
the distinction between cell log-expression, population means and control-standardized
responses. Query-specific control values are measured biological inputs.

## Molecular pretraining mix

The fixed input index is `data/derived/slp11-cell-world-training-v5/`.
These are source-sampling probabilities per update, not proportions of unique
cells visited or a claim that every accessible cell was consumed.

| Source | Accessible indexed observations | Update share |
|---|---:|---:|
| Replogle K562 individual cells and controls | 197,804 | 15% |
| Replogle RPE1 individual cells and controls | 152,951 | 15% |
| Frangieh paired RNA/protein cells | Up to 93,397 fitting-split cells, with additional global gene filtering | 15% |
| Nadal-Ribelles yeast, Control and NaCl | 38,978 population views | 20% |
| Replogle K562 essential populations | 1,443 | 4% |
| Replogle RPE1 essential populations | 1,666 | 4% |
| Replogle K562 genome-wide populations | 7,438 | 10% |
| Nadig HepG2 populations | 1,758 | 4% |
| Norman single/combination populations | 107 | 7% |
| MCF10A full medium day 0 | 28 | 2% |
| MCF10A full medium day 6 | 27 | 2% |
| MCF10A TGF-beta1 day 6 | 23 | 2% |

Human population sources total 12,490 fitting views. The paired-cell source
contains 103,862 cells before its reconstruction split; 20 measured protein
channels accompany RNA. The global human intervention exclusion contains 1,506
genes. Yeast training and development genes are separately partitioned.

RNA/protein reconstruction, RNA-only to protein imputation, molecular response
prediction, latent endpoint prediction, flow matching, and combination
composition train one shared model. The generative stage also trains RNA
detection and positive expression, and alternates the two orders of measured
combinations. Controls and perturbed cells are unpaired; minibatch matching is
a computational population coupling, not an observed cell trajectory.

Human cell RNA is log1p(CP10K). Yeast targets are already population means of
per-cell log1p(CP10K); no second log transform is applied. Other population
sources retain their recorded native response space. Full source hashes,
normalization, rights receipts, seeds, draws and module captures accompany runs.

## What the API does

`WorldModel(bundle_directory)` loads the actual safetensor weights and verifies
the bundle. The methods have distinct meanings:

- `encode`: represent supplied molecular observations and biological context.
- `intervene`: apply an action set to latent state; empty actions are exact identity.
- Repeated `intervene` calls: compose different interventions in persistent state.
- `decode`: predict an anchored molecular change in native assay units.
- `reconstruct`: decode a state relative to its basal profile, including withheld coordinates.
- `sample`: draw an alternative latent endpoint using the conditional flow.
- `generate`: draw human RNA/protein measurements from the learned observation distributions.

Individual-cell observation distributions are trained for the human cell assays.
Yeast supports molecular population-state prediction, not a claimed yeast
individual-cell observation likelihood. RNA generation has an explicit zero mass
and nonnegative positive component. It generates processed expression, not raw
sequencing counts; partial panels do not enforce whole-cell library totals.

The deterministic forecast, latent variation and observation variance must not
be interpreted as interchangeable or automatically calibrated uncertainty.
Flow integration time is computational time. Simultaneous combination endpoints
do not establish biological temporal dynamics.

## Evaluation and artifacts

Full molecular development evaluation completed on the final checkpoint.
Scores below use matched supported query coordinates. GWPS/HepG2 rows pool
multiple views of the same held gene; full source-view results are also saved.
Landscape correlation removes each query's average perturbation effect.

| Held-gene endpoint | Model MSE | Unchanged state | Training mean | Centered landscape r |
|---|---:|---:|---:|---:|
| K562, 305 genes | .003952 | .004373 | .004021 | .230 |
| RPE1, 360 genes | .009851 | .012422 | .009368 | .212 |
| K562 genome-wide, 1,491 genes | .012032 | .012575 | .012352 | .176 |
| HepG2, 361 genes | .064356 | .069988 | .064567 | .184 |
| Yeast Control, 346 genes | .024072 | .023268 | .022321 | .144 |
| Yeast NaCl, 346 genes | .022721 | .022068 | .020981 | .156 |

The model improves all four human endpoints over unchanged state and three
over the training-mean comparator. RPE1 still favors the training mean; yeast
also favors the simpler controls in MSE. These comparisons do not establish
improvement over the strongest historical fitted response models.

For 23 Norman held combinations, direct MSE is .020784 versus .040451 for
unchanged state and .021025 for predicted additive. Persistent latent
composition gives .021307. The three MCF10A direct MSEs are
.003667/.004656/.006123; their corresponding unchanged controls are
.004130/.004659/.007119. Composition benefits vary by route and environment.

Generated cells were checked in six intervention/context groups containing 56
held cells. Protein energy distance improves over sampled controls in five
groups; RNA improves in three. RNA sparsity and observation variance are
reported per group. These small groups establish executable, nontrivial
generation, not calibrated uncertainty or broad cell-generation superiority.

The separate globally held Frangieh intervention evaluation is weaker: its
control-mean-conditioned protein predictions exceed unchanged-control MSE in
all three contexts. This route conditions a nonlinear cell model on a population
mean; it does not integrate predictions over the observed control-cell
distribution. It is reported explicitly rather than treated as demonstrated
held-gene protein transfer. RNA-only protein reconstruction is also reported
for the held-cell groups, with mixed improvements.

The full report is
`results/slp11-transition/cell-world-v1-generative-development-final/report.json`,
SHA-256 `71a62bd15f3de634319ce2785f0ddce89c9ca6573eb7e18b02bb0cf3b026e8a4`.

These molecular development sets have been inspected during prior work and this
run. They are not independent prospective confirmation. The molecular core uses no synthetic-lethality labels. The separate SL
application uses the explicit training-fold and label-free protocols above.

The completed standalone artifact is
`results/slp11-transition/cell-world-v1-generative-research-export-v1/`.
It includes weights, configuration, normalizer, a real molecular example,
source-rights receipts, inference code and dependency contracts. Native/Linux
CPU replay agrees within `4.76837158203125e-7`; empty interventions preserve
observations exactly. [experiment-cell-world-replay.yaml](experiment-cell-world-replay.yaml)
materializes the already-trained bundle and checks its Linux replay through
OMF 2. It performs zero optimization steps; native CUDA training and OMF artifact
replay are reported separately. Standalone export is distinct from an OMF
ModelPackage service deployment.

Prior molecular and supervised SL results remain in [docs/results.md](docs/results.md).
Frozen SLp-1 evidence remains in [docs/model-card.md](docs/model-card.md).

The OMF 2 artifact replay completed successfully as run
`01a074cc-b93d-78be-8ae9-be8db3cf6096` with no evaluation failures.
Its verified export, including weights and immutable evidence, is
`results/slp11-transition/cell-world-v1-generative-omf2-export-v3/`.
