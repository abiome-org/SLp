# SLp molecular world models

The current development target is above-SOTA human SL prediction across the
combined benchmark suite under CV3: both test genes are withheld from SL-label
fitting and all human perturbation fitting in each fold. Human/nonhuman
perturbation world modeling is the proposed route to that target, followed by
human-only post-training. The [next-model design](docs/development.md#next-model-design-derived-from-strict-cv3)
is a proposal; the artifacts and measured capabilities below remain unchanged.

## SLp-1.2 pretrained base

A **344,953,859-parameter SLp-1.2-XL** base trained for **54,971 of the planned
60,000 updates**, stopping at its allocated time limit. Repeated development
evaluation selected update **26,000** for inference; the final optimizer
checkpoint is retained for continuation. It uses the same admitted mixed-species
corpus as the 142M artifact described below. The recovered local bundle is
`results/slp12-xl-345m-r1-bundle-recovered`, and its complete manifest matches
the native B200 export. macOS CPU replay and isolated Linux OMF replay/export
passed against the exported CUDA example.
Human post-training and external SL benchmarking remain deferred. A matched
evaluation on frozen development panels found lower XL error in **3 of 12
source/modality scores**. Human fitness MSE increased from **0.124835 to
0.137873** (+10.4%); Norman MSE decreased from **0.674013 to 0.585321**
(-13.2%) but still exceeded the fitting-mean baseline (**0.563122**).
XL also improved K562 and HepG2 population prediction; the remaining scores
worsened. Wrong-gene substitutions barely affected single-cell predictions.
This run did not produce a general scaling gain; the 142M artifact remains the
recommended default base. These are selected checkpoints on retrospective
development data, not an independent test or a controlled scaling curve.
The [results ledger](docs/results.md) records the matched comparison, execution,
recovery and costs. All SLp RunPod resources have been deleted after local
artifact verification.

SLp-1.2 is a trained **142,311,171-parameter** shared set transformer for
molecular endpoint and quantitative fitness prediction. Individual intervention
tokens, molecular observations and RNA/protein/fitness queries use one backbone.
Static descriptors support unknown genes; optional learned entity indices use
25% dropout during training. There is no frozen 1.1 simulation bridge or fitted
SL classifier. The implementation and inference contract are in
[`modules/slp-1-2`](modules/slp-1-2/CONTRACT.md).

This artifact is a **mixed-species pretrained base**. The completed schedule
mixed human and yeast quantitative data throughout; it did not contain a
separate post-training stage. Yeast supplies pretraining data and diagnostic
evaluations. The intended post-training stage and final application selection
use human data only. That stage has not been run.

The first campaign completed 40,000 AdamW updates on one RTX 4090; repeated
development evaluation selected update 27,000. It used the pinned 1.1 data
release, with 2,806 human and 1,070 yeast intervention genes excluded jointly
from fitting across sources. The existing sequence descriptors were retained.
After update 1,573, training added 10% measured-control/reference examples in
a captured continuation. No human SL benchmark labels entered fitting.

The final evaluation used 128 batches per source on retrospective development
data. Mean predictions improved over the unchanged-control comparator and
the same model with masked intervention tokens in all eleven measured sources.
Human fitness MSE was **0.136761 versus 0.261167** for the neutral comparator;
yeast fitness MSE was **0.059122 versus 0.066956**. We did not compare 1.2 with
1.1 on matched panels or evaluate an independent benchmark.

A subsequent frozen-base evaluation adds fitting-only means and wrong-gene
swaps on fixed development panels. Human fitness MSE is **0.124835 versus
0.219549** for the fitting-context mean and **0.334593** with wrong genes.
Human K562, RPE1 and HepG2 population predictions also beat their fitting means.
K562 single-cell RNA ties its fitting mean, Norman is worse, and yeast molecular
accuracy barely changes after swapping interventions. Beating an unchanged or
masked-action comparator alone therefore does not establish intervention-specific
accuracy across sources.

The human-only frozen-representation probe provides stronger evidence for the
base's usefulness. With **128 human adaptation labels**, its ridge readout
scores **0.115334 MSE**, beating a descriptor/context readout given **8,192
labels** (0.164430). At 8,192 labels, pretrained features score **0.111505**
versus **0.178959** for a matched random frozen backbone. Adaptation and
evaluation interventions are disjoint and excluded from base fitting. This is
one retrospective development partition within human quantitative fitness;
it does not measure human SL transfer or isolate the contribution of yeast.
The base weights stayed frozen. The [evaluation workflow](docs/development.md)
preserves fixed panels, baselines and readout budgets for future comparisons.

The limitations matter for use. On 32,768 held-gene yeast double-deletion draws,
predicted interaction residuals had correlation **-0.0014** with observations
and MSE **0.006056**, worse than the measured-single additive comparator's
**0.003710**. The latter uses observed single-mutant fitness. In finite sampled
single-cell panels, generated endpoint energy distance improved over sampled
controls only for RPE1 RNA; K562 RNA and Frangieh RNA/protein were worse. The
model supports endpoint-prediction research; this campaign did not demonstrate
useful yeast nonadditivity or broad gains in generated cell distributions.
Human combination-fitness outputs remain extrapolations because this corpus
contains no human double-knockout fitness supervision.

The local candidate bundle is `results/slp12-joint-142m-r1-bundle`; selected
weights have SHA-256
`78becf7fb4b6d1b60d0fdd5ed80c39e5739ab11bb2c9ddf0a146880344bdaca0`.
Training/evaluation source and configuration receipts travel with the artifact.
The final weights passed native CUDA replay, macOS CPU replay and isolated
Linux CPU OMF replay/export. Maximum CUDA-to-CPU error was below 7e-6 in the
shipped real molecular example, including an eight-step generated sample.
Full source metrics, distribution checks, runtime evidence and costs are in
[the results ledger](docs/results.md). Public artifact pointers continue to name
the released 1.1 model. Original code/weights use MIT; data and descriptors retain
their source terms.

## SLp-1.1 published release

SLp-1.1 learns molecular state, genetic-intervention dynamics, RNA/protein
observation distributions, and a nonlinear functional viability landscape.
Its two components contain **15,123,593 learned parameters**: 14,118,917 in the
generative molecular world and 1,004,676 in the functional world. They are
trained on quantitative observations; human SL classification is downstream.
The functional action encoder consumes molecular-world simulations and static
biological descriptors. This is a factorized world; its components were trained
in stages, with separate latent spaces.

The implementation is [cell-world-v1](modules/slp-1-1-cell-world-v1/CONTRACT.md).
The earlier population models and supervised SL readouts remain historical
baselines. The following SL scores are measured on the world components.


## Functional world and SL decoding

The [functional world](modules/slp-1-1-genomic-fitness-world-v1/CONTRACT.md)
represents observed basal context, applies interventions in a common functional
state, and decodes one nonlinear viability function V. Conditional fitness is
V(z+A+B)-V(z+A); the double effect is V(z+A+B)-V(z). Their finite difference
defines the fixed excess-loss SL readout. Signed curvature supports aggravating
and rescuing effects. This is a learned observation of state, with persistent
intervention composition and exact fitness-path consistency.

Functional pretraining uses **4,182,121 observed human gene/cell effects** for
5,034 fitting genes across 843 contexts and **1,818,947 yeast double-deletion
measurements**, including their raw single-mutant fitness. Human and yeast
identifiers, assays and endpoints remain native. No human SL-pair label enters
this training. The capacity phase ran 20,000 updates in 683 seconds on one
RTX 4070; molecular development loss selected update 9,000. Its weights are
`a0968b69b4409ac7faa34687b7ae3c3725c34d958e6587e783fa42fb9f1db818`.

| Official MuSL CV3, ten cold-SL-gene folds | AUROC | Average precision |
|---|---:|---:|
| Trained world + downstream SL decoder | **.787830** | **.780602** |
| Untrained functional component + same decoder | .734081 | .729673 |
| Direct descriptors + same decoder protocol | .798579 | .802964 |
| Trained world, fixed excess-fitness-loss decoder | .554577 | .579712 |
| Untrained functional component, fixed excess-loss decoder | .476810 | .482464 |

The trained world improves the downstream decoder in **all ten folds**. Its mean
AUROC gain over the matched untrained functional component is **.053749**, with
a fixed-model gene-bootstrap interval **[.036096,.072308]**. This establishes
measurable SL transfer through a downstream decoder in this retrospective
protocol. The functional control retains the same frozen molecular signatures;
the comparison isolates functional quantitative pretraining, not the entire
molecular pretraining history. The direct-descriptor classifier remains slightly
stronger. These results are not an independent SOTA determination.

The fixed excess-loss decoder uses no human SL training. Its pooled CV3 AUROC
is .550026, interval [.522725,.580480]. A five-initialization untrained-control
score ensemble reaches .532974; the trained-minus-ensemble interval
[-.017970,.055588] includes zero. Thus the strongest attribution evidence is
the downstream-decoder comparison, not a claim that label-free attribution is
settled. Label-free SLAMR performance is weaker than the earlier RNA response
readout: MRR .143896 in Jurkat and .100902 in K562. All controls remain in the ledger.

Held fitness MSE is .127015 for human gene effects versus .218405 for the
fitting-context mean. Yeast double log-fitness MSE is .062741 versus .069742
for its fitting mean; interaction-residual MSE .004269 does not beat the
fitting residual mean .004242. The world predicts useful quantitative state
and SL features, while fine interaction generalization remains a limitation.

The standalone research artifact is
`results/slp11-transition/cellular-genomic-world-predictor-v2/`.
`functional_predict.SLpWorld` exposes joint `encode`, `intervene`, and `decode`,
as well as molecular generation, native functional queries, descriptor-driven
new-gene action encoding, and named-gene SL predictions. The SL decoder reads
783 coordinates from predicted functional states and fitness observations;
there is no raw-descriptor bypass. Its fitted SL labels are restricted to each
training fold. All twenty decoder/control artifacts replay all 22,175 held-out
prediction occurrences exactly.

OMF 2 run `01a0781b-935a-75e9-a73c-55e9487d83fc` passes complete Linux
artifact replay: CUDA/CPU SL score error 0 and feature error 1.14e-5. The
immutable research export is
`results/slp11-transition/cellular-genomic-world-omf2-export-v1/artifacts/model`.
This verifies standalone inference; it is separate from a production service release.

## Molecular training

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
Frozen SLp-1 evidence remains in [model/v1/MODEL_CARD.md](model/v1/MODEL_CARD.md).

The OMF 2 artifact replay completed successfully as run
`01a074cc-b93d-78be-8ae9-be8db3cf6096` with no evaluation failures.
Its verified export, including weights and immutable evidence, is
`results/slp11-transition/cell-world-v1-generative-omf2-export-v3/`.

## Distribution

The published research release is [SLp-1.1 r1 on Hugging Face](https://huggingface.co/potteryrage/SLp/tree/main/checkpoints/v1.1/SLp-1.1-r1).
[artifacts.lock.json](artifacts.lock.json) pins the exact remote revision and
checksums. Original SLp code and weights use [MIT](LICENSE); source data and
bundled descriptors retain the [third-party terms](release/THIRD_PARTY_NOTICES.md).
Prepared training inputs, separate MuSL decoder inputs and evaluation evidence
are distributed in [SLp-1.1-data](https://huggingface.co/datasets/potteryrage/SLp-1.1-data),
with an explicit per-file inventory and source-specific terms.
The [development guide](docs/development.md) covers downloads, training and contribution.
