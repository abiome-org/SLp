# SLp-1.1

## What this model is

SLp-1.1 predicts population-level molecular states after genetic intervention.
Its main scientific question is whether intervention-conditioned state modeling
transfers to unseen action genes and biological contexts. Synthetic lethality is
a downstream use case, not the training target.

The current line has two retained components:

- a strong linear response backbone for K562 and RPE1;
- a shared neural population-state model spanning eight human assay/environment contexts.

SLp-1 is frozen proof-of-concept work. Its architecture and gene universe are
not defaults here. Its historical card is [docs/model-card.md](docs/model-card.md).
The full experiment and correction ledger is [docs/results.md](docs/results.md).

## What users can do

For a supported context, users supply one or more statically described genetic
actions and a molecular background state. The model predicts the resulting
population-mean state on that context's native measured query panel.

The current bundle supports:

- single-action forecasts in K562 essential CRISPRi, RPE1 essential CRISPRi,
  K562 genome-wide CRISPRi, and HepG2 CRISPRi;
- single-action and composed two-action forecasts in Norman K562 CRISPRa;
- MCF10A Cas9 knockout forecasts in full medium at day 0/day 6 and TGF-beta1
  medium at day 6;
- direct two-action prediction and sequential action composition;
- prediction from an observed intermediate state;
- explicit query-support masks for every context;
- deterministic local loading from safetensors and NumPy artifacts.

Outputs outside a context's supported-query mask are numerical placeholders.
The model does not produce individual cells, count distributions, temporal
trajectories, viability, synthetic-lethality probabilities, or clinical advice.

## Inputs and outputs

Actions use 642 global descriptors:

- 577 static sequence, protein, and Gene Ontology features;
- the public SLIM STRING64 embedding;
- an explicit STRING-coverage bit.

Stable Ensembl identifiers and NCBI taxonomy IDs carry gene identity. There is
no learned gene-ID vocabulary. Each context adapter defines native query IDs,
support mask, basal anchor, assay, intervention mechanism, response scale, and
frozen linear prior.

The neural API accepts raw action features, an exchangeable action-set mask, a
full native-panel basal state, optional observed-background values, and optional
aligned control-context values and mask. It returns the observed background plus
a frozen action prior and learned state update. Empty action sets return the
observed background exactly.

## Current architecture

The current eight-context model has 910,725 parameters, width 128, four state
slots, four attention heads, three mechanism IDs, and five assay IDs. It
distinguishes CRISPRi, CRISPRa and Cas9 knockout and keeps these endpoint systems
separate:

- log1p mean CP10K for K562 and RPE1 essential panels;
- Norman control-z expression;
- author-normalized K562 genome-wide control-z expression;
- separately normalized HepG2 control-z expression;
- MCF10A mean per-cell ln1p(CP10K), distinct from ln1p mean CP10K.

A value-bound masked encoder preserves gene-response pairing while remaining
invariant to permutation of whole query/value pairs. A separately masked control
expression context supplies basal information; it is distinct from the zero
basal anchor of control-z endpoints.

Training uses population endpoints with residual MSE and masked observed-state
reconstruction. There is no single-cell likelihood. Neural outputs correct
frozen reduced-rank action priors.

## Strong retained linear backbone

The strongest native backbone is a rank-16 reduced-rank map from the 642 action
descriptors. Rank and regularization were selected by fitting-only three-fold
cross-validation, including fold-local normalization.

| Native development panel | MSE | Independently query-centered r |
|---|---:|---:|
| K562 essential, 305 held actions | .00334073 | .27656 |
| RPE1 essential, 360 held actions | .00795145 | .32614 |

It improves the earlier static577 rank-32 backbone in both contexts and remains
the strongest retained RPE1 predictor.

## Corrected canonical SLIM comparison

Frozen models were scored on the official GEARS simulation split: 273
K562 and 386 RPE1 unseen-single conditions. The canonical comparison now uses
SLIM's pinned population scaffold: fitting/control cells are sampled with seed
1, rescaled gene-wise to each frozen predicted mean, clipped per cell to
[0,14.99], and scored by the authors' per-condition definition.

| Frozen canonical model | K562 Pearson-delta | K562 MSE | RPE1 Pearson-delta | RPE1 MSE |
|---|---:|---:|---:|---:|
| Published-default SLIM, K10/lambda .1 | .476035 | .00585907 | .636037 | .01016149 |
| Fitting-CV SLIM | .501233 | .00569426 | .645247 | .01002488 |
| static577+STRING64+presence reduced rank | **.516688** | **.00539291** | **.652638** | **.00944712** |

Against fitting-CV SLIM, paired condition-bootstrap 95% intervals for Pearson
gain are [.004568,.026433] in K562 and [.002304,.012975] in RPE1. MSE reduction
intervals are [.00012969,.00048137] and [.00031991,.00084358].

These are corrected results. The first join omitted valid official STRING
vectors for 3 K562 fitting genes, 17 RPE1 fitting genes, and 4 RPE1 test genes.
The correction loads STRING64 directly from the official HDF5, joins static577
through stable IDs, and reuses every frozen hyperparameter and scoring choice.

The frozen implementation uses deterministic full SVD; reconstructing its
means matches within 8e-15. The authors' unseeded randomized PCA changes mean
values by only 3e-6 to 1.4e-5 across recorded seeds 1/2/3. The repository does
not contain the README-linked manuscript result CSV, and this exact pipeline
does not reproduce its .499/.613 headlines. Those paper values are not treated
as matched scores. This remains retrospective evidence on source biology used
during development, not prospective state-of-the-art evidence.

## Shared neural evidence

The five-context generation completed all three Norman combination folds with
seed 731, 20,000 updates per fold, and identical numerical architecture. Fold 0
ran through OMF 2 on Linux CUDA; folds 1/2 ran on native Windows CUDA. All 59
held pairs occur exactly once, on the same 7,182 common measured queries.

| Five-context Norman forecast | Pooled MSE | Centered nonadditive r |
|---|---:|---:|
| Unchanged control state | .0444222 | .436245 |
| Frozen additive prior | .0166031 | .419930 |
| Direct two-action model | .0158734 | .434552 |
| Model-predicted additive | .0165772 | .407463 |
| Autonomous two-order average | **.0155180** | **.450436** |
| Observed-parent prior | .0162434 | .419930 |
| Observed-parent model | **.0150617** | .442280 |

Autonomous composition improves pooled MSE by 6.39% over predicted additive.
Paired condition-bootstrap 95% intervals are [.000252,.001897] for MSE reduction
and [.01457,.07224] for centered nonadditive correlation gain. Fold 2 favors
direct prediction over autonomous rollout; all fold results remain in the
ledger. The three folds are one seed, not independent biological replications.

The eight-context generation adds three MCF10A knockout environments and was
trained for 20,000 updates in 994 seconds on the RTX 4070 (2,477 MiB peak allocated
GPU memory). Its seed-731/fold-0 development performance is:

| Source | MSE | Perturbation-specific centered r |
|---|---:|---:|
| K562 essential, 305 held actions | .00331249 | .282763 |
| RPE1 essential, 360 held actions | .00808808 | .320072 |
| K562 genome-wide, 1,491 unique genes | .01176522 | .092601 |
| HepG2, 361 unique genes | .05632339 | .247783 |

RPE1 still favors the retained 642-feature linear model (.00795145 MSE).
For MCF10A held pairs, autonomous/predicted-additive MSE is .00549782/.00664156
in full medium day 0 (10 views), .00629195/.00764973 in full medium day 6
(10 views), and .00902911/.01243533 in TGF-beta1 day 6 (7 views). These are
known-gene combinations; guide-specific views can share a stable gene pair.
Unchanged-control MSE is lower still (.00412984/.00465950/.00711903), so the
MCF10A composition gains over additive prediction do not yet establish improved
absolute intervention prediction over this basic baseline.

Minimal-medium day-6 outcomes were absent from training. A frozen full-medium
adapter received the minimal-medium control profile, without refitting. Across
27 double-intervention views, autonomous MSE is .00961428 versus .01218559 for
predicted additive, but the unchanged control state is substantially better
(.00428156). For 10 single-intervention views, model/control MSE is
.00552547/.00303003. The model therefore learns a composition correction relative to its additive
forecast, while reliable MCF10A intervention prediction and transfer to this
held medium are not established. Observed-parent forecasts use measured single-intervention outcomes
at inference and are a separate information setting.

## Data scope

Training uses Replogle K562/RPE1 essential CRISPRi, author K562 genome-wide
CRISPRi, HepG2 CRISPRi, Norman K562 CRISPRa, and GSE164996 MCF10A Cas9 knockout
populations. The eight-context run uses 1,443, 1,666, 7,438, 1,758, and 107
fitting populations from the original five sources, plus 28/27/23 fitting
MCF10A views. Eight active MCF10A genes remain after held-gene exclusions.
A shared stable-gene-pair hash withholds fold-zero pairs in every MCF10A fitting
environment. All minimal-medium outcomes are physically absent from training.
MCF10A data are public GEO research measurements with conditional redistribution
rights; they are not relabeled as CC0.

Held intervention genes are excluded from fitting quantitative trajectories for
the corresponding evaluation. Population-view metrics weight source/GEM views
separately; unique-gene metrics first average views by intervention gene.
Perturbation-specific centered correlation removes the mean across interventions
and each query profile mean.

The GEARS comparison uses the official simulation split, seed 1. Replogle and
Norman archives are CC0 1.0 with scoped rights records. SLIM is pinned at commit
`5a7e9ade5d0a6b6331e6dbc81181450605047bcc`; the STRING HDF5 SHA-256 is
`789416877b8701ef6f800106d26bf7bb97ea8e72744e6ab93e24933a717f247d`.

## Artifacts and API status

The earlier static577 rank-32 model has a verified OMF 2.0.0 captured-script
build. OMF trains, evaluates, reproduces, and exports its 4.77 MB model directory.
Standalone CPython 3.12 inference reproduced all 5,761,355 development values
within 8e-15 without OMF installed.

A local unpromoted OMF v2 research release exists as
`slp11-response-rank32-omf2-20260905`, revision
`sha256:283f75a26c26433681a54fb9b9fcdf007806487d0c037f7c50e185d7e004c115`.
It is a usable research artifact, not a deployed service.

The five-context neural OMF run `01a07387-19fa-76f1-8e7a-bb9568b9d870`
completed training and evaluation and was exported through OMF's supported
experiment export command. Its artifact is retained under
`results/slp11-transition/joint-world-expanded-fold0-omf2-export-v1/`.

The current eight-context standalone bundle is
`results/slp11-transition/joint-world-eight-context-research-export-v1/`.
It contains 26 hash-verified payload files, including all eight adapters and
priors, the actual checkpoint, inference code and pinned Linux requirements.
Its API defaults to the checkpoint selected in the export manifest. A clean
Linux replay with OMF unavailable verified all eight contexts, empty/single/double
actions and support masks. Maximum Windows/Linux prediction drift was 2.16e-7;
empty actions preserved the supplied observation exactly in each runtime.
No training dataset is required for inference.

`experiment-context-world.yaml` is validated for OMF 2 replay. The eight-context
weights were trained natively; that new OMF experiment has not been executed.
These are local unpromoted research bundles. OMF ModelPackage service deployment
and external publication have not been completed.

## Limitations

- Results are population-mean forecasts on measured native panels.
- The neural model does not define a cell distribution or temporal process.
- RPE1 currently favors the linear backbone over the neural model. MCF10A
  currently favors retaining the measured control state in absolute MSE.
- Norman composition is measured across three folds at one seed; the eight-context
  extension has one completed fold.
- GWPS and HepG2 are adaptive development contexts.
- The canonical SLIM comparison uses the authors' population scorer and remains
  retrospective. It is not a SynLeaf synthetic-lethality benchmark.
- No result establishes general SOTA, prospective transfer, synthetic-lethality
  performance, clinical utility, or deployment readiness.

Exact protocols, hashes, outputs, corrections, and historical
decisions remain in [docs/results.md](docs/results.md).
