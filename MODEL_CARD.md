# SLp-1.1

## What this model is

SLp-1.1 predicts population-level molecular states after genetic intervention.
Its main scientific question is whether intervention-conditioned state modeling
transfers to unseen action genes and biological contexts. Synthetic lethality is
a downstream use case, not the training target.

The current line has two retained components:

- a strong linear response backbone for K562 and RPE1;
- a shared neural population-state model spanning five human contexts.

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

The expanded shared model has 877,444 parameters, width 128, four state slots,
four attention heads, two mechanism IDs, and four assay IDs. It distinguishes
CRISPRi from CRISPRa and keeps these endpoint systems separate:

- log1p mean CP10K for K562 and RPE1 essential panels;
- Norman control-z expression;
- author-normalized K562 genome-wide control-z expression;
- separately normalized HepG2 control-z expression.

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

Eight frozen models were scored on the official GEARS simulation split: 273
K562 and 386 RPE1 unseen-single conditions. The direct-mean scorer adds the
training-control mean, clips to [0,14.99], and averages per-condition
correlation across measured genes.

| Frozen canonical model | K562 Pearson-delta | K562 MSE | RPE1 Pearson-delta | RPE1 MSE |
|---|---:|---:|---:|---:|
| Published-default SLIM, K10/lambda .1 | .476276 | .00585545 | .636390 | .01014863 |
| Fitting-CV SLIM | .501496 | .00569064 | .645608 | .01001188 |
| STRING64 reduced rank | .492983 | .00568554 | .651180 | .01010347 |
| static577+STRING64+presence reduced rank | **.516913** | **.00538925** | **.653012** | **.00943376** |

Against fitting-CV SLIM, paired condition-bootstrap 95% intervals for Pearson
gain are [.004531,.026404] in K562 and [.002322,.012966] in RPE1. MSE reduction
intervals are [.00012974,.00048148] and [.00032024,.00084411].

These are corrected results. The first join omitted valid official STRING
vectors for 3 K562 fitting genes, 17 RPE1 fitting genes, and 4 RPE1 test genes.
The correction loads STRING64 directly from the official HDF5, joins static577
through stable IDs, and reuses every frozen hyperparameter and scoring choice.

This is a retrospective comparison on source biology used during development.
It does not reproduce SLIM's stochastic synthetic-cell scaffold and is not
prospective state-of-the-art evidence.

## Shared neural evidence

The earlier three-context 642-feature generation reached MSE/centered r
.0033355/.2800 in K562 and .0081827/.3216 in RPE1 after 20,000 updates. On
Norman fold 0, autonomous-average MSE was .0151073, 8.38% below its own
predicted-additive forecast.

The first completed five-context run adds K562 genome-wide and HepG2. At 20,000
updates on seed 731 and Norman fold 1:

| Development source | MSE | Perturbation-specific centered r |
|---|---:|---:|
| K562 essential | .00332139 | .281796 |
| RPE1 essential | .00815964 | .320993 |
| K562 genome-wide, 1,613 population views | .01210323 | .084511 |
| K562 genome-wide, 1,491 unique genes | .01176667 | .090593 |
| HepG2, 396 population views | .05860674 | .232930 |
| HepG2, 361 unique genes | .05642275 | .248465 |

On 19 held Norman fold-1 combinations, autonomous-average MSE is .01788604:
8.15% below predicted additive and 5.51% below direct two-action prediction.
Observed-parent-average MSE is .01747486, 9.35% below its observed-parent prior.

The shared model merits completing the remaining folds. RPE1 MSE still trails
the frozen 642-feature prior (.00795168). Fold 2 and an OMF fold-0 run remain
underway. These are adaptive development results, not aggregate composition or
unseen-context evidence.

## Data scope

Training uses Replogle K562/RPE1 essential CRISPRi, author K562 genome-wide
CRISPRi, HepG2 CRISPRi, and Norman K562 CRISPRa populations. The completed
five-context fold-1 run uses 1,443, 1,666, 7,438, 1,758, and 111 fitting
populations respectively.

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

The five-context fold-1 export contains inference code, configuration,
normalization, five adapters, five priors, a safetensor checkpoint, and pinned
Linux requirements under
`results/slp11-transition/joint-world-expanded-fold1-research-export-v1/`.
Native Windows inference is complete. Linux standalone portability is pending,
and no completed OMF neural release is claimed.

## Limitations

- Results are population-mean forecasts on measured native panels.
- The neural model does not define a cell distribution or temporal process.
- RPE1 currently favors the linear backbone over the neural model.
- Norman composition evidence is not yet aggregated across all folds.
- GWPS and HepG2 are adaptive development contexts.
- The canonical SLIM comparison is retrospective and uses a direct-mean scorer.
- No result establishes general SOTA, prospective transfer, synthetic-lethality
  performance, clinical utility, or deployment readiness.

Exact protocols, hashes, outputs, negative results, corrections, and historical
decisions remain in [docs/results.md](docs/results.md).