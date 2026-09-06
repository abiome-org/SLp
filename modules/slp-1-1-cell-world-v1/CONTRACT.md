# SLp cellular world model

This module learns molecular observations, intervention-conditioned latent state
transitions and a conditional generative flow. It has no classifier, SL labels,
learned gene-ID lookup, external fitted response prior or application score.

The observation encoder consumes variable molecular query panels with static
gene descriptors, explicit assay/modality, organism and perturbation mechanism.
Paired RNA and surface-protein measurements share a cell state. Protein assays
retain their measured antibody-channel identities using fixed barcode features;
they are not silently equated to gene expression.

The deterministic transition accepts a set of actions or an ordered sequence of
sets. All state transitions operate on persistent latent state. Empty actions
preserve state exactly. The decoder queries molecular coordinates and predicts
normalized observation residuals and reconstruction variance. Endpoint forecasts
add decoded state differences to the observed molecular anchor. They contain no
per-context fitted linear response model.

The generative path learns a conditional latent residual distribution by flow
matching to encoded observed cells. Control and perturbed cells are unpaired;
minibatch latent matching supplies a computational coupling, not observed
before/after cell trajectories. Flow integration time is computational time.
Simultaneous double interventions do not supply biological time courses.

Training combines single-cell state reconstruction and cross-modal imputation,
population molecular transition losses, latent endpoint prediction, conditional
flow matching and measured-parent/composed intervention endpoints. Gene-held
outcomes remain outside fitting. Human and yeast identities and assay units
remain separate. Sampling variability is an implemented model output; its
calibration requires empirical evaluation and is not assumed.

The final architecture has 14,118,917 trainable parameters: width 256, 32 latent
slots, a three-block state encoder, four intervention blocks and four conditional
flow blocks. Frozen ESM2-8M protein descriptors, shared GO features and available
STRING descriptors are inputs. The world model itself is trained on molecular
measurements, not sequence language modeling or synthetic-lethality labels.

## Observation distributions

For individual human cells, a Bernoulli detection head and positive-lognormal
head generate processed RNA log-expression. The positive distribution is clipped
at the per-coordinate log1p(CP10K) upper bound. Protein uses a Gaussian decoder
in the measured, isotype-corrected expression space. These are observation models
for processed assays, not sequencing-count likelihoods; partial query panels do
not enforce a whole-cell library-size constraint.

`generate` samples both latent state and fresh molecular measurements. It uses
the learned observation decoder and supplied control context. `sample` samples
latent state only. `decode` anchors a predicted state change to the supplied
observation; `reconstruct` decodes the state directly relative to its basal
profile. These operations have different meanings and are reported separately.

Individual-cell observation distributions are trained for human assay IDs 5
and 6. Yeast supplies population mean log-expression in Control and NaCl; the
model does not claim a learned yeast individual-cell observation likelihood.
Taxonomy IDs are 9606 and 4932. Mechanisms are CRISPRi=0, CRISPRa=1, knockout=2.
Assay IDs and source-specific units are recorded in the exported metadata.

## Standalone API

`WorldModel(directory, device)` verifies the bundle and loads safetensors.
`encode(observed, basal, query_descriptors, ...)` returns a `MolecularState`.
`intervene(state, actions, mask)` returns a new persistent state; call it again
with a different action to compose interventions. `decode(state)` returns native
values and observation reconstruction variance. `generate(state, actions, mask,
seed=731)` returns sampled human RNA/protein measurements.

Observations have shape B by Q. RNA query descriptors concatenate 642 gene
features and 60 zero measurement features; protein descriptors use zero gene
features and the one-hot 15-base antibody barcode. Actions have shape B by A by
642 with an explicit Boolean mask. There is no fixed learned gene vocabulary.
Raw descriptors use the exported static-feature normalizer; an explicit
`normalized_descriptors=True` accepts already-normalized inputs. Assay scales,
query order and observed masks must align with the supplied native measurements.

`make_example.py` builds a small request from the admitted Frangieh control
profile. `replay.py` executes that request using only the exported model. The
OMF replay experiment materializes already-trained weights and verifies Linux
inference; it performs zero optimization steps. Native CUDA training and OMF
artifact replay are distinct operations.

`requirements-native.lock` records the exercised Windows training packages.
`requirements-linux.lock` is the hash-locked Linux inference environment;
`requirements-training-linux.lock` additionally includes SciPy for corpus loading
and minibatch matching. The original, observation-head-free pretraining checkpoint
remains loadable through its recorded configuration.
