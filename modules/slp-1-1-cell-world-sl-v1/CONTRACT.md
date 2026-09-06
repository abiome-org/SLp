# Cellular world model with an SL decoder

The unchanged 14,118,917-parameter cellular world checkpoint is trained on
molecular observations. This application simulates single and double genetic
interventions in three fixed fitting-control contexts: K562, RPE1 and HepG2.
The application does not train the molecular encoder, dynamics or RNA/protein
decoder on SL labels.

For each context, single-intervention features contain a fixed 64-dimensional
projection of the mean latent change and a fixed 32-dimensional projection of
512 queried molecular changes. Pair features are symmetric sum, absolute
difference and product of these states; projected double-intervention response;
projected double-minus-single-A-minus-single-B response; and eight molecular
geometry summaries. The resulting 1,080 coordinates contain no direct raw gene
descriptors, public pair relations or legacy SL classifiers. Projections and
query panels are fixed by seeds before label access.

The supervised decoder is a fold-local LightGBM readout. Its tree count is
selected by a second two-new-gene split inside each training fold. Both genes
of an outer test pair are absent from that decoder's SL training data. Molecular
pretraining can contain single-intervention measurements involving those genes.
All ten models and their test predictions are frozen before scoring. Controls
use the same decoder protocol over raw biological descriptors or an untrained
world architecture. The untrained control uses a nonzero random dynamics output
projection because the normal training initializer has an exactly zero one.

A separate label-free decoder is the mean cosine of predicted single molecular
responses across the three contexts. Positive response similarity represents
functional redundancy. This rule uses no SL labels, calibration, sign fitting or
context selection. It is a score, not a viability probability. Decoder-supervised
transfer and prediction without SL labels are distinct measured capabilities.

No feature or checkpoint is chosen using test scores. These public benchmarks
have been inspected historically, so their results remain retrospective. Report
all arms and folds; compare average precision separately from trapezoidal PR-AUC.
The ten overlapping splits are not ten independent biological replications.

`SLPredictor` loads actual world weights plus the ten trained SL decoders. It
accepts raw 642-coordinate gene descriptors and integer pair indices, runs the
world model, and returns an ensemble SL score plus the label-free response
similarity. This ensemble is for subsequent research inference; the benchmark
uses each held-out fold's own decoder. Synthetic-lethality scores are derived
application outputs and do not replace molecular generation in the world API.

Data, simulation features and trained decoder payloads are artifact contents,
not repository contents. The bridge accepts an explicit self-contained world
artifact, and inference needs no repository-relative imports or training data.
