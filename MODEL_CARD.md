# SLp-1.1

## Purpose and current status

SLp-1.1 is a species-aware molecular world-model research program. Its primary
question is whether intervention-conditioned molecular-state modeling transfers
to unseen intervention genes and biological contexts. Synthetic lethality is
a downstream stress test, not the model's pretraining target or identity.

Biological models are trained on the local RTX 4070. The investigator authorized
autonomous dataset acquisition, bounded native CUDA experiments, and scientific
redesign. Exploratory development does not require the independent signed OMF
handoff discussed in older audit entries. Release certification is separate
from making and testing the model.

No candidate is launch-ready or demonstrated state of the art. The strongest
retained measured-panel response model in the current count comparison uses
32 fitted response coordinates: it improves development MSE over static ridge
by 4.24% in K562 and 3.48% in RPE1, with better perturbation-specific
correlation in both. It is available as a local research inference bundle.
The OMF 2.0.0 captured-script build now trains, evaluates, reproduces and exports
that baseline successfully. Standalone inference in a clean environment without
OMF reproduces all 665 development intervention forecasts within 8e-15. A local
unpromoted v2 research release is saved as
`slp11-response-rank32-omf2-20260905`; no service deployment is claimed.
The joint neural count model remains worse than ridge. Supplying fitting-derived
query descriptors improves the matched neural model modestly, but it remains
6.11%/10.73% worse than the retained rank-32 model in K562/RPE1.
Exact experiments, versions, decisions and limitations are in
[the evidence ledger](docs/results.md).

A new SLp-1-inspired composition pilot explicitly trains observed single-to-double
molecular endpoint relations, which frozen SLp-1 did not train. On 59 Norman
K562 CRISPRa combinations, three fixed pair folds and three seeds, the primary
observed-background correction fails: MSE .0177891 versus a symmetric-state
ridge baseline .0161508. The predeclared autonomous rollout ensemble reaches
.0152034, a 5.87% pooled gain, but regresses in one fold and each individual
seed is worse than ridge. The locked-weight follow-up finds 5.40% pooled gain
over the operator's own predicted-additive output, but only 1.86% over a matched
direct-endpoint ensemble; both descriptive gene-weighted intervals include zero.
The signal is not stable across folds and does not reverse the primary rejection.
All constituent singles are available in training, so this is known-gene
combination interpolation, not intervention-cold or temporal emergence.

Earlier, a three-seed human response-query ensemble passes its fixed development
rule in K562 and RPE1 but fails its first frozen held-gene confirmation:
likelihood gains over ridge are positive and below the fixed threshold.

The later matched three-versus-four-context experiment also fails its fixed
advancement rule across seeds 731–733 and their predetermined average. Adding
HepG2 improves its adaptive development MSE by 10.43%, but RPE1 MSE regresses
by 1.67%. A new cell-state encoder has now been trained directly on paired
RNA/protein cells. It improves co-culture protein forecasting over every
existing baseline, but fails the other five environment/modality comparisons.
Its RNA reconstruction also fails the fixed advancement threshold.

Subsequent paired-cell experiments reinforce that reconstruction alone does
not establish intervention forecasting: PCA reconstructs RNA better, while
five of six held-gene forecasting comparisons still fail. Adding transcript
query features changes forecasts little, and a nonlinear PCA-state residual
worsens every comparison. That branch is stopped. A matched biological-process
annotation neural pair also misses its advancement rule in all three human
contexts. These are retained negative results, not release candidates.

A subsequent fixed response-basis transition improves RPE1 MSE by 4.58%
relative to its matched learned-query neural model and raises its centered
correlation to .3244. It still fails the all-context rule: K562 correlation
regresses slightly and genome-wide MSE changes by only .03%. Its output basis
comes from fitting molecular responses on the measured panel, so this result
does not establish prediction of unmeasured genes or assays.

The first neural pilot on rebuilt yeast raw-count RNA also fails: full-panel
MSE is 4.14%/4.51% worse than batch ridge in Control/NaCl, and its
batch-reference-subtracted perturbation correlations are near zero. This
candidate is rejected. Adding action-aligned control expression to human BP
ridge yields only .012%–.035% MSE gains. Neither result supports promotion.

## Model implemented

The self-contained experimental module is
`modules/slp-1-1-world-transition-v1/`. It has:

- an intervention feature-set encoder with no learned gene-ID vocabulary;
- an optional measured basal-state encoder;
- a shared latent transition and independently queried molecular decoder;
- Gaussian measurement means and scales, with optional low-rank shared factors;
- portable safetensor loading, context-aware inference and joint sampling.

A separate `modules/slp-1-1-cell-state-v1/` experiment learns an 802,050-parameter
paired-cell encoder and affine molecular observation heads. Its first fitted
intervention readout is context-local latent ridge. This tests a learned output
representation; it does not establish nonlinear intervention dynamics or a
validated cell generator.

An additional 516,129-parameter prototype in
`modules/slp-1-1-count-latent-state-v1/` combines a conditional Gaussian state,
a variational cell encoder and feature-queried negative-binomial count factors.
Its empty-action expected rate exactly matches the supplied smoothed control
rate. Its first K562 raw-cell pilot trains on 197,804 fitting/control cells
and 8,563 queries. On 305 development genes it improves MSE over the anchored
mean by 5.26%, but is 5.18% worse than static ridge; centered residual
correlation is .1878 versus ridge .2222. It fails the advancement rule.
Fitting diagnostics also trail ridge, and variance effects are modest. A
matched 4,000-update continuation adds molecular population-mean supervision
while preserving cell likelihood. It improves development MSE from .0037580
for the count-only continuation to .0037003 and raises centered residual
correlation from .1888 to .2113, but static ridge remains better at
.0035846/.2222. Paired-gene resampling places the auxiliary model's relative
MSE regression against ridge between .84% and 5.69%; the candidate is rejected.
Conditioning on an observed library total makes the count-factor objective an
approximation, not a coherent joint library generator.

Applying the frozen K562 checkpoint to measured RPE1 controls without RPE1
perturbation fitting is descriptive only. On the 7,226 common queries and all
1,666 RPE1 fitting genes, its MSE/correlation is .01358/.1402, versus
.01306/.1296 for a frozen K562 ridge transfer. Correlation falls to .0071 on
the 223 RPE1-only intervention genes and .0074 on the 1,523 RPE1-only queries.
The construction demonstrates portable control substitution, not useful
unfitted-context transfer. RPE1 development and test counts were not used in
this comparison.

A metadata-only registry now aligns K562 and RPE1 through one static577 ENSG
space while retaining their native query axes, count denominators and 104
source-plus-GEM contexts. A self-contained shared count/population training
step now trains both native panels. At the same 16,000-update budget, the
joint model reaches development MSE/correlation .0036798/.2039 in K562 and
.0090944/.2423 in RPE1. Static ridge remains better at .0035846/.2222 and
.0084534/.2652. K562 MSE improves only .38% over the matched K562-only model,
below the fixed 1% requirement. Reconstruction preservation passes, but the
forecast advancement rule fails. Both checkpoints and CPU replay are retained
as research artifacts. RPE1 development counts are now part of adaptive
development; its protected test counts remain unused.

A fitting-only reduced-rank diagnostic improves ridge OOF MSE by 3.24% in
K562 and 2.87% in RPE1 with 32 supervised response coordinates. A subsequent
fixed full-fit/development test passes in both contexts: MSE/correlation is
.0034327/.2488 in K562 and .0081592/.2859 in RPE1, improving full ridge MSE
by 4.24% and 3.48%. This is the retained measured-panel response model and
stronger comparator for neural count models. A local inference bundle requires
explicit control-group weights and preserves the original signed log1p-profile
predictions. The model remains linear and panel-specific; it does not establish
cell generation, unmeasured-query transfer or independent confirmation.
A separate checkpoint audit finds aligned count/mean decoder gradients in all
sampled fitting batches, so it does not support a decoder-gradient-conflict
explanation for the neural forecast gap.

A fitting-only decoder experiment replaces the exact supervised rank-32 query
loadings with predictions from static gene features. Its fixed 174,656-parameter
decoder regresses held-fitting-gene MSE by 3.58%/3.47% in K562/RPE1. That supports
a query-representation limitation under the tested architecture and optimizer.
The matched count experiment then supplies fitting-derived native-panel query
descriptors explicitly. It improves MSE over its equal-size control by
1.06%/1.82% in K562/RPE1 and improves K562 reconstruction, but remains worse
than rank32 in both MSE and centered correlation. The count candidate is
rejected. The remaining problem is intervention-to-response generalization;
adding measured query descriptors alone does not solve it.

Protein sequence and functional annotations describe interventions and readout
queries. The optional response-query arm adds descriptors fitted only from
training molecular responses. These descriptors require a measured assay
panel; transfer to unmeasured readouts is not established. Query predictions
are invariant to query ordering and chunking within numerical tolerance.

Multiple intervention tokens are supported structurally. Training on singles
does not establish combination effects. Current context-specific references
also mean that unseen-context prediction is not established. The current
pseudobulk likelihood describes uncertainty in aggregate measurements, not a
validated generator of individual cells. No time-dependent trajectory claim
has been demonstrated.

The earlier Gaussian transition training compresses basal context to 64 query tokens selected
by variation among the source control profiles. The larger fixed panel defines
their normalization and support; the context encoder does not consume every
measured gene. The new count-state pilot encodes all 8,563 measured control
queries, with unique experimental groups encoded once per optimizer update.
Broad molecular-state reconstruction and transfer across
unmeasured modalities remain unestablished.

Measurement exposure can change likelihood uncertainty through
`biological_variance + sampling_variance / num_cells`. Cell count never enters
the predicted molecular mean or latent state. Core controls estimate sampling
variation; fitting-only out-of-fold residuals estimate remaining variation.

## Chosen build direction

The next SLp-1.1 model is one shared observation-and-transition model, not a
collection of unrelated endpoint pilots. This is a prospective design; no
performance is claimed for it yet. A query-set observation encoder will map a
measured molecular population into a compact state using values, masks, native
query descriptors and assay context. A mechanism-aware operator will update
that state from an exchangeable set of interventions, preserving CRISPRi and
CRISPRa as explicit modes rather than pooling them as equivalent actions.
Queried assay decoders will map the resulting state back onto each native RNA
panel, using a count likelihood for raw cells and the appropriate continuous
likelihood for normalized molecular endpoints. The observation encoder will
be part of both training and inference so an observed control or perturbed
background can be encoded and acted upon directly.

The retained rank-32 response result supports a useful low-dimensional response
backbone: supervised response coordinates improve over full static ridge in
both K562 and RPE1. It does not support keeping those fitted, panel-specific
coordinates as the final state definition. Conversely, the count models show
that raw-cell reconstruction and a feature-queried decoder alone do not learn
competitive intervention responses. The Norman composition result shows that
state conditioning changes forecasts and that autonomous rollout contains a
tentative signal, while also showing that a small fold-fitted endpoint basis
and separately trained operator are insufficient. Together these results favor
learning observation, action update and queried decoding jointly, while using
the rank-32 model as a strong supervised backbone and comparator.

The attainable first implementation trains the shared encoder and operator on
the available K562 and RPE1 CRISPRi raw-cell populations, retaining their native
8,563- and 8,749-query axes and 104 source-plus-GEM control contexts. Norman
K562 CRISPRa contributes control-to-single, control-to-double and observed
single-to-double endpoint relations on its native 7,182-query panel. Frozen
ESM and GO descriptors define intervention and query identity without a learned
gene vocabulary; small learned projections, mode embeddings and assay heads
connect them to the shared state. The training unit is a molecular population,
because these experiments do not provide paired before-and-after cells.

This corpus can teach a coherent state representation and known-gene
composition, but it cannot establish broad emergence. That requires many more
measured combinations with varied pathways and both constituents held from
combination fitting, plus independent cell contexts and perturbation modes.
Adamson CRISPRi combinations can extend mode coverage once their count endpoint
is prepared; additional independent combination datasets are needed before a
claim that the operator predicts emergent interactions beyond K562 or CRISPRa.

## Data actually available

Yeast development uses 3,811 single-deletion proteome records, 3,679 intervention
genes and 1,850 queried proteins from one source/context. Composite stable
identifiers and taxonomy IDs are preserved. Processed source-wide normalization
limits these results to retrospective evidence.

Human development uses Replogle 2022 essential-gene CRISPRi in K562 day 6 and
RPE1 day 7, with 7,226 common ENSG readouts. The first raw-pseudobulk adapter
produced 3,281 training and 726 development-validation records, with 713 records
in a separate unscored test artifact. Its initial CP10k/log2 transform faithfully
reconstructs raw mean abundance but retains unequal sampling noise and batch
effects. It is now a diagnostic arm, not the target dataset for advancement.

The authors' per-gemgroup control-normalized summaries now form the corrected
dataset used for training. Target measurements and measured raw basal
expression remain distinct modalities. Core-control summaries and contributing
cell counts support explicit uncertainty estimation. Source-specific adapters
record missingness, normalization and exclusions.

Available static human features include ESM2-t6 protein vectors, direct archived
GO molecular-function/cellular-component features, and their aligned fusion.
Identity joins use stable ENSG/UniProt relations, never cross-species display
symbols. Missing proteins and missing annotation coverage remain explicit.
No yeast phenotype is relabeled as a human measurement.

Frangieh2021 Perturb-CITE-seq processed counts have also been acquired:
218,331 paired RNA/protein cell profiles across control, interferon-gamma and
T-cell co-culture environments. The source has 23,712 RNA columns and 24 antibody
channels, comprising 20 molecular targets and 4 isotype controls. Stable source
ENSG mappings cover 18,063 RNA queries and 237 primary intervention genes.
The paired development adapter now contains 1,802 guide-resolved pseudobulks
from 84,121 target cells, with controls from 39,347 verified non-targeting cells.
It retains 151 fitting and 43 validation genes across three environments.
A first shared RNA/protein endpoint model has now been trained and fails its
six-stratum development comparison. RNA normalization and the authors'
matched-isotype protein transform remain separate observation spaces; guide
mixtures and ambiguous assignments are excluded.

Direct cell-state training now has 103,862 paired fitting/control cells in 51
bounded sparse RNA/dense protein shards. The within-fitting-cell reconstruction
split contains 93,397 training and 10,465 validation cells. This reconstruction
validation is distinct from held-intervention evaluation. Original validation
and test intervention cells are excluded from these shards.

Nadal-Ribelles 2025 yeast RNA summaries add 3,419 development records for 1,732
species-native deletion genes in control and 0.4 M NaCl/15 minute environments,
with 6,340 SGD query IDs and explicit missingness. The authors' upstream
`logfoldchanges` estimator is not reconstructable from the archived summaries.
Their extreme values are strongly tied to cell count, so that endpoint is
excluded from joint training and retained as diagnostic evidence.

The author's raw Seurat counts now provide a replacement development corpus:
410,048 eligible cells, 2,013 deletion genes, 6,683 exact SGD RNA queries and
29 environment/batch strata. Counts are normalized per cell as ln1p(CP10k)
using all 6,951 biological source rows in the denominator. Streaming shards
retain absolute sums, squared sums and counts for 47,498 genotype/batch
populations, including 958 verified WT cells. There is no immediate WT
subtraction: some batch controls contain only two or three cells. Independent
WT moment calculations agree exactly. Protected and development-test actions
were excluded before selected count decoding.

Shared-coordinate human/yeast static packs combine ESM320, protein presence
and one jointly fitted static MF/CC GO256 basis. Every eligible yeast action
now has a full-protein ESM representation; 273 non-ORF RNA queries retain
explicit missing protein features. Species identities remain separate. These
resources enable a controlled cross-species experiment; they do not themselves
demonstrate transfer.

## Development protocol and results

The current exploratory protocol fixes a stable gene hash split with seed 731.
Repeated records and the same intervention across contexts share a partition.
Validation and test intervention genes do not contribute quantitative fitting
trajectories. The original protected molecular snapshots and SL benchmarks
remain unopened in this development path.

Checkpoints minimize development gene-macro Gaussian NLL. The exploratory rule
requires at least 0.02 nats per observed target improvement against both fitted
mean and feature-ridge, adjusted profile Pearson at least 0.10, and no failing
represented context. This is a development filter, not the historical OMF
promotion protocol: earlier thresholds and evidence remain versioned in their
original resources. The ensemble and confirmation decision rule were frozen
before its reserved molecular gene holdout was evaluated.

Mean, feature-ridge, nonlinear kernel and reduced-rank molecular-response
comparisons are implemented. Their uncertainty scales use gene-grouped
training-only out-of-fold residuals. Learned response bases are refitted inside
calibration folds. SL signs, thresholds, labels and benchmark-dependent choices
do not enter model selection.

Initial yeast feature/decoder trials failed the development rule. On the
initial human raw-bulk dataset, the response-query model reached adjusted
held-gene correlations of 0.150 in K562 and 0.182 in RPE1. Its NLL improved over
matched feature-ridge by 0.00699 and 0.00113 nats respectively, below the rule.
These are adaptive development results from one seed and a subsequently
diagnosed noisy dataset, not independent confirmation.

On corrected human measurements, the fusion response-query model improves
NLL against a broader ridge/reduced-rank grid across three seeds in both
contexts. K562 adjusted correlations are 0.231â€“0.236; RPE1 correlations are
0.246â€“0.265. The RPE1 likelihood gain still misses the 0.02-nat development
threshold. Gene bootstrap supports positive likelihood improvement, while
the correlation advantage over ridge remains uncertain. A ridge-reference
correction model also improves likelihood but does not pass both contexts.
Training-only neural OOF uncertainty calibration improves likelihood further.
An ensemble of the three independently trained means, with ensemble OOF
calibration, reaches NLL -0.55823/adjusted r 0.2473 in K562 and
-0.25759/0.2693 in RPE1. NLL gains over ridge are 0.03612 and 0.02314;
both contexts pass the unchanged development rule. These are adaptive
development results, not independent confirmation. Member latent coordinates
are retained separately; their average is not a meaningful shared state.

Frozen confirmation on 713 reserved records (323 K562, 390 RPE1) gives NLL
gains over ridge of 0.01465 and 0.01034, respectively, with adjusted r
0.2282/0.2515. Both contexts fail the 0.02-nat rule. The model and rule are
unchanged, and this holdout is now retired from model selection. Positive
same-source transfer is supported; the specified advancement claim is rejected.

Norman 2019 K562 CRISPRa single/double molecular perturbations have also been
acquired from official GEO sources. The static feature universe now covers all
105 intervention genes without changing previously computed feature vectors.
A per-cell/control-normalized pilot has completed. Additive feature ridge
outperforms both randomly initialized and human-initialized neural models;
the transfer rule fails. Its 49 validation constructs each have one held
constituent; there is no two-held-constituent stratum. The normalization is
computed by SLp from author raw counts, not an author-precomputed matrix.

The genome-scale K562 day-8 screen expands the development corpus to 13,058
records across three assay contexts, with 10,719 training and 2,339 validation
records. Shared fully observed fitting queries number 7,036. The expanded
ESM/GO cache has 10,231 genes with previous rows unchanged. The seed731 model
reaches adjusted r 0.2580/0.2825/0.1024 in K562 essential/RPE1/GWPS but fails
the all-context likelihood rule. This is development evidence, separate from
the frozen two-context confirmation.

Adding direct physical-neighbor static features improves this neural model in
all three contexts to adjusted r 0.2802/0.3043/0.1077. Matched feature-ridge also
improves; neural NLL margins over it are 0.01202/0.00201/0.00475, below the
required 0.02 everywhere. A larger control-anchored alternative satisfies
the empty-intervention identity but regresses on molecular prediction and is
not advanced. Doubling the original latent state to128 improves adjusted r to
0.2906/0.3311/0.1081 but still fails the likelihood margin in all contexts.

A minimal control-conditioned module,
`modules/slp-1-1-control-transition-v2/`, removes the need for a new context's
perturbation-derived reference amplitude. It accepts measured controls and
uses a shared fitting-only query amplitude, with exact empty-action identity.
Its physical1156/state128 candidate reaches adjusted r
0.2841/0.3156/0.1048. It fails the development margin and regresses in GWPS
likelihood relative to its base577/state64 predecessor. Its portable runtime
has passed source reload and target-free new-context inference checks;
these engineering checks do not establish biological prediction accuracy.

A fixed larger-protein-encoder screen is complete: ESM2-650M with matched
PCA320 improves the two essential-gene contexts but regresses genome-scale
K562; its all-context replacement rule fails. Equal-context/equal-gene
training weights also fail the matched neural rule, trading a K562 improvement
for regressions elsewhere. Neither is adopted as the default. The independent
state-difference decoder also fails the matched performance rule, despite
passing its zero-effect consistency checks. A new observed-state module adds
measured-response reconstruction and latent prediction during fitting;
forecasting still accepts only controls and intervention features. Its first
run improves the revised decoder in the essential-gene contexts but also fails
the all-context rule: adjusted r is .2842/.3208/.1002, and every source misses
the .02-nat ridge margin. This candidate is not promoted. Model-specific OOF
uncertainty calibration of frozen v2 improves NLL to -.553117/-.255993/-.914383,
with unchanged mean predictions; all three ridge margins still fail.

A stricter landscape audit removes prediction and truth query centroids
separately before scoring gene profiles. Under this definition, v2 reaches
.2949/.3061/.1079, versus ridge .2693/.3150/.0844. RPE1's earlier advantage
therefore does not survive removal of the models' systematic average effects.
Subsequent experiments require nonregression under this metric as well.

The nonlinear state/query observation decoder also fails: source NLL is
-.544380/-.240803/-.913206, and independently centered RPE1 correlation is
.2913 versus ridge .3150. Increasing observation flexibility alone has not
resolved transfer. A separate paired-state module now supports distinct RNA
and antibody observation heads over one intervention-conditioned state. Its
Frangieh pilot selects epoch 10 and fails all six environment/modality gates.
Only Control protein reaches adjusted r above .10, and its MSE is worse than
the physical-feature ridge baseline. RNA landscape correlations are .003â€“.011.
Portable CPU reload from stored transformations, raw intervention features and
explicit controls passes; this is an engineering result, not biological success.

An explicit per-gene state core with two sparse physical-interaction message
steps has now been trained on all three source contexts. The static-feature
pilot fails every context's advancement rule; independently centered
correlations are .1604, .1260 and .0974 in K562 essential, RPE1 and K562 GWPS.
A controlled response-descriptor augmentation also fails all three contexts,
so this graph configuration is shelved. A larger
fitting-derived response-query representation
also fails on the existing global-state model: independently centered RPE1
correlation .3103 remains below ridge .3150, with likelihood regressions in
both essential-gene contexts.

A four-context adaptive-development dataset now adds the retired HepG2
diagnostic to Replogle, retaining only fitting and validation gene partitions.
It contains 15,212 records and preserves every original Replogle prefix.
HepG2 target-space control sampling estimates are unavailable, so its distinct
normalization and uncertainty boundary require an explicit training adapter.
Future HepG2 results cannot be claimed as unseen-context confirmation.

The independent data audit found cell counts ranging from 2 to several
thousand per summary, strong count-dependent control noise, retained batch
effects, and differing construct efficacy. Those findings drive the corrected
data and exposure-aware likelihood. Duplicate library constructs are not
biological replicates or a noise ceiling.

## Work toward launch

The active experiment directly tests context generalization in acquired
Nadig2025 HepG2 CRISPRi data. Metadata define2390 genes and2544 construct
populations;1665 genes have source-fitting responses and725 do not. World and
strong transferred-baseline forecasts were frozen before outcome processing.
The endpoint is explicitly SLp-computed per-GEM control-standardized expression,
not the authors' DESeq2 log-fold change. The first frozen HepG2 diagnostic is
complete: seen-gene MSE improves by at least2.31% over every
baseline, but profile correlation trails same-gene response transfer
(.2625 versus .2812). For unseen fitting genes, world correlation .1711 exceeds
equal-source ridge .1639, while MSE is1.04% worse. The fixed test fails both
strata. The descriptive paired 95% bootstrap interval for unseen-gene MSE
improvement against equal-source ridge is [-1.73%, -0.30%], supporting a
regression for this frozen candidate. Those outcomes cannot
support an untouched claim for later models or enter their selection.
Common basal descriptors use pooled control
counts on one fixed 6,789-gene panel measured across all four sources. An
earlier full-library compatibility assumption was unproven and superseded;
the fixed-panel denominator is explicit. Downstream SL
assessment remains a separate later evaluation.

A launch claim requires supported molecular performance, uncertainty and
compatibility evidence, complete data/model lineage, current training and
redistribution rights, security evidence and the required independent release
approval. Retrospective development cannot replace untouched or prospective
confirmation.

The native tensor-file inference path has passed target-free reload and
sampling contract checks. OMF 2.0 script experiments can now export an actual
model directory containing fitted weights, inference source, dependencies and
a manifest. ModelPackage artifact materialization and deployment compatibility
for the future full world model remain separate integration work. No metadata-
embedded weights, absolute-run-path adapter workaround, fabricated signature
or unsupported policy key is used. These release limitations do not stop local
model development.

## Reproducibility and historical evidence

Fresh run directories retain protocols, source hashes/copies, checkpoints,
references and reports. Real datasets and artifacts remain outside Git in
ignored data/results stores. The native tested runtime and module contract are
recorded beside the implementation. Scientific claims follow the ledger rather
than the scale of the architecture or dataset.

SLp-1 remains frozen in `model/v1/`, `docs/model-card.md` and the historical
results ledger. Its gene universe, feature concatenation, decoder family,
training mixtures and benchmark scores are not defaults for SLp-1.1. Earlier
sparse modules and OMF evidence remain historical contracts; they are not the
active biological training implementation.
