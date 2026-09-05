# Experimental molecular transition module

This self-contained module runs native local CUDA development. It is not yet
an admitted OMF workload or a certified model release. Python 3.11.9 and the
CUDA 12.8 Torch runtime were exercised on one RTX 4070. No repository-relative
imports outside this module are required.

`transition_model.py` maps intervention feature sets and optional measured
basal molecular tokens to a latent state. Independently queried feature vectors
decode measurement means, diagonal uncertainty, and optional shared low-rank
Gaussian factors. It has no learned gene-ID vocabulary. The application-neutral
network receives no SL labels or scoring thresholds. Set inputs support
multiple interventions structurally; biological combination generalization
requires separate molecular data and evaluation.

`train.py` consumes the pinned yeast fitting composite. `train_human.py`
consumes a checksum-pinned human development NPZ with disjoint gene-grouped
training and validation indices and empty test indices. Each creates a fresh
output directory, saves its protocol before fitting, copies source, trains
with a bounded wall-clock allowance, and selects checkpoints by development
gene-macro Gaussian NLL. These choices do not establish independent performance.

The default protein-only decoder uses static features for intervention and
query genes. The optional response-basis decoder also requires training-derived
measurement descriptors for query genes. These are assay features, not static
sequence features: no claim is made that its response basis transfers to a new
unmeasured assay panel. Context-specific fitted references are currently
required; unseen-context inference is not yet established.

Artifacts are `model.safetensors`, `model-config.json`, `reference.npz`, copied
`source/`, `protocol.json`, and `report.json`. `inference.Predictor` reloads tensor
files without a corpus, database, or OMF runtime. Its `predict` accepts raw
action/query features, aligned references and scales, and optional measured
context; its `sample` draws from the specified joint Gaussian. Current
pseudobulk training supports uncertainty over aggregate molecular measurements,
not a validated single-cell population generator. OMF portable inference is a
separate untested integration.

Verification covers feature/query permutation and chunk invariance, missing
observations, exact low-rank Gaussian likelihood against a dense reference,
artifact reload, sampling covariance, grouped fitting-only calibration, and
source-specific preparation. Numerical revisions are identified by copied
source hashes, with new artifacts rather than overwritten experiments.
