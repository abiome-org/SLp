# SLp-1.2 inference contract

The bundle contains actual shared-transformer weights, source, normalized static
entity descriptors, optional entity indices, DepMap context encodings, source
receipts and SHA-256 hashes. `inference.World` verifies the manifest before
loading. The model itself needs no training data or prior SLp model.

Use Python 3.12. Linux CUDA 12.8 dependencies are hash-pinned in
`requirements-linux-cu128.lock`; `requirements-linux-cpu.lock` supplies the
matching Linux CPU environment. `requirements.in` lists direct versions for
Apple platforms. Install with `pip install --require-hashes -r
requirements-linux-cu128.lock` for CUDA, choose the CPU lock for Linux CPU,
or resolve the direct versions for macOS. CPU/GPU results may differ slightly.

`World.predict(batch)` accepts NumPy arrays with observation, action and query
fields matching `model.WorldModel.forward`. Masks are boolean; embedding indices
are int64; numeric values are float32. Feature vectors have 702 coordinates;
context vectors have 128. Values and anchors must use the assay's recorded
normalization in `corpus.json`; outputs use the same units. Query scale can
convert normalized predictions back to the source units. A missing observation
is masked, rather than treated as a measured zero. Action dose/delay/duration
have individual known flags. An empty action set is permitted, but exact identity
under no intervention is not imposed. Query panels can influence each other.

Assay, modality and species remain explicit. Modality indices are RNA=0,
protein=1, fitness=2. Taxon indices are human=0 (NCBI 9606), yeast=1 (4932).
Mechanisms are CRISPRi=0, CRISPRa=1, knockout=2. Flow time is computational
denoising time, never a biological time course. Mean prediction does not read
targets. `sample=True` performs fixed-seed Euler flow sampling; use it only for
the single-cell assays that supplied distributional training. Samples are in
normalized measurement space and are not clipped to physical bounds.

`World.fitness([[gene_a], [gene_a, gene_b]], taxon=4932)` predicts yeast log
relative fitness. Human prediction requires one admitted DepMap context ID per
experiment and returns the source gene-effect score, not an SL probability.
The bootstrap corpus has no human double-knockout fitness supervision: human
combination predictions are research extrapolations. Unknown genes require
normalized descriptors and use entity index zero. Learned IDs are optional.

Checkpoint selection uses retrospective development data. The bundle is a
research artifact, not a validated clinical predictor. Training and downstream
SL application labels remain separate. Data and descriptors retain the supplied
third-party terms; original model code and weights use MIT.
