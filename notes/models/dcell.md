# dcell: DCell-style visible neural network (Ma et al. 2018), retrained on SLB train

battery: dcell__faithful (per-gene input weights, as published), dcell__tied (gene-agnostic ontotype-style
input); species: every species with a GO bundle and train labels (SLB-1.3: human, scer, spom, bsub, cele, dmel,
mmus; SLB-1.2 also spne); needs: data/interim/bundle/<sp>/go.parquet, torch (external/models/dcell/.venv), GPU
recommended (scer ~1 min/epoch on GPU, ~4 min on 8 CPU threads).

## Paper / code
Ma J, Yu MK, Fong S, Ono K, Sage E, Demchak B, Sharan R, Ideker T. 2018, Nat Methods 15:290, "Using deep
learning to model the hierarchical structure and function of a cell". Code: github.com/idekerlab/DCell
(MIT, commit 110c8a7, Lua Torch 7) cloned to external/models/dcell/upstream; the PyTorch successor DrugCell
(idekerlab/DrugCell, MIT, commit c507e1d) at external/models/dcell/drugcell_upstream. Lua Torch does not run
on current CUDA, so scripts/models/dcell/dcell.py re-implements the DrugCell/DCell VNN in PyTorch (one module
per GO term: children states + directly annotated genes -> Linear -> tanh -> BatchNorm, 6 hidden units,
auxiliary 1-unit head per term with loss weight 0.2, root head). The published predictions for Costanzo 2010
(chianti.ucsd.edu/~kono/ci/data/deep-cell/predictions/prediction.tar.gz) are 404; they would be leaky anyway
(trained on Costanzo scer GI).

## Changes vs the paper
- Target: SLB train label (BCE) instead of Costanzo growth/GI regression; negatives resampled per epoch to
  10 per positive.
- Ontology: GO from the bundle (IGI dropped), terms with >= 6 and <= 30% of annotated genes, redundant
  child terms collapsed, synthetic ROOT over the three namespaces (~3.1k terms scer, ~9.9k human).
- `tied` variant: each term receives the COUNT of its disrupted direct genes instead of one weight per gene.
  With SLB's held-out gene families, the input weights of dev/test genes are never trained in the faithful
  variant, so only the ontology structure can generalise; tied makes the model gene-agnostic.
- Early stopping on train pairs with both genes in a random 15% of train genes (small species without
  positives there: random 15% of train pairs). No dev labels used.

## Leakage: clean
SLB train labels only; GO without IGI.

## Results
SLB-1.2 dev: faithful 0.519 (human 0.495, scer 0.528, spom 0.535, spne 0.517); tied 0.534 (human 0.452,
scer 0.585, spom 0.540, spne 0.559).
SLB-1.3 dev: faithful 0.499 (human 0.439, scer 0.522, spom 0.534; aux bsub 0.440, cele 0.389, dmel 0.517);
tied 0.507 (human 0.443, scer 0.557, spom 0.521; aux bsub 0.563, cele 0.639, dmel 0.618; aux species have
< 20 dev positives and are not scored by the evaluator). Reports: results/models/slb1.3/dcell__*_dev.txt.
Human is below 0.5 in every run: the human VNN (~9.9k terms, 156k train pairs) memorises train genes.
Both are far below the GO feature models (ontotype 0.62-0.63, go_ppi_gbm 0.63-0.65): the VNN with ~10^4
modules overfits the train genes within 1-3 epochs (internal validation AUROC peaks at 0.57-0.67, then falls).
