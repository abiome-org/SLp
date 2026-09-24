# fba: flux-balance double-gene-deletion SL score (COBRApy)

battery: fba (score = min(f_a, f_b) - f_ab), fba__mult (score = f_a*f_b - f_ab); species: human, scer, spom,
spne, bsub, mmus, dmel, cele, ecol (any species with a loader in scripts/models/fba/gems.py); needs: a
genome-scale metabolic model per species (no labels, no bundle). Setup: scripts/models/fba/setup.sh (+
export_maps.py for the mouse/fly/worm gene maps); run: scripts/models/fba/run.sh.

## What it is
Label-free mechanistic baseline. For every SLB pair whose two genes are both in the species' GEM, knock
out gene a, gene b and both (GPR-aware `gene.knock_out()`), maximise the biomass objective, and use relative
growth f = growth / wild type. Default score `fba_min` = how far the double mutant falls below the sicker
single mutant; `fba__mult` = negative epistasis vs the multiplicative expectation. Pairs outside the model
score 0 (`in_model=False` in the output) so the file covers every example.

## Models, versions, licences
| species | GEM | source | version / sha256 | licence | gene-ID mapping |
|---|---|---|---|---|---|
| scer | yeast-GEM (Yeast9 lineage), 1,143 genes | github.com/SysBioChalmers/yeast-GEM | commit 2d594ae (2026-08-30) | CC-BY-4.0 | ORFs = SGD systematic IDs (identity) |
| spom | pomGEM (Elsemman et al. 2022, PLoS Comput Biol), 718 genes | zenodo 10.5281/zenodo.6513462 (`pomGEM_updated_editedManually.xml`) | sha256 a46cf82e... | CC-BY-4.0 | `SPCC965_14c` -> `SPCC965.14c` checked against PomBase IDs: 716/718 |
| spne | iDS372 (Dias et al. 2019, Front Microbiol), strain R6, 372 genes | Frontiers suppl. Data_Sheet_2.zip | sha256 60a3f6d1... | CC-BY | R6 `spr` tags -> D39V canonical IDs via the D39V GenBank notes (slpbench spne resolver), else DIAMOND reciprocal best hit (>=95% id, >=90% cov): 371/372 |
| human | Human-GEM (Human1 lineage), 2,848 genes | github.com/SysBioChalmers/Human-GEM | commit 28c85ef (2026-09-09) | CC-BY-4.0 | Ensembl -> HGNC symbol (HGNC complete set): 2,845 |
| ecol | iML1515 | BiGG | sha256 2555e0f7... | BiGG terms | b-numbers (identity; SLB ecol scheme) |
| bsub | iYO844 | BiGG | sha256 ae819c25... | BiGG terms | BSU tags (identity; SLB bsub scheme) |
| mmus | Mouse-GEM, 2,846 genes | github.com/SysBioChalmers/Mouse-GEM | commit fc769ac | CC-BY-4.0 | MGI symbols via ids_extra.mmus: 2,845 |
| dmel | Fruitfly-GEM, 1,738 genes | github.com/SysBioChalmers/Fruitfly-GEM | commit a7c0b5f | CC-BY-4.0 | symbols -> FBgn via ids.dmel: 1,687 |
| cele | Worm-GEM, 1,604 genes | github.com/SysBioChalmers/Worm-GEM | commit 3f4655b | CC-BY-4.0 | sequence names -> WBGene via ids_extra.cele: 1,603 |
All files and checksums: external/models/fba/ (gems/, annot/, SHA256SUMS); `scripts/models/fba/setup.sh` re-fetches.
SpoMBEL1693 (Sohn 2012, BioModels MODEL1507180061) was downloaded but its SBML carries no gene associations, so pomGEM is used.

Media: yeast-GEM and pomGEM default (minimal glucose, aerobic); iDS372 as distributed (its exchange bounds);
Human-GEM: glucose-limited Ham's F-12-like medium (glucose 1, other organics 0.1, inorganics unlimited, plus
serum retinol / tocopherols / linolenate / cholesterol / fatty acids, which the Human-GEM biomass needs; without
them growth is 0). No cell-line-specific models: ftINIT / tINIT need MATLAB+Gurobi or Gurobi in Python, which
we do not have; a generic model is used for all 50 lines. Solver GLPK with a 60 s LP time limit (a few knock-out
LPs stall; those get NaN -> score 0).

Mouse/fly/worm GEMs give ~0 growth on the Ham's medium (different biomass requirements) and are run on
their distributed medium (all exchanges open).

## Leakage
None. No labels of any kind; GEMs are built from biochemistry + single-gene essentiality (the GEM papers
curated against single-KO data, which is a permitted input). Label-free, so dev scores are clean.

## Results (dev, SLB-1.2)
Full split (uncovered pairs = 0): SLB 0.509 (human 0.527, scer 0.499, spom 0.500, spne 0.512); fba__mult 0.504.
Coverage (both genes in the GEM): human 1,576/15,048 dev pairs (42 pos), scer 1,798/91,939 (12 pos),
spom 118/24,171 (0 pos), spne 76/1,564 (11 pos).
Covered subset only (`scripts/models/mech_common/subset_eval.py`, same SLB weighting):
| | human | scer | spom | spne |
|---|---|---|---|---|
| fba | 0.741 | 0.465 | n/a (0 pos) | 0.619 |
| fba__mult | 0.803 | 0.511 | n/a | 0.741 |
| paralog_identity on the same pairs | 0.503 | 0.627 | n/a | 0.5 |
Few pairs are ever predicted SL (dev: 19 human, 1 scer, 0 spom/spne with score > 0.01). On train (label-free, so
a fair extra check; unadjusted AUROC since train has no propensities): covered-pair AUROC is ~0.5 (human 0.48,
scer 0.54, spom 0.49, spne 0.57), but the pairs FBA calls SL are enriched: scer 134 predicted, 11.2% SL vs 0.8%
base (14x); human 91 predicted, 7.7% vs 2.8%; spne 5 predicted, 1 SL. So FBA is a high-precision, very
low-recall SL caller; the dev covered-subset human number rests on 42 positives and is not confirmed on train.

SLB-1.3 dev: fba 0.510 (human 0.524, scer 0.503, spom 0.502; fba__mult 0.508; auxiliary bsub 0.509 (4 pos), cele 0.518,
dmel 0.501, mmus n/a); covered pairs: human 1,906 (44 pos) 0.730, scer 2,816 (19 pos) 0.605, spom 183 (2 pos),
bsub 30, cele 4, dmel 5, mmus 0.
Reports: results/models/fba_dev.txt, fba__mult_dev.txt; results/models/slb1.3/fba*_dev.txt. Runtime: ~1 min for dev (cached knock-outs in
external/models/fba/cache/); ~15 min for all train pairs at 24-32 processes.

## Problems / notes
- Metabolic coverage is structural: 2-20% of SLB genes are in the GEMs; SLB pairs are dominated by non-metabolic
  genes (Costanzo, E-MAP, paralog screens).
- iDS372 is strain R6 (D39-derived, near-identical metabolism); mapping to D39V via locus-tag notes / RBH.
- Human: a context-specific (per cell line) variant would need ftINIT + Gurobi; not done.
