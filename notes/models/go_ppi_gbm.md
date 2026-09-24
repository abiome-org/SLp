# go_ppi_gbm: GO + physical PPI + STRING functional-association GBM (Wong 2004 / Pandey 2010 style)

battery: go_ppi_gbm (one model per species), go_ppi_gbm__pooled (one model over all species; also the
fallback for species without train labels); species: every species with a bundle (human, scer, spom, spne
now; cele, dmel, mmus, ecol, bsub ready); needs: data/interim/bundle/<sp>/{go,ppi,fitness}.parquet.

## Papers / lineage
- Wong SL et al. 2004, PNAS 101:15682, "Combining biological networks to predict genetic interactions"
  (probabilistic decision tree over PPI, GO, co-expression, localisation, "2-hop" network features; yeast SL).
- Pandey G et al. 2010, BMC Syst Biol 4:117, "An integrative multi-network and multi-classifier approach to
  predict genetic interactions" (yeast; GO + PPI + co-expression features, ensemble classifiers).
- Also Paladugu et al. 2008 (BMC Syst Biol, PPI topology features), Madhukar et al. 2015 (network features).
No original code is released for either paper in a usable form (Wong's decision tree was built in a
commercial package; Pandey's scripts are not public), so this is a re-implementation of the feature
families with a modern learner (LightGBM), fitted on SLB train only. Script: scripts/models/go_ppi_gbm/.

## Features (29)
GO BP/CC/MF (propagated over is_a + part_of): shared terms, Jaccard, max IC of a shared term (Resnik), shared
direct terms. BioGRID physical evidence count. STRING neighbourhood, fusion, co-occurrence, co-expression,
curated database. Shared physical / STRING (>= 400) neighbours and Jaccard, degrees (min/max). n GO terms
(min/max). Single-loss effect (min/max, bundle fitness). same_family. species code.

## Leakage: clean (after one fix)
Fitted on SLB train labels only; inputs are single-gene data / networks. Three GI back-doors were closed:
1. GO evidence IGI (inferred from genetic interaction) is dropped.
2. BioGRID rows with Experimental System Type "genetic" are dropped.
3. **STRING "experimental" channel is dropped**: STRING imports "biochemical, biophysical and genetic
   assays" from BioGRID/IMEx into it. On scer 20% of STRING-experimental edges are BioGRID GIs vs 2.7% of
   BioGRID physical edges. With it the first version scored scer 0.669 (SLB 0.657); without it scer 0.61.
   STRING textmining and combined_score are also dropped (paper co-mentions of SL pairs).
Remaining indirect channel: STRING "database" (KEGG/Reactome/complexes) and co-expression are functional
annotations that correlate with GI (as all these features do) but do not encode GI measurements.

## Results (dev, SLB-1.2, all examples scored)
| variant | SLB | human | scer | spom | spne |
|---|---|---|---|---|---|
| go_ppi_gbm | 0.650 | 0.671 | 0.611 | 0.638 | 0.678 |
| go_ppi_gbm__pooled | 0.647 | 0.672 | 0.609 | 0.640 | 0.667 |
| (leaky first version with STRING experimental) | 0.657 | 0.684 | 0.669 | 0.645 | 0.629 |
Reference baselines on dev: lgbm 0.544, fitness 0.5. Paired family-bootstrap vs lgbm (`slpbench compare`,
200 reps): go_ppi_gbm +0.105, 95% CI [+0.036, +0.156], P(not better) 0.000; per species human +0.075,
scer +0.103, spom +0.129, spne +0.114. Pooled: +0.103 [+0.007, +0.157].
Top features: human: shared physical neighbours (Jaccard), fitness, co-expression; scer: fitness,
co-expression, CC max IC; spom: STRING degree, fitness, physical degree; spne: fitness, STRING degree.
Internal (train-gene-held-out) validation AUROC (unbalanced): human 0.79, scer 0.80, spom 0.70, spne 0.96.
Reports: results/models/go_ppi_gbm_dev.txt, go_ppi_gbm__pooled_dev.txt.
Runtime: ~15 min on 32 threads (feature construction for 1.9M scer train pairs dominates).

## Notes
- spne STRING is strain R6 (STRING has no D39/D39V); mapped through spr tags; BioGRID has 6 spne edges.
- spne's high unbalanced internal AUROC (0.96) is fitness: SL calls there sit on the sickest genes; the
  SLB balancing removes that, leaving 0.67-0.68.

## Results (dev, SLB-1.3; headline = human/scer/spom)
| variant | SLB | human | scer | spom | aux bsub (4 pos) | cele (12) | dmel (10) | mmus (0) |
|---|---|---|---|---|---|---|---|---|
| go_ppi_gbm | 0.626 | 0.637 | 0.602 | 0.638 | 0.498 | 0.523 | 0.522 | n/a |
| go_ppi_gbm__pooled | 0.617 | 0.614 | 0.601 | 0.637 | 0.489 | 0.552 | 0.524 | n/a |
Auxiliary species have < 20 dev positives, so the evaluator does not score them; the numbers are noise-level.
Reports: results/models/slb1.3/go_ppi_gbm*_dev.txt.
Paired family bootstrap vs lgbm on SLB-1.3 (lgbm 0.532): +0.094, 95% CI [+0.046, +0.126]; human +0.079, scer +0.087, spom +0.116.
