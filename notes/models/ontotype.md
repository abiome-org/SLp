# ontotype: GO ontotype classifier (Yu et al. 2016), retrained on SLB train

battery: ontotype (one model per species), ontotype__pooled (allspecies: one model on a shared GO-term
space; scores species without train labels by transfer); species: every species with a GO bundle (human,
scer, spom, spne now; cele, dmel, mmus, ecol, bsub ready); needs: data/interim/bundle/<sp>/go.parquet.

## Paper / code
Yu MK, Kramer M, Dutkowski J, ... Ideker T. 2016, Cell Systems 2:77, "Translation of genotype to phenotype
by a hierarchy of cell subsystems". Code: github.com/michaelkyu/ontotype (commit 71ba8e3, 2016; licence not
stated in the repo), cloned to external/models/ontotype. The ontotype of a genotype = per ontology term, the
number of disrupted genes annotated to it; Yu et al. fed GO (and CliXO) ontotypes of Costanzo 2010 double
mutants to a random forest regressing double-mutant fitness. The released notebook is Python 2 and
builds ontotypes only; there are no released trained models.

## Changes
- Ontology: GO from the bundle (go-basic, is_a + part_of, propagated; IGI evidence dropped); terms with
  >= 6 genes and <= 30% of the species' annotated genes; ontotype values 0/1/2.
- Learner: LightGBM binary classifier on the sparse ontotype (a random forest on 1.9M x 3.5k was too slow);
  target = SLB train label instead of Costanzo fitness. Early stopping on train pairs whose both genes
  lie in a random 15% of train genes. No dev labels used for fitting.

## Leakage: clean
Trained on SLB train labels only; GO without IGI. (The original Yu et al. model is trained on Costanzo 2010
and would be leaky for scer; it was not released anyway.)

## Results (dev)
| variant | SLB | human | scer | spom | spne |
|---|---|---|---|---|---|
| ontotype | 0.627 | 0.609 | 0.637 | 0.714 | 0.549 |
| ontotype__pooled | 0.633 | 0.592 | 0.626 | 0.697 | 0.615 |
vs lgbm (paired family bootstrap): +0.083, 95% CI [-0.002, +0.178]; per species human +0.014, scer +0.129,
spom +0.205, spne -0.015. The best single model on S. pombe among everything run so far (0.714).
Internal validation AUROC: human 0.63, scer 0.72, spom 0.72, spne 0.44 (spne GO is sparse: 660 terms).
Runtime: ~20 min (scer).

## Results (dev, SLB-1.3)
| variant | SLB | human | scer | spom | aux bsub (4 pos) | cele (12) | dmel (10) | mmus (0) |
|---|---|---|---|---|---|---|---|---|
| ontotype | 0.616 | 0.581 | 0.621 | 0.647 | 0.396 | 0.509 | 0.652 | n/a |
| ontotype__pooled | 0.600 | 0.534 | 0.610 | 0.655 | 0.473 | 0.433 | 0.675 | n/a |
SLB-1.2 ontotype__pooled: 0.633 (human 0.592, scer 0.626, spom 0.697, spne 0.615).
Reports: results/models/slb1.3/ontotype*_dev.txt.
Paired family bootstrap vs lgbm on SLB-1.3 (lgbm 0.532): +0.085, 95% CI [+0.007, +0.174]; human +0.022, scer +0.107, spom +0.125.
