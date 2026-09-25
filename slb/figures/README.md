# SLB figures

Rebuild from `slb/`:

```bash
uv run python figures/prepare_data.py   # collects every plotted number into figures/data/ (gitignored)
cd figures && for f in fig1_hero fig2_labels fig3_grader fig4_results; do uv run python $f.py; done
```

`prepare_data.py` reads the built benchmark, the parsed measurements, the replication report, the
robustness builds and the pinned leaderboard results. It uses train/dev labels and aggregate test scores
only. `style.py` holds the shared palette: amber = test / SL, teal = dev, slate = train.

1. **[Held-out gene families](fig1_hero.png).** (a) All 18,723 benchmark genes on a circle, by species
   and then by split. Curves are the homology edges between them: orthologs cross the circle; paralogs and
   overlapping ORFs loop at the rim. Every curve stays inside one split, which is the benchmark's
   hold-out guarantee. Inner ring: split of each gene; outer ring: measured pairs per gene (log).
   (b–d) Every measured pair of three screens as a gene × gene map, genes ordered by number of SL
   partners; blue = measured, amber = SL call. (e) Human: 50 cell lines (grouped by genetic ancestry) ×
   the 1,400 most-screened gene pairs.
2. **[Labels](fig2_labels.png).** (a) Score distributions of three screens with the label each measured
   pair receives; the gap between the SL and neutral bands is unscored. (b) Every source's label recovery
   by an independent re-measurement; sources are admitted above AUROC 0.65, and excluded ones are grouped
   by reason. (c) SL rate by chromosomal distance relative to pairs on different chromosomes: linked yeast
   pairs are up to 14× enriched, so pairs within 200 kb are unscored. (d) From 15.8 M measured pairs to
   7.6 M scored pairs and their splits.
3. **[Grader](fig3_grader.png).** (a) Balance of single-gene fitness and per-gene row counts between SL
   and non-SL pairs before and after the SLB weights (dev). (b) Reject probes on dev; the row-count probe
   scored 0.65 (dev) and 0.64 (test) before the fix and 0.50 / 0.51 now. (c) The noise floor: 60 random
   per-gene scorings per split, beside the dev scores of the best models. (d) The six baselines on the
   official split and 4 alternative family splits.
4. **[Results](fig4_results.png).** (a) The held-out test leaderboard with 95% family-bootstrap CIs, and
   each entry's per-species and per-pair-type AUROC ("tie" = the model does not score that species).
   (b) Dev against test SLB score for every entry; the band is ±0.03. (c) Human test AUROC within cell line
   and screen by genetic ancestry for the top ranked models; African-ancestry support is 3 cell lines.
