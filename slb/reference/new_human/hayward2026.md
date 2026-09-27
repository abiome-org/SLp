# hayward2026: notes

Hayward, Vaitsiankova et al. 2026 bioRxiv, doi 10.64898/2026.06.07.728858 (Ciccia lab with Hart lab).
License: the preprint is CC BY-NC-ND 4.0. The data portal
(ddr-genetic-interactions.ciccialab-database.com) gives no explicit license. The counts come from GEO GSE343308.

## Design
- enCas12a in4mer, all-by-all of the DDR genes (232 unique symbols in the published tables, all resolve in HGNC).
  Three arrays per pair, three replicates, T0/T9/T17 (MCF10A) and T0/T9/T18 (hTERT-RPE1 TP53-/-). Drug-free.
- The count library also contains 49 essential-gene controls, 13 paralog positive-control pairs and 50
  non-essential control genes (single-gene arrays only, e.g. CXorf66). There are no non-targeting x non-targeting
  arrays, so the replicate re-scoring centres LFC on the non-essential single-gene arrays.
- Contexts are `MCF10A` and `RPE1`. `RPE1` is the same string spidr2025 and ryanlab_zdlfc use, and both resolve
  through contexts.lookup, to MCF-10A and hTERT-RPE1.

## Scores and labels
- score: the authors' GI z-score (local-std normalised) at the final timepoint, T17 for MCF10A and T18 for RPE1.
  signif: FDR_negative.
- Positive: z <= -5 and FDR_negative < 0.05. Negative: |z| < 1 and FDR_negative > 0.25. Everything else is null.
- Why -5: the authors' FDR<0.1 call (z about -3.5) replicates less well. Single-replicate re-calls reach about
  0.80 at -3, 0.92 at -4 and 0.98-0.99 at -5. SPIDR's recovery of the RPE1 positives rises from 0.65 (-3)
  to 0.68 (-5) and 0.73 (-6). -6 would halve the positives (MCF10A would keep 17).
- Positives: RPE1 110 (0.42%), MCF10A 36 (0.14%). The MCF10A rate is below the 0.3% guide. That follows from
  the strict cut-off and is not an implausible call rate: z <= -4 would give 77 (0.29%).

## Checks (full numbers in hayward2026.json)
- Replicate GI (SPIDR recipe) on raw additive GI is dominated by pairs of strongly depleted essential-like
  genes whose counts sit at the floor. With that score, labels defined from RPE1 rep1 are recovered by rep2 and rep3 at only 0.44-0.57; the other directions give 0.63-0.86.
  After z-scoring each replicate's GI within 50 bins of f_a+f_b (mirroring the authors' local std), the
  split-rule AUROCs at z<-5 are 0.97-1.00 (MCF10A) and 0.93-1.00 (RPE1). Two replicates against the third
  gives 0.98-1.00.
- Final labels scored by single replicates: MCF10A 0.93/0.95/0.94, RPE1 0.97/0.97/0.99.
- Cross-study, RPE1 vs spidr2025 (CRISPRi, RPE1), 17.7k shared pairs: SPIDR scores our positives at 0.68
  (85 pos). Our score recovers SPIDR's labels at 0.78, but only 15 positives overlap. The whole-map Spearman
  is about 0.03, so the two screens agree only on the strong tail.
- MCF10A vs SPIDR RPE1 is a different line and weaker evidence: 0.63 (24 pos).
- Cross-line within the study is 0.99 (MCF10A labels by RPE1) and 0.90 (RPE1 labels by MCF10A). The same
  library and pipeline were used in both lines, so this is inflated.
- Fitness: DepMap -(f_a+f_b) gives 0.52 overall; the screen's own SMF gives 0.57-0.60. Labels are not confounded by fitness.
- Paralog fraction: 0.43% of pairs and 6.8% of positives.

## Caveats
- The in-house local-z replicate score is my re-implementation, not the authors' exact pipeline.
- The only independent overlap is SPIDR, which uses a different modality (CRISPRi vs Cas12a KO). Thompson 2021 RPE1 shares only 14 pairs.
- replicates() returns all arrayed pairs, including the essential controls. Join to load() for DDR x DDR only.
