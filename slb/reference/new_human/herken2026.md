# herken2026 — Herken, Norman & Gilbert 2026 Mol Cell (K562 CRISPRi)

- Data: GEO GSE312636 `GI_expt1_counts.txt.gz` (Rep1/2 x T0/UT/DRUG) and `GI_expt2_counts.txt.gz`
  (t0/dmso/etop/keto x r1/r2). 638 sgRNAs (312 genes x 2, 1 gene x 1, 13 non-targeting), all-by-all, both orders.
  Zenodo 10.5281/zenodo.17822522 holds only the notebooks (no processed GI), so scores are re-derived.
- Arms used: expt1 T0->UT and expt2 T0->DMSO only. ATR-inhibitor, etoposide and ketoconazole arms are never read.
- Scoring: `call_genetic_interactions.ipynb` re-implemented (Horlbeck 2018 method, same as SLKB horlbeck2018):
  end-count median filter (35), pseudocount 10, NTxNT-centred log2 enrichment / doublings (authors' growthScores),
  replicate average, ABBA symmetrisation, per-sgRNA quadratic fit with intercept fixed at the query single
  phenotype, z by non-targeting residual SD, gene GI = mean of the 2x2 sgRNA block. expt2 has ~400 empty count
  fields per column; those constructs are treated as missing.
- Score = mean of the two maps; 48,828 pairs (313 genes; paper reports 293 genes because it intersects with the
  drug maps too). 24 old symbols renamed by HGNC resolver, 0 unresolved.
- Label: pos = mean GI <= -3 and < 0 in both maps (872, 1.79%); neg = |mean GI| < median (24,414).
  |GI| < 1 was tried first but labels 72% of pairs negative; switched to the horlbeck2018 median rule
  (decided after seeing the distribution and the cross-study contradiction counts: 21 -> 13 Horlbeck positives in our negative band).
- Replication: map1->map2 split-rule AUROC 0.964 / 0.970 (Spearman 0.58); single biological replicates 0.93-0.98.
- Cross-study (K562): horlbeck2018 0.797 (their score, our 46 pos) / 0.843 (our score, their 75 pos);
  simpson2023 0.95 / 0.97 (165/170 pos). Jurkat horlbeck2018 (different line) only 0.61/0.62.
  Herken, Simpson and Horlbeck share the Weissman/Gilbert/Adamson CRISPRi platform and pipeline: agreement is
  partly method-shared, not fully independent.
- Fitness confound: screen singles -(f_a+f_b) AUROC 0.66, DepMap 0.55. Paralog fraction 0.06% (31 pairs, 8 pos).
- CRISPRi is hypomorphic knockdown (mechanism CRISPRi). Verdict: include.
