# simpson2023 — Simpson, Ling, Jing & Adamson 2023 bioRxiv (preprint; K562 CRISPRi)

- Data: github.com/simpsondl/parpi-manuscript `manuscript_data/counts/screen{2022,2023}_raw_counts.zip`
  (verified: 29.6 / 28.6 MB; 1,301,881 constructs = 1141^2 sgRNA orders; 543 genes + 55 non-targeting;
  columns T0/DMSO/NIRAP x R1/R2). Zenodo 10.5281/zenodo.17666844 is a snapshot of the same repo, no scores.
- Arm used: untreated gamma (T0 -> DMSO) only; NIRAP columns are not even loaded.
- Scoring: authors' R pipeline (config.yaml + workflow/scripts) re-implemented: T0 median filter (35) and
  construct filter (50), pseudocount 10, NT+NT-centred log2 fraction ratio / doublings, orientation-averaged
  phenotypes, single = mean with non-targeting partners, correlation filter (< 0.25 dropped; 37 sgRNAs per
  screen), per-query quadratic lm (free intercept), GI.z by non-targeting residual SD, gene GI = mean over
  non-identical sgRNA combinations. Genes (not the authors' pseudogenes) are the unit.
  No R available locally, so the port is not checked number-for-number against the authors' outputs.
- Score = mean of the 2022 and 2023 screens: 139,274 pairs, 529 genes after filters (STRA13 is ambiguous in
  HGNC and dropped; 36 old symbols renamed).
- Label: pos = mean GI <= -3 and < 0 in both screens (499, 0.36%); neg = |mean GI| < median (69,615).
  The authors' discriminant hit call is sign-agnostic and was not reproduced.
- Replication: screen 2022 -> 2023 split rule 0.975, reverse 0.943 (Spearman 0.36); replicates 0.88-0.97.
  Orientation (AB-only vs BA-only) replicates poorly: 0.65-0.70, Spearman 0.10-0.14, while AB-2022 vs AB-2023
  is 0.34. So there is a reproducible sgRNA-position component; the orientation-averaged score is used.
- Cross-study (K562): horlbeck2018: our score for their labels 0.731 (97 pos); their score for our labels 0.875
  (only 14 pos). herken2026: 0.97 / 0.95 (170 / 165 pos). All three are from the same CRISPRi lab lineage and
  pipeline family, so agreement is partly shared method. 23 of 97 Horlbeck positives in the overlap sit in our negative band.
- Fitness confound: screen singles 0.62, DepMap 0.50. Paralog fraction 0.11% (149 pairs, 12 pos).
- Verdict: include (preprint caveat).
