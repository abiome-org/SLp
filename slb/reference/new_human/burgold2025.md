# burgold2025: Burgold et al. 2025 Nat Commun, Sanger ENCORE "COLO1" (HT-29)

doi 10.1038/s41467-025-67256-9, CC BY 4.0. Parser: `src/slbench/sources/staging/burgold2025.py`.

## Data and design
- Raw counts: GitHub `ibarrioh/DualGuide_COLO1`, `Analysis/ENCORE/input/COLO1_RUNMERGED_EXACT_ANNOTATED.txt`
  (plain text, not stored in LFS). Also stored: the repo readme, stats files, the authors' R script, and MOESM3/MOESM6.
- Zenodo 10.5281/zenodo.17191951 is only a GitHub release snapshot, so it was not fetched. MOESM3 is the pilot
  library and MOESM6 is source data. **The authors publish no GI call table for COLO1.** The GEMINI scores
  mentioned in the text are not released.
- Columns: `lib.COLO.1` is the plasmid. `CPID2437/2440/2443` are Cas9-negative parental HT-29: essential x control
  pairs are not depleted relative to plasmid. `CPID1020/1023/1026` are HT-29 Cas9: essential x control about -1.4 log2.
  The roles were inferred from these data. The paper's Fig 3A legend says "day 3 vs day 14", but the released file
  has no day-3 sample.
- Library: 40 anchors x 404 library genes, with 2x2 guide pairs per gene pair. There are also 40x40 anchor pairs
  in both orientations, and GI-control pairs (e.g. ASF1A/B, CNOT7/8, HDAC1/2, MAPK1/3). Each gene's single effect
  comes from its guides paired with 3 cutting-control genes (ADAD1, CYLC2, KLK12).
  After HGNC resolution there are 16,993 gene pairs and 479 genes. The "18,962 pairs" in the brief probably
  counts the control pairs.

## Scoring decisions
- Reference ("start") = depth-normalised mean of the 3 Cas9-negative parental replicates. These cells have the
  same transduction and culture time, so only Cas9 cutting differs. The authors used the plasmid. With a plasmid
  reference, the Spearman correlation of the score is only 0.63 against the parental-reference score, and the
  plasmid-referenced score agrees worse with the other HT-29 studies (Flister our-score AUROC 0.58 vs 0.68 before
  normalisation).
- LFC is centred on cutting control x control pairs (non-essential x non-essential and intergenic x intergenic).
  These constructs make two cuts, like every gene x gene and gene x control construct.
- Additive GI per replicate follows spidr_replicate_gi. I added one step: **per-anchor robust standardisation.**
  Raw GI has anchor offsets (median -0.45 for WEE1, +0.20 for CHEK1) and MAD about 2x larger for essential anchors.
  In the raw scoring, WEE1 and PRMT1 were in 56 of the 80 positives, and DepMap fitness predicted the labels at
  AUROC 0.79. After normalisation: 58 positives, fitness AUROC 0.61, and better replication and Flister agreement
  (see the normalisation comparison below).
- Label: z <= -3 on the pooled (mean of 3 reps) standardised GI **and** GI < 0 in every replicate gives 1.
  |z| < 1 gives 0. Everything else is null. This gives 58 positives and 12,235 negatives, a positive rate of 0.34%.
  The positives include CNOT7/8, ASF1A/B, MAPK1/3, CDK4/6, HDAC1/2, BCL2L1/MCL1, CTNNB1/JUP and EGFR/MAP2K1.
  8 of the 200 paralog pairs are positive.

## Checks
- **Shared reference inflates replication.** When all 3 Cas9 replicates are scored against the same parental
  mean, replicate Spearman is 0.22 and single-vs-single split AUROC is 0.69. When each Cas9 replicate is scored
  against its own parental replicate, Spearman is 0.07 and split AUROC is 0.63. For the independent-reference
  version, leave-one-replicate-out (labels from 2 reps, scored by the third) gives AUROC 0.68 / 0.68 / 0.71.
  `replicates()` returns the independent (paired) version.
- Same line, HT-29:
  - Flister 2025 (Cas12a, 65 shared paralog pairs): their score for our 4 positives, AUROC 0.98. Our score for their
    16 'Lethal' pairs, AUROC 0.72.
  - Dede 2020 zdLFC (11 pairs): 1.0 in both directions (3-4 positives).
  - No overlap reaches 20 positives, so there can be no formal contradiction.
- Other lines (weaker evidence):
  - Agreement is mostly >= 0.7 for chou2025, flister2025 (other lines), harle2025, parrish2021 and thompson2021.
  - It is poor for shen2017 (496 shared pairs, about 0.4-0.5; noisy older screen in other lines) and wong2016
    (4 positives).
- Fitness-only AUROC is 0.61 with DepMap single effects and 0.49 with the screen's own singles.

## Normalisation comparison (independent references; Flister HT-29 = our score for their 16 positives)
| variant | pos | split | LOO | fitness | Flister |
|---|---|---|---|---|---|
| raw additive | 80 | 0.59 | 0.66-0.69 | 0.79 | 0.68 |
| per-anchor centre + scale (used) | 58 | 0.63 | 0.68-0.71 | 0.61 | 0.72 |
| two-way median polish + scale | 59 | 0.64 | 0.69-0.72 | 0.73 | 0.82, but Dede drops to 0.48 because the polish erases GI-control pairs |

## Caveats
- The cross-study evidence covers only paralog pairs. Most positives are anchor x library pairs that no other
  study measured, and WEE1 is still a hub (19 of 58).
- Symbol `TAZ` resolves to TAFAZZIN. I checked the guide coordinates via the WGE API (chrX:154.41 Mb), so the
  guides target tafazzin, even though a colorectal anchor set probably meant WWTR1/TAZ. `ICK` resolves to CILK1.
  No symbols fail to resolve.
- Verdict: **include (borderline)**. Independent-reference LOO is at least 0.65, and the same-line cross-study
  AUROCs are 0.72 and 0.98 with small n. The single-replicate split is 0.63.
