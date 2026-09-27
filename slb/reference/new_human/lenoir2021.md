# lenoir2021: Lenoir et al. 2021 Nat Commun, enCas12a lipid-metabolism GI screen in AML

- doi 10.1038/s41467-021-26867-8 (CC BY). Data: github.com/PeterDeWeirdt/FASTS at commit 1fa5601
  (gene_residuals.csv, guide_residuals.csv, library csv, lognorm table with per-replicate reads, notebooks).
- Design: 8 anchors (ACACA, C12orf49/SPRING1, FASN, GPX4, PSTK, PTEN, SREBF1, TP53) x 99 library genes,
  3x3 guides per pair. MOLM-13 and NOMO-1 (both Japanese-donor AML lines). Two replicate infections (A/B), day 14 and day 21.
- The "1,658 pairs" in gene_residuals.csv include 891 gene x control-gene rows (the 15 non-expressed "control genes")
  and 10 self pairs. Once those are removed there are **756 real pairs per line** (1,512 rows).
  The 10 SL control paralog pairs in the library (ARID1A/B etc.) have no scores from the authors and are not used.

## Decisions
- Measurement is the day 21 authors' `pair_z_score` from gnt (a linear anchor-guide model, with the two anchor
  orientations combined, computed on the A/B mean LFC). signif is BH FDR within the condition. Negative means synergistic.
- Labels: positive z <= -3 and FDR < 0.05; negative |z| < 1; else null. I chose day 21 because it is the later
  timepoint and the one the authors focus on. Day 14 z recovers the day 21 labels at AUROC 0.99.
- replicates(): additive GI per replicate infection (A = gi_rep1, B = gi_rep2) at day 21, recomputed from the lognorm
  table. Method: pDNA |z|<3 filter as the authors did; LFC centred on non-essential x control-gene constructs;
  f_g = median over gene x control-gene constructs; median over guide pairs. These are true independent replicates.

## Results
- 68 pos / 719 neg / 725 null (4.5%). MOLM-13: 42 pos (5.6%). NOMO-1: 26 pos (3.4%). 99 genes; 3 paralog pairs.
- Own replication (per line): 2% tail split, rep A->B / B->A: MOLM-13 0.99/0.94, NOMO-1 0.93/0.96. Spearman A vs B is
  0.52-0.54. Single-replicate AUROC for the final labels is 0.84-0.90. Positives in the -4 < z <= -3 band still
  replicate (single-replicate AUROC about 0.80).
- Cross-line (MOLM-13 vs NOMO-1) label AUROC is 0.77/0.82, so many hits are shared between the lines.
- Fitness: the DepMap -(f_a+f_b) AUROC is 0.59, but DepMap covers only 365 labelled rows. Using the screen's own
  singles gives 0.58. No confounding.
- Cross-study: none in the same lines. There are 5 overlapping pairs, all in other lines. SREBF1-SREBF2 is positive here
  and positive in 8/13 flister2025 lines. ACACA-ACACB is negative here but positive in 7/11 flister2025 lines, while it is
  neutral in harle2025 and ryanlab. These are too few pairs to count as evidence.

## Verdict: include
Own replication is strong and the hit rate is plausible. Caveats: the dataset is small and hub-structured, and the
positives are concentrated on anchors (ACACA 15, PTEN 12, SPRING1 12, GPX4 11). The MOLM-13 positive rate is slightly above 5%.
C12orf49 resolves to SPRING1. No symbols are unresolved.
