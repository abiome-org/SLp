# nebenfuehr2026: check only, EXCLUDE

Nebenfuehr et al. 2026 Cell Rep (doi 10.1016/j.celrep.2025.116850, CC BY-NC-ND). eHAP iCas9 (p53-null),
all-by-all 461 DNA-repair genes. Data: GEO GSE290153 `fullbasal_grna1_coupled_counts.tsv.gz`
(864,187 guide pairs x day0/day14 x 3 reps). The IR arm (GSE290153_IR_*) was not fetched: SLB uses drug-free /
unirradiated arms only.

## What was checked
- Re-scored every replicate with the spidr_replicate_gi recipe (day14 vs same-replicate day0, centred on
  nt x nt, f_g = median gene x nt, GI = LFC - f_a - f_b, median over ~8 guide pairs per gene pair), 105,111 gene pairs.
- Replicate agreement: guide-pair LFC Spearman -0.035 to 0.029; gene-pair GI mean Spearman -0.014;
  tail reproducibility 0.088 (chance is 0.10); split-rule AUROC (z <= -3 or 2% tail) 0.40 to 0.62 in all six
  directions. Using a common pooled day0 does not rescue it.
- Sanity: single-gene effects from gene x nt guides do not correlate with DepMap (Spearman -0.09 to 0.06),
  and DepMap-essential genes (n=69, e.g. replication genes) have median LFC around 0 in every replicate.
  A working Cas9 screen at day 14 would deplete them strongly, so this file carries no fitness signal.
- Count quality: day14_rep1 has 54% zero counts (bottleneck); rep3 day0 has only 8.0M reads and 42% zeros.
  Only reps 1-2 exist as GEO samples; rep3 is present only in the combined file.

## Other notes
- Library symbols use mouse-style capitalisation (Abl1, Trp53, 4930447C04Rik) though the samples are human eHAP;
  453/459 resolve to HGNC after upper-casing (unresolved: 4930447C04Rik, Morc2b, Supt16, Trp53, Trp53bp1, Zfp365).
  The guides themselves are human-targeting (all 4 BRCA1 and 4 PCNA guides match human genomic sequence
  via Ensembl REST, none match mouse), so the missing essential-gene depletion is not a species mismatch.
- Consistent with the prior check (replicate LFC correlation ~0, day-14 bottlenecking).

## Verdict
Exclude. No staging parser written (the brief requires one only if the source would pass). The scores are
not useful even as measurements-only training data.
