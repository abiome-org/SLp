# koo2025 — *B. subtilis* 168 double-CRISPRi envelope GI map (Koo et al. 2025)

Source key: `koo2025` · raw dir: `data/raw/koo2025_bsub_dcrispri/` (downloaded by a helper agent) ·
parser: `slpbench.sources.bacteria_extra.koo2025()` · **new species code `bsub`**

## Citation

Koo B-M, Todor H, Sun J, van Gestel J, Hawkins JS, Hearne CC, Banta AB, Huang KC, Peters JM,
Gross CA. **Comprehensive genetic interaction analysis of the *Bacillus subtilis* envelope using
double-CRISPRi.** *Cell Systems* 16(11):101406 (2025). PMID 41045937, PMC12716459,
doi:10.1016/j.cels.2025.101406. Preprint: bioRxiv 2024.08.14.608006.

License: PMC author manuscript (NIHMS2115161), NIH public access. Supplementary tables carry no
separate licence statement; treated as academic-use research data.

## Files

| file | contents |
|---|---|
| NIHMS2115161-supplement-MMC2.xlsx | Table S1: 1,318 sgRNAs (BSU locus tag, gene, essentiality flag, cloned position sgRNA1/sgRNA2) |
| NIHMS2115161-supplement-MMC3.xlsx | **Table S2: relative fitness (RF) matrices**, 8 time-interval comparisons, 319 × 1,310 |
| NIHMS2115161-supplement-MMC4.xlsx | **Table S3: GI-score matrices**, unfiltered and filtered, same 8 intervals |
| NIHMS2115161-supplement-MMC5.xlsx | Table S4: GI-score correlation matrices |
| NIHMS2115161-supplement-MMC6/7/8.xlsx | strains, primers, time-point key |
| NIHMS2115161-supplement-MMC1.pdf | supplementary figures (includes the GI-score schematic, Figure S4) |

sha256 of the files used by the parser: MMC2
46bb00e508f82fc1fa6fb5c256d47932e375ed0830d9d083b5807a1de7d05c84, MMC3
e640e36941ca0ff727135971d4376b137b3eb30e50f32062d3c6e3c2516cf018, MMC4
36e881f95f8308dafea168a523f37cd23b3e19a868102b824ef6e723da2f77ea. URLs in
`koo2025_bsub.fetch.tsv` (PMC author-manuscript bin paths). Also needed:
`data/raw/bsub_annot/AL009126.3.gb` (NCBI efetch, sha256
42ef156403f68c267497cc032c597421a31832255011793401e552d32f9709e3) for the BSU ID mapping.

## Design

One strain = two sgRNAs (sgRNA1 from a 319-member set, sgRNA2 from a 1,310-member set) plus a
barcode; 419,485 sgRNA combinations including single-sgRNA controls. The pool was grown with 1%
xylose (CRISPRi ON) and sampled at T0–T7 (three independent 1 L flasks per time point; Figure S2).
Per interval, RF = change in a strain's relative abundance, and the **GI score is a robust z-score
of the strain's observed RF against the distribution of RF expected from its two single
knockdowns** (Figure S4). Negative = worse than expected = aggravating / synthetic sick.

The intervals are not equivalent conditions: T0→T1, T1→T2, T2→T3 are successive 5-generation
exponential-growth windows with CRISPRi induced; T0→T4 and T0→T6 are overnight/uninduced controls;
T4→T5 and T6→T7 measure recovery from stationary phase. Only the three exponential windows are used.

Because 316 of the targets appear as both sgRNA1 and sgRNA2, ~32k gene pairs are measured in **both
orientations** with different constructs — an independent re-measurement inside the screen.

Gene IDs: sgRNA labels are `BSU#####_name`; `bsub_map` resolves old-style BSU tags, the current
`BSU_#####` tags, names and synonyms (AL009126.3) to the old-style tag, which is the canonical
`bsub` ID here (and SubtiWiki's).

## Parser and label rule

`koo2025()`:

- Score = **mean GI score over the three exponential intervals** (the filtered `* GI scores`
  sheets). Merging them is essential: single intervals replicate poorly across orientations
  (0.62–0.77) while the mean reaches 0.87–0.89.
- **Positive:** mean GI ≤ −1.5.
- **Negative:** |mean GI| < median |mean GI| of the merged matrix (0.12).
- **Ambiguous:** everything between, and all positive (alleviating) deviations.

| rows | positives | negatives | ambiguous |
|---|---|---|---|
| 198,391 | 822 | 99,195 | 98,374 |

Hit rate 822/100,017 = **0.8%**, plausible for a targeted envelope-focused library.

Overlap with a per-interval "extreme" rule: 541 pairs reach GI ≤ −3 in at least one of the three
intervals, and 314 of those are also SLB positives (38% of the 822 SLB positives). The two rules
therefore agree on a core but the mean-based rule is the one with the orientation-replication
evidence, and a single-interval extreme is exactly what the anti-correlated intervals make
unreliable.

The threshold was chosen from the orientation-replication curve, not from the paper (the Cell
Systems methods are paywalled and both the journal and bioRxiv rate-limited this machine, so the
exact published cut-off could not be read; the supplementary figures show the authors using "max GI
score > 3/5" across intervals for some analyses, which is stricter than the SLB rule and applies to
a different quantity, the per-interval maximum).

## Replication evidence

**Orientation swap** (same gene pair, sgRNA1↔sgRNA2, different constructs and barcodes), on the
mean-of-three-intervals score, 21,783 canonical gene pairs measured both ways. Recomputed by
`koo2025_checks()`:

| positive cut-off | positives (sgRNA1-side / sgRNA2-side) | AUROC (scored by the other orientation) |
|---|---|---|
| ≤ −0.75 | 623 / 619 | 0.793 / 0.757 |
| ≤ −1.0 | 424 / 392 | 0.820 / 0.783 |
| ≤ −1.25 | 299 / 259 | 0.875 / 0.830 |
| **≤ −1.5 (SLB rule)** | 224 / 186 | **0.892 / 0.871** |
| ≤ −2.0 | 149 / 124 | 0.928 / 0.913 |

(The first number of each pair is labels-from-the-sgRNA1-side-measurement scored by the sgRNA2-side
one, the second the reverse.) Spearman between orientations over all pairs is ~0.16 — the signal is
in the tail, and stronger thresholds replicate better, exactly as in both yeasts.

**Single-interval checks** (weaker, for reference; also in `koo2025_checks()`): with labels from
T0→T1 at ≤ −1.5, AUROC is 0.642 from T1→T2, 0.645 from T2→T3, 0.811 from T4→T5 (an independent
culture phase) and 0.443 from T6→T7.
Successive intervals are anti-correlated (Spearman −0.39 for T0→T1 vs T1→T2), as expected for
increments of relative abundance, which is why the mean over intervals rather than any pair of them
is the better score.

**Cross-study:** none possible. There is no second pairwise GI screen in *B. subtilis* with
comparable coverage. Koo 2017's deletion libraries give single-gene essentiality only.

**Fitness-only AUROC:** 0.763 with `bsub_single()` (below), covering 99.2% of labelled pairs.
Higher than Costanzo (0.73) but below dual CRISPRi-seq's 0.85; the benchmark's propensity balancing
is what handles this.

## Recommendation

**LEAD DECISION 2026-09-24: included as an auxiliary species** (`bsub`). Rationale below.

**Include** as the benchmark's *B. subtilis* source (new species `bsub`):

1. Orientation replication of the SLB rule is 0.87–0.89, comfortably over the 0.65 bar, and it is a
   genuinely independent re-measurement (different sgRNA pair construct).
2. Hit rate 0.8%, ~822 positives and ~99k negatives — enough for a species stratum.
3. It adds a second bacterium and a second CRISPRi-based GI map, which also gives the orthology
   agent a *bsub*–*spne* ortholog axis to hold out families across.

Cautions for the lead:
- `signif` is null (the source publishes no per-pair p-value), so the negative band rests on effect
  size alone.
- Gene space is envelope-biased (1,079 genes, mostly cell-envelope and essential-adjacent), so it
  is not a random sample of the genome.
- The library targets operons with single sgRNAs; polar effects on downstream genes in an operon are
  not deconvolved, the same caveat as the *S. pneumoniae* CRISPRi sources.

## Single-gene fitness

`bsub_single()` — per gene, log2 of the median relative fitness over all double-CRISPRi strains in
which the gene is one of the two targets, using the 10-generation induced interval (`T0 to T2 RF`);
0 = no cost, negative = sicker. 1,125 genes, covering 1,071 of the 1,079 genes in the pairs.
It disagrees with the authors' deletion-essentiality flag (AUROC 0.35) because that flag is about
deletability while these sgRNAs knock down partially; the genes it calls sickest are the expected
knockdown-sensitive ones (ftsH, gcaD, rasP, mbl, dlt operon, accC, spoVE, ponA, cpgA). Documented
in the function's docstring.
