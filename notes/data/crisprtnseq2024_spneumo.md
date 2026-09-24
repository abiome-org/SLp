# crisprtnseq2024 — *S. pneumoniae* D39V CRISPRi–TnSeq (Jana et al. 2024)

Source key: `crisprtnseq2024` · raw dir: `data/raw/crisprtnseq2024_spneumo/` · parser:
`slpbench.sources.bacteria_extra.crisprtnseq2024()` · species code `spne`

## Citation

Jana B, Liu X, Dénéréaz J, Park H, Leshchiner D, Liu B, Gallay C, Zhu J, Veening JW, van Opijnen T.
**CRISPRi–TnSeq maps genome-wide interactions between essential and non-essential genes in bacteria.**
*Nature Microbiology* 9:2395–2409 (2024). PMID 39030344, PMC11371651, doi:10.1038/s41564-024-01759-x.

License: the article is © the authors, published under a Nature Research licence; the supplementary
tables carry no separate licence statement. Treated here as academic-use research data (same footing
as the other Elsevier/Springer supplements already in `data/raw`). Sequencing reads: SRA BioProject
PRJNA813307 (not used).

## Files

| file | bytes | sha256 |
|---|---|---|
| MOESM3.xlsx (Suppl. Tables 1–5: strains, libraries, IPTG levels) | 26,642 | ce7dc53008d1be7231694f53c2066b290eb1dab28eac8a1d80bdec76f83600ff |
| MOESM5.xlsx (growth / qPCR validation) | 26,761 | da52b19b78629e947e84d3d0b9cbae5c0b45b455cd2cc5c9000626df903fb59b |
| **MOESM6.xlsx (Suppl. Data: 33 CRISPRi–TnSeq library tables + called interactions)** | 15,842,163 | a6f8bb8a1e50b6d57a633a1552b55f598f476f0717ff6b077d3f837f57cd8559 |
| MOESM7.xlsx (random 50k control) | 29,622 | ae1e67e2c730279affa1a135aa763765a44a2ce96676dc93bb662fc51380f627 |
| MOESM8.xlsx (GSEA enrichment) | 43,496 | 54ac73a4b33d69012f986fd9cb4cd67da08091f79ea2e86e7fad0ad805fefb71 |
| MOESM9.xlsx (Antibiotic–TnSeq, 15 libraries) | 7,049,738 | c9089d2de770a01f75bdfd0ac6c94cef2b7c594e5f93ec927e56a0754efa8e7b |
| MOESM10.xlsx | 10,384 | 42cd6c162e52ef015f6eed37303fa4528cbe3c0387eed26dbcbd985bed396d48 |

Only MOESM6 is parsed. URLs in `crisprtnseq2024_spneumo.fetch.tsv`.

ID mapping also needs `data/raw/spne_annot/` (NCBI efetch GenBank flatfiles; see
`spne_annot.fetch.tsv`): CP027540.1 (D39V, SPV_ tags + gene names + SPD_/SP_/spr cross-references)
and NZ_CP027540.1 (RefSeq SPV_RS* tags with `/old_locus_tag` = SPV_ tags). sha256:
CP027540.1.gb cd83dc05a33418815d48a012bd5e439e0e800c22f0e1086d53fdf4b42014a50b;
NZ_CP027540.1.gb af61b6ee1e45b8912466beb72937d681eadfdca08a8cdeace1d33e9a3d427d30.
(efetch output is not guaranteed byte-stable across annotation updates.)

## Design

13 CRISPRi strains, each knocking down one essential gene/operon (adk, atpF, clpP, cozE, fabH, folA,
ftsH, ftsZ, gyrA, parC, pbp2x, rpoB, rpoC) in D39V, each transformed with a genome-saturating
*magellan6* transposon library. Each library was grown ± IPTG at 1–4 inducer levels (partial
knockdown; some queries have true experimental repeats: adk15/adk15R, atpF15/15R, atpF30/30R,
cozE100/cozE100(2), parC20/parC20(2)) — 33 library × condition tables in total. Per non-essential
gene, Tn-seq fitness W is computed with and without induction; the interaction score is the
**fitness difference** value = W(+IPTG) − W(−IPTG), with a two-sided t-test p-value and BH padj
per table. Negative value = the transposon knockout is sicker when the essential partner is knocked
down = aggravating / synthetic sick. ~24,000 essential × non-essential pairs; the authors call 1,334
interactions (754 negative, 580 positive), listed in the `genetic interaction` tab with a z-score.

Gene IDs: tables key on RefSeq `SPV_RS*` plus `locus_tag_SPV` (SPV_*), `OldLocus` (SPD_*) and
`TIGR4` (SP_*). `bacteria_extra.spne_map` (canonical IDs from `ids_extra.spne`, lead decision 2026-09-24) maps all of these to the benchmark's canonical D39V ID
(the name/tag used by `dualcrispri2025`). 217 of 1,873 RS tags do not map (RS tags retired from the
current RefSeq annotation, mostly RNA/pseudo features); those rows are dropped.

## Parser and label rule

`crisprtnseq2024()` aggregates over the 1–4 tables per query (mean fitness difference, min p) and
labels:

- **Positive (1):** on the authors' called-interaction list with z < 0 (their negative interactions).
- **Negative (0):** not on the list, |fitness difference| < 0.0215 (the dataset median |value|) and
  p > 0.05. The neutral band mirrors the house rule ("|score| below the screen median and clearly
  non-significant").
- **Ambiguous (null):** called positive/alleviating interactions, and everything in between.

Rationale: the authors' call combines effect size (|value| > 0.1) with padj < 0.05 per library and
concordance across libraries, which is stricter than any single-column rule I could reconstruct;
using it keeps the positives comparable to the published network. Alleviating interactions are
dropped rather than made negatives, because they are measured interactions, not non-interactions.

Counts (`uv run python -c "from slpbench.sources import bacteria_extra as B; d=B.crisprtnseq2024()"`):

| rows | positives | negatives | ambiguous |
|---|---|---|---|
| 19,186 | 666 | 8,898 | 9,622 |

Hit rate among labelled pairs 666/9,564 = 7.0% — plausible for essential × non-essential pairs at
partial knockdown.

## Replication evidence

**Within-study (split-half over libraries).** For each query, labels were re-derived from one subset
of its library tables (value < −0.1 & p < 0.05 as positives, |value| < median & p > 0.05 as
negatives) and scored with the mean fitness difference of the held-out tables. 34 splits over the 13
queries: **median AUROC 0.917** (per-query medians 0.81–0.999; recomputed by
`crisprtnseq2024_checks()`), median Spearman of the fitness
difference between halves 0.64. The one pure experimental repeat with enough positives
(cozE100 vs cozE100(2), same IPTG) gives AUROC 0.917. So the measurement is internally reproducible.

**Cross-study (dual CRISPRi-seq 2025, same strain D39V).** 2,798 pairs are measured by both sources
(the 9 queries that dual CRISPRi-seq also targets with a single-gene sgRNA: adk, clpP, cozE, folA,
ftsH, gyrA, parC, rpoB, rpoC).

| direction | n | positives | AUROC |
|---|---|---|---|
| CRISPRi–TnSeq labels recovered by dual CRISPRi-seq εSum | 1,601 | 87 | **0.502** (95% CI 0.441–0.566) |
| dual CRISPRi-seq labels recovered by CRISPRi–TnSeq fitness difference | 756 | 22 | 0.657 (95% CI 0.506–0.787) |

Both directions are recomputed by `spne_cross_checks()`, which also covers `dualtnseq2025`
(no overlapping pairs with this source: it needs viable transposon insertions where this one
targets essential genes).

Spearman between the two interaction scores on the shared pairs: **0.045** (per-query −0.05 to 0.09).
Tail overlap: of the 2% most negative CRISPRi–TnSeq pairs, 8.9% fall in the 10% most negative dual
CRISPRi-seq pairs (chance = 10%). Stricter positive rules do not fix it: requiring the call to be
significant in ≥2 libraries raises cross-study AUROC to 0.675 but leaves only 11 positives in the
overlap; |value| < −0.2 gives 0.62 with 15 positives.

This is not an ID-mapping artifact: the two studies' **single-gene** fitness measures for the same
non-essential genes (CRISPRi–TnSeq W(−IPTG) vs dual CRISPRi-seq single-sgRNA log2FC) correlate at
Spearman 0.50 over 411 shared genes, so the gene identities line up; it is the pairwise interaction
scores that do not.

**Low-throughput validation (the authors').** Supplementary Table 11 ('Correlation TnSeq vs
Mt-Growth' tab) compares the screen's fitness difference with growth curves of 32 individually
constructed deletion mutants in the matching CRISPRi background: Pearson r = 0.86, Spearman 0.86,
sign agreement 32/32. All 32 are called interactions, so there are no negative controls and no
AUROC can be computed, but the called effects are real in individually built strains.

Interpretation: the assays differ (transposon knockout + titrated knockdown vs two knockdowns,
operon-level sgRNAs, different media/IPTG regimes), and each is reproducible on its own, but they
give essentially independent answers on shared pairs. This is the ito2021 situation: internally
consistent, not recovered by an independent study of the same context.

## Recommendation

**LEAD DECISION 2026-09-24: measurements only, not benchmark labels** (all three *S. pneumoniae*
screens), which matches the recommendation below.

**Measurements yes, benchmark labels no** (keep in `data/interim/measurements` as optional training
data, like ito2021/thompson2021). Rule 1 of the audit prefers cross-study evidence where a study of
the same context exists; here it exists and gives 0.50.

If the lead prefers to include it anyway, the defensible variant is positives restricted to calls
significant in ≥2 libraries (cross-study 0.675, ~11 positives in the overlap, 141 positives overall),
and it should then be a separate screen/stratum from `dualcrispri2025` so the disagreement does not
propagate into merged labels. Note also the flip side of the same table: dual CRISPRi-seq's own
labels are recovered by CRISPRi–TnSeq at only 0.657 on 22 overlapping positives, so this overlap is
weak evidence in both directions and does not by itself argue for dropping `dualcrispri2025`
(which passes on its own replicates at 0.95).

## Extra artifacts from this source

- `spne_single_tnseq()` — independent single-gene fitness for *spne*: median over libraries of the
  transposon-mutant fitness W without induction, minus 1 (negative = sicker), 1,655 genes.
  Complements `fitness.spne_single()` (dual CRISPRi-seq single-sgRNA log2FC); Spearman 0.50 on the
  411 shared genes.
- `spne_table()` / `spne_resolver()` — D39V ID mapping (SPV_, SPV_RS*, SPD_, SP_ TIGR4, spr R6,
  names and synonyms → canonical D39V ID), reusable by other *S. pneumoniae* sources.
- MOESM9 (Antibiotic–TnSeq) is gene × drug, not gene × gene, so it is not a GI source.
