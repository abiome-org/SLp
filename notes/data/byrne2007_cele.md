# byrne2007 (C. elegans RNAi x mutant growth GI map, Byrne et al. 2007)

**Citation.** Byrne AB, Weirauch MT, Wong V, Koeva M, Dixon SJ, Stuart JM, Roy PJ. A global analysis of
genetic interactions in *Caenorhabditis elegans*. *J Biol* 6:8 (2007). doi:10.1186/jbiol58. PMC2373897.

**Files** (`data/raw/byrne2007_cele/`, from the Europe PMC supplementary-files zip; `byrne2007_cele.fetch.tsv`)

| file | sha256 |
|---|---|
| jbiol58-S4.xls (all interaction strengths; SGI network strengths) | 528e41411b34b0aea4bf6c9b310064044bfae4b3fcf18595423523e0649a91f0 |
| jbiol58-S1.xls (growth matrix, same data as a matrix) | dfe045b2f68cefaf75571af2b99781482edbce3448c1d292d1d4e205f3166639 |
| jbiol58-S3.xls (SGI network + Lehner/other networks) | bf1a518958f4ad811fd6535c61530eef8df45e98a6ddf6a3a84b9319f60a4546 |

**License.** BioMed Central open access (CC BY).

**Design.** 11 query mutant strains (let-60, let-23, sem-5, sos-1, glp-1, bar-1, daf-2, sma-6, clk-2,
egl-15, let-756; mostly hypomorphic signalling alleles) x ~860 feeding-RNAi clones (a signalling set
and a chromosome III set), growth scored semi-quantitatively in several rounds. Interaction strength
= enhancement of the RNAi phenotype in the mutant over wild type, averaged over usable rounds (range
-2.5 to 6; higher = sicker). ~7,000 unique pairs tested. The authors' 1,246-edge SGI network holds the
pairs that passed their internal consistency criteria. Species `cele` (WBGene via
`ids_extra.cele`, from CGC name / sequence name), context `N2` (the wild-type background of the
query strains), mechanism `RNAi x mutant`.

**Parser.** `eukaryotes_extra.byrne2007()`; score = -strength (SLB sign convention).

**Counts.** 7,231 pairs over 837 genes; **994 positives, 3,574 negatives**, 2,663 ambiguous.
192 of 7,582 rows have an RNAi target that does not resolve to a live WS298 gene.

**Label rule.** Positive: in the authors' SGI network and strength >= 2. Negative: not in the network
and |strength| < median |strength| (0.25). Everything else ambiguous.

**Replication evidence** (`byrne2007_checks()`)

| check | pairs | pos | AUROC |
|---|---|---|---|
| within-study: pairs measured twice (two batches or reciprocal), labels from measurement 1 scored by 2 | 154 | 63 | 0.874 |
| same, 2 -> 1 | 154 | 23 | 0.933 |
| cross-study: Lehner 2006 hit calls scored by Byrne strength | 1,127 | 26 | **0.703** |
| cross-study: Byrne labels scored by Lehner 2006 binary hit/no-hit | 766 | 182 | 0.533 |

Pearson between duplicate measurements 0.66. The Lehner 2006 comparison is asymmetric by design:
Lehner is a binary, low-sensitivity screen (1% of tested pairs called), so it recovers only 11 of 179
Byrne positives on shared pairs (AUROC of a binary score = 0.5 + (TPR-FPR)/2); Lehner's own hits are
recovered by Byrne's quantitative strength at 0.70. Fitness-only AUROC (WormBase lethality/sterility
of the two genes, `cele_single()`): 0.697.

**Hit rate caveat.** 994 / 4,568 labelled pairs are positive (22%; 14% of all tested pairs; the
authors report 1,246 SGIs in ~7,000 tests, 18%). The queries are sensitised signalling mutants and the
RNAi set is enriched for essential/signalling genes, so a high rate is expected, but it is at the
level at which SLB excluded Tang 2022 (22%). Raising the cut to strength >= 3 or 4 does not change
the Lehner comparison (0.70 / 0.53).

**Recommendation: CONDITIONAL INCLUDE (lead's call).** It passes the AUROC rule both within-study
(0.87-0.93, small n) and cross-study (0.70 vs Lehner), and is not contradicted (the reverse direction
is uninformative given Lehner's sensitivity). Against it: the hit rate (18-22%) and the small
within-study sample. If included, it makes a C. elegans species with 11 query genes; after the family
split, test pairs need a held-out query, so expect ~2 test queries and a few hundred test pairs.
It would be the only metazoan species besides human.

**Single-gene effect for `cele`:** `eukaryotes_extra.cele_single()` - WormBase WS298 phenotype
annotations (RNAi and alleles; `data/raw/wormbase/`, `wormbase.fetch.tsv`): -1 if any positive
annotation to lethal (WBPhenotype:0000062) or larval arrest (0000059) or their descendants, -0.5 for
sterile (0000688) or slow growth (0000031) and descendants, 0 for any other assayed gene (incl. genes
with only NOT annotations). 22,479 genes: 5,407 at -1, 783 at -0.5, 16,289 at 0. Same scale as
the SLB S. pombe viability effect.
sha256: phenotype_association.WS298.wb 3459ca02779b93557a5fde3c58c978036f0c5e0e943affebfe2b3b6b89b034ab,
phenotype_ontology.WS298.obo ac8fae2c31975d77b1b32b7e2a531df2d4d8491dd510b363b22ac3ec19668e85.
