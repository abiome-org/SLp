# billmann2026: Billmann, Costanzo et al. 2026 Cell, HAP1 global GI map

- Data: Mendeley 10.17632/bpcpfns6vb v2 (CC BY 4.0). Data S4 sheets `qGI_scores` and `qGI_FDR` are wide
  matrices of 17,804 library genes x 298 screens. Each column is named `<QUERY>_<GIN>_<min|rich>`. S1 is
  the screen table and S5 lists replicate screens with MCMC consensus probabilities for 7 queries (not
  used). SHA-256s match Mendeley.
- Parser: `src/slbench/sources/staging/billmann2026.py`. The xlsx sheets are read with a regex streaming
  reader (openpyxl is too slow on the 770 MB of XML), then cached as long parquet in
  `data/raw/billmann2026/_parsed/` (about 20 s the first time).
- Arm: `rich` = IMDM, the standard HAP1 medium, with 166 screens and 128 queries. `min` = DMEM with 10 mM
  glucose and 1 mM Gln, a metabolic sensitising condition with 132 screens; it is excluded.
- Replicate screens (28 queries, 2 to 5 rich screens each) are averaged: mean qGI, median FDR.
  `replicates()` returns one column per screen (`gi_rep1..5`, `fdr_rep1..5`). The query symbols there are
  unresolved raw names.
- Label: positive if qGI < -0.5 and FDR < 0.05. Negative if |qGI| < the screen's median |qGI| and
  FDR > 0.5. Otherwise null.
- Counts: 2,245,995 pairs, 8,261 positives (0.37%) and 1,143,296 negatives, with 17,749 genes.
  - 39 symbols do not resolve. Most are RP11-/AC clone IDs, plus QARS, STRA13 and B3GNT1. The query SLX1
    (SLX1A/SLX1B) also fails, which leaves 127 queries.
  - Paralog pairs make up 0.017% of all pairs (376), but 16% of them are positive.
- Degree (hub structure): every query has about 17,748 pairs. Positives per query have a median of 43 and
  range from 2 to 305; the top queries are BCL2, CDKN2B, ATP5MG, NDUFA2 and VPS52. On the library side,
  3,710 genes carry at least one positive, with at most 24 (RIC8A). 8,001 pairs have both genes as queries,
  and their two orientations are merged by finalize.
- Own replication:
  - SLB rule on screen i, scored by screen j of the same query: AUROC 0.87 pooled; the 20 ordered screen
    pairs range from 0.79 to 0.95.
  - Per query (rep1 to rep2) the mean is 0.79. Scored instead by an unrelated query's screen it is 0.49, so
    replication is query-specific.
  - split_rule_auroc on rep1 vs rep2: 0.78 (1% tail) and 0.80 (qGI < -0.5). Spearman is 0.17 genome-wide,
    which is expected because most pairs are null.
  - Min-medium screens of the same query recover the rich labels at 0.69 (1,459 positives), a weaker
    check because the condition differs.
  - Scoring the final labels against a single screen is circular for replicated queries, so it is not
    reported.
- Cross-study:
  - HAP1: only 7 pairs overlap with CHyMErA, so this is not testable.
  - Other cell lines (weak evidence): pooled 0.54 over 166 positives. The paralog libraries do better:
    Harle 0.85, Chou 0.67 / 0.86, Dede 0.92 (their labels scored by us). Ito is 0.39 over about 14 unique
    pairs.
- Fitness: DepMap -(fa+fb) gives 0.61, but covers only 22% of pairs. The HAP1 WT LFC of the library gene
  gives 0.69. Both are below 0.8.
- Subsets: the Aregger 2020 lipid queries (FASN, ACACA, C12orf49, SREBF1/2, LDLR, PDK1, SLC16A3) are
  present but all in the `min` arm, so they are excluded here. Xiao 2024 could not be identified from the
  metadata.
- Verdict: include, on own-replication evidence.
