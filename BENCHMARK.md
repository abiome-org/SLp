# SLB-1.1: a held-out, reproducibility-checked synthetic-lethality benchmark

**Task.** Given a genetic context (species + cell line or strain) and a gene pair, score how likely
losing both genes is synthetically lethal or sick: a strong negative genetic interaction beyond the
two single-loss effects.

**Data.** Every example is a measured outcome from a published combinatorial screen. Positives and
negatives were both tested. There are no "unknown = negative" pairs, no literature-mined labels and no
synthetic data. Every source that supplies labels passed a reproducibility check (next section).

The v1.1 build covers:
- **Human:** 50 cell lines from 8 studies, annotated with genetic ancestry.
- ***S. cerevisiae***, ***S. pombe*** and ***S. pneumoniae***: one source each.

Counts are in [DATA_CARD.md](DATA_CARD.md). The replication evidence for every source, included or
not, is in [REPLICATION.md](REPLICATION.md).

**Held out.** Genes are grouped into families: paralogs with ≥ 30% protein identity, plus orthologs
across human, both yeasts and fly (reciprocal best hits, or ones supported by at least 3 prediction
methods). Whole families are hashed into train, dev or test. In a test pair, neither gene has a
paralog above 30% identity or any ortholog in train, in any species.

There is only one evaluation: both genes held out. Splits that hold out a pair but let its genes
appear in training (sometimes called CV1/CV2) are not offered. Models can memorise which genes have
many SL partners, so those splits measure training-set lookup, not prediction.

## Quick start

```bash
uv run slpbench baseline lgbm --split dev --out results/lgbm_dev.parquet
uv run slpbench eval results/lgbm_dev.parquet --split dev
uv run slpbench compare results/fitness_dev.parquet results/lgbm_dev.parquet --split dev
uv run slpbench check-leakage my_training_pairs.parquet
```

A model reads `data/bench/slb1.1/{split}.parquet` (or `test_inputs.parquet`) and writes one `score`
per `example_id`. Higher means more likely SL.

## Label quality

The same pair measured twice, in replicates or by two labs, often gets a different SL call. The
benchmark therefore admits a source's labels only if they reproduce.

**Rule.** A source's labels are used if all three hold:
1. An independent re-measurement recovers them at AUROC ≥ 0.65. The re-measurement is another study in
   the same cell line where one exists; otherwise the source's own replicates, alleles or query/array
   orientations, each re-scored separately.
2. They are not contradicted by the cross-study consensus.
3. The hit rate is plausible.

`slpbench audit` re-runs every check and regenerates REPLICATION.md.

**Included sources:**

| Source | Evidence | AUROC |
|---|---|---|
| Dede 2020 | Cross-study, vs the other included studies | 0.93 |
| Chou 2025 | Cross-study | 0.81 |
| Harle 2025 | Cross-study | 0.80 |
| Flister 2025 | Cross-study | 0.76 |
| Horlbeck 2018 | Own replicates | 0.71–0.81 |
| Parrish 2021 | Own replicates | 0.90–0.95 |
| Zhao 2018 | Own replicates | 0.69–0.72 |
| SPIDR 2025 | Own replicates, after re-scoring | 0.86 |
| Costanzo 2016 (*S. cerevisiae*) | Opposite orientation, at the SLB cut-off | 0.74 |
| Ryan 2012 (*S. pombe*) | Independent alleles or orientation, at the SLB cut-off | 0.76 |
| Dual CRISPRi-seq 2025 (*S. pneumoniae*) | Own replicates | 0.95 |

**Excluded sources.** Their measurements are kept in `data/interim/measurements` as optional training
data, but they supply no benchmark labels.

| Source | Reason |
|---|---|
| Ito 2021 | Agrees with itself (0.91–1.00), but contradicted by 4 concordant studies (0.55) |
| Thompson 2021 | Cross-study 0.60; its calls are largely predictable from single-gene fitness (0.87) |
| Shen 2017 | Its own replicates don't recover its labels (0.56–0.72) |
| Han 2017, Wong 2016, CHyMErA 2020 | Unverifiable: no usable replicate counts and no overlap with other studies |
| Fischer 2015, Heigwer 2023 (fly) | The two studies contradict each other (0.57 / 0.41). Same lab, same cell line |
| Diehl 2021, Tang 2022 | Implausible hit rates: 63% and 22% of tested pairs called SL |

**Where published calls were replaced:**
- **Budding and fission yeast:** stronger interactions replicate better, so the positive cut-offs are
  stricter than the authors' (ε < −0.2 instead of −0.12; S < −3 instead of −2.3).
- **SPIDR:** its published GEMINI calls are recovered by its own replicates at only 0.62. Its raw
  counts are therefore re-scored with the same additive zdLFC recipe as the other paralog screens.

Remaining disagreement: where two included studies measured the same pair in the same cell line and
at least one called SL, all of them agreed 43% of the time. The earlier set, with the non-replicating
sources included, managed 20%.

## Label rules

- **Negative:** a tested pair that was not called, and whose score falls in that screen's neutral band.
- **Ambiguous:** anything between positive and negative. It is dropped.
- **Merging:** the same (species, context, pair) measured by several included sources is merged;
  conflicting labels are dropped.

| Source | Positive | Negative |
|---|---|---|
| Horlbeck 2018, Zhao 2018 (SLKB original calls) | Author SL call | Not called, and \|score\| < median for that screen |
| Dede 2020, Parrish 2021 (Ryan-lab uniform zdLFC) | zdLFC ≤ −3 | \|zdLFC\| < 1 |
| SPIDR 2025 (RPE1, CRISPRi; re-scored from counts) | z(additive GI) ≤ −3 and GI < 0 in both replicates | \|z\| < 1 |
| Chou 2025 | ZdLFC < −2 (authors) | \|ZdLFC\| < 1 |
| Flister 2025 | Author "Lethal" call; diff_z ≤ −2 for lines with no call | \|diff_z\| < 1 |
| Harle 2025 | Author binary hit matrix | Not a hit, FDR > 0.25, \|GI\| < median for that line |
| Costanzo 2016 SGA | ε < −0.2 and p < 0.05 | p > 0.25 and \|ε\| < median |
| Ryan 2012 *S. pombe* E-MAP | S < −3 | \|S\| < 1 |
| Dual CRISPRi-seq 2025 (*S. pneumoniae*) | Authors' "Negative" call; single-gene sgRNAs, essential × essential pairs dropped | "Neutral", \|ε\| < median |

## Metric

**SLB score** is the primary hill-climbing number.

1. **Compare within a stratum.** A stratum is one context (cell line or strain) × one screen (source
   set). Using AUROC, SL pairs are compared only with non-SL pairs from the same stratum. Knowing which
   cell lines or libraries have high hit rates therefore earns nothing: a library-prior baseline
   scores exactly 0.500.
2. **Human species score:** the mean over genetic-ancestry groups with at least 20 positives in the
   split. Ancestry is the donor's genotype-inferred majority super-population from Cellosaurus
   (Kessler et al. 2019), with more than 50% as the cut-off. Lines with no estimate (hTERT-RPE1, C092)
   are reported but not averaged in. Each ancestry group counts equally, however many cell lines it has.
3. **SLB score** = the mean of the species scores. Each species counts equally.

`eval` reports every species, ancestry group, cell line and paralog/non-paralog stratum:
- **fitness-matched AUROC** (below), plus its own aggregate, `SLB fitness-matched`.
- **within-gene AUROC**: stratified additionally by gene. It asks "given gene A, rank its partners".
- **AP lift**: average precision ÷ prevalence.

Use `--boot N` for a gene-family cluster-bootstrap 95% CI. Use `slpbench compare A B` for a paired
bootstrap of the difference on both scores. Use `compare` on dev to decide whether a change helped.

### The single-gene fitness confound

SL calls concentrate on genes that are already sick on their own. That is real biology
(interaction degree tracks single-mutant fitness), but it means SLB can be climbed by predicting
sickness rather than interactions. The `fitness` baseline uses only -(f_a + f_b), and its test scores
are on the [leaderboard](LEADERBOARD.md).

**Fitness-matched AUROC** additionally requires the compared SL and non-SL pairs to have both genes in
the same within-species quintiles of a reference single-loss effect:

| Species | Reference single-loss effect |
|---|---|
| Human | DepMap mean Chronos score |
| *S. cerevisiae* | SGA single-mutant fitness |
| *S. pneumoniae* | Single-sgRNA log2FC |
| *S. pombe* | None available, so the metric reduces to SLB there |

The quintiles are in `gene_single_effects.parquet`. Read both scores together. A gain in SLB that does
not show up in the fitness-matched score is better single-gene modelling, not better pair modelling.

**Protocol.** Hill-climb on `dev`. Evaluate on `test` only at milestones, and record every test
evaluation in `leaderboard.yaml`; `slpbench leaderboard` regenerates [LEADERBOARD.md](LEADERBOARD.md).

## Leakage contract

A model scored on `test` must not have been fitted on any record involving a gene from a test family,
in any species:

- **Not allowed:** any combinatorial or multi-gene perturbation readout; any SL or genetic-interaction
  label or edge from any source (SynLethDB, BioGRID, SLKB, knowledge graphs with SL or GI edges,
  papers). This includes the excluded sources above.
- **Allowed, but declare it:** single-gene data (DepMap and other single-KO screens, expression,
  sequence, GO, PPI), and pretrained models whose training data follows the same rule.

`slpbench check-leakage pairs.parquet` (columns `species, gene_a, gene_b`) flags records touching
held-out families, including genes absent from the benchmark that are paralogs or orthologs of
held-out genes. Add `--allow-dev` for final models trained on train + dev. Models fitted on public SL
databases without this filter are listed as `leaky` and not ranked.

What "held out" leaves behind:
- 31% of human test genes (15% in *S. cerevisiae*, 19% in *S. pombe*) have a distant paralog in
  train, always below 30% identity (median 23%).
- No test gene has an ortholog in train at any level of algorithm support.

## Files (`data/bench/slb1.1/`)

| File | Contents |
|---|---|
| `train.parquet` | Labelled. Both genes in train families. |
| `dev.parquet`, `dev_semi.parquet` | Labelled. Hill-climbing and model selection. `*_semi`: one gene held out. |
| `test_inputs.parquet`, `test_semi_inputs.parquet` | No labels. |
| `hidden/test*_labels.parquet` | Test labels. Only `slpbench eval --split test` reads them. |
| `contexts.parquet` | Per context: Cellosaurus accession, DepMap ID, disease, sex, ancestry group and fractions. |
| `held_out_families.parquet` | Gene → family → bucket. Used by the leakage checker. |
| `gene_single_effects.parquet` | Reference single-loss effect and quintile per gene. A permitted input. |
| `manifest.json` | Build parameters, excluded sources, row counts, sha256 of every file. |

Example columns: `example_id, species, context_id, ancestry_group, gene_a, gene_b, same_family,
sources, label`. Gene IDs:

| Species | ID |
|---|---|
| Human | HGNC symbol |
| *S. cerevisiae* | SGD systematic ORF |
| *S. pombe* | PomBase systematic ID |
| *S. pneumoniae* | D39V gene name or locus tag |

`slpbench.ids.resolve` maps other names.

## Known limitations

- **Ancestry coverage is thin.** European 38 lines, East Asian 7, African 3 (RKO, HeLa, NCI-H23),
  with 30 test positives for the African group. This reflects the public screens. New screens in
  non-European lines are the highest-value additions.
- **No fly or other metazoan besides human.** The only two fly GI maps contradict each other.
- **Human screens mostly test paralog pairs,** so same-family pairs are overrepresented.
- **Inclusion is binary.** Every included source cleared the bar, but not equally: Dede 0.93 versus
  Zhao 0.69. Labels are not weighted by source reliability.

## Rebuilding

`uv run python -m slpbench.fetch` downloads the raw sources. `uv run slpbench build` parses, audits
decisions, merges and splits. Then run `uv run slpbench audit` and `uv run slpbench card`.
`reference/raw_sha256sums.txt` pins the raw inputs. The split depends only on `SALT`, the family
graph and the bucket fractions (all in `manifest.json`). A robustness study across 4 alternative
salts is in [ROBUSTNESS.md](ROBUSTNESS.md).

## Changes from SLB-1

- Label sources are now filtered by the replication audit. Ito, Thompson, Shen, Han, Wong, CHyMErA
  and both fly maps are out.
- Yeast positive cut-offs are stricter, and SPIDR is re-scored from its raw counts.
- Strata are context × screen, which removes a library-composition shortcut worth up to 0.57 AUROC
  on human.
- Ancestry uses a majority rule (> 50%), and a group needs at least 20 positives to count.
- Orthologs supported by 3 or more methods join families.
