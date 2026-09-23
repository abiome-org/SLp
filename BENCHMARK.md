# SLB-1: a held-out, real-data synthetic-lethality benchmark

**Task.** Given a genetic context (a species plus a cell line or strain) and a gene pair, score how
likely it is that losing both genes is synthetically lethal or sick, i.e. a strong negative genetic
interaction beyond the two single-loss effects.

**Data.** Every example is a measured outcome from a published combinatorial screen. Both positives
and negatives were tested in the lab. There are no random "unknown = negative" pairs, no
literature-mined labels and no synthetic data. The v1 build draws on 19 sources across 5 species:
human (60 cell lines, 16 studies), *S. cerevisiae*, *S. pombe*, *D. melanogaster* and
*S. pneumoniae*. Counts are in [DATA_CARD.md](DATA_CARD.md).

**Held out.** Genes are grouped into families: close paralogs plus reciprocal-best orthologs across
species. Whole families are hashed into train, dev or test. A test pair has both genes in test
families, so neither gene, nor any of its paralogs, nor its orthologs in any other species appear in
the train split.

## Quick start

```bash
uv run slpbench baseline lgbm --split dev --out results/lgbm_dev.parquet
uv run slpbench eval results/lgbm_dev.parquet --split dev
uv run slpbench compare results/fitness_dev.parquet results/lgbm_dev.parquet --split dev
uv run slpbench check-leakage my_training_pairs.parquet
```

Your model reads `data/bench/slb1/{split}.parquet` (or `test_inputs.parquet`) and writes one
`score` per `example_id`. Higher means more likely SL. Rebuilding from raw sources:
`uv run python -m slpbench.fetch && uv run slpbench build && uv run slpbench card` (see
[Rebuilding](#rebuilding)).

## Files (`data/bench/slb1/`)

| File | Contents |
|---|---|
| `train.parquet` | Labelled training pairs. Both genes in train families. |
| `dev.parquet`, `dev_semi.parquet` | Labelled. Use for hill-climbing and model selection. |
| `test_inputs.parquet`, `test_semi_inputs.parquet` | No labels. |
| `hidden/test*_labels.parquet` | Test labels. Only `slpbench eval --split test` reads them. |
| `contexts.parquet` | Per context: Cellosaurus accession, DepMap ID, disease, sex, ancestry group and fractions. |
| `held_out_families.parquet` | Gene → family → bucket. Used by the leakage checker. |
| `manifest.json` | Build parameters, row counts and sha256 of every file. |

Example columns: `example_id, species, context_id, ancestry_group, gene_a, gene_b, same_family,
sources, label`.

Gene IDs:
- human: HGNC symbols
- *S. cerevisiae*: SGD systematic ORFs
- *S. pombe*: PomBase systematic IDs
- fly: FBgn IDs
- *S. pneumoniae*: D39V gene names or locus tags

`slpbench.ids.resolve` maps other names to these.

## Metric

**SLB score** is the primary hill-climbing number.

1. Scores are compared only within a context: the AUROC over (SL, non-SL) pairs from the same cell
   line or strain. A model gains nothing by learning which screens or cell lines have high hit rates.
2. Each species gets one score. For human, the score is the mean over genetic-ancestry groups (each
   with at least 10 positives in the split) of that group's context-stratified AUROC. Each ancestry
   group counts equally, however many cell lines it has.
3. SLB score is the mean of the species scores. Each species counts equally, so yeast's 185k test
   pairs cannot drown out fly's 11k.

`eval` also reports these for every species, ancestry group, human cell line and paralog vs.
non-paralog stratum:
- **within-gene AUROC**: stratified by (context, gene). It asks "given gene A, rank its partners".
- **AP lift**: average precision ÷ prevalence.

Use `--boot N` for a gene-family cluster-bootstrap 95% CI. Use `slpbench compare A B` for a paired
bootstrap of the difference. That is the right tool for deciding whether a change helped.

**Protocol.** Hill-climb on `dev`. Evaluate on `test` only at milestones, and record every test
evaluation in `results/LEADERBOARD.md`. The test labels are local files, so holding back from
peeking is on us.

## Leakage contract

A model scored on `test` must not have been fitted on any record that involves a gene from a test
family, in any species:

- **Not allowed:** any combinatorial or multi-gene perturbation readout (double knockout, knockdown or
  CRISPRi screen); any SL or genetic-interaction label or edge, from any source (SynLethDB, BioGRID,
  SLKB, knowledge graphs containing SL or GI edges, papers). This covers the benchmark's own sources
  and any other copy of the same screens.
- **Allowed, but declare it:** single-gene data (DepMap and other single-KO screens, expression,
  sequence, GO, PPI, pathways), and pretrained models whose training data follows the same rule.

`slpbench check-leakage pairs.parquet` (columns `species, gene_a, gene_b`) flags records that touch
held-out families. That includes genes absent from the benchmark that are paralogs or orthologs of
held-out genes. Add `--allow-dev` for final models trained on train + dev.

Models trained on public SL databases without this filter can still be scored, but they are marked
`leaky` on the leaderboard. Most published SL predictors fall in that group.

## Label rules

Each source is labelled with its own interaction score and its authors' threshold wherever one
exists.

- **Negative:** a tested pair that was not called, and whose score falls in that screen's neutral band.
- **Ambiguous:** anything between positive and negative. It is dropped, not guessed.
- **Merging:** measurements of the same (species, context, pair) from several sources are merged.
  Pairs where the sources disagree (at least one says SL and another non-SL) are dropped.

| Source | Positive | Negative |
|---|---|---|
| Horlbeck 2018, Han 2017, Shen 2017, Zhao 2018, Wong 2016 (SLKB original calls) | Author SL call | Not called, and \|score\| < median for that screen |
| Dede 2020, CHyMErA 2020, Parrish 2021, Thompson 2021, Ito 2021 (Ryan-lab uniform zdLFC, final time point) | zdLFC ≤ −3 | \|zdLFC\| < 1 |
| Chou 2025 | ZdLFC < −2 (authors) | \|ZdLFC\| < 1 |
| Flister 2025 | Author "Lethal" call; diff_z ≤ −2 for lines with no call | \|diff_z\| < 1 |
| Harle 2025 | Author binary hit matrix | Not a hit, FDR > 0.25, \|GI\| < median for that line |
| SPIDR / Fielden 2025 (CRISPRi, RPE1) | GEMINI sensitive score ≤ −1 | Score = 0 (authors' no-evidence value) |
| Costanzo 2016 SGA | ε < −0.12 and p < 0.05 (authors' stringent cut) | p > 0.25 and \|ε\| < median |
| Ryan 2012 *S. pombe* E-MAP | S < −2.3 | \|S\| < 1 |
| Fischer 2015, Heigwer 2023 (fly S2 RNAi, cell count) | π < 0 at FDR < 1% / 10% (per paper) | FDR > 0.25 and \|π\| < median |
| Dual CRISPRi-seq 2025 (*S. pneumoniae*) | Authors' "Negative" call; single-gene sgRNA targets only | "Neutral", \|ε\| < median, not essential × essential |

Exclusions:
- **Diehl 2021 (RPE1):** 63% of tested pairs are called SL.
- **Tang 2022 (22Rv1):** 22% are called SL. Both rates are implausible for an unbiased pairwise screen.
- **Najm 2018:** no effect size.
- **SLKB's own Ito 2021 calls:** the score direction is ambiguous, so the uniform re-scoring is used instead.

## Known limitations

- **Ancestry coverage is thin.** The human test set is 27k pairs:

  | Ancestry group | Cell lines |
  |---|---:|
  | European (EUR) | 41 |
  | East Asian (EAS) | 11 |
  | African (AFR) | 1 (RKO) |
  | Admixed | 2 (HeLa, NCI-H23) |
  | Unknown | 5 (hTERT-RPE1, HAP1, HEK293T, Mel202, C092) |

  The AFR and admixed strata are small and noisy. This reflects the public screens, and new screens
  in non-European lines are the highest-value additions.
- **Human screens mostly test paralog pairs,** so human same-family pairs are overrepresented.
- **Label agreement across studies is low.** Where two studies screened the same pair in the same
  cell line and at least one called it SL, the other agreed 20% of the time. A perfect model of
  biology would still score well below 1.0.
- **Fission yeast, fly and pneumococcus have no single-gene feature tables** in the reference
  baselines, so those baselines score 0.5 there. That is a property of the baselines, not the benchmark.

## Rebuilding

1. `uv run python -m slpbench.fetch` downloads every raw source. The URLs are in
   `src/slpbench/fetch.py`.
2. `uv run slpbench build` unpacks the archives, then parses, merges and splits.
3. `uv run slpbench card` rewrites DATA_CARD.md.

`reference/raw_sha256sums.txt` pins the raw files this build used.

The split depends only on `SALT`, the family graph and the bucket fractions, all recorded in
`manifest.json`. Changing any of them creates a new benchmark version, not an update to SLB-1.
