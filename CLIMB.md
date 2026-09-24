# SLB-1.3 benchmark and first model climb

## Frozen grader

The benchmark manifest is SHA-256 `b1ac12d961b80d917b518d96834acd8289b954751f61833d37d26738631b41a9`;
the scorer is **1.3.2**. The headline number is fitness-balanced AUROC within context × screen,
averaged over human ancestry groups and then equally over human, *S. cerevisiae* and *S. pombe*.
Higher is better. Auxiliary species are reported separately. The point-score definition is SLB-1.3;
Scorer 1.3.1 hardened uncertainty and submission validation; 1.3.2 orders
families canonically in the bootstrap so its intervals reproduce across runs.
Neither patch changes valid point scores.

The dev split has 196,523 measured pairs; the private test split has 284,153. Test families are
disjoint from train/dev families across paralogs and orthologs. Labels are strong negative genetic
interactions versus measured neutral outcomes, with ambiguous pairs removed. `slpbench verify --raw`
validated all 14 built artifacts and all 292 pinned raw files. The public release archive is
`data/release/slb1.3-public.tar.gz`, pinned by `reference/slb1.3-public.sha256`. It includes
test inputs and dev scoring weights, but no test labels or test propensity files. A model generated
test predictions from this public bundle, which the private evaluator scored successfully.
Dev `eval` and `compare` also run using only this bundle.

**Pass/fail gates:** published source replication audit; both classes measured; family split
integrity; no GI-derived model inputs for ranked models; finite scores for unique, known example IDs;
complete prediction coverage unless the evaluator explicitly records imputation; artifact and
prediction and result hashes on the test leaderboard. The 27 repository tests include direct homology-edge
checks and adversarial submission checks.

The dev grader probes (`reference/grader_probe_slb1.3.json`) show:

| Probe | SLB score | Expected response |
|---|---:|---|
| Exact label oracle | 1.0000 | accepts a correct ordering |
| Inverted labels | 0.0000 | rejects a wrong ordering |
| Constant scores | 0.5000 | rejects skipped work |
| Context × screen hit-rate lookup | 0.5000 | rejects a library prior |
| Random, 20 seeds | 0.4992 ± 0.0106 SD | establishes a practical noise scale |
| Fitness-only LightGBM | 0.5179 | measures residual fitness signal |

Duplicate, unknown and null IDs; NaN and infinity scores; missing rows; a corrupted release file;
and a tampered leaderboard result all fail their respective gates. The official evaluator's point
score is deterministic for a fixed file; uncertainty is reported with a family-cluster bootstrap.

## Model comparison and climb

`MODELS.md` compares **87 dev variants**: 79 non-leaky entries, 6 known leaky entries, and 2
possibly leaky entries. It includes graph and matrix methods from the [Feng et al. benchmark](https://www.nature.com/articles/s41467-024-52900-7),
recent [SynLeaF](https://arxiv.org/abs/2603.22369) and [MuSL](https://github.com/JieZheng-ShanghaiTech/MuSL),
mechanistic and statistical baselines, and our two ensemble finalists. Published architecture
adaptations are identified in `notes/models/`; only ranked models use permitted inputs. The
complete dev search, including all 34 scored trials and prediction hashes, is in
`reference/slp_fusion_dev_search.json` and reproducible with
`uv run python scripts/models/slp_fusion/search_dev.py`.

The working hypothesis was that network/ontology, KG and paralog scores rank different gene
pairs well. Source scores were converted to within-species percentile ranks before averaging.
The following ledger shows the sequence and instructive rejected changes:

| Candidate | Dev SLB | Outcome |
|---|---:|---|
| GO/PPI GBM alone | 0.6256 | strongest individual all-species starting point |
| GO/PPI ×2 + Ontotype | 0.6392 | kept; ontology adds signal |
| GO/PPI ×2 + SynLeaF | 0.6463 | kept for exploration; KG adds dev signal |
| GO/PPI ×2 + Ontotype + SynLeaF | 0.6506 | kept |
| **Core:** preceding + De Kegel | **0.6550** | finalist; +0.0293 over GO/PPI, paired dev 95% CI +0.0091 to +0.0635 |
| Core + MuSL | 0.6497 | rejected; lower score |
| Core + SLMGAE in yeasts | 0.6537 | rejected; lower score |
| Core + MVGCN-iSL in yeasts | 0.6545 | rejected; lower score |
| **Loss:** core + DepMap OLS in human | **0.6663** | finalist; +0.0113 over core, paired dev CI −0.0078 to +0.0241 |

Both finalists and the scorer were committed in `52f2aa4` before their test predictions were
generated. Their weights were not changed after seeing test scores. Both are reported below; the
loss variant has the higher held-out point score among these frozen finalists.

## Held-out results

Test scores use the frozen scorer and 200 family-cluster bootstrap replicates. The leaderboard
re-checks each result against its prediction file and benchmark digest.

| Model | Dev SLB | Test SLB (95% CI) | Human | *S. cerevisiae* | *S. pombe* |
|---|---:|---:|---:|---:|---:|
| **SLP Fusion (loss)** | **0.6663** | **0.6448 (0.5946–0.6837)** | 0.6840 | 0.5810 | 0.6694 |
| SLP Fusion (core) | 0.6550 | 0.6394 (0.5880–0.6805) | 0.6678 | 0.5810 | 0.6694 |
| Ontotype | 0.6163 | 0.6405 (0.5975–0.6726) | 0.6250 | 0.5886 | 0.7078 |
| Ontotype (pooled) | 0.5999 | 0.6315 (0.5905–0.6656) | 0.6395 | 0.5806 | 0.6744 |
| GO/PPI GBM | 0.6256 | 0.6247 (0.5763–0.6603) | 0.6630 | 0.5902 | 0.6207 |
| De Kegel (all-species) | 0.5865 | 0.6027 (0.5510–0.6466) | 0.6993 | 0.5520 | 0.5568 |
| MuSL (all-species) | 0.5802 | 0.5935 (0.5487–0.6336) | 0.6356 | 0.5526 | 0.5922 |
| SynLeaF (all-species) | 0.5999 | 0.5543 (0.5082–0.6048) | 0.5372 | 0.5234 | 0.6024 |
| Reference LightGBM | 0.5315 | 0.5513 (0.5091–0.5815) | 0.6102 | 0.5176 | 0.5262 |

With 400 paired family-bootstrap replicates, Fusion (loss) exceeds reference LightGBM by
**0.0935** (95% CI **+0.0562 to +0.1226**). Its difference from Ontotype is **+0.0043**
(CI **−0.0341 to +0.0426**); these two models are statistically unresolved on this test set.
Adding the human loss scan to the core changes test score by **+0.0054**
(CI **−0.0032 to +0.0151**). The core exceeds GO/PPI GBM by **+0.0148**
(CI **−0.0071 to +0.0399**). SynLeaF's 0.046 dev-to-test drop illustrates why the final ranking
uses the held-out set rather than dev alone.

For Fusion (loss), the test diagnostic is 0.651 on same-family pairs and 0.583 on
different-family pairs. Human ancestry strata score 0.716 (AFR, 32 positives), 0.687 (EAS,
92), and 0.649 (EUR, 442); the headline averages these groups equally.

On 2026-09-24 the Ryan adapter regenerated its De Kegel test dependency from
the current bundle. De Kegel's point score changed by 0.0002. The frozen
Fusion formulas were then rerun with the same weights and scorer, and all
three result files and leaderboard pins were refreshed. No test-label tuning
or change to the Fusion recipe was made.

The scorer 1.3.2 maintenance pass subsequently refreshed every test-result
JSON and public-release lock after fixing nondeterministic bootstrap family
indexing. Point scores and frozen model weights are unchanged; confidence
intervals and result hashes use the canonical family order.

## 2026 literature extension

The dated [2026 method audit](notes/models/2026_landscape.md) added clean
test adapters for PAGAN, GiGCN, the SL-Predict MAE branch, and Ryan's context-specific
paralog refits; Cilantro-SL's pre-existing dev adapter is also run on test.
The two strongest *human* components among these are SL-Predict MAE at 0.724
and Ryan's full clean refit at 0.712. Both are human-only, giving headline
three-species scores of 0.575 and 0.571. PAGAN scores human and budding yeast
at 0.591 and 0.469, for a 0.520 headline. These comparisons are adaptations
under common benchmark rules; source papers' original scores use different
datasets and often easier split or negative definitions.

GiGCN's binary, human-only architecture adaptation scores 0.569 on the human
component and 0.523 across the three headline species. Its released GO
similarity matrix is unavailable, so it uses the benchmark's evidence-filtered
GO features; the 2026 audit records that substitution and the binary-label
change.

Cilantro-SL's Geneformer/FiLM/SLNet adaptation scores 0.599 on the human
component and 0.533 overall, with 11,065/20,210 human test pairs natively
covered. Its five random train-pair calibration folds have AUROC 0.814–0.848,
while its family-held-out SLB dev and test human components are 0.552 and
0.599. The train-fold and SLB metrics use different samples and weighting,
so the contrast is suggestive rather than an exact CV1-to-CV3 estimate.

The SL-Predict MAE branch was a surprising win relative to the older
co-dependency baseline. A fixed ablation under the same SLB classifier gives
human test components of 0.688 for MAE vectors alone, 0.543 for raw
coessentiality alone, and 0.724 for both. The source encoder sees single-gene
profiles, including some SLB test genes, without their pair labels. This is
the sort of input the family-holdout task allows. The ablation compares
feature groups, not the effect of MAE pretraining, and its test evaluations
followed inspection of the full branch's test result. Both are registered
as exploratory, unranked test reads.

One exploratory dev climb added the MAE branch to the frozen Fusion's human
rank blend at weights 0.25, 0.5, 1 and 2. The best tried weight, 2, raised dev
SLB from 0.6663 to 0.6729. Its paired 95% interval was −0.0011 to +0.0203,
so it did not clear the benchmark's noise gate; no new Fusion was promoted or
tested. This experiment followed the independent MAE branch's test readout,
so it is recorded as exploratory rather than a new confirmatory finalist.

## Running the benchmark

```bash
uv sync
uv run slpbench verify --raw
uv run pytest -q
uv run slpbench export-public data/release/slb1.3
SLB_BENCH=data/release/slb1.3 uv run slpbench verify
SLB_BENCH=data/release/slb1.3 uv run slpbench eval results/models/slb1.3/slp_fusion__loss_dev.parquet --split dev
uv run slpbench battery --split dev
SLB_BENCH=data/bench/slb1.3 SLB_SPLIT=test bash scripts/models/slp_fusion/run.sh
uv run slpbench eval results/models/slb1.3/slp_fusion__loss_test.parquet --split test --boot 200
SLB_BENCH=data/bench/slb1.3 uv run python scripts/refresh_test_results.py
uv run slpbench leaderboard
```

The benchmark measures recovery of strong negative interactions in published combinatorial
screens. Human ancestry coverage is uneven, and bacterial screens currently disagree too much
for a headline bacterial score. Fitness overlap weights balance their fitted covariate means,
with the residual shown by the fitness-only control. These limits are documented by source and
species in `BENCHMARK.md`, `DATA_CARD.md` and `REPLICATION.md`.
