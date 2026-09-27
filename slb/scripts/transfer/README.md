# Cross-species transfer track

Can a model trained with **no human pair labels** predict human synthetic lethality as well as one trained
with them? Protocol, arms and the pre-registered pass criterion: [`PROTOCOL.md`](PROTOCOL.md).

```bash
bash scripts/transfer/run.sh                      # arms -> fits (6 in parallel) -> results/transfer/report.{md,json}
uv run python scripts/transfer/ablate.py --arm h0 --group net          # feature-group ablation
uv run python scripts/transfer/run_model.py --arm h0 --model gbm_nocode --seed 1   # refit noise
uv run python scripts/transfer/crossfit.py build && ...fit --h0 / --pair i j ... && ...score
```

Arms are `data/transfer/<arm>/`, copies of `data/slb/` where only `train.parquet` differs. Every other file
is a symlink, so the family hold-out and the scorer are unchanged. Every test readout is appended to
`results/test_evals.jsonl`.

## Results, 2026-09-25 (today's pooled SLB models; SLp-1.1 not included)

Human score = the SLB fitness-balanced AUROC, averaged over ancestry groups. "Pooled" = 0.38 × dev + 0.62 × test
(341 + 557 positives). 95% CIs come from a family-cluster bootstrap.

| model | H+ (all labels) | H0 (no human pairs) | H0 − H+ (95% CI) | verdict |
|---|---|---|---|---|
| GO/PPI GBM, no species code | 0.649 | 0.634 | −0.016 (−0.056 to +0.021) | fail: CI too wide |
| same, 5 GBM seeds (mean) | 0.646 | 0.626 | −0.020 (every seed −0.007 to −0.037) | |
| GO/PPI GBM, within-species percentiles | 0.634 | 0.616 | −0.018 (−0.054 to +0.015) | fail: CI too wide |
| GO/PPI GBM with species code (leaderboard recipe) | 0.644 | 0.615 | −0.029 (−0.082 to +0.019) | fail |
| Ontotype, pooled | 0.608 | 0.531 | −0.077 (−0.133 to −0.022) | fail: does not transfer |

Label-free human baselines (pooled): DepMap OLS 0.621, paralog identity 0.600, codependency 0.580,
fitness 0.497. The ortholog-open lookup of measured GI scores between orthologs (`ortholog_gi_transfer`)
scores 0.539.

Cross-fitted human train pairs (`crossfit.py`, 3,301 positives, gbm_nocode):

| | H+ cross-fit | H0 | H0 − H+ |
|---|---|---|---|
| human score | 0.618 (0.565–0.666) | 0.592 (0.548–0.641) | −0.026 (−0.051 to +0.005) |
| H0 − paralog identity | | | +0.048 (+0.002 to +0.097) |
| H0 − codependency | | | +0.043 (+0.002 to +0.090) |

All human pairs as evaluation (`crossfit.py --scope all`: train + dev + test, 4,199 positives, 923 families,
H+ cross-fitted; ortholog-open):

| | human score | pairs without a measured non-human ortholog pair |
|---|---|---|
| H+ cross-fit | 0.616 (0.571–0.657) | 0.604 |
| H0 | 0.603 (0.559–0.646) | 0.583 |
| H0 − H+ | −0.013 (−0.038 to +0.013) | |
| H0 − paralog identity / codependency | +0.048 (+0.008 to +0.087) / +0.047 (+0.011 to +0.083) | |
| ortholog lookup | 0.517 | |

What this shows:

- **The all-human readout comes closest:** H0 is 0.013 behind H+ and its interval (±0.026) only just
  misses the 0.03 margin at the lower end. Lookup explains little (0.517). On pairs with no measured ortholog pair
  the gap is slightly larger (−0.021), so some of H0's human signal comes through yeast orthologs.
- **No model passes, and with only dev+test human data none could.** The pooled paired interval is about ±0.04
  wide, wider than the 0.03 margin. The larger cross-fit readout narrows it to about ±0.028 and puts H0 at
  −0.026, with the interval just missing −0.03 at its lower end.
- **A species-agnostic feature model keeps most of the human signal without human labels.** Above chance,
  H0 keeps about 85–90% of H+'s human signal on the primary readout and 78% on the cross-fit. On the
  cross-fit, H0 is significantly better than paralog identity and codependency, but on the primary
  readout it is not significantly better than DepMap OLS.
- **The signal that transfers is relational, not about which genes are involved.** Holding the network
  features (PPI/STRING neighbourhood, co-expression) constant costs H0 −0.043 (−0.071 to −0.014).
  Holding `same_family` constant costs nothing, so this is not a "paralogs are SL" rule learned from
  yeast. Ontotype describes a pair by which GO terms it disrupts, and it loses 0.077 without human rows.
  A world model that learns gene identities rather than relations risks behaving like Ontotype here.
- **Human labels are worth little to these models in either direction.** Non-human data adds nothing
  measurable to human (`full` − `human_only` +0.012, −0.025 to +0.050). The dose curve (1–30% of human
  positives) sits between H0 and H+ without a trend, so no exchange rate can be read from it.
- **Adding species-level percentiles (`gbm_xs`) did not help transfer.**
