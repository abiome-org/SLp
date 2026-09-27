# Cross-species transfer track: protocol

Written 2026-09-25, before any transfer-arm result was scored. Criteria below are not to be changed after
seeing results; any later change is added as a dated amendment beneath the original text.

## Question

Human cancer-cell-line combinatorial screens are scarce. Can a model trained with **no human pair labels**
predict human synthetic lethality as well as the same model trained with them? If so, human screens are
needed only for evaluation.

## Arms (`scripts/transfer/arms.py`)

Every arm keeps the SLB family hold-out: a human test gene has no paralog or ortholog in any arm's train
rows, in any species. Only the human train rows change.

| Arm | Human pair labels | Non-human pair labels |
|---|---|---|
| `full` (H+) | all SLB train (3,486 positives) | all |
| `h0` (H0) | none | all |
| `human_only` | all | none |
| `dose_pXX_sN` | pairs among a random gene subset keeping about XX% of positives (1, 3, 10, 30%; seeds 0, 1) | all |

Human single-gene data (DepMap effects, GO, PPI, STRING, expression) stays allowed in every arm, including
H0: it is abundant and not combinatorial. An H0 with no human data at all is out of scope here.

## Models (`scripts/transfer/run_model.py`)

The pooled multi-species SLB adapters, unchanged except that they train once per arm:
`ontotype` (pooled Ontotype), `gbm_nocode` (pooled GO/PPI GBM without the species code, which H0 never
sees for human) and `gbm_xs` (the same with each gene-level feature replaced by its within-species
percentile; label-free). `gbm` (with species code) is run on `full` and `h0` only, to tie the harness to
the leaderboard. No arm selects on human dev: the only model selection is each adapter's internal
validation split of the arm's own train rows.

## Evaluation (`scripts/transfer/score.py`)

- **Human score**: the SLB human species score (fitness-balanced AUROC, mean over ancestry groups with at
  least 20 positives). H0 never uses human dev, so human dev and test are both evaluation data here.
  **Pooled human** = positive-weighted mean of the dev and test human scores; its CI comes from
  independent family-cluster bootstraps of the two splits (dev and test families are disjoint).
- **Paired differences** use the same bootstrap draws for both predictions (1,000 reps).
- **Label-free human baselines H0 must beat**: `paralog_identity`, DepMap OLS (`depmap_ols__loss`),
  `codependency`, `fitness`.
- **Ortholog-open ceiling** (reported, not a pass criterion): `ortholog_gi_transfer`, the best measured
  negative GI score among cross-species ortholog pairs. It reads measurements of held-out families, so it
  shows how much of human SL a direct conservation lookup can recover, not a clean model.

## Pass criterion (primary, per model)

H0 transfers if, on pooled human, **the lower 95% bound of H0 − H+ is above −0.03** (non-inferiority,
δ = 0.03: the dev noise SD of the human score is 0.043) **and** H0's pooled human score is above every
label-free baseline's, with the lower 95% bound of H0 − DepMap OLS above 0.

Secondary readouts: the dose curve (human score vs human positives in train; the exchange rate is the
dose at which the curve reaches H+), `human_only` vs `full` (does non-human data help human at all), and
the non-human species scores of H0 vs H+ (does removing human labels cost the other species anything).

A pass by today's models would be surprising; the track exists so that SLp-2+ can be scored on it. A fail
is informative only if H+ itself beats the label-free baselines; otherwise there is nothing to transfer.

## Amendment, 2026-09-25 (after the first results; the criterion above is unchanged)

1. **The primary criterion cannot be met with the current human data.** On pooled human (dev + test, about
   900 positives), the paired H0 − H+ 95% interval is about ±0.04 wide, so its lower bound stays below
   −0.03 even when the true difference is 0. So the primary verdict can come out non-inferior only if the
   human evaluation set grows (new screens), or if the margin is widened, which would be a new protocol.
2. **Added readout (secondary): cross-fitted human train pairs** (`crossfit.py`). There are 3,301
   ancestry-grouped positives, and H+ is cross-fitted over 5 family folds. It is ortholog-open for both
   arms equally. Its balance weights are refitted on that split with row counts over scored rows only.
   The balance is weaker there: `fitness` scores 0.543, not 0.50.
3. **Added: refit noise.** `gbm_nocode` was refitted on `full` and `h0` with GBM seeds 1–4.
4. **Added: feature-group ablations** (`ablate.py`) on `full` and `h0`.
5. **Added: all-human cross-fit** (`crossfit.py --scope all`). Every scored human pair (train, dev and
   test) is used for evaluation. H0 never sees human labels, and H+ is cross-fitted over 5 human family
   folds. That gives 4,199 ancestry-grouped positives from 923 families (1,262 non-paralog unique
   pairs), against 232 families on dev+test. It is ortholog-open: a strict hold-out would drop 69–78% of
   yeast training rows. It also reports the pairs with no measured non-human ortholog pair. H+ fold
   models here train on human test labels, so they must never feed the leaderboard.
