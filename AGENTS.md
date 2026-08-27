# Agent instructions

## Objective

Advance state of the art in cold-start synthetic-lethality prediction, especially the two-new-gene setting, while keeping every claim reproducible and scientifically narrow. The central question is whether intervention-conditioned cellular-state modeling transfers to unseen genes; benchmark-supervised prediction is a separate secondary track.

## Scientific rules

- Prevent leakage before optimizing performance. For intervention-isolated experiments, remove every quantitative trajectory involving any validation or test gene, not only exact test pairs.
- Never use benchmark test labels for architecture choice, feature choice, target choice, sign selection, calibration, early stopping, or hyperparameter tuning.
- Select the world model and label-free readouts on fixed molecular validation objectives. Open a locked external benchmark only after the candidate and decision rule are frozen.
- Keep label-free transition scores, quantitative-decoder scores, and fold-local supervised readouts separate in code and reporting.
- Compare against simple degree, feature, and published baselines on the identical split. Report all folds and seeds, not only their mean.
- Treat retrospective evidence as retrospective. A state-of-the-art claim requires a locked protocol and untouched or prospective confirmation.
- Pin or checksum raw inputs, split files, feature packs, and checkpoints. Record exclusions and accessible feature modalities for every result.

## Training workflow

1. Read `docs/model-card.md` and the latest section of `docs/results.md` before proposing an experiment.
2. State one falsifiable hypothesis and a fixed advancement criterion. Prefer the smallest experiment that can reject it.
3. Prepare data with scripts in `src/training/`; write generated arrays and audit JSON to `results/`, never into source or documentation directories.
4. Train through `src/training/run_modal.py` or a focused local entry point. The retained compact configuration is `d=384`, `latent=128`, `layers=6`, with 12 pretraining, 10 perturbation, and 3 reinforcement epochs where applicable.
5. Evaluate molecular validation first. Advance to MuSL, SLAMR, Sanger, HAP1, or Feng only when the prespecified criterion passes.
6. Append the hypothesis, protocol, hashes, compute, fold-level metrics, interpretation, and decision to `docs/results.md`. Update the model card only when the supported model or claim changes.

For the retained intervention-isolated training path:

```bash
python src/training/sl_predict.py prepare_spectral_safe
python src/training/sl_predict.py benchmarks
modal run src/training/run_modal.py --spectral-safe-intervention \
  --pretrain-epochs 12 --perturb-epochs 10 --rl-epochs 3
```

Run `modal run src/training/run_modal.py --help` before changing flags. Do not silently substitute data versions, split seeds, or benchmark semantics.

## Engineering rules

- Keep the repository extremely lean. Prefer editing an existing function over adding a framework, abstraction layer, configuration system, or duplicate runner.
- Put training and data preparation in `src/training/`; put only evaluation logic in `src/benchmarks/`.
- Keep `docs/` limited to `litreview.md`, `model-card.md`, and `results.md`.
- Never commit `data/`, `results/` artifacts, model weights, caches, vendored repositories, credentials, or `ontology/`. The ontology is local research infrastructure, not part of the public codebase.
- Preserve deterministic seeds and leakage assertions. Add a focused check when changing split, exclusion, or benchmark logic.
- Write like a scientist: name the dataset, population, endpoint, uncertainty, and limitation. Avoid internal process jargon.
- Do not refactor unrelated code during an experiment.

## Compute

Local compute is one RTX 4070. Run a local job without asking only when it should finish within one hour. For longer work, use available remote compute within the assigned allowance or ask the user before starting. Stop failed or scientifically invalid runs promptly; do not spend compute to complete an experiment whose advancement criterion can no longer pass.

## Completion standard

A change is complete when the relevant preparation or evaluation command runs, leakage constraints are checked, outputs are reproducible from recorded inputs, and `docs/results.md` states what changed and what the evidence does and does not support. Code-only changes should include the narrowest practical syntax or smoke check.
