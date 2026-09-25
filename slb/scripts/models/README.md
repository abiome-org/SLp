# Model adapters

One directory per published SL model (or model family) that SLB can run. Each adapter trains only on
`data/slb/train.parquet` labels plus permitted single-gene inputs (see the leakage rules in the SLB
README), never reads `hidden/`, and writes
`results/models/<bench>/<model>[__<variant>]_<split>.parquet` with columns `example_id, score`.

```bash
SLB_BENCH=data/slb SLB_SPLIT=dev bash scripts/models/<model>/run.sh   # one model
bash scripts/models/run_battery.sh [model ...]                         # all of battery.yaml, then score
uv run slbench battery --split dev                                     # -> results/reports/models_dev.md
```

- `battery.yaml` (in `slb/`) lists every scored model, its family, and whether it is leaky and why.
  When an entry's `adapter:` is set, that directory runs it (for example `statsl` runs DAISY and ISLE).
- `_common/` holds shared helpers: `slb.py` (load splits, write predictions, evaluate), `threads.sh`
  (thread caps, `SLB_OUT`), `eval.sh` (dev scoring with median fill for models that cover only some
  species), `bundle.py` / `bundle_io.py` (per-species GO, PPI, fitness and ESM-2 feature bundle in
  `data/interim/bundle/`), `omics.py` and `depmap_feats.py` (single-gene human features).
- `feng_suite/` builds the shared inputs and Docker images for the Feng et al. 2024 graph methods
  (cmfw, ddgcn, gcatsl, grsmf, kg4sl, mge4sl, nsf4sl, pilsl, ptgnn, sl2mf, slgnn, slmgae).
- `dev_rank_ensemble/` is the frozen dev-selected rank blend; `search_dev.py` reproduces its search
  ledger (`reference/dev_rank_ensemble_dev_search.json`).
- `slp11/` scores the prior SLp-1.1 world model on SLB, using the environment of its clone in
  `external/models/slp11`: `score.py` scores gene pairs, and `to_preds.py` turns those scores into
  prediction files.

Upstream code lives in `external/models/<repo>/` (gitignored), with the adapter's work caches under
`external/models/<repo>/_slb/<bench>/` and `external/models/_slb_work/<bench>/`.
