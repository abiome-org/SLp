# SLP: synthetic lethality prediction

Restarted on 2026-09-22. Step one is a single evaluation that current and future models are all
measured against.

- [BENCHMARK.md](BENCHMARK.md): **SLB-1**, a benchmark built only from measured combinatorial
  screens. Positives and negatives were both tested in the lab. Test gene families (paralogs plus
  orthologs) are held out across 5 species and 60 human cell lines annotated with genetic ancestry.
  This file covers the task, metric, leakage contract and label rules.
- [DATA_CARD.md](DATA_CARD.md): per-source, per-split, per-ancestry and per-cell-line counts. Generated.
- [LEADERBOARD.md](LEADERBOARD.md): test-split results. Generated from `leaderboard.yaml`.
- `reference/old-slp/`: the prior repo's model card, literature review and SL results
  (abiome-org/slp at the commit in `SOURCE_COMMIT`).

```bash
uv sync
uv run python -m slpbench.fetch      # raw sources → data/raw (about 3 GB)
uv run slpbench build                # → data/bench/slb1
uv run slpbench baseline lgbm --split dev --out results/lgbm_dev.parquet
uv run slpbench eval results/lgbm_dev.parquet --split dev
uv run pytest -q
```

`scripts/score_slp11.py` scores SLB pairs with the frozen SLp-1.1 model, using the old repo's
environment in `external/old-slp` (not tracked in git).
