# SLP: synthetic lethality prediction

Restarted on 2026-09-22. Step one is a single evaluation that current and future models are all
measured against.

- [BENCHMARK.md](BENCHMARK.md): **SLB-1.3**, a benchmark built only from measured combinatorial
  screens. Positives and negatives were both tested in the lab. Test gene families (paralogs plus
  orthologs across 12 proteomes) are held out. The SLB score averages human (50 cell lines annotated
  with genetic ancestry), *S. cerevisiae* and *S. pombe*. *B. subtilis*, *C. elegans*, fly and mouse
  are auxiliary species.
  Every label source passed a reproducibility audit, and the score is fitness-balanced: a model
  that only knows which genes are sick on their own scores 0.5. This file covers the task, metric,
  leakage contract and label rules.
- [REPLICATION.md](REPLICATION.md): per-source reproducibility evidence and include/exclude decisions. Generated.
- [ROBUSTNESS.md](ROBUSTNESS.md): baseline ranking across 5 random family splits.
- [DATA_CARD.md](DATA_CARD.md): per-source, per-split, per-ancestry and per-cell-line counts. Generated.
- [LEADERBOARD.md](LEADERBOARD.md): test-split results. Generated from `leaderboard.yaml`.
- `notes/data/`, `notes/models/`: per-source data notes and per-model notes for the SL model battery
  (`scripts/models/<name>/run.sh`). Agent brief: `notes/AGENT_BRIEF.md`.
- `reference/old-slp/`: the prior repo's model card, literature review and SL results
  (abiome-org/slp at the commit in `SOURCE_COMMIT`).

```bash
uv sync
uv run python -m slpbench.fetch      # raw sources → data/raw (about 3 GB)
uv run slpbench build                # → data/bench/slb1.3
uv run slpbench audit                # → REPLICATION.md
uv run slpbench baseline lgbm --split dev --out results/lgbm_dev.parquet
uv run slpbench eval results/lgbm_dev.parquet --split dev
uv run pytest -q
```

`scripts/score_slp11.py` scores SLB pairs with the frozen SLp-1.1 model, using the old repo's
environment in `external/old-slp` (not tracked in git).
