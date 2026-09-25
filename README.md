# SLP

SLp world models for genetic intervention research, and SLB, the benchmark they are measured on.

- [`slb/`](slb/README.md): the synthetic-lethality benchmark. It contains the `slbench` package, data
  build, scorer, leaderboard, model adapters and figures, and has its own `pyproject.toml` (run
  `uv sync` inside `slb/`).
- SLp model work goes at the top level, next to `slb/`. The previous SLp-1.x code is
  [abiome-org/slp](https://github.com/abiome-org/slp). A local clone, used to score SLp-1.1 on SLB, is at
  `slb/external/models/slp11`.
