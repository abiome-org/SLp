# SLP

SLp world models for genetic intervention research, and SLB, the benchmark they are measured on.

- [`slb/`](slb/README.md): the synthetic-lethality benchmark. It contains the `slbench` package, data
  build, scorer, leaderboard, model adapters and figures, and has its own `pyproject.toml`. To run it,
  follow the Quickstart in [`slb/README.md`](slb/README.md): clone, `uv sync`, download the public data
  bundle from the [`slb` release](https://github.com/abiome-org/slp/releases/tag/slb).
- SLp model work goes at the top level, next to `slb/`. The SLp-1.x code is in this repository's history
  (before the merge that added `slb/`) and on its `slp-1.2-*` branches.

Code: MIT (`LICENSE`). Data: see "Sources and citation" in `slb/README.md`.
