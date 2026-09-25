#!/usr/bin/env bash
# MVGCN-iSL on SLB. Variants (SLB_VARIANTS, default "human allspecies"): human = faithful (DepMap cell-line omics),
# allspecies = bundle PPI/co-expression views + fitness/ESM-2 node features, per species.
# env: SLB_BENCH, SLB_SPLIT (default dev), SLB_GPU=1 to use CUDA (claim it on the board first)
set -euo pipefail
T=${SLB_THREADS:-8}; export SLB_THREADS=$T OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T POLARS_MAX_THREADS=$T NUMBA_NUM_THREADS=$T
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
PY=external/models/MVGCNiSL/.venv/bin/python
[ -x "$PY" ] || (cd external/models/MVGCNiSL && uv venv -q -p 3.11 .venv && VIRTUAL_ENV=.venv uv pip install -q torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 && VIRTUAL_ENV=.venv uv pip install -q torch_geometric==2.6.1 pandas pyarrow scikit-learn scipy networkx tqdm matplotlib)
for v in ${SLB_VARIANTS:-human allspecies}; do
  if [ "$v" = human ]; then "$PY" scripts/models/mvgcn_isl/run.py; else "$PY" scripts/models/mvgcn_isl/run_allspecies.py; fi
done
