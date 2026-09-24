#!/usr/bin/env bash
# Struct2SL on SLB (human only). env: SLB_BENCH, SLB_SPLIT (default dev), SLB_VARIANT=slbtrain|released (default: both)
set -euo pipefail
T=${SLB_THREADS:-8}; export SLB_THREADS=$T OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T POLARS_MAX_THREADS=$T NUMBA_NUM_THREADS=$T
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
PY=external/models/MVGCNiSL/.venv/bin/python
[ -x "$PY" ] || (cd external/models/MVGCNiSL && uv venv -q -p 3.11 .venv && VIRTUAL_ENV=.venv uv pip install -q torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 && VIRTUAL_ENV=.venv uv pip install -q torch_geometric==2.6.1 pandas pyarrow scikit-learn scipy networkx tqdm matplotlib)
if [ ! -f data/raw/struct2sl/bestmodel.pt ]; then
  mkdir -p data/raw/struct2sl
  for x in "68221633 Human_SL.csv" "68221639 Human_SL_ff.csv" "68221627 Human_nonSL.csv" "68221648 ppi_features.npz" "68221654 sequence_features.npz" "68221651 struct_features.zip" "68221642 bestmodel.pt"; do
    set -- $x; curl -sSL -o data/raw/struct2sl/$2 https://ndownloader.figshare.com/files/$1; done
fi
for v in ${SLB_VARIANT:-slbtrain released}; do SLB_VARIANT=$v "$PY" scripts/models/struct2sl/run.py; done
