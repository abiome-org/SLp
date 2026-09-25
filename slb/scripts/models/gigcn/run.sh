#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
export SLB_BENCH=${SLB_BENCH:-data/slb}
export SLB_SPLIT=${SLB_SPLIT:-dev}
source scripts/models/_common/threads.sh
PY=${GIGCN_PY:-external/models/synleaf/.venv/bin/python}
"$PY" scripts/models/gigcn/run.py
