#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
source scripts/models/_common/threads.sh
export SLB_BENCH=${SLB_BENCH:-data/slb}
V=external/models/ryan_paralog_seq_similarity/.venv
[ -x $V/bin/python ] || (uv venv -q -p 3.12 $V && VIRTUAL_ENV=$V uv pip install -q xgboost scikit-learn pandas pyarrow)
$V/bin/python scripts/models/dennler2025/run.py "${SLB_SPLIT:-${1:-dev}}"
