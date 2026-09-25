#!/usr/bin/env bash
# EXP2SL retrained on SLB (CPU torch). Human only (L1000 shRNA signatures). Venv: external/models/exp2sl/.venv
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
source scripts/models/_common/threads.sh
export SLB_BENCH=${SLB_BENCH:-data/slb}
V=external/models/exp2sl/.venv
[ -x $V/bin/python ] || (uv venv -q -p 3.12 $V && VIRTUAL_ENV=$V uv pip install -q h5py pandas pyarrow scikit-learn numpy torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple)
$V/bin/python scripts/models/exp2sl/run.py "${SLB_SPLIT:-${1:-dev}}"
