#!/usr/bin/env bash
# DAISY / ISLE re-implementations + single-statistic diagnostics on SLB. Human only.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
source scripts/models/_common/threads.sh
export SLB_BENCH=${SLB_BENCH:-data/slb}
uv run python scripts/models/statsl/score.py "${SLB_SPLIT:-${1:-dev}}"
