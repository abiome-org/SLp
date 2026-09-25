#!/usr/bin/env bash
# Delta Dependency (paralogSL) and DepMap OLS paralog scan on SLB; human only (DepMap).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
source scripts/models/_common/threads.sh
export SLB_BENCH=${SLB_BENCH:-data/slb}
uv run python scripts/models/deltadep/run.py "${SLB_SPLIT:-${1:-dev}}"
