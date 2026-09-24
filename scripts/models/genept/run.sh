#!/usr/bin/env bash
# GenePT embedding pair classifier retrained on SLB (human only).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
source scripts/models/_common/threads.sh
export SLB_BENCH=${SLB_BENCH:-data/bench/slb1.2}
uv run python scripts/models/genept/run.py "${SLB_SPLIT:-${1:-dev}}"
