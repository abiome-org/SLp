#!/usr/bin/env bash
# SINaTRA released human predictions (leaky; human only).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
source scripts/models/_common/threads.sh
export SLB_BENCH=${SLB_BENCH:-data/slb}
uv run python scripts/models/sinatra/run.py "${SLB_SPLIT:-${1:-dev}}"
