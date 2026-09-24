#!/usr/bin/env bash
# SLIdR (Srivatsa 2022) re-implementation on SLB; human only (DepMap/DEMETER2 cell-line screens + genotypes).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
source scripts/models/_common/threads.sh
export SLB_BENCH=${SLB_BENCH:-data/bench/slb1.2}
uv run python scripts/models/slidr/run.py "${SLB_SPLIT:-${1:-dev}}"
