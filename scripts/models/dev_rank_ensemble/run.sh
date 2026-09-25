#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export SLB_BENCH=${SLB_BENCH:-data/bench/slb1.3}
source scripts/models/_common/threads.sh
uv run python scripts/models/dev_rank_ensemble/run.py --variant core
uv run python scripts/models/dev_rank_ensemble/run.py --variant loss
