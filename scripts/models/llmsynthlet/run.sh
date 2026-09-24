#!/usr/bin/env bash
# LLMsynthlet released predictions (CPU). For the GPU zero-shot runs see run_llm.sh.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
source scripts/models/_common/threads.sh
export SLB_BENCH=${SLB_BENCH:-data/bench/slb1.2}
uv run python scripts/models/llmsynthlet/released.py "${SLB_SPLIT:-${1:-dev}}"
