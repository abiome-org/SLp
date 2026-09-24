#!/usr/bin/env bash
# Run every models-features CPU adapter for one benchmark/split (SLB_BENCH, SLB_SPLIT). GPU ones (LLM, fm/mm
# sub-agent models) are separate.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
source scripts/models/_common/threads.sh
for m in dekegel2021 dennler2025 statsl discoversl sinatra slidr deltadep ryan2026_context llmsynthlet exp2sl genept; do
  echo "=== $m $(date +%T)"
  scripts/models/$m/run.sh 2>&1 | grep -E "SLB score|H. sapiens|Error|error|Traceback" | head -40
done
