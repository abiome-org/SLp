#!/usr/bin/env bash
# Ontotype (Yu et al. 2016) GBM on GO ontotypes, fit on SLB train. Env: SLB_BENCH, SLB_SPLIT (default dev).
# Variants: ontotype (per species), ontotype__pooled (allspecies, shared GO-term space).
set -euo pipefail
cd "$(dirname "$0")/../../.."
source scripts/models/_common/threads.sh
uv run python scripts/models/ontotype/run.py
uv run python scripts/models/ontotype/run.py --pooled
if [ "$SLB_SPLIT" = dev ]; then
  for v in ontotype ontotype__pooled; do
    uv run slpbench eval "$SLB_OUT/${v}_dev.parquet" --split dev --allow-missing > "$SLB_OUT/${v}_dev.txt"
  done
fi
