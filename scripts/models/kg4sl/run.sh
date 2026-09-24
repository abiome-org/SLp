#!/usr/bin/env bash
# KG4SL (Feng et al. 2024 SL_benchmark implementation) on SLB. See notes/models/kg4sl.md and notes/models/feng_suite.md.
# env: SLB_BENCH (default data/bench/slb1.2), SLB_SPLIT (default dev), SLB_THREADS (default 8),
#      SLB_VARIANTS (default: "human"), SLB_DOCKER_IMAGE / SLB_GPU=1 for the GPU image.
# Outputs: results/models/kg4sl_<split>.parquet (human)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
for v in ${SLB_VARIANTS:-human}; do
  SLB_BALANCE=${SLB_BALANCE:-1} bash "$ROOT/scripts/models/feng_suite/run_model.sh" KG4SL kg4sl "$v"
done
