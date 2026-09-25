#!/usr/bin/env bash
# NSF4SL (Feng et al. 2024 SL_benchmark implementation) on SLB.
# env: SLB_BENCH (default data/slb), SLB_SPLIT (default dev), SLB_THREADS (default 8),
#      SLB_VARIANTS (default: "human"), SLB_DOCKER_IMAGE / SLB_GPU=1 for the GPU image.
# Outputs: results/models/nsf4sl_<split>.parquet (human)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
for v in ${SLB_VARIANTS:-human}; do
  bash "$ROOT/scripts/models/feng_suite/run_model.sh" NSF4SL nsf4sl "$v"
done
