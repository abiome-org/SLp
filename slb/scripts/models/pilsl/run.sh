#!/usr/bin/env bash
# PiLSL (Feng et al. 2024 SL_benchmark implementation) on SLB.
# env: SLB_BENCH (default data/slb), SLB_SPLIT (default dev), SLB_THREADS (default 8),
#      SLB_VARIANTS (default: "human"), SLB_DOCKER_IMAGE / SLB_GPU=1 for the GPU image.
# Outputs: results/models/pilsl_<split>.parquet (human)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
for v in ${SLB_VARIANTS:-human}; do
  # CPU-only (the dgl 0.4 code cannot use the RTX 3090); bounded to SLB_PILSL_TIMEOUT seconds (default 6 h)
  timeout "${SLB_PILSL_TIMEOUT:-21600}" bash "$ROOT/scripts/models/feng_suite/run_model.sh" PiLSL pilsl "$v"
done
