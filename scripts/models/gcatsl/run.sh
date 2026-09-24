#!/usr/bin/env bash
# GCATSL (Feng et al. 2024 SL_benchmark implementation) on SLB. See notes/models/gcatsl.md and notes/models/feng_suite.md.
# env: SLB_BENCH (default data/bench/slb1.2), SLB_SPLIT (default dev), SLB_THREADS (default 8),
#      SLB_VARIANTS (default: "human allspecies"), SLB_DOCKER_IMAGE / SLB_GPU=1 for the GPU image.
# Outputs: results/models/gcatsl_<split>.parquet (human), results/models/gcatsl__allspecies_<split>.parquet
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
for v in ${SLB_VARIANTS:-human allspecies}; do
  bash "$ROOT/scripts/models/feng_suite/run_model.sh" GCATSL gcatsl "$v"
done
