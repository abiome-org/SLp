#!/usr/bin/env bash
# GO + PPI + STRING GBM (Wong 2004 / Pandey 2010 style), fit on SLB train. Env: SLB_BENCH, SLB_SPLIT (default dev).
# Needs the shared bundle (scripts/models/_common/bundle.py). Variants: go_ppi_gbm (per species), go_ppi_gbm__pooled.
set -euo pipefail
cd "$(dirname "$0")/../../.."
source scripts/models/_common/threads.sh
uv run python scripts/models/go_ppi_gbm/run.py
uv run python scripts/models/go_ppi_gbm/run.py --pooled
if [ "$SLB_SPLIT" = dev ]; then
  for v in go_ppi_gbm go_ppi_gbm__pooled; do
    uv run slpbench eval "$SLB_OUT/${v}_dev.parquet" --split dev --allow-missing > "$SLB_OUT/${v}_dev.txt"
  done
fi
