#!/usr/bin/env bash
# LEAKY diagnostic: GI transfer through cross-species orthologs. Env: SLB_BENCH, SLB_SPLIT (default dev).
set -euo pipefail
cd "$(dirname "$0")/../../.."
source scripts/models/_common/threads.sh
uv run python scripts/models/ortholog_gi_transfer/run.py --split "$SLB_SPLIT"
if [ "$SLB_SPLIT" = dev ]; then
  { uv run slpbench eval "$SLB_OUT/ortholog_gi_transfer_dev.parquet" --split dev --allow-missing
    echo; echo "== covered subset (in_model) =="
    uv run python scripts/models/mech_common/subset_eval.py "$SLB_OUT/ortholog_gi_transfer_dev.parquet"; } \
    > "$SLB_OUT/ortholog_gi_transfer_dev.txt"
fi
