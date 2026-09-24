#!/usr/bin/env bash
# FBA double-deletion SL score (label-free). Env: SLB_BENCH, SLB_SPLIT (default dev), FBA_PROCS (default 8).
# Species with a GEM loader in scripts/models/fba/gems.py are simulated; others get score 0 (in_model=False).
# Run setup.sh once. Writes $SLB_OUT/fba_<split>.parquet and fba__mult_<split>.parquet.
set -euo pipefail
cd "$(dirname "$0")/../../.."
source scripts/models/_common/threads.sh
PY=external/models/fba/.venv/bin/python
[ -x $PY ] || scripts/models/fba/setup.sh
$PY scripts/models/fba/run.py --split "$SLB_SPLIT" --procs "${FBA_PROCS:-8}" --score fba_min
$PY scripts/models/fba/run.py --split "$SLB_SPLIT" --procs "${FBA_PROCS:-8}" --score fba_mult \
    --out "$SLB_OUT/fba__mult_${SLB_SPLIT}.parquet"
if [ "$SLB_SPLIT" = dev ]; then
  for v in fba fba__mult; do
    { uv run slpbench eval "$SLB_OUT/${v}_dev.parquet" --split dev --allow-missing
      echo; echo "== covered subset (in_model) =="
      uv run python scripts/models/mech_common/subset_eval.py "$SLB_OUT/${v}_dev.parquet"; } > "$SLB_OUT/${v}_dev.txt"
  done
fi
