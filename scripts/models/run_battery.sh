#!/usr/bin/env bash
# Re-run every SL model adapter on one benchmark version and split, then score them all.
#   SLB_BENCH=data/bench/slb1.3 SLB_SPLIT=dev bash scripts/models/run_battery.sh [model ...]
# Each adapter is scripts/models/<model>/run.sh (reads SLB_BENCH / SLB_SPLIT); GPU adapters queue
# on the GPU themselves. Failures are logged and skipped. Takes many hours for the full battery.
set -uo pipefail
cd "$(dirname "$0")/../.."
export SLB_BENCH=${SLB_BENCH:-data/bench/slb1.3} SLB_SPLIT=${SLB_SPLIT:-dev}
source scripts/models/_common/threads.sh
log=results/models/$(basename "$SLB_BENCH")/battery_run_${SLB_SPLIT}.log
mkdir -p "$(dirname "$log")"
models=${*:-$(uv run python -c "import yaml; r=yaml.safe_load(open('models/battery.yaml'))['models']; print(' '.join([*dict.fromkeys(e.get('adapter', m) for m, e in r.items() if e['family'] not in ('baseline', 'ensemble')), 'dev_rank_ensemble']))")}
failed=0
for m in $models; do
  [ -f scripts/models/$m/run.sh ] || { echo "$m: no adapter" | tee -a "$log"; failed=1; continue; }
  echo "== $m $(date)" | tee -a "$log"
  bash scripts/models/$m/run.sh >> "$log" 2>&1 || { echo "$m: FAILED (see $log)" | tee -a "$log"; failed=1; }
done
for b in random fitness fitness_lgbm paralog_identity codependency lgbm; do
  uv run slpbench baseline $b --split "$SLB_SPLIT" --out results/$(basename "$SLB_BENCH")/${b}_${SLB_SPLIT}.parquet >> "$log" 2>&1 || {
    echo "$b baseline: FAILED (see $log)" | tee -a "$log"; failed=1;
  }
done
uv run slpbench battery --split "$SLB_SPLIT" || failed=1
exit "$failed"
