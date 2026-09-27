#!/usr/bin/env bash
# Cross-species transfer track: build arms, fit every (arm, model), score. See PROTOCOL.md.
# Env: SLB_JOBS parallel fits (default 6), SLB_THREADS threads per fit (default 8).
set -euo pipefail
cd "$(dirname "$0")/../.."
export SLB_THREADS=${SLB_THREADS:-8} OMP_NUM_THREADS=${SLB_THREADS:-8}
mkdir -p results/transfer/logs
[ -f data/transfer/h0/train.parquet ] || uv run python scripts/transfer/arms.py
for s in dev test; do
  [ -f results/transfer/_baselines/ortholog_gi_transfer_$s.parquet ] || \
    uv run python scripts/models/ortholog_gi_transfer/run.py --split $s \
      --out results/transfer/_baselines/ortholog_gi_transfer_$s.parquet > results/transfer/logs/ortholog_$s.log 2>&1
done
jobs() {
  for arm in full h0 human_only dose_p01_s0 dose_p03_s0 dose_p10_s0 dose_p30_s0 dose_p01_s1 dose_p03_s1 dose_p10_s1 dose_p30_s1; do
    for m in ontotype gbm_nocode gbm_xs; do echo "$arm $m"; done
  done
  echo "full gbm"; echo "h0 gbm"
}
jobs | while read -r arm m; do
  [ -f "results/transfer/$arm/${m}_test.parquet" ] || pgrep -f -- "--arm $arm --model $m\$" > /dev/null || echo "$arm $m"
done | xargs -P "${SLB_JOBS:-6}" -L 1 bash -c \
  'uv run python scripts/transfer/run_model.py --arm "$0" --model "$1" > "results/transfer/logs/${0}_${1}.log" 2>&1 || echo "FAILED $0 $1"'
uv run python scripts/transfer/score.py
