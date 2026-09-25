#!/usr/bin/env bash
# Context-specific paralog SL RF (Ryan lab 2026) on SLB. Human only (DepMap cell-line omics).
# Needs the De Kegel 2021 pretrained scores for train + split (scripts/models/dekegel2021/run.sh produces them).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
source scripts/models/_common/threads.sh
export SLB_BENCH=${SLB_BENCH:-data/slb}
SPLIT=${SLB_SPLIT:-${1:-dev}}
W=external/models/dekegel_paralog_sl/_slb/$(basename "$SLB_BENCH")
[ -s $W/${SPLIT}_pretrained.csv ] || SLB_SPLIT=$SPLIT scripts/models/dekegel2021/run.sh
if [ ! -s $W/train_pretrained.csv ]; then
  docker run --rm --user "$(id -u):$(id -g)" -v "$ROOT/external/models/dekegel_paralog_sl":/model -v "$ROOT/$W":/w \
    -v "$ROOT/scripts/models/dekegel2021":/code slb/dekegel2021 python /code/score_pretrained.py /w/train_feat.csv /w/train_pretrained.csv
fi
uv run python scripts/models/ryan2026_context/run.py "$SPLIT"
