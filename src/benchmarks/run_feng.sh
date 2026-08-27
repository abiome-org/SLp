#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$ROOT/data/models/SL_benchmark/src"
LOG="$ROOT/results/feng_runs"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate slbench
export WANDB_MODE=offline
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1
mkdir -p "$LOG"
cd "$SRC"
MODELS="${MODELS:-GRSMF SL2MF CMFW DDGCN KG4SL SLMGAE NSF4SL GCATSL SLGNN PTGNN MGE4SL PiLSL}"
SPLITS="${SPLITS:-CV1 CV2 CV3}"
for m in $MODELS; do
  for ds in $SPLITS; do
    out="$LOG/${m}_${ds}_Random_1.log"
    if [[ -f "$LOG/${m}_${ds}_Random_1.ok" ]]; then
      echo "skip $m $ds"
      continue
    fi
    echo "=== $m $ds $(date -Is) ==="
    if python main.py -m "$m" -ns Random -ds "$ds" -pn 1 >"$out" 2>&1; then
      touch "$LOG/${m}_${ds}_Random_1.ok"
    else
      echo "FAIL $m $ds" | tee -a "$LOG/failures.txt"
    fi
  done
done
echo "done $(date -Is)"
