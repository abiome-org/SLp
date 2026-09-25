#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
export SLB_BENCH=${SLB_BENCH:-data/slb}
export SLB_SPLIT=${SLB_SPLIT:-dev}
source scripts/models/_common/threads.sh
RAW=data/raw/sl_predict_2026
mkdir -p "$RAW"
[ -s "$RAW/mae_encoder_d256_leak_repaired.ckpt" ] || curl -fLsS --retry 3 -o "$RAW/mae_encoder_d256_leak_repaired.ckpt" \
  https://huggingface.co/potteryrage/sl-predict/resolve/main/mae_encoder_d256_leak_repaired.ckpt
[ -s "$RAW/CRISPRGeneEffect_26Q1.csv" ] || curl -fLsS --retry 3 -o "$RAW/CRISPRGeneEffect_26Q1.csv" \
  https://huggingface.co/datasets/ChanghaoKan/crispr-depmap/resolve/main/CRISPRGeneEffect_26Q1.csv
PY=${SLPRED_PY:-external/models/synleaf/.venv/bin/python}
"$PY" scripts/models/sl_predict_2026/extract.py
uv run python scripts/models/sl_predict_2026/run.py
