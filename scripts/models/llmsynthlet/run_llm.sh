#!/usr/bin/env bash
# Zero-shot LLM SL scoring (GPU; claim it on the board first). Human only (the prompts are human/cancer-specific).
#   llmsynthlet__qwen25_7b          LLMsynthlet prompt, one prompt per pair (no cell-line context)
#   llmsynthlet__qwen25_7b_context  LLMsynthlet prompt per (cell line, pair) with the line's damaging mutations
#   basis_a1__qwen25_7b             BASIS SL-Bench A1 yes/no question, P(Yes)
#   basis_a1__qwen25_7b_allspecies  same question, organism-aware, every species (yeast standard names)
# MODEL can be overridden (e.g. Qwen/Qwen2.5-32B-Instruct-AWQ, the paper's model is Qwen2.5-32B-Instruct).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
source scripts/models/_common/threads.sh
export SLB_BENCH=${SLB_BENCH:-data/bench/slb1.2}
SPLIT=${SLB_SPLIT:-${1:-dev}}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
TAG=${TAG:-qwen25_7b}
MODES=${MODES:-"yesno yesno_all synthlet"}
export HF_HOME=${HF_HOME:-$ROOT/external/models/_hf}
V=external/models/llmsynthlet/.venv
W=external/models/llmsynthlet/_slb/$(basename "$SLB_BENCH"); mkdir -p $W
for m in $MODES; do
  case $m in
    yesno) $V/bin/python scripts/models/llmsynthlet/llm_run.py --mode yesno --split $SPLIT --model $MODEL --out $W/${SPLIT}_basis_a1_$TAG.csv
           uv run python scripts/models/llmsynthlet/finalize_llm.py $W/${SPLIT}_basis_a1_$TAG.csv basis_a1__$TAG $SPLIT ;;
    yesno_all) $V/bin/python scripts/models/llmsynthlet/llm_run.py --mode yesno --allspecies --split $SPLIT --model $MODEL --out $W/${SPLIT}_basis_a1_all_$TAG.csv
           uv run python scripts/models/llmsynthlet/finalize_llm.py $W/${SPLIT}_basis_a1_all_$TAG.csv basis_a1__${TAG}_allspecies $SPLIT ;;
    synthlet) $V/bin/python scripts/models/llmsynthlet/llm_run.py --mode synthlet --split $SPLIT --model $MODEL --out $W/${SPLIT}_synthlet_$TAG.csv
           uv run python scripts/models/llmsynthlet/finalize_llm.py $W/${SPLIT}_synthlet_$TAG.csv llmsynthlet__$TAG $SPLIT ;;
    synthlet_context) $V/bin/python scripts/models/llmsynthlet/llm_run.py --mode synthlet --context --split $SPLIT --model $MODEL --out $W/${SPLIT}_synthlet_ctx_$TAG.csv
           uv run python scripts/models/llmsynthlet/finalize_llm.py $W/${SPLIT}_synthlet_ctx_$TAG.csv llmsynthlet__${TAG}_context $SPLIT ;;
  esac
done
