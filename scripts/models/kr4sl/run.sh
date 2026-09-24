#!/usr/bin/env bash
# KR4SL on SLB (human only; needs CUDA: claim the GPU on the board first). env: SLB_BENCH, SLB_SPLIT (default dev)
set -euo pipefail
T=${SLB_THREADS:-8}; export SLB_THREADS=$T OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T POLARS_MAX_THREADS=$T
export SLB_GPU=${SLB_GPU:-1}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
PY=external/models/KR4SL/.venv/bin/python
[ -x "$PY" ] || (cd external/models/KR4SL && uv venv -q -p 3.11 .venv && VIRTUAL_ENV=.venv uv pip install -q torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 && VIRTUAL_ENV=.venv uv pip install -q "transformers<4.46" scipy pandas pyarrow "numpy<2" scikit-learn && VIRTUAL_ENV=.venv uv pip install -q torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html)
BN=$(basename "${SLB_BENCH:-data/bench/slb1.2}"); W=${SLB_WORK:-$ROOT/external/models/_slb_work/$BN}
[ -f "$W/human_train_pairs.parquet" ] || uv run python scripts/models/_common/slb_pairs.py
[ -f "$W/kr4sl/all_entities.txt" ] || "$PY" scripts/models/kr4sl/build.py
[ -f "$W/kr4sl/all_entities_pretrain_emb.npy" ] || "$PY" scripts/models/kr4sl/embed.py
"$PY" scripts/models/kr4sl/run.py
