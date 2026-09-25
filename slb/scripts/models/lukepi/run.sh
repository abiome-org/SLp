#!/usr/bin/env bash
# LukePi on SLB (human only: PrimeKG is a human KG). env: SLB_BENCH, SLB_SPLIT (default dev), SLB_GPU=1 optional
set -euo pipefail
T=${SLB_THREADS:-8}; export SLB_THREADS=$T OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T POLARS_MAX_THREADS=$T
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
PY=external/models/LukePi/.venv/bin/python
# the released checkpoint uses the pre-2.3 PyG HGTConv layout (per-type k_lin/q_lin/v_lin), hence torch 1.13 + PyG 2.2
[ -x "$PY" ] || (cd external/models/LukePi && uv venv -q -p 3.10 .venv && VIRTUAL_ENV=.venv uv pip install -q torch==1.13.1 --index-url https://download.pytorch.org/whl/cu117 && VIRTUAL_ENV=.venv uv pip install -q "numpy<2" torch_geometric==2.2.0 pandas pyarrow scikit-learn scipy tqdm && VIRTUAL_ENV=.venv uv pip install -q "torch_scatter==2.1.1+pt113cu117" "torch_sparse==0.6.17+pt113cu117" -f https://data.pyg.org/whl/torch-1.13.1+cu117.html)
[ -f data/raw/lukepi/Configuration_LukePi/kgdata.pkl ] || (mkdir -p data/raw/lukepi && cd data/raw/lukepi && ../../../external/models/_gdown_venv/bin/gdown -q --folder https://drive.google.com/drive/folders/1-IMvqc6O_6f4C9vjtFZC918dxycaOqGg)
[ -f data/raw/mit4sl/data/data/MultiOmics_feature/kg_data/Primenode.csv ] || { echo "need PrimeKG node table: see scripts/models/mit4sl (MiT4SL data.zip)"; exit 1; }
"$PY" scripts/models/lukepi/run.py
