#!/usr/bin/env bash
# MuSL (Fang et al., IEEE JBHI 2026) retrained on SLB train labels.
#   musl__human : full MuSL (CNN on TCGA joint-expression histograms + 35 stat features + GraphSAGE over
#                 BioGRID physical PPI initialised with the authors' ESM2 embeddings; cross-attention,
#                 adaptive fusion, contrastive loss). Human only (TCGA expression + human ESM2 file).
# Env: SLB_BENCH (default data/bench/slb1.2), SLB_SPLIT (default dev). Uses the GPU if visible (claim it first).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
export SLB_BENCH=${SLB_BENCH:-data/bench/slb1.2}
SPLIT=${SLB_SPLIT:-${1:-dev}}
M=external/models/musl; D=scripts/models/musl; PY=$ROOT/$M/.venv/bin/python
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8} NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-8} POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-8}

# 1. environment (environment.yml pins: python 3.10, torch 2.1.2+cu121, torch-geometric 2.6.1, numpy 1.26.4,
#    pandas 2.0.3, scikit-learn 1.5.2, scanpy 1.11.1)
if [ ! -x "$PY" ]; then
  (cd $M && uv venv -p 3.10 .venv -q && VIRTUAL_ENV=.venv uv pip install -q torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121 \
    && VIRTUAL_ENV=.venv uv pip install -q torch-geometric==2.6.1 numpy==1.26.4 pandas==2.0.3 scikit-learn==1.5.2 scanpy==1.11.1 tqdm pyarrow wandb)
fi
# 2. authors' Zenodo data (record 17098066, CC-BY-4.0): ESM2 embeddings + TCGA expression
for f in protein_embeddings.pt tcga_all.h5ad; do
  [ -s data/raw/musl/$f ] || uv run python scripts/models/_common/fetch.py musl "https://zenodo.org/api/records/17098066/files/$f/content" $f
done
# 3. train on SLB train (human) + score the split
W=$M/_slb; mkdir -p $W
$PY $D/train_slb.py "$SPLIT" $W/${SPLIT}_scores_$(basename "$SLB_BENCH").csv 2>&1 | tee $W/train_${SPLIT}.log | grep -vE "it/s|Processing rows" || true
[ -s $W/${SPLIT}_scores_$(basename "$SLB_BENCH").csv ]
uv run python scripts/models/dekegel2021/finalize.py $W/${SPLIT}_scores_$(basename "$SLB_BENCH").csv musl__human $SPLIT nan
V=(musl__human)
if [ "${MUSL_ALLSPECIES:-1}" = 1 ] && ls data/interim/bundle/*/esm2.parquet >/dev/null 2>&1; then
  $PY $D/train_bundle.py "$SPLIT" $W/${SPLIT}_allspecies.csv 2>&1 | tee $W/train_bundle_${SPLIT}.log
  uv run python scripts/models/dekegel2021/finalize.py $W/${SPLIT}_allspecies.csv musl__allspecies $SPLIT nan
  V+=(musl__allspecies)
fi
if [ "$SPLIT" = dev ]; then  # slb.evaluate: results/models[/<bench>]/<name>_dev.{txt,json}
  for n in "${V[@]}"; do
    uv run python -c "import sys; sys.path.insert(0, 'scripts/models/_common'); import slb; slb.evaluate('$n')"
  done
fi
