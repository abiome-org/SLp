#!/usr/bin/env bash
# Cilantro-SL (Hu et al., bioRxiv 2026, doi 10.64898/2026.02.25.708096; github kaileyhh/Cilantro-SL @c674b55)
# retrained on SLB train.  Output: results/models/cilantro_sl_<split>.parquet (example_id, score).
#   stage 0 (GPU): Geneformer gf-12L-30M-i2048 in-silico deletion -> 512-d delta cell embedding per
#                  (SLB human context, SLB gene)                          isp_embed.py
#   stage 1: FiLM viability pretraining on DepMap CRISPR gene effect (single-gene data only) with Gene2vec
#   stage 2: pair classifier (SLNet) on SLB train + 5-fold ensemble + Mondrian conformal p-values  train_slb.py
# Human-only by construction (needs DepMap cell-line RNA-seq + CRISPR gene effect): non-human rows are left
# missing; human rows whose (context, gene) has no Geneformer delta (gene not expressed / outside the 2048-token
# window / context without DepMap data) are also missing (eval median-fills).
# Env: SLB_BENCH (default data/bench/slb1.3), SLB_SPLIT (default dev).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
export SLB_BENCH=${SLB_BENCH:-data/bench/slb1.3}
SPLIT=${SLB_SPLIT:-dev}
C=external/models/cilantro_sl; PY=$ROOT/$C/.venv/bin/python; D=$ROOT/scripts/models/cilantro_sl
W=$ROOT/$C/_slb/$(basename "$SLB_BENCH"); mkdir -p "$W"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 POLARS_MAX_THREADS=8

# 0. environment: uv venv (python 3.11, torch 2.5.1+cu121, transformers 4.46.3) + kaileyhh/geneformer fork
if [ ! -x "$PY" ]; then
  [ -d $C/geneformer_fork ] || git clone https://github.com/kaileyhh/geneformer $C/geneformer_fork
  uv venv -p 3.11 $C/.venv
  VIRTUAL_ENV=$C/.venv uv pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
  VIRTUAL_ENV=$C/.venv uv pip install "transformers==4.46.3" "numpy<2" "pandas>=2" datasets anndata scanpy loompy \
    tdigest pandarallel tables pyarrow scikit-learn scipy peft optuna optuna-integration statsmodels hyperopt \
    seaborn matplotlib bitsandbytes tensorboard accelerate gdown
  VIRTUAL_ENV=$C/.venv uv pip install --no-deps -e $C/geneformer_fork
fi
# Geneformer V1 12L model + gc30M dictionaries from the last HF revision that still ships them
R=https://huggingface.co/ctheodoris/Geneformer/resolve/01d3ea8993c2
G=$C/gf_weights; mkdir -p $G/gf-12L-30M-i2048 $G/dicts
for f in config.json pytorch_model.bin; do
  [ -s $G/gf-12L-30M-i2048/$f ] || curl -fL -o $G/gf-12L-30M-i2048/$f $R/gf-12L-30M-i2048/$f; done
for f in token_dictionary_gc30M.pkl gene_median_dictionary_gc30M.pkl; do
  [ -s $G/dicts/$f ] || curl -fL -o $G/dicts/$f $R/geneformer/gene_dictionaries_30m/$f; done
# Gene2vec (128-d) as distributed in the Cilantro-SL data folder
[ -s data/raw/cilantro_sl/gene2vec_embs.pt ] || { mkdir -p data/raw/cilantro_sl
  $C/.venv/bin/gdown 1vx4vZYieTIm96F7BiVyju5KU6f53dkne -O data/raw/cilantro_sl/gene2vec_embs.pt; }

# stage 0 (cached per benchmark version; ~15-25 min on a 3090)
[ -s $W/deltas.parquet ] || $PY $D/isp_embed.py $W/deltas.parquet 2> $W/isp.log
# stages 1-2
$PY $D/train_slb.py $W/deltas.parquet $SPLIT $W 2> $W/train_$SPLIT.log
tail -8 $W/train_$SPLIT.log
uv run python - <<EOF
import sys; sys.path.insert(0, 'scripts/models/_common'); import slb, pandas as pd
d = slb.load('$SPLIT')[['example_id', 'species']]
s = d.merge(pd.read_parquet('$W/${SPLIT}_scores.parquet')[['example_id', 'score']], on='example_id', how='left')
slb.write(s, 'cilantro_sl', '$SPLIT')
if '$SPLIT' == 'dev':
    slb.evaluate('cilantro_sl')
EOF
