#!/usr/bin/env bash
# SynLeaF (Xing et al. 2026, arXiv 2603.22369) retrained on SLB train labels, pan-cancer setting.
#   synleaf__human : full dual-stage model (umt): stage 1 = omics teacher (cross-VAE PoE over TCGA
#                    cna/exp/mut) + KG teacher (RGCN over SynLethKG 2.0 minus SL/nonSL/SR relations);
#                    stage 2 = distilled joint model. Human only (TCGA omics + human KG).
#   synleaf__allspecies : KG/RGCN branch only (task only_kg, same code/hyper-parameters), one model per species
#                    with SLB train rows, on a KG built from the shared bundle (GO + GO hierarchy + BioGRID
#                    physical + STRING non-experimental channels; data/interim/bundle/<species>). Set
#                    SYNLEAF_ALLSPECIES=0 to skip. Large species' training rows are subsampled to 300k.
# Env: SLB_BENCH (default data/bench/slb1.2), SLB_SPLIT (default dev). Uses the GPU if visible
# (claim it on the board first); the omics teacher is cheap enough for CPU.
# Stage checkpoints are cached per benchmark dir (external/models/synleaf/result/<ct>_*); delete to retrain.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
export SLB_BENCH=${SLB_BENCH:-data/bench/slb1.2}
SPLIT=${SLB_SPLIT:-${1:-dev}}
M=external/models/synleaf; D=scripts/models/synleaf; PY=$ROOT/$M/.venv/bin/python
export SYNLEAF_CT=${SYNLEAF_CT:-slb_$(basename "$SLB_BENCH")}
CT=$SYNLEAF_CT
EPOCHS=${SYNLEAF_EPOCHS:-50}; PATIENCE=${SYNLEAF_PATIENCE:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8} NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-8} POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-8}

# 1. environment (versions: python 3.11, torch 2.4.0+cu124, torch_geometric 2.8.0, accelerate 0.34.2)
if [ ! -x "$PY" ]; then
  (cd $M && uv venv -p 3.11 .venv -q && VIRTUAL_ENV=.venv uv pip install -q torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124 \
    && VIRTUAL_ENV=.venv uv pip install -q accelerate==0.34.2 pandas matplotlib tqdm scikit-learn biopython torch_geometric pyarrow "numpy<2.1")
fi
# 2. authors' packaged raw data (Google Drive, linked from the repo README); only KG, UniProt, pan TCGA are used
R=data/raw/synleaf
if [ ! -s $R/extracted/data_raw/SLKG2/raw_kg.tsv ]; then
  uv run python scripts/models/_common/fetch.py synleaf \
    "https://drive.usercontent.google.com/download?id=1IFvkEcOWdfmlkg60hB5hZr3VXT5EQJrB&export=download&confirm=t" data_raw.tar.gz
  mkdir -p $R/extracted && tar -xzf $R/data_raw.tar.gz -C $R/extracted data_raw/SLKG2 data_raw/uniprot data_raw/TCGA/pan
fi
# 3. model inputs (gene set, KG w/o SL relations, omics matrices, SLB train folds) + eval pairs
$PY $D/prep.py "$SPLIT"
# 4. training (train.py of the repo, unchanged; single process via accelerate's default config)
cd $M/src
train() {  # task folder [extra args]
  local t=$1 f=$2; shift 2
  [ -s ../result/$f/checkpoint.pth ] && return 0
  $PY train.py --cancer_type $CT --task_type $t --epochs $EPOCHS --patience $PATIENCE --omics_types cna exp mut \
    --specify_result_saving_folder $f "$@" > ../result/$f.log 2>&1
}
mkdir -p ../result
# omics teacher: repo defaults (batch 384) but <= 50 epochs, patience 8 (CPU is fine).
# KG teacher and distillation stage: batch 2048 and few epochs to bound GPU time (the RGCN runs on a 2-hop
# subgraph of the 3.3M-edge KG per batch; ~2.3-3.2 min/epoch at batch 2048 on an RTX 3090).
train only_omics ${CT}_only_omics
train only_kg ${CT}_only_kg --batch_size 2048 --epochs ${SYNLEAF_KG_EPOCHS:-6} --patience 2
train umt ${CT}_umt --batch_size 2048 --epochs ${SYNLEAF_UMT_EPOCHS:-8} --patience 3 \
  --omics_ckpt_path ../result/${CT}_only_omics/checkpoint.pth --kg_ckpt_path ../result/${CT}_only_kg/checkpoint.pth
# 5. predict + write results
W=$ROOT/$M/_slb; mkdir -p $W
$PY $ROOT/$D/predict.py $CT ../result/${CT}_umt $SPLIT $W/${SPLIT}_umt.csv
cd "$ROOT"
uv run python scripts/models/dekegel2021/finalize.py $W/${SPLIT}_umt.csv synleaf__human $SPLIT nan

# 6. allspecies variant (KG branch on bundle KGs)
V=(synleaf__human)
if [ "${SYNLEAF_ALLSPECIES:-1}" = 1 ]; then
  export SYNLEAF_CT_PREFIX=${SYNLEAF_CT_PREFIX:-bundle_$(basename "$SLB_BENCH")}
  SPECIES=$($PY $D/prep_bundle.py "$SPLIT")
  cd $M/src
  for sp in $SPECIES; do
    ct=${SYNLEAF_CT_PREFIX}_$sp
    train only_kg ${ct}_only_kg --cancer_type $ct --omics_types none --batch_size 1024 --epochs ${SYNLEAF_KG_EPOCHS:-8} --patience 3
    $PY $ROOT/$D/predict.py $ct ../result/${ct}_only_kg $SPLIT $W/${SPLIT}_allspecies_$sp.csv
  done
  cd "$ROOT"
  uv run python -c "import pandas as pd,sys; pd.concat([pd.read_csv(f) for f in sys.argv[2:]]).to_csv(sys.argv[1],index=False)" \
    $W/${SPLIT}_allspecies.csv $(for sp in $SPECIES; do echo $W/${SPLIT}_allspecies_$sp.csv; done)
  uv run python scripts/models/dekegel2021/finalize.py $W/${SPLIT}_allspecies.csv synleaf__allspecies $SPLIT nan
  V+=(synleaf__allspecies)
fi
if [ "$SPLIT" = dev ]; then  # slb.evaluate: results/models[/<bench>]/<name>_dev.{txt,json}
  for n in "${V[@]}"; do
    uv run python -c "import sys; sys.path.insert(0, 'scripts/models/_common'); import slb; slb.evaluate('$n')"
  done
fi
