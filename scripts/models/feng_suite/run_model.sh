#!/usr/bin/env bash
# Run one model of the Feng et al. 2024 SL_benchmark code base (patched copy, see slb.patch) on SLB.
# Usage: run_model.sh <FENG_MODEL_NAME> <out_name> [variant]
#   FENG_MODEL_NAME in SL2MF GRSMF CMFW DDGCN GCATSL SLMGAE KG4SL SLGNN NSF4SL PTGNN PiLSL MGE4SL
#   variant: human (default; faithful inputs: SynLethKG w/o SL, gene2go GO sim, BioGRID physical PPI; human rows only)
#            allspecies (per-species inputs from data/interim/bundle/<sp>/; every species in the benchmark;
#                        only for models that need no KG: SL2MF GRSMF CMFW DDGCN GCATSL SLMGAE)
# Output: results/models/<out_name>_<split>.parquet (human) or results/models/<out_name>__allspecies_<split>.parquet
# env: SLB_BENCH (default data/bench/slb1.2), SLB_SPLIT (default dev), SLB_THREADS (default 8),
#      SLB_DOCKER_IMAGE (default slb-feng:cpu; slb-feng:gpu + SLB_GPU=1 for the GPU image; claim the GPU first)
set -euo pipefail
# run from a private copy so that editing this file never corrupts a running job (bash reads scripts incrementally)
if [ -z "${_SLB_RUNCOPY:-}" ]; then
  export SLB_ROOT=${SLB_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
  _t=$(mktemp /tmp/slb_run_model.XXXXXX.sh); cp "$0" "$_t"; _SLB_RUNCOPY=1 exec bash "$_t" "$@"
fi
MODEL=$1; NAME=$2; VARIANT=${3:-human}
ROOT=$SLB_ROOT
export SLB_BENCH=${SLB_BENCH:-data/bench/slb1.2}
export SLB_SPLIT=${SLB_SPLIT:-dev}
BN=$(basename "$SLB_BENCH")
RES=$ROOT/results/models; [ "$BN" = "slb1.2" ] || RES=$ROOT/results/models/$BN; mkdir -p "$RES"
export SLB_WORK=${SLB_WORK:-$ROOT/external/models/_slb_work/$BN}
WORK=$SLB_WORK
THREADS=${SLB_THREADS:-8}; export SLB_THREADS=$THREADS
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS POLARS_MAX_THREADS=$THREADS NUMBA_NUM_THREADS=$THREADS
IMG=${SLB_DOCKER_IMAGE:-slb-feng:cpu}
PREP=$ROOT/external/models/SL_benchmark/.venv-prep/bin/python
cd "$ROOT"
[ -x "$PREP" ] || (cd external/models/SL_benchmark && uv venv -q -p 3.11 .venv-prep && VIRTUAL_ENV=.venv-prep uv pip install -q numpy pandas pyarrow scipy numba scikit-learn)
docker image inspect "$IMG" > /dev/null 2>&1 || docker build -t slb-feng:cpu -f scripts/models/feng_suite/Dockerfile scripts/models/feng_suite

run_one() {  # $1 = species, $2 = feng dir (contains data/), $3 = out parquet
  local SP=$1 FD=$2 OUTP=$3
  local CL=slb; [ "${SLB_BALANCE:-0}" = "1" ] && CL=slbbal
  # split pickle rebuilt every run (cheap); SLB_BALANCE=1 subsamples fit negatives 1:1 (KG4SL's original protocol)
  docker run --rm -u "$(id -u):$(id -g)" -e SLB_BALANCE="${SLB_BALANCE:-0}" -v "$FD":/work -v "$ROOT/scripts/models/feng_suite":/s "$IMG" python /s/make_pkl.py
  local N; N=$("$PREP" -c "import pandas as pd; print(len(pd.read_parquet('$FD/data/universe.parquet')))")
  mkdir -p "$FD/results/Random_score_mats" "$FD/results/score_dist" "$FD/logs"
  exec 9>"$FD/logs/${NAME}.lock"; flock -n 9 || { echo "another $NAME run holds $FD/logs/${NAME}.lock"; return 1; }
  local GPUARG=""; [ "${SLB_GPU:-0}" = "1" ] && GPUARG="--gpus all"
  rm -rf "$FD/src_$NAME"; cp -r external/models/SL_benchmark/src "$FD/src_$NAME"
  local lm; lm=$(echo "$MODEL" | tr 'A-Z' 'a-z')
  rm -rf "$FD/results/Random_score_mats/$lm"   # never extract a stale matrix from an earlier run
  mkdir -p "$FD/data/preprocessed_data/gcatsl_data"; [ "$MODEL" = "GCATSL" ] && rm -f "$FD/data/preprocessed_data/gcatsl_data/"*.txt
  local LOG="$FD/logs/${NAME}.log"
  echo "[$(date)] $MODEL species=$SP N=$N threads=$THREADS img=$IMG" | tee "$LOG"
  local start; start=$(date +%s)
  docker rm -f "slbg_${NAME}_${SP}" > /dev/null 2>&1 || true
  docker run --rm --name "slbg_${NAME}_${SP}" $GPUARG --cpus "$THREADS" -u "$(id -u):$(id -g)" -e HOME=/tmp -e WANDB_MODE=disabled -e SLB_NUM_NODE="$N" -e SLB_KFOLD=1 \
    -e TF_FORCE_GPU_ALLOW_GROWTH=true -e SLB_THREADS="$THREADS" -e OMP_NUM_THREADS="$THREADS" -e OPENBLAS_NUM_THREADS="$THREADS" -e NUMEXPR_NUM_THREADS="$THREADS" -e MKL_NUM_THREADS="$THREADS" -e TF_NUM_INTRAOP_THREADS="$THREADS" -e NUMBA_NUM_THREADS="$THREADS" \
    -v "$FD":/work -w "/work/src_$NAME" "$IMG" \
    python -u main.py -m "$MODEL" -ns Random -ds CV1 -pn 50 --indep_test --cell_line "$CL" --save_mat --output_name slb >> "$LOG" 2>&1 || true
  if [ "$MODEL" = "GCATSL" ] && [ ! -f "$FD/results/Random_score_mats/$lm/${lm}_fold_0_pos_neg_0.02_CV1_Random_classify.npy" ]; then
    # released train_gcatsl.py only builds its random-walk global matrix on the first call and skips training
    docker run --rm --name "slbg_${NAME}_${SP}" $GPUARG --cpus "$THREADS" -u "$(id -u):$(id -g)" -e HOME=/tmp -e WANDB_MODE=disabled -e SLB_NUM_NODE="$N" -e SLB_KFOLD=1 \
      -e TF_FORCE_GPU_ALLOW_GROWTH=true -e SLB_THREADS="$THREADS" -e OMP_NUM_THREADS="$THREADS" -e OPENBLAS_NUM_THREADS="$THREADS" -e NUMEXPR_NUM_THREADS="$THREADS" -e MKL_NUM_THREADS="$THREADS" -e NUMBA_NUM_THREADS="$THREADS" \
      -v "$FD":/work -w "/work/src_$NAME" "$IMG" \
      python -u main.py -m "$MODEL" -ns Random -ds CV1 -pn 50 --indep_test --cell_line "$CL" --save_mat --output_name slb >> "$LOG" 2>&1 || true
  fi
  echo "runtime_sec $(( $(date +%s) - start ))" | tee -a "$LOG"
  local MAT="$FD/results/Random_score_mats/$lm/${lm}_fold_0_pos_neg_0.02_CV1_Random_classify.npy"
  [ -f "$MAT" ] || MAT="$FD/results/Random_score_mats/$lm/${lm}_fold_0_pos_neg_0.02_CV1_Random_classify.npz.npy"  # GCATSL's naming
  SLB_SPECIES=$SP "$PREP" scripts/models/feng_suite/extract_scores.py "$MAT" "$FD/data/slb_gene_index.parquet" "$OUTP"
}

if [ "$VARIANT" = "human" ]; then
  [ -f "$WORK/human_train_pairs.parquet" ] || uv run python scripts/models/_common/slb_pairs.py
  [ -f "$WORK/feng/data/preprocessed_data/final_gosim_mf_from_r_9845.npy" ] || "$PREP" scripts/models/feng_suite/build_inputs.py
  case $MODEL in
    NSF4SL) [ -f "$WORK/feng/data/preprocessed_data/kg_TransE_l2_entity.npy" ] || bash scripts/models/feng_suite/build_transe.sh ;;
    SLGNN)  "$PREP" scripts/models/feng_suite/make_score_pairs.py ;;
    PTGNN)  [ -f "$WORK/feng/data/preprocessed_data/ptgnn_data/ptgnn_encod_by_word_sl_9845_800.npy" ] || "$PREP" scripts/models/feng_suite/build_ptgnn_inputs.py ;;
    PiLSL)  if [ ! -f "$WORK/feng/data/preprocessed_data/pilsl_data/pilsl_random_feature.npy" ]; then
              "$PREP" scripts/models/feng_suite/build_pilsl_inputs.py
              docker run --rm -u "$(id -u):$(id -g)" -v "$WORK/feng":/work -v "$ROOT/scripts/models/feng_suite":/s "$IMG" python /s/make_pilsl_feat.py
            fi ;;
  esac
  run_one human "$WORK/feng" "$RES/${NAME}_${SLB_SPLIT}.parquet"
else
  SPECIES=$("$PREP" -c "import pandas as pd; print(' '.join(sorted(pd.read_parquet('$ROOT/$SLB_BENCH/train.parquet' if not '$SLB_BENCH'.startswith('/') else '$SLB_BENCH/train.parquet', columns=['species']).species.unique())))")
  parts=()
  for SP in $SPECIES; do
    [ -f "$WORK/${SP}_train_pairs.parquet" ] || SLB_SPECIES=$SP uv run python scripts/models/_common/slb_pairs.py
    [ -f "$WORK/feng_$SP/data/preprocessed_data/final_gosim_mf_from_r_9845.npy" ] || SLB_SPECIES=$SP "$PREP" scripts/models/feng_suite/build_inputs_bundle.py
    P="$WORK/feng_$SP/${NAME}__allspecies_${SLB_SPLIT}.parquet"
    run_one "$SP" "$WORK/feng_$SP" "$P" || echo "WARNING: $MODEL failed for $SP"
    [ -f "$P" ] && parts+=("$P")
  done
  "$PREP" -c "import pandas as pd,sys; d=pd.concat([pd.read_parquet(p) for p in sys.argv[2:]]); d.to_parquet(sys.argv[1], index=False); print('wrote', sys.argv[1], len(d))" \
    "$RES/${NAME}__allspecies_${SLB_SPLIT}.parquet" "${parts[@]}"
fi
