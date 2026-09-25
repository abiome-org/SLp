#!/usr/bin/env bash
# ESM4SL (Dai/Yang et al., EMBC 2025; github JieZheng-ShanghaiTech/ESM4SL @348b2bf) retrained on SLB train.
#   esm4sl__human      : ESM4SL proper = per-residue ESM-2 650M embeddings -> MLP_3D -> self/cross attention ->
#                        attention pooling -> MLP, with the CellCNN cell-line branch (DepMap 24Q4 tensor per
#                        context, cellfeat.py). One model pooled over all human SLB contexts.
#   esm4sl__allspecies : the paper's ESM-2 + MLP baseline (mean ESM-2 embeddings of both genes -> MLP), trained
#                        separately per species on that species' SLB train rows; every species of the benchmark
#                        that has sequences in data/interim/orthology_extra/fasta.
# Both use the original coach_pl training entry point + ClsModule/AttnModule/AttnWrap/MLP and the original
# config files (esm4sl/configuration/{attn,mlp}/new.yaml); overrides are passed on the command line below.
# Env: SLB_BENCH (default data/slb), SLB_SPLIT (default dev), ESM4SL_ATTN_EPOCHS (default 1), ESM4SL_MAXLEN (default 1000),
#      ESM4SL_VARIANTS (default "human allspecies"; human needs the GPU, allspecies runs on CPU).
# GPU: embedding (~60 min first time, cached/resumable) + attention training (~15 min/epoch on a 3090).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
export SLB_BENCH=${SLB_BENCH:-data/slb}
SPLIT=${SLB_SPLIT:-dev}
E=external/models/esm4sl; PY=$ROOT/$E/.venv/bin/python; D=$ROOT/scripts/models/esm4sl
W=$ROOT/$E/_slb/$(basename "$SLB_BENCH")/$SPLIT; mkdir -p "$W"
export ESM4SL_MAXLEN=${ESM4SL_MAXLEN:-1000}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 POLARS_MAX_THREADS=8

# 0. environment (uv venv; torch 2.5.1+cu121, pytorch-lightning 2.4.0, fair-esm 2.0.0)
if [ ! -x "$PY" ]; then
  uv venv -p 3.10 $E/.venv
  VIRTUAL_ENV=$E/.venv uv pip install "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu121
  VIRTUAL_ENV=$E/.venv uv pip install "numpy<2" pandas scikit-learn "pytorch-lightning==2.4.0" fair-esm fvcore \
    omegaconf easydict scipy tensorboard biopython pyarrow matplotlib seaborn transformers peft "rich==13.7.1"
fi
[ -s $E/_weights/esm2_t33_650M_UR50D.pt ] || { mkdir -p $E/_weights
  curl -fL -o $E/_weights/esm2_t33_650M_UR50D.pt https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
  curl -fL -o $E/_weights/esm2_t33_650M_UR50D-contact-regression.pt https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt; }

# 1. sequences -> ESM-2 embeddings (mean: data/interim/esm2_650m/<sp>.parquet, shared bundle; per-residue: human)
uv run python $D/seqs.py $E/_slb/seqs.tsv
ESM_BENCH_ONLY=${ESM_BENCH_ONLY:-1} $PY $D/embed.py $E/_slb/seqs.tsv data/interim/esm2_650m $E/_slb/perres human
CF=$ROOT/$E/_slb/$(basename "$SLB_BENCH")/cellfeat.npz
[ -s $CF ] || uv run python $D/cellfeat.py $CF

train() {  # <workdir> <config> <extra overrides...>  (ESM4SL_MAXLEN env truncates proteins in the attention model)
  local wd=$1 cfg=$2; shift 2
  rm -rf "$wd/out"
  (cd $E && $PY $D/train_slb.py --config-file $cfg --num-gpus 1 \
     OUTPUT_DIR "$wd/out" DATASET.TRAIN_FILE "$wd/train.csv" DATASET.VAL_FILE "$wd/val.csv" \
     DATASET.TEST_FILE "$wd/test.csv" DATALOADER.TRAIN.NUM_WORKERS 2 "$@" > "$wd/train.log" 2>&1)
  tail -3 "$wd/train.log" | cut -c1-200
}

VARIANTS=${ESM4SL_VARIANTS:-human allspecies}
# 2. esm4sl__human (attention + cell branch; GPU)
if [[ " $VARIANTS " == *" human "* ]]; then
uv run python $D/prep.py human $SPLIT $W/attn_human whole
rm -rf $W/attn_human/out
(cd $E && $PY $D/train_attn.py $W/attn_human $CF ${ESM4SL_ATTN_EPOCHS:-1} > $W/attn_human/train.log 2>&1)
tail -2 $W/attn_human/train.log
uv run python $D/collect.py esm4sl__human $SPLIT $W/attn_human
fi

# 3. esm4sl__allspecies (ESM-2 + MLP per species; CPU, 8 threads)
if [[ " $VARIANTS " == *" allspecies "* ]]; then
SPECIES=$(uv run python -c "
import sys; sys.path.insert(0,'scripts/models/_common'); import slb, os
sp = sorted(slb.load('train').species.unique())
print(' '.join(s for s in sp if os.path.exists(f'data/interim/esm2_650m/{s}.parquet')))")
DIRS=()
for sp in $SPECIES; do
  uv run python $D/prep.py $sp $SPLIT $W/mlp_$sp mean
  N=$(($(wc -l < $W/mlp_$sp/train.csv) - 1))
  # original: batch 16, 100 epochs. Scaled for SLB's larger species (lr is scaled by BS/16 inside ClsModule):
  BS=16; [ $N -gt 50000 ] && BS=64; [ $N -gt 500000 ] && BS=256
  EP=$(( 1500000 / (N + 1) )); [ $EP -gt 100 ] && EP=100; [ $EP -lt 5 ] && EP=5
  ESM4SL_CPU=1 CUDA_VISIBLE_DEVICES= train $W/mlp_$sp esm4sl/configuration/mlp/new.yaml DATASET.NAME SLBMeanDataset \
    DATASET.ESM_ROOT $W/mlp_$sp/emb.parquet DATALOADER.TRAIN.BATCH_SIZE $BS TRAINER.MAX_EPOCHS $EP
  DIRS+=($W/mlp_$sp)
done
uv run python $D/collect.py esm4sl__allspecies $SPLIT "${DIRS[@]}"
fi

if [ "$SPLIT" = dev ]; then
  for v in $VARIANTS; do
    uv run python -c "import sys; sys.path.insert(0, 'scripts/models/_common'); import slb; slb.evaluate('esm4sl__$v')"
  done
fi
