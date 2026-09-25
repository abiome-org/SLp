#!/usr/bin/env bash
# DCell-style VNN (Ma et al. 2018) retrained on SLB train, per species with a GO bundle.
# Env: SLB_BENCH, SLB_SPLIT (default dev), DCELL_DEVICE (default cuda if available; post gpu:claim first).
# Variants: faithful (per-gene inputs, as published) and tied (ontotype-style gene-agnostic inputs).
# Writes results/models/dcell__{faithful,tied}_<split>.parquet
set -euo pipefail
cd "$(dirname "$0")/../../.."
source scripts/models/_common/threads.sh

PY=external/models/dcell/.venv/bin/python
DEV="${DCELL_DEVICE:-$($PY -c 'import torch;print("cuda" if torch.cuda.is_available() else "cpu")')}"
TMP=$(mktemp -d)
SPECIES=$($PY -c "
import sys; sys.path.insert(0,'scripts/models/_common'); import bundle_io as B
ev=B.load_split(); tr=B.load_train()
print(' '.join(s for s in ev.species.unique() if B.has_bundle(s,'go') and (tr.species==s).any()))")
for variant in faithful tied; do
  flag=""; [ $variant = tied ] && flag="--tied"
  for s in $SPECIES; do
    $PY scripts/models/dcell/dcell.py --species "$s" --split "$SLB_SPLIT" --device "$DEV" --epochs 15 --patience 4 \
        --bs 8192 --lr 2e-3 $flag --out "$TMP/${variant}_$s.parquet"
  done
  $PY -c "
import glob, pandas as pd, sys; sys.path.insert(0,'scripts/models/_common'); import bundle_io as B
ev=B.load_split(columns=['example_id'])
d=pd.concat([pd.read_parquet(f) for f in glob.glob('$TMP/${variant}_*.parquet')])
out=ev.merge(d, on='example_id', how='left'); p=B.out_path('dcell', variant='$variant'); out.to_parquet(p)
print('wrote', p, out.score.notna().sum(), '/', len(out))"
  if [ "$SLB_SPLIT" = dev ]; then
    uv run slbench eval "$SLB_OUT/dcell__${variant}_dev.parquet" --split dev --allow-missing > "$SLB_OUT/dcell__${variant}_dev.txt"
  fi
done
rm -rf "$TMP"
