#!/usr/bin/env bash
# SBSL (Seale, Tepeli, Goncalves, Bioinformatics 2022) retrained on SLB train labels only.
#   sbsl__en : SBSL molecular pair features + caret elastic net (SBSL-EN)
#   sbsl__rf : same features + caret random forest
# Human-only (DepMap / TCGA / GTEx features); other species left missing (median-filled by eval).
# env: SLB_BENCH (default data/slb), SLB_SPLIT (default dev), SBSL_CPUS (default 8).
# Needs the ELISL adapter's python env + omics caches (scripts/models/elisl, external/models/elisl/.venv).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
export SLB_BENCH=${SLB_BENCH:-data/slb}
SPLIT=${SLB_SPLIT:-${1:-dev}}
export SBSL_CPUS=${SBSL_CPUS:-8}
export OMP_NUM_THREADS=$SBSL_CPUS OPENBLAS_NUM_THREADS=$SBSL_CPUS MKL_NUM_THREADS=$SBSL_CPUS NUMEXPR_NUM_THREADS=$SBSL_CPUS ELISL_CPUS=$SBSL_CPUS
M=external/models/sbsl
D=scripts/models/sbsl
PY=$ROOT/external/models/elisl/.venv/bin/python
[ -x "$PY" ] || { uv venv --python 3.8 external/models/elisl/.venv; VIRTUAL_ENV=external/models/elisl/.venv uv pip install -r scripts/models/elisl/requirements.lock.txt; }
docker image inspect slb/sbsl >/dev/null 2>&1 || docker build -t slb/sbsl $M/slb_docker
F="uv run python scripts/models/_common/fetch.py"
for f in c2.cp.kegg.v7.0.symbols.gmt c2.cp.reactome.v7.0.symbols.gmt c2.cp.pid.v7.0.symbols.gmt; do
  $F msigdb_v7 https://data.broadinstitute.org/gsea-msigdb/msigdb/release/7.0/$f >/dev/null
done
# other raw inputs are the ELISL adapter's (see scripts/models/elisl/run.sh) + data/raw/demeter2/D2_gene_effect.csv

for S in train "$SPLIT"; do
  $PY $D/features.py $S
  $PY $D/discover_run.py $S
  $PY $D/prep_csv.py csv $S
done
B=$(basename $SLB_BENCH)
docker run --rm --cpus "$SBSL_CPUS" -v "$ROOT/$M/_slb/feat/$B":/w -v "$ROOT/$D":/code:ro slb/sbsl \
  Rscript /code/train.R /w/train.csv /w/$SPLIT.csv /w/pred_$SPLIT | tee $M/_slb/train_${B}_$SPLIT.log
for v in en rf; do
  # shared writer: constant fill for species SBSL cannot score, .coverage.json, dev evaluation
  uv run python scripts/models/elisl/finalize.py $M/_slb/feat/$B/pred_${SPLIT}_$v.csv sbsl__$v "$SPLIT"
done
