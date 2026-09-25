# Source from every run.sh: cap thread pools (machine is shared by several agents; lead order 2026-09-24).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-8} POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-8} RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-8}
# Benchmark + output dir: results/models/<bench name>/.
export SLB_BENCH="${SLB_BENCH:-data/slb}" SLB_SPLIT="${SLB_SPLIT:-dev}"
_b=$(basename "$SLB_BENCH")
export SLB_OUT="results/models/$_b"
mkdir -p "$SLB_OUT"
