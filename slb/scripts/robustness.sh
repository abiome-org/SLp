# Rebuild the splits under 4 alternative salts and score the baselines on each test split.
# Then: uv run python scripts/robustness.py  (writes results/reports/robustness.md)
set -e
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 POLARS_MAX_THREADS=16
for salt in alt-a alt-b alt-c alt-d; do
  out=data/robustness/$salt
  SLB_SALT=$salt SLB_OUT=$out uv run python -m slbench.build --stage splits > /dev/null
  for b in random paralog_identity codependency fitness fitness_lgbm lgbm; do
    SLB_BENCH=$out uv run slbench baseline $b --split test --out $out/pred_$b.parquet > /dev/null
    SLB_BENCH=$out uv run slbench eval $out/pred_$b.parquet --split test --out $out/res_$b.json > /dev/null
  done
  echo "$salt done"
done
