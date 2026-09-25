#!/usr/bin/env bash
# Score <results dir>/<name>_<split>.parquet with the SLB evaluator.
# Examples the model does not score (non-human species for human-only models, genes without features) are given the
# median of the model's own scores -- exactly what `slbench eval --allow-missing` does; the fill is done here because
# the evaluator refuses files that miss more than half of the examples (a version-mismatch guard) and human-only
# models always miss ~85-90% of rows. The filled copy is temporary; the stored prediction file is untouched.
# Usage: eval.sh <name>   env: SLB_BENCH (default data/slb), SLB_SPLIT (default dev; test is refused here)
# Results dir: results/models/<bench name>/.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
SPLIT=${SLB_SPLIT:-dev}
export SLB_BENCH=${SLB_BENCH:-data/slb}
D=results/models/$(basename "$SLB_BENCH")
[ "$SPLIT" = "test" ] && { echo "refusing to evaluate on test"; exit 1; }
TMP=$(mktemp /tmp/slbg_eval_XXXXXX.parquet)
uv run python - "$D/$1_${SPLIT}.parquet" "$SLB_BENCH" "$SPLIT" "$TMP" <<'PY'
import sys, pandas as pd
p, bench, split, out = sys.argv[1:]
d = pd.read_parquet(p)
ids = pd.read_parquet(f"{bench}/{split}_inputs.parquet", columns=["example_id"]).example_id
miss = ~ids.isin(d.example_id)
fill = pd.DataFrame({"example_id": ids[miss].values, "score": d.score.median()})
pd.concat([d[["example_id", "score"]], fill]).to_parquet(out, index=False)
print(f"{len(d)} scored, {int(miss.sum())} of {len(ids)} {split} examples filled with the model's median score", file=sys.stderr)
PY
uv run slbench eval "$TMP" --split "$SPLIT" --out "$D/$1_${SPLIT}.json" > "$D/$1_${SPLIT}.txt" 2>&1 || { cat "$D/$1_${SPLIT}.txt"; rm -f "$TMP"; exit 1; }
rm -f "$TMP"
grep -E "SLB score|sapiens|missing" "$D/$1_${SPLIT}.txt"
