#!/usr/bin/env bash
# De Kegel et al. 2021 paralog-SL random forest on SLB.
#   dekegel2021__pretrained : released RF_model.pickle (trained on DepMap-derived paralog SL labels; no
#                            combinatorial-screen labels) applied to Ens111 paralog features.
#   dekegel2021__slbtrain   : same features + hyper-parameters, refit on SLB train labels only.
# Human pairs without Ensembl paralog features get score 0 (the model's prior: not a paralog pair);
# other species are left missing (median-filled by eval).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
source scripts/models/_common/threads.sh
export SLB_BENCH=${SLB_BENCH:-data/slb}
SPLIT=${SLB_SPLIT:-${1:-dev}}
W=external/models/dekegel_paralog_sl/_slb/$(basename "$SLB_BENCH"); mkdir -p $W
D=scripts/models/dekegel2021
docker image inspect slb/dekegel2021 >/dev/null 2>&1 || \
  docker build -q -t slb/dekegel2021 -f external/models/dekegel_paralog_sl/Dockerfile.slb external/models/dekegel_paralog_sl
[ -s $W/train_feat.csv ] || uv run python $D/features.py train $W/train_feat.csv
uv run python $D/features.py $SPLIT $W/${SPLIT}_feat.csv
docker run --rm --user "$(id -u):$(id -g)" -v "$ROOT/external/models/dekegel_paralog_sl":/model -v "$ROOT/$W":/w -v "$ROOT/$D":/code \
  slb/dekegel2021 python /code/score_pretrained.py /w/${SPLIT}_feat.csv /w/${SPLIT}_pretrained.csv
(cd $D && uv run --project "$ROOT" python retrain.py "$ROOT/$W/train_feat.csv" "$ROOT/$W/${SPLIT}_feat.csv" "$ROOT/$W/${SPLIT}_slbtrain.csv")
uv run python $D/finalize.py $W/${SPLIT}_pretrained.csv dekegel2021__pretrained $SPLIT 0
uv run python $D/finalize.py $W/${SPLIT}_slbtrain.csv dekegel2021__slbtrain $SPLIT 0
for n in dekegel2021__pretrained dekegel2021__slbtrain; do
  uv run python -c "import sys; sys.path.insert(0, 'scripts/models/_common'); import slb; slb.evaluate('$n', '$SPLIT')"
done
# all-species variant (bundle features; one RF per species)
uv run python scripts/models/dekegel2021/allspecies.py "$SPLIT"
