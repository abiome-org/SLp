#!/usr/bin/env bash
# DiscoverSL (Das et al. 2019) on SLB. Human only (TCGA). Needs docker image slb/r-stats
# (external/models/_rdocker/Dockerfile) and data/raw/tcga_pancan + data/raw/discoversl.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
source scripts/models/_common/threads.sh
export SLB_BENCH=${SLB_BENCH:-data/bench/slb1.2}
SPLIT=${SLB_SPLIT:-${1:-dev}}
docker image inspect slb/r-stats >/dev/null 2>&1 || docker build -q -t slb/r-stats external/models/_rdocker
[ -s external/models/discoversl/export/model.rds ] || docker run --rm --user "$(id -u):$(id -g)" -v "$ROOT/external/models/discoversl":/d slb/r-stats \
  Rscript -e 'e<-new.env(); load("/d/pkg/DiscoverSL/R/sysdata.rda", envir=e); write.table(e$genesets, "/d/export/genesets.tsv", sep="\t", quote=F, row.names=F, col.names=F); saveRDS(e$model, "/d/export/model.rds")'
uv run python scripts/models/discoversl/run.py "$SPLIT"
