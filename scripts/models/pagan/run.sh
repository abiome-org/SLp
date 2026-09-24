#!/usr/bin/env bash
# PAGAN genes-to-pairs: single-gene essentiality -> unsupervised pair nodes.
set -euo pipefail
cd "$(dirname "$0")/../../.."
source scripts/models/_common/threads.sh
PY=external/models/synleaf/.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "Build the PyTorch/PyG environment with scripts/models/synleaf/run.sh first" >&2
  exit 1
fi
"$PY" scripts/models/pagan/run.py
