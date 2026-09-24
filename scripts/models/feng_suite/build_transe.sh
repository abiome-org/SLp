#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$ROOT/external/models/MVGCNiSL/.venv/bin/python   # torch 2.4 env (any modern torch works)
[ -x "$PY" ] || { (cd "$ROOT/external/models/SL_benchmark" && uv venv -q -p 3.11 .venv-torch && VIRTUAL_ENV=.venv-torch uv pip install -q torch pandas numpy); PY=$ROOT/external/models/SL_benchmark/.venv-torch/bin/python; }
"$PY" "$ROOT/scripts/models/feng_suite/transe.py"
