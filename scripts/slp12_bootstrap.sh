#!/usr/bin/env bash
# Run inside the authorized pod. Account credentials are not required here.
set -euo pipefail
cd /workspace
if [[ ! -d SLp/.git ]]; then
  git clone https://github.com/abiome-org/SLp.git SLp
fi
cd SLp
git checkout --detach d668aaea44ee9d462092d3e618a80411d3f62dd2
mkdir -p modules/slp-1-2
cp /workspace/slp12-source/* modules/slp-1-2/
# Packages are reproducible from the lock; keep their many small files on local
# disk. Only datasets, source, logs and model artifacts need persistent storage.
python3 scripts/fetch_artifacts.py data > /workspace/slp12-ops/data-download.log 2>&1 &
slp_download_pid=$!
trap 'kill "$slp_download_pid" 2>/dev/null || true' EXIT
python3 -m venv /opt/slp12-env
/opt/slp12-env/bin/pip install --no-cache-dir --no-compile --require-hashes -r modules/slp-1-2/requirements-linux-cu128.lock
/opt/slp12-env/bin/python -m pip freeze > /workspace/slp12-ops/installed-requirements.txt
wait "$slp_download_pid"
/opt/slp12-env/bin/python modules/slp-1-2/prepare.py --root /workspace/SLp --output /workspace/slp12-ops/corpus
touch /workspace/slp12-ops/bootstrap-complete
