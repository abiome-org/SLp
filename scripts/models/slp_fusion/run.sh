#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
source scripts/models/_common/threads.sh
uv run python scripts/models/slp_fusion/run.py --variant core
uv run python scripts/models/slp_fusion/run.py --variant loss
