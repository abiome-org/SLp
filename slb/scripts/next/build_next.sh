#!/usr/bin/env bash
# Side-by-side SLB build with the candidate human sources: data/slb_next (data/slb is not touched).
set -euo pipefail
cd "$(dirname "$0")/../.."
uv run python scripts/next/stage_measurements.py "$@"
export SLB_OUT=data/slb_next SLB_WORK=data/interim/next SLB_EXTRA_MEASUREMENTS=data/interim/measurements_next
uv run python -m slbench.build --stage examples
uv run python -m slbench.build --stage splits
