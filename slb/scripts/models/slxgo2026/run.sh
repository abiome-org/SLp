#!/usr/bin/env bash
# SLxGO GO-PCA branch: authors' 50-dimensional BioBERT/PCA gene features and
# symmetric HistGradientBoosting classifier, retrained on SLB train labels.
set -euo pipefail
cd "$(dirname "$0")/../../.."
source scripts/models/_common/threads.sh
uv run python scripts/models/slxgo2026/run.py
