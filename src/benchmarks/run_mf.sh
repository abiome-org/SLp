#!/usr/bin/env bash
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate slbench
python -m pip install -U 'numpy<2'
export MODELS="${MODELS:-GRSMF SL2MF CMFW}"
bash "$(dirname "$0")/run_feng.sh"
